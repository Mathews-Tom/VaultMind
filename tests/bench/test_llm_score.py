"""Tests for LLM cite-or-decline scoring — prompt building, deterministic
response parsing, and scoring, all against a fake LLMClient (zero network)."""

from __future__ import annotations

from typing import Any

from vaultmind.bench.golden import GoldenQuestion
from vaultmind.bench.llm_score import (
    LLMAnswer,
    build_prompt,
    make_decline_scorer,
    parse_llm_answer,
    score_decline,
)
from vaultmind.llm.client import LLMResponse, Message


class FakeLLMClient:
    """Minimal LLMClient double returning a canned response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[list[Message], str, int]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, model, max_tokens))
        return LLMResponse(text=self.response_text, model=model, usage={})

    def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


def _q(answerable: bool, expected: tuple[str, ...] = ("a.md",)) -> GoldenQuestion:
    return GoldenQuestion(
        id="q1",
        question="Where is note A?",
        answerable=answerable,
        expected_notes=expected if answerable else (),
    )


def _hit(note_path: str) -> dict[str, Any]:
    return {"metadata": {"note_path": note_path}, "content": "some content"}


class TestBuildPrompt:
    def test_includes_question_and_context(self) -> None:
        prompt = build_prompt("Where is A?", [_hit("a.md")])
        assert "Where is A?" in prompt
        assert "a.md" in prompt

    def test_no_hits_notes_empty_context(self) -> None:
        prompt = build_prompt("Where is A?", [])
        assert "no notes retrieved" in prompt

    def test_truncates_long_content(self) -> None:
        long_hit = {"metadata": {"note_path": "a.md"}, "content": "x" * 5000}
        prompt = build_prompt("Q", [long_hit])
        assert len(prompt) < 5000


class TestParseLLMAnswer:
    def test_decline_only_response(self) -> None:
        assert parse_llm_answer("DECLINE") == LLMAnswer(declined=True, cited_paths=())

    def test_decline_case_insensitive(self) -> None:
        assert parse_llm_answer("decline") == LLMAnswer(declined=True, cited_paths=())

    def test_cited_single_path(self) -> None:
        answer = parse_llm_answer("The answer is X.\nCITED: a.md")
        assert answer.declined is False
        assert answer.cited_paths == ("a.md",)

    def test_cited_multiple_paths(self) -> None:
        answer = parse_llm_answer("Answer.\nCITED: a.md, b.md")
        assert answer.cited_paths == ("a.md", "b.md")

    def test_empty_response_declines(self) -> None:
        assert parse_llm_answer("").declined is True

    def test_whitespace_only_response_declines(self) -> None:
        assert parse_llm_answer("   \n  ").declined is True

    def test_answer_without_cited_or_decline_marker(self) -> None:
        answer = parse_llm_answer("Just an answer with no marker.")
        assert answer.declined is False
        assert answer.cited_paths == ()

    def test_cited_prefix_case_insensitive(self) -> None:
        answer = parse_llm_answer("Answer.\ncited: a.md")
        assert answer.cited_paths == ("a.md",)


class TestScoreDecline:
    def test_answerable_correct_citation_scores_true(self) -> None:
        client = FakeLLMClient("Answer.\nCITED: a.md")
        result = score_decline(_q(True, ("a.md",)), [_hit("a.md")], client, model="m")
        assert result is True

    def test_answerable_wrong_citation_scores_false(self) -> None:
        client = FakeLLMClient("Answer.\nCITED: z.md")
        result = score_decline(_q(True, ("a.md",)), [_hit("a.md")], client, model="m")
        assert result is False

    def test_answerable_declining_scores_false(self) -> None:
        client = FakeLLMClient("DECLINE")
        result = score_decline(_q(True, ("a.md",)), [_hit("a.md")], client, model="m")
        assert result is False

    def test_unanswerable_declining_scores_true(self) -> None:
        client = FakeLLMClient("DECLINE")
        result = score_decline(_q(False), [_hit("a.md")], client, model="m")
        assert result is True

    def test_unanswerable_hallucinating_scores_false(self) -> None:
        client = FakeLLMClient("Answer.\nCITED: a.md")
        result = score_decline(_q(False), [_hit("a.md")], client, model="m")
        assert result is False

    def test_calls_llm_with_given_model_and_max_tokens(self) -> None:
        client = FakeLLMClient("DECLINE")
        score_decline(_q(False), [], client, model="my-model", max_tokens=42)
        assert client.calls[0][1] == "my-model"
        assert client.calls[0][2] == 42


class TestMakeDeclineScorer:
    def test_returns_bound_callable(self) -> None:
        client = FakeLLMClient("Answer.\nCITED: a.md")
        scorer = make_decline_scorer(client, model="m")
        assert scorer(_q(True, ("a.md",)), [_hit("a.md")]) is True

    def test_reuses_same_client_across_calls(self) -> None:
        client = FakeLLMClient("DECLINE")
        scorer = make_decline_scorer(client, model="m")
        scorer(_q(False), [])
        scorer(_q(False), [])
        assert len(client.calls) == 2
