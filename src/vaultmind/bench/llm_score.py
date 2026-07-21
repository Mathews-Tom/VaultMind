"""LLM cite-or-decline scoring for the retrieval self-benchmark (`--llm`).

Generates one single-shot answer per golden question from the retrieved
context, then deterministically parses the response into a decline/citation
signal. The LLM is never asked to grade its own correctness — this module
does that deterministically from the parsed citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vaultmind.llm.client import Message

if TYPE_CHECKING:
    from vaultmind.bench.golden import GoldenQuestion
    from vaultmind.bench.runner import DeclineScorer
    from vaultmind.llm.client import LLMClient

DECLINE_MARKER = "DECLINE"
CITED_PREFIX = "CITED:"
_MAX_CONTEXT_CHARS = 800
_DEFAULT_MAX_TOKENS = 300


def _note_path(hit: dict[str, Any]) -> str:
    return str(hit.get("metadata", {}).get("note_path", ""))


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    """Build the single-shot cite-or-decline prompt for one golden question."""
    if not hits:
        context = "(no notes retrieved)"
    else:
        context = "\n\n".join(
            f"[Note path: {_note_path(h)}]\n{str(h.get('content', ''))[:_MAX_CONTEXT_CHARS]}"
            for h in hits
        )
    return (
        "Context from the vault:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the context above. If the context does not answer "
        f"the question, respond with exactly the single word {DECLINE_MARKER} "
        "and nothing else. Otherwise, answer concisely and end your response "
        f"on a new line formatted exactly as:\n{CITED_PREFIX} <note path 1>, <note path 2>"
    )


@dataclass(frozen=True, slots=True)
class LLMAnswer:
    """Deterministically-parsed shape of an LLM cite-or-decline response."""

    declined: bool
    cited_paths: tuple[str, ...]


def parse_llm_answer(raw: str) -> LLMAnswer:
    """Parse a cite-or-decline response into a decline flag + cited paths."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return LLMAnswer(declined=True, cited_paths=())

    cited_paths: tuple[str, ...] = ()
    for line in lines:
        if line.upper().startswith(CITED_PREFIX):
            raw_paths = line[len(CITED_PREFIX) :].strip()
            cited_paths = tuple(p.strip() for p in raw_paths.split(",") if p.strip())
            break

    declined = not cited_paths and any(line.upper() == DECLINE_MARKER for line in lines)
    return LLMAnswer(declined=declined, cited_paths=cited_paths)


def score_decline(
    question: GoldenQuestion,
    hits: list[dict[str, Any]],
    llm_client: LLMClient,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> bool:
    """Score one question's cite-or-decline correctness via a single LLM call.

    Correct when: an unanswerable question is declined, or an answerable
    question is answered while citing at least one expected note path.
    Hallucinated answers (wrong citation) and unnecessary declines both
    score incorrect.
    """
    prompt = build_prompt(question.question, hits)
    response = llm_client.complete(
        [Message(role="user", content=prompt)], model=model, max_tokens=max_tokens
    )
    answer = parse_llm_answer(response.text)

    if not question.answerable:
        return answer.declined
    if answer.declined:
        return False
    return any(path in question.expected_notes for path in answer.cited_paths)


def make_decline_scorer(
    llm_client: LLMClient, model: str, max_tokens: int = _DEFAULT_MAX_TOKENS
) -> DeclineScorer:
    """Build a `DeclineScorer` (see `vaultmind.bench.runner`) bound to one client/model."""

    def scorer(question: GoldenQuestion, hits: list[dict[str, Any]]) -> bool:
        return score_decline(question, hits, llm_client, model, max_tokens)

    return scorer
