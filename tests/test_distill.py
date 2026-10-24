"""Tests for conversation distillation into qa-artifact notes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import frontmatter
import pytest

from vaultmind.config import Settings, VaultConfig
from vaultmind.knowledge.intake import SourceIntake, StageDisposition
from vaultmind.llm.client import LLMError, LLMResponse
from vaultmind.memory.models import OutcomeStatus
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.distill import (
    DISTILL_SYSTEM_PROMPT,
    DISTILLED_AUTHORITY,
    DistillationGeneration,
    distill_retained_conversation,
    extract_and_store_episodes,
    generate_distillation,
    preserve_conversation_snapshot,
    render_conversation,
)
from vaultmind.vault.models import Note, NoteType
from vaultmind.vault.parser import VaultParser
from vaultmind.wiki.workspace import WikiError, WikiWorkspace, proposal_digest


def _make_llm(response_obj: object) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text=json.dumps(response_obj), model="fake-model", usage={"total_tokens": 40}
    )
    return client


_GOOD_TURNS = [
    {"user": "Should we use ChromaDB or pgvector?", "assistant": "ChromaDB for zero-infra now."},
    {"user": "What about scale later?", "assistant": "Migrate to pgvector past 1M chunks."},
    {"user": "Sounds good, let's do that.", "assistant": "Documented as a decision."},
]

_GOOD_RESPONSE = {
    "question": "ChromaDB vs pgvector for vector storage?",
    "summary": "Discussed tradeoffs; chose ChromaDB now, pgvector as a future migration path.",
    "resolution": "Use ChromaDB, revisit pgvector past 1M chunks.",
    "systems": ["ChromaDB", "pgvector"],
    "participants": ["user"],
}


def _generate(llm: MagicMock, text: str | None = None) -> DistillationGeneration:
    return generate_distillation(
        render_conversation(_GOOD_TURNS) if text is None else text,
        llm,
        "fake-model",
        source_ref="telegram-thinking:1:1234",
        occurred_at="2026-07-21T10:00:00Z",
    )


def _input(intake: SourceIntake, event: str = "chat:1:actor"):
    return preserve_conversation_snapshot(
        intake,
        dict(
            version=1,
            history=_GOOD_TURNS,
            participants=[dict(role="user", user_id=1), dict(role="assistant")],
            summaries=[dict(summary="Historical generated summary")],
        ),
        origin="telegram:chat=1:actor=1",
        adapter="telegram-distill",
        event_id=event,
    )


def _distill(intake: SourceIntake, captured, llm: MagicMock):
    return distill_retained_conversation(
        intake,
        captured,
        llm,
        "fake-model",
        source_ref="telegram-thinking:1:1234",
        occurred_at="2026-07-21T10:00:00Z",
        output_folder="qa-artifacts",
    )


class TestDistillationGeneration:
    def test_full_schema_wikilinks_and_no_filesystem_effects(self, tmp_path: Path) -> None:
        result = _generate(_make_llm(_GOOD_RESPONSE))
        assert result.success and result.error == ""
        assert list(tmp_path.iterdir()) == []
        post = frontmatter.loads(result.content)
        assert post.metadata["type"] == "qa-artifact"
        for name in ("question", "summary", "resolution", "systems", "participants"):
            assert post.metadata[name] == _GOOD_RESPONSE[name]
        assert post.metadata["source_ref"] == "telegram-thinking:1:1234"
        assert post.metadata["occurred_at"] == "2026-07-21T10:00:00Z"
        assert post.metadata["authority"] == DISTILLED_AUTHORITY == 4
        assert post.metadata["approved"] is False
        assert post.metadata["status"] == "unapproved"
        assert all(link in post.content for link in ("[[ChromaDB]]", "[[pgvector]]", "[[user]]"))

    def test_optional_sections_and_unresolved_resolution(self) -> None:
        response = dict(_GOOD_RESPONSE, systems=[], participants=[], resolution="")
        result = _generate(_make_llm(response))
        assert result.success and result.frontmatter["resolution"] == ""
        assert "## Systems" not in result.content
        assert "## Participants" not in result.content
        assert "_Unresolved._" in result.content

    @pytest.mark.parametrize(
        "response,error",
        [(["not", "object"], "JSON object"), (dict(_GOOD_RESPONSE, question=""), "no question")],
    )
    def test_invalid_structured_output(self, response: object, error: str) -> None:
        result = _generate(_make_llm(response))
        assert not result.success and error in result.error

    def test_invalid_json(self) -> None:
        llm = _make_llm({})
        llm.complete.return_value = LLMResponse(text="not json", model="fake", usage={})
        assert "Invalid JSON" in _generate(llm).error

    def test_empty_input_and_provider_failure(self) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        assert "No conversation turns" in _generate(llm, "").error
        llm.complete.assert_not_called()
        llm.complete.side_effect = LLMError("boom", provider="anthropic")
        assert _generate(llm).error == "LLM call failed"


class TestPreservedDistillation:
    def test_retains_request_before_call_and_projects_parented_unapproved_note(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(vault=VaultConfig(path=tmp_path))
        with SourceIntake(settings) as intake:
            captured = _input(intake)
            llm = _make_llm(_GOOD_RESPONSE)
            response = llm.complete.return_value

            def complete(**kwargs):
                requests = [
                    json.loads(path.read_text())
                    for path in (tmp_path / "_wiki/sources").glob("*.md")
                    if path.read_text().startswith('{"authority":')
                ]
                assert len(requests) == llm.complete.call_count
                assert requests[0]["messages"][0]["content"] == kwargs["messages"][0].content
                assert requests[0]["system"] == kwargs["system"] == DISTILL_SYSTEM_PROMPT
                assert requests[0]["model"] == kwargs["model"]
                assert requests[0]["max_tokens"] == kwargs["max_tokens"]
                return response

            llm.complete.side_effect = complete
            result = _distill(intake, captured, llm)
            assert result.success and result.derivative is not None
            assert result.derivative.approved is False
            retained = intake.read(result.derivative.source_id)
            assert retained.provenance["original_retained"] is False
            assert retained.dependencies[0].source_id == result.request.source_id
            assert retained.dependencies[0].dependencies[0].source_id == captured.source_id
            assert (tmp_path / result.output_path).read_bytes() == retained.content
            assert not result.derivative.indexed
            parser = VaultParser(settings.vault)
            note = parser.parse_file(tmp_path / result.output_path)
            assert note.note_type is NoteType.QA_ARTIFACT
            assert set(note.wikilinks) == {"ChromaDB", "pgvector", "user"}
            assert note.authority == DISTILLED_AUTHORITY
            second = _distill(intake, _input(intake, "chat:2:actor"), llm)
            assert second.success and second.output_path != result.output_path

    @pytest.mark.parametrize("failure", ["provider", "invalid", "derivative", "projection"])
    def test_failure_preserves_original_without_claiming_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        settings = Settings(vault=VaultConfig(path=tmp_path))
        with SourceIntake(settings) as intake:
            captured = _input(intake)
            original = intake.read(captured.source_id).dependencies[0]
            llm = _make_llm(_GOOD_RESPONSE)
            if failure == "provider":
                llm.complete.side_effect = RuntimeError("provider unavailable")
            elif failure == "invalid":
                llm.complete.return_value = LLMResponse(text="bad json", model="fake", usage={})
            elif failure == "derivative":
                preserve = intake.preserve_text

                def fail_derivative(content, **kwargs):
                    if kwargs.get("role") == "distillation":
                        raise OSError("derivative unavailable")
                    return preserve(content, **kwargs)

                monkeypatch.setattr(intake, "preserve_text", fail_derivative)
            else:
                monkeypatch.setattr(
                    intake, "project_text", lambda *_: StageDisposition("projection", "failed")
                )
            result = _distill(intake, captured, llm)
            assert not result.success and result.output_path == ""
            assert intake.read(original.source_id).content == original.content
            assert not (tmp_path / "qa-artifacts").exists()
            if failure in {"provider", "invalid"}:
                assert result.generation_disposition.status == "failed"
                replay = _distill(intake, _input(intake), llm)
                assert replay.generation is None
                assert llm.complete.call_count == 1
            if failure == "derivative":
                assert result.generation.success
                assert result.derivative is None
                assert result.derivative_disposition.status == "failed"

    def test_missing_parent_refuses_before_completion(self, tmp_path: Path) -> None:
        with SourceIntake(Settings(vault=VaultConfig(path=tmp_path))) as intake:
            captured = _input(intake)
            original = intake.read(captured.source_id).dependencies[0]
            (tmp_path / original.path).unlink()
            llm = _make_llm(_GOOD_RESPONSE)
            with pytest.raises(WikiError):
                _distill(intake, captured, llm)
            llm.complete.assert_not_called()

    @pytest.mark.parametrize(
        "text", ["---\ndo_not_retain: true\n---\nprivate", "\x00private", "x" * (256 * 1024 + 1)]
    )
    def test_original_policy_is_not_hidden_by_envelope(self, tmp_path: Path, text: str) -> None:
        with SourceIntake(Settings(vault=VaultConfig(path=tmp_path))) as intake:
            with pytest.raises(WikiError):
                preserve_conversation_snapshot(
                    intake,
                    dict(history=[dict(user=text, assistant="")]),
                    origin="test",
                    adapter="test",
                    event_id="one",
                )
            assert not (tmp_path / "_wiki").exists()

    def test_retained_producer_id_requires_exact_wiki_review(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        settings = Settings(vault=VaultConfig(path=vault))
        with WikiWorkspace(settings) as workspace:
            intake = SourceIntake.from_workspace(workspace)
            result = _distill(intake, _input(intake), _make_llm(_GOOD_RESPONSE))
            assert result.derivative is not None
            source_id = result.derivative.source_id
            draft = tmp_path / "draft.md"
            draft.write_text(
                "# Storage\n\n## Current understanding\n\n"
                f"Use ChromaDB now. [[_wiki/sources/{source_id}]]\n"
            )
            proposal = workspace.propose("wiki/storage.md", [source_id], draft_file=draft)
            assert not (vault / "wiki/storage.md").exists()
            assert "Exact before/after unified diff" in workspace.review(proposal.proposal_id)
            with pytest.raises(WikiError):
                workspace.apply(proposal.proposal_id, "0" * 64)
            workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
            assert (vault / "wiki/storage.md").exists()


class TestExtractAndStoreEpisodes:
    def _note(self, tmp_path: Path) -> Note:
        return Note(
            path=Path("qa-artifacts/2026-07-21-test.md"),
            title="Test qa-artifact",
            content="x" * 150,
            note_type=NoteType.QA_ARTIFACT,
        )

    def test_extracts_and_persists_episodes(self, tmp_path: Path) -> None:
        note = self._note(tmp_path)
        llm = _make_llm(
            [
                {
                    "decision": "Use ChromaDB",
                    "context": "vector storage choice",
                    "outcome": "Works well",
                    "outcome_status": "success",
                    "lessons": ["measure before migrating"],
                    "entities": ["ChromaDB"],
                }
            ]
        )
        store = EpisodeStore(tmp_path / "episodes.db")

        count = extract_and_store_episodes(note, llm, "fake-model", store)

        assert count == 1
        resolved = store.query_resolved()
        assert len(resolved) == 1
        assert resolved[0].decision == "Use ChromaDB"
        assert resolved[0].outcome_status == OutcomeStatus.SUCCESS
        assert resolved[0].source_notes == [str(note.path)]
        store.close()

    def test_no_op_when_episode_store_not_configured(self, tmp_path: Path) -> None:
        note = self._note(tmp_path)
        llm = _make_llm([{"decision": "x"}])
        count = extract_and_store_episodes(note, llm, "fake-model", None)
        assert count == 0
        llm.complete.assert_not_called()

    def test_empty_extraction_returns_zero(self, tmp_path: Path) -> None:
        note = self._note(tmp_path)
        llm = _make_llm([])
        store = EpisodeStore(tmp_path / "episodes.db")
        count = extract_and_store_episodes(note, llm, "fake-model", store)
        assert count == 0
        assert store.query_resolved() == []
        store.close()


class TestGenerationResultDataclass:
    def test_defaults(self) -> None:
        result = DistillationGeneration(success=False)
        assert result.content == ""
        assert result.frontmatter == {}
        assert result.error == ""


def test_absolute_ordinary_output_folder_returns_relative_projection_path(tmp_path: Path) -> None:
    with SourceIntake(Settings(vault=VaultConfig(path=tmp_path))) as intake:
        result = distill_retained_conversation(
            intake,
            _input(intake),
            _make_llm(_GOOD_RESPONSE),
            "fake-model",
            source_ref="selected-session",
            occurred_at="2026-10-24T12:00:00Z",
            output_folder=str(tmp_path / "nested"),
        )
        assert result.success
        assert result.output_path.startswith("nested/")
        assert (tmp_path / result.output_path).is_file()
