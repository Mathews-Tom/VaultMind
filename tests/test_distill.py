"""Tests for conversation distillation into qa-artifact notes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import frontmatter

from vaultmind.llm.client import LLMError, LLMResponse
from vaultmind.memory.models import OutcomeStatus
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.distill import (
    DISTILLED_AUTHORITY,
    DistillResult,
    distill_conversation,
    extract_and_store_episodes,
)
from vaultmind.vault.models import Note, NoteType
from vaultmind.vault.parser import VaultParser


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


class TestDistillConversationSuccess:
    def test_writes_note_with_full_schema(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        result = distill_conversation(
            _GOOD_TURNS,
            llm,
            "fake-model",
            tmp_path,
            "qa-artifacts",
            source_ref="telegram-thinking:1:1234",
            occurred_at="2026-07-21T10:00:00Z",
        )

        assert result.success is True
        assert result.error == ""
        output_path = tmp_path / result.output_path
        assert output_path.exists()

        post = frontmatter.loads(output_path.read_text())
        assert post.metadata["type"] == "qa-artifact"
        assert post.metadata["question"] == _GOOD_RESPONSE["question"]
        assert post.metadata["summary"] == _GOOD_RESPONSE["summary"]
        assert post.metadata["resolution"] == _GOOD_RESPONSE["resolution"]
        assert post.metadata["systems"] == _GOOD_RESPONSE["systems"]
        assert post.metadata["participants"] == _GOOD_RESPONSE["participants"]
        assert post.metadata["source_ref"] == "telegram-thinking:1:1234"
        assert post.metadata["occurred_at"] == "2026-07-21T10:00:00Z"
        assert post.metadata["authority"] == DISTILLED_AUTHORITY == 4

    def test_note_parses_as_qa_artifact_note_type(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", source_ref="ref-1"
        )
        assert result.success is True

        from vaultmind.config import VaultConfig

        parser = VaultParser(VaultConfig(path=tmp_path))
        note = parser.parse_file(tmp_path / result.output_path)
        assert note.note_type == NoteType.QA_ARTIFACT
        assert note.authority == DISTILLED_AUTHORITY

    def test_body_wikilinks_systems_and_participants(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", source_ref="ref-wikilinks"
        )
        assert result.success is True

        post = frontmatter.loads((tmp_path / result.output_path).read_text())
        assert "[[ChromaDB]]" in post.content
        assert "[[pgvector]]" in post.content
        assert "[[user]]" in post.content

    def test_wikilinks_extracted_by_note_wikilinks_property(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", source_ref="ref-wl2"
        )
        assert result.success is True

        from vaultmind.config import VaultConfig

        parser = VaultParser(VaultConfig(path=tmp_path))
        note = parser.parse_file(tmp_path / result.output_path)
        assert set(note.wikilinks) == {"ChromaDB", "pgvector", "user"}

    def test_no_systems_or_participants_omits_wikilink_sections(self, tmp_path: Path) -> None:
        response = dict(_GOOD_RESPONSE)
        response["systems"] = []
        response["participants"] = []
        llm = _make_llm(response)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", source_ref="ref-no-links"
        )
        assert result.success is True

        post = frontmatter.loads((tmp_path / result.output_path).read_text())
        assert "## Systems" not in post.content
        assert "## Participants" not in post.content

    def test_unresolved_resolution_kept_empty(self, tmp_path: Path) -> None:
        response = dict(_GOOD_RESPONSE)
        response["resolution"] = ""
        llm = _make_llm(response)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", source_ref="ref-2"
        )
        assert result.success is True
        assert result.frontmatter["resolution"] == ""

    def test_avoids_filename_collision(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        first = distill_conversation(
            _GOOD_TURNS,
            llm,
            "fake-model",
            tmp_path,
            "qa-artifacts",
            source_ref="ref-a",
            occurred_at="2026-07-21T10:00:00Z",
        )
        second = distill_conversation(
            _GOOD_TURNS,
            llm,
            "fake-model",
            tmp_path,
            "qa-artifacts",
            source_ref="ref-b",
            occurred_at="2026-07-21T10:00:00Z",
        )
        assert first.output_path != second.output_path
        assert (tmp_path / first.output_path).exists()
        assert (tmp_path / second.output_path).exists()


class TestDistillConversationFailure:
    def test_empty_turns_fails_without_llm_call(self, tmp_path: Path) -> None:
        llm = _make_llm(_GOOD_RESPONSE)
        result = distill_conversation([], llm, "fake-model", tmp_path, "qa-artifacts", "ref")
        assert result.success is False
        assert "No conversation turns" in result.error
        llm.complete.assert_not_called()

    def test_llm_error_returns_failure(self, tmp_path: Path) -> None:
        llm = MagicMock()
        llm.complete.side_effect = LLMError("boom", provider="anthropic")
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", "ref"
        )
        assert result.success is False
        assert result.output_path == ""
        assert not (tmp_path / "qa-artifacts").exists()

    def test_invalid_json_returns_failure(self, tmp_path: Path) -> None:
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(text="not json", model="fake-model", usage={})
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", "ref"
        )
        assert result.success is False
        assert "Invalid JSON" in result.error

    def test_non_object_json_returns_failure(self, tmp_path: Path) -> None:
        llm = _make_llm(["not", "an", "object"])
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", "ref"
        )
        assert result.success is False
        assert "JSON object" in result.error

    def test_missing_question_returns_failure(self, tmp_path: Path) -> None:
        response = dict(_GOOD_RESPONSE)
        response["question"] = ""
        llm = _make_llm(response)
        result = distill_conversation(
            _GOOD_TURNS, llm, "fake-model", tmp_path, "qa-artifacts", "ref"
        )
        assert result.success is False
        assert "no question" in result.error


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


class TestDistillResultDataclass:
    def test_defaults(self) -> None:
        result = DistillResult(success=False)
        assert result.output_path == ""
        assert result.frontmatter == {}
        assert result.error == ""
