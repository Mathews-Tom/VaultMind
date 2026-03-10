"""Tests for episodic memory — EpisodeStore and episode extractor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from vaultmind.llm.client import LLMResponse
from vaultmind.memory.extractor import extract_episodes
from vaultmind.memory.models import Episode, OutcomeStatus
from vaultmind.memory.store import EpisodeStore
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


def _make_note(
    path: str = "test.md",
    title: str = "Test Note",
    content: str = "",
) -> Note:
    return Note(path=Path(path), title=title, content=content)


def _make_fake_llm(response_json: object) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text=json.dumps(response_json),
        model="fake-model",
        usage={"total_tokens": 50},
    )
    return client


# ---------------------------------------------------------------------------
# EpisodeStore tests
# ---------------------------------------------------------------------------


class TestCreateEpisode:
    def test_create_and_retrieve_by_id(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep = store.create(
            decision="Use SQLite for persistence",
            context="Evaluating storage options",
            entities=["SQLite", "storage"],
            source_notes=["notes/decision.md"],
            tags=["architecture"],
        )

        assert isinstance(ep, Episode)
        assert ep.decision == "Use SQLite for persistence"
        assert ep.context == "Evaluating storage options"
        assert ep.outcome_status == OutcomeStatus.PENDING
        assert ep.entities == ["SQLite", "storage"]
        assert ep.source_notes == ["notes/decision.md"]
        assert ep.tags == ["architecture"]
        assert ep.resolved is None

        fetched = store.get(ep.episode_id)
        assert fetched is not None
        assert fetched.episode_id == ep.episode_id
        assert fetched.decision == ep.decision
        assert fetched.outcome_status == OutcomeStatus.PENDING

        store.close()

    def test_episode_id_is_12_hex_chars(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep = store.create(decision="Some decision")
        assert len(ep.episode_id) == 12
        assert all(c in "0123456789abcdef" for c in ep.episode_id)
        store.close()


class TestResolveEpisode:
    def test_resolve_sets_outcome_status_lessons(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep = store.create(decision="Switch to async IO")

        store.resolve(
            ep.episode_id,
            outcome="Latency dropped by 40%",
            status=OutcomeStatus.SUCCESS,
            lessons=["Async pays off for IO-heavy workloads"],
        )

        resolved = store.get(ep.episode_id)
        assert resolved is not None
        assert resolved.outcome == "Latency dropped by 40%"
        assert resolved.outcome_status == OutcomeStatus.SUCCESS
        assert resolved.lessons == ["Async pays off for IO-heavy workloads"]
        assert resolved.resolved is not None

        store.close()

    def test_resolve_with_empty_lessons(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep = store.create(decision="Deploy on Friday")

        store.resolve(ep.episode_id, outcome="Went fine", status=OutcomeStatus.PARTIAL, lessons=[])

        resolved = store.get(ep.episode_id)
        assert resolved is not None
        assert resolved.lessons == []
        assert resolved.outcome_status == OutcomeStatus.PARTIAL

        store.close()


class TestQueryPending:
    def test_only_pending_episodes_returned(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep1 = store.create(decision="Decision A")
        ep2 = store.create(decision="Decision B")
        ep3 = store.create(decision="Decision C")

        # Resolve ep2
        store.resolve(ep2.episode_id, outcome="Done", status=OutcomeStatus.SUCCESS, lessons=[])

        pending = store.query_pending()
        pending_ids = {ep.episode_id for ep in pending}

        assert ep1.episode_id in pending_ids
        assert ep3.episode_id in pending_ids
        assert ep2.episode_id not in pending_ids

        store.close()

    def test_pending_ordered_by_created_desc(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep_a = store.create(decision="First")
        store.create(decision="Second")
        ep_c = store.create(decision="Third")

        pending = store.query_pending()
        ids = [ep.episode_id for ep in pending]

        # Most recent first
        assert ids.index(ep_c.episode_id) < ids.index(ep_a.episode_id)

        store.close()

    def test_respects_limit(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for i in range(5):
            store.create(decision=f"Decision {i}")

        pending = store.query_pending(limit=3)
        assert len(pending) == 3

        store.close()


class TestSearchByEntity:
    def test_finds_episodes_with_matching_entity(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ep_match = store.create(decision="Use ChromaDB", entities=["ChromaDB", "vector search"])
        ep_other = store.create(decision="Use Postgres", entities=["Postgres", "SQL"])

        results = store.search_by_entity("ChromaDB")
        result_ids = {ep.episode_id for ep in results}

        assert ep_match.episode_id in result_ids
        assert ep_other.episode_id not in result_ids

        store.close()

    def test_returns_empty_when_no_match(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.create(decision="Some decision", entities=["Redis"])

        results = store.search_by_entity("MongoDB")
        assert results == []

        store.close()


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


class TestExtractEpisodesFromNote:
    def test_extracts_episodes_from_note(self) -> None:
        llm = _make_fake_llm(
            [
                {
                    "decision": "Adopt TDD",
                    "context": "Team discussion on quality",
                    "outcome": "Defect rate dropped",
                    "outcome_status": "success",
                    "lessons": ["Write tests first"],
                    "entities": ["TDD", "testing"],
                }
            ]
        )
        note = _make_note(content="A" * 200)

        results = extract_episodes(note, llm, "fake-model")  # type: ignore[arg-type]

        assert len(results) == 1
        assert results[0]["decision"] == "Adopt TDD"
        assert results[0]["outcome_status"] == "success"
        assert "TDD" in results[0]["entities"]

    def test_passes_correct_model_and_system_prompt(self) -> None:
        llm = _make_fake_llm([])
        note = _make_note(content="B" * 200)

        extract_episodes(note, llm, "my-model")  # type: ignore[arg-type]

        call_kwargs = llm.complete.call_args
        assert call_kwargs.kwargs.get("model") == "my-model" or call_kwargs.args[1] == "my-model"
        assert call_kwargs.kwargs.get("system") is not None


class TestExtractSkipsShortNotes:
    def test_returns_empty_for_short_body(self) -> None:
        llm = _make_fake_llm([{"decision": "something"}])
        note = _make_note(content="Too short")

        results = extract_episodes(note, llm, "fake-model")  # type: ignore[arg-type]

        assert results == []
        llm.complete.assert_not_called()

    def test_returns_empty_for_exactly_99_chars(self) -> None:
        llm = _make_fake_llm([])
        note = _make_note(content="x" * 99)

        results = extract_episodes(note, llm, "fake-model")  # type: ignore[arg-type]

        assert results == []
        llm.complete.assert_not_called()

    def test_proceeds_for_exactly_100_chars(self) -> None:
        llm = _make_fake_llm([])
        note = _make_note(content="x" * 100)

        extract_episodes(note, llm, "fake-model")  # type: ignore[arg-type]

        llm.complete.assert_called_once()


class TestExtractHandlesInvalidJson:
    def test_bad_json_returns_empty_list(self) -> None:
        client = MagicMock()
        client.complete.return_value = LLMResponse(
            text="not valid json at all }{",
            model="fake-model",
            usage={"total_tokens": 10},
        )
        note = _make_note(content="C" * 200)

        results = extract_episodes(note, client, "fake-model")  # type: ignore[arg-type]

        assert results == []

    def test_non_array_json_returns_empty_list(self) -> None:
        client = MagicMock()
        client.complete.return_value = LLMResponse(
            text='{"decision": "something"}',
            model="fake-model",
            usage={"total_tokens": 10},
        )
        note = _make_note(content="D" * 200)

        results = extract_episodes(note, client, "fake-model")  # type: ignore[arg-type]

        assert results == []

    def test_llm_exception_returns_empty_list(self) -> None:
        client = MagicMock()
        client.complete.side_effect = RuntimeError("API timeout")
        note = _make_note(content="E" * 200)

        results = extract_episodes(note, client, "fake-model")  # type: ignore[arg-type]

        assert results == []
