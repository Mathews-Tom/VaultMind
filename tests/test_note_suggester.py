"""Tests for NoteSuggester — context-aware link suggestion engine."""

from __future__ import annotations

import pytest

from vaultmind.indexer.note_suggester import (
    _SUGGESTION_MAX_DISTANCE,
    _SUGGESTION_MIN_DISTANCE,
    NoteSuggester,
    NoteSuggestion,
)
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSuggestionsConfig:
    """Minimal NoteSuggestionsConfig stand-in."""

    def __init__(
        self,
        enabled: bool = True,
        min_content_length: int = 100,
        entity_weight: float = 0.1,
        graph_weight: float = 0.05,
    ) -> None:
        self.enabled = enabled
        self.min_content_length = min_content_length
        self.entity_weight = entity_weight
        self.graph_weight = graph_weight


class FakeStore:
    """VaultStore stand-in returning controlled search results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        return self._results


class FakeGraph:
    """KnowledgeGraph stand-in with controllable path results."""

    def __init__(
        self,
        paths: dict[tuple[str, str], list[str]] | None = None,
        entities: set[str] | None = None,
    ) -> None:
        self._paths = paths or {}
        self._entities = entities or set()

    def find_path(self, source: str, target: str) -> list[str] | None:
        key = (source.strip().lower(), target.strip().lower())
        return self._paths.get(key)

    def get_entity(self, name: str) -> dict | None:
        if name.lower() in self._entities:
            return {"id": name.lower(), "label": name, "type": "concept"}
        return None


def _make_note(
    path: str = "test/note.md",
    title: str = "Test Note",
    content: str = "",
    entities: list[str] | None = None,
) -> Note:
    if not content:
        content = "---\ntype: fleeting\n---\n\n" + "A " * 100
    return Note(
        path=path,
        title=title,
        content=content,
        entities=entities or [],
    )


def _make_hit(
    note_path: str,
    title: str = "Other Note",
    distance: float = 0.25,
    entities: str = "",
    heading: str = "",
) -> dict:
    return {
        "chunk_id": f"{note_path}::0",
        "content": "some content",
        "metadata": {
            "note_path": note_path,
            "note_title": title,
            "heading": heading,
            "entities": entities,
        },
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# Tests — Threshold constants
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_suggestion_min_distance(self) -> None:
        # Min distance = 0.20 → similarity < 0.80 (above = merge territory)
        assert _SUGGESTION_MIN_DISTANCE == 0.20

    def test_suggestion_max_distance(self) -> None:
        # Max distance = 0.30 → similarity ≥ 0.70
        assert _SUGGESTION_MAX_DISTANCE == 0.30


# ---------------------------------------------------------------------------
# Tests — NoteSuggestion dataclass
# ---------------------------------------------------------------------------


class TestNoteSuggestion:
    def test_frozen(self) -> None:
        s = NoteSuggestion(
            source_path="a.md",
            source_title="A",
            target_path="b.md",
            target_title="B",
            similarity=0.75,
            shared_entities=[],
            graph_distance=None,
            composite_score=0.75,
        )
        with pytest.raises(AttributeError):
            s.similarity = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — suggest_links core
# ---------------------------------------------------------------------------


class TestSuggestLinks:
    def test_finds_suggestion_in_band(self) -> None:
        store = FakeStore([
            _make_hit("other/related.md", "Related", distance=0.25),
        ])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note)

        assert len(results) == 1
        assert results[0].similarity == pytest.approx(0.75)

    def test_excludes_above_merge_threshold(self) -> None:
        # distance=0.10 → similarity=0.90 → merge territory
        store = FakeStore([
            _make_hit("other/dup.md", distance=0.10),
        ])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note)
        assert len(results) == 0

    def test_excludes_below_suggestion_threshold(self) -> None:
        # distance=0.40 → similarity=0.60 → too dissimilar
        store = FakeStore([
            _make_hit("other/far.md", distance=0.40),
        ])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note)
        assert len(results) == 0

    def test_self_match_excluded(self) -> None:
        store = FakeStore([
            _make_hit("test/note.md", "Self", distance=0.25),
            _make_hit("other/real.md", "Real", distance=0.25),
        ])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note(path="test/note.md")

        results = suggester.suggest_links(note)
        assert len(results) == 1
        assert results[0].target_path == "other/real.md"

    def test_short_content_skipped(self) -> None:
        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        suggester = NoteSuggester(
            FakeSuggestionsConfig(min_content_length=100),  # type: ignore[arg-type]
            store,  # type: ignore[arg-type]
        )
        note = _make_note(content="---\ntype: fleeting\n---\n\nShort")

        results = suggester.suggest_links(note)
        assert len(results) == 0

    def test_max_results_respected(self) -> None:
        hits = [
            _make_hit(f"other/{i}.md", f"Match {i}", distance=0.25)
            for i in range(20)
        ]
        store = FakeStore(hits)
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note, max_results=3)
        assert len(results) == 3

    def test_deduplicates_same_note_chunks(self) -> None:
        store = FakeStore([
            _make_hit("other/note.md", "Same", distance=0.22),
            _make_hit("other/note.md", "Same", distance=0.24),
        ])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests — Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:
    def test_entity_boost(self) -> None:
        store = FakeStore([
            _make_hit("other/a.md", "A", distance=0.25, entities="python,ml"),
            _make_hit("other/b.md", "B", distance=0.25, entities="unrelated"),
        ])
        config = FakeSuggestionsConfig(entity_weight=0.1)
        suggester = NoteSuggester(config, store)  # type: ignore[arg-type]
        note = _make_note(entities=["python", "ml"])

        results = suggester.suggest_links(note)

        # Both in band, but A should have higher score due to shared entities
        assert len(results) == 2
        assert results[0].target_path == "other/a.md"
        assert results[0].composite_score > results[1].composite_score
        assert len(results[0].shared_entities) == 2

    def test_graph_distance_boost(self) -> None:
        store = FakeStore([
            _make_hit("other/close.md", "Close", distance=0.25),
            _make_hit("other/far.md", "Far", distance=0.25),
        ])
        graph = FakeGraph(paths={
            ("test note", "close"): ["test note", "bridge", "close"],  # dist=2
        })
        config = FakeSuggestionsConfig(graph_weight=0.05)
        suggester = NoteSuggester(config, store, graph=graph)  # type: ignore[arg-type]
        note = _make_note()

        results = suggester.suggest_links(note)

        # Close should be ranked higher due to graph distance boost
        assert len(results) == 2
        assert results[0].target_path == "other/close.md"
        assert results[0].graph_distance == 2
        assert results[1].graph_distance is None

    def test_no_graph_distance_for_disconnected(self) -> None:
        store = FakeStore([
            _make_hit("other/note.md", "Note", distance=0.25),
        ])
        graph = FakeGraph()  # empty graph, no paths
        suggester = NoteSuggester(
            FakeSuggestionsConfig(), store, graph=graph,  # type: ignore[arg-type]
        )
        note = _make_note()

        results = suggester.suggest_links(note)
        assert len(results) == 1
        assert results[0].graph_distance is None
        # Score should just be similarity (no graph boost)
        assert results[0].composite_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Tests — Entity extraction from graph
# ---------------------------------------------------------------------------


class TestEntityFromGraph:
    def test_extracts_frontmatter_entities(self) -> None:
        store = FakeStore([
            _make_hit("other/hit.md", distance=0.25, entities="python"),
        ])
        graph = FakeGraph(entities={"python"})
        suggester = NoteSuggester(
            FakeSuggestionsConfig(), store, graph=graph,  # type: ignore[arg-type]
        )
        note = _make_note(entities=["Python"])

        results = suggester.suggest_links(note)
        assert len(results) == 1
        assert "python" in results[0].shared_entities


# ---------------------------------------------------------------------------
# Tests — Results cache
# ---------------------------------------------------------------------------


class TestResultsCache:
    def test_results_stored(self) -> None:
        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        suggester.suggest_links(_make_note())

        assert suggester.result_count == 1
        assert len(suggester.get_results("test/note.md")) == 1

    def test_clear_results(self) -> None:
        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        suggester.suggest_links(_make_note())

        suggester.clear_results()
        assert suggester.result_count == 0


# ---------------------------------------------------------------------------
# Tests — Event bus callback
# ---------------------------------------------------------------------------


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_on_note_changed_runs_suggestions(self) -> None:
        from vaultmind.vault.events import NoteCreatedEvent

        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        event = NoteCreatedEvent(path=note.path, note=note, chunks_indexed=1)
        await suggester.on_note_changed(event)

        assert suggester.result_count == 1

    @pytest.mark.asyncio
    async def test_on_note_changed_disabled(self) -> None:
        from vaultmind.vault.events import NoteCreatedEvent

        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        config = FakeSuggestionsConfig(enabled=False)
        suggester = NoteSuggester(config, store)  # type: ignore[arg-type]
        note = _make_note()

        event = NoteCreatedEvent(path=note.path, note=note, chunks_indexed=1)
        await suggester.on_note_changed(event)

        assert suggester.result_count == 0


# ---------------------------------------------------------------------------
# Tests — Batch scan
# ---------------------------------------------------------------------------


class TestScanVault:
    def test_batch_scan(self) -> None:
        store = FakeStore([_make_hit("other/hit.md", distance=0.25)])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]

        notes = [
            _make_note(path="a.md", title="A"),
            _make_note(path="b.md", title="B"),
        ]
        results = suggester.scan_vault(notes)

        assert len(results) == 2

    def test_batch_scan_empty(self) -> None:
        store = FakeStore([])
        suggester = NoteSuggester(FakeSuggestionsConfig(), store)  # type: ignore[arg-type]

        results = suggester.scan_vault([_make_note()])
        assert len(results) == 0
