"""Tests for DuplicateDetector — semantic duplicate and merge detection."""

from __future__ import annotations

import pytest

from vaultmind.indexer.duplicate_detector import (
    _DUPLICATE_MAX_DISTANCE,
    _MERGE_MAX_DISTANCE,
    DuplicateDetector,
    DuplicateMatch,
    MatchType,
)
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDuplicateConfig:
    """Minimal DuplicateDetectionConfig stand-in."""

    def __init__(
        self,
        enabled: bool = True,
        min_content_length: int = 100,
    ) -> None:
        self.enabled = enabled
        self.min_content_length = min_content_length


class FakeStore:
    """Minimal VaultStore stand-in that returns controlled search results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []
        self.last_query: str | None = None

    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        self.last_query = query
        return self._results


def _make_note(path: str = "test/note.md", title: str = "Test Note", content: str = "") -> Note:
    if not content:
        content = "---\ntype: fleeting\n---\n\n" + "A" * 200
    return Note(path=path, title=title, content=content)


def _make_hit(
    note_path: str,
    title: str = "Other Note",
    distance: float = 0.05,
    heading: str = "",
) -> dict:
    return {
        "chunk_id": f"{note_path}::0",
        "content": "some content",
        "metadata": {
            "note_path": note_path,
            "note_title": title,
            "heading": heading,
        },
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# Tests — Threshold constants
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_duplicate_threshold(self) -> None:
        assert _DUPLICATE_MAX_DISTANCE == 0.08
        assert pytest.approx(0.92) == 1 - _DUPLICATE_MAX_DISTANCE

    def test_merge_threshold(self) -> None:
        assert _MERGE_MAX_DISTANCE == 0.20
        assert pytest.approx(0.80) == 1 - _MERGE_MAX_DISTANCE


# ---------------------------------------------------------------------------
# Tests — MatchType
# ---------------------------------------------------------------------------


class TestMatchType:
    def test_values(self) -> None:
        assert MatchType.DUPLICATE == "duplicate"
        assert MatchType.MERGE == "merge"


# ---------------------------------------------------------------------------
# Tests — DuplicateMatch
# ---------------------------------------------------------------------------


class TestDuplicateMatch:
    def test_frozen(self) -> None:
        m = DuplicateMatch(
            source_path="a.md",
            source_title="A",
            match_path="b.md",
            match_title="B",
            similarity=0.95,
            match_type=MatchType.DUPLICATE,
        )
        with pytest.raises(AttributeError):
            m.similarity = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — DuplicateDetector.find_duplicates
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    def test_duplicate_match(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/dup.md", "Duplicate", distance=0.05),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.DUPLICATE
        assert matches[0].similarity == pytest.approx(0.95)

    def test_merge_match(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/merge.md", "Merge Candidate", distance=0.15),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.MERGE
        assert matches[0].similarity == pytest.approx(0.85)

    def test_below_threshold_excluded(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/far.md", "Unrelated", distance=0.30),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note)
        assert len(matches) == 0

    def test_self_match_excluded(self) -> None:
        store = FakeStore(
            [
                _make_hit("test/note.md", "Same Note", distance=0.0),
                _make_hit("other/real.md", "Real Match", distance=0.05),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note(path="test/note.md")

        matches = detector.find_duplicates(note)
        assert len(matches) == 1
        assert matches[0].match_path == "other/real.md"

    def test_deduplicates_same_note_multiple_chunks(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/dup.md", "Dup", distance=0.04),
                _make_hit("other/dup.md", "Dup", distance=0.06),  # same note, second chunk
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note)
        assert len(matches) == 1

    def test_short_content_skipped(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/dup.md", distance=0.05),
            ]
        )
        detector = DuplicateDetector(
            FakeDuplicateConfig(min_content_length=100),  # type: ignore[arg-type]
            store,  # type: ignore[arg-type]
        )
        # Note body (after frontmatter) is very short
        note = _make_note(content="---\ntype: fleeting\n---\n\nShort")

        matches = detector.find_duplicates(note)
        assert len(matches) == 0
        assert store.last_query is None  # search never called

    def test_mixed_duplicate_and_merge(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/dup.md", "Duplicate", distance=0.03),
                _make_hit("other/merge.md", "Merge", distance=0.15),
                _make_hit("other/far.md", "Unrelated", distance=0.40),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note)
        assert len(matches) == 2
        assert matches[0].match_type == MatchType.DUPLICATE
        assert matches[1].match_type == MatchType.MERGE

    def test_max_results_respected(self) -> None:
        hits = [_make_hit(f"other/{i}.md", f"Match {i}", distance=0.05) for i in range(20)]
        store = FakeStore(hits)
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        matches = detector.find_duplicates(note, max_results=3)
        assert len(matches) == 3


# ---------------------------------------------------------------------------
# Tests — Results cache
# ---------------------------------------------------------------------------


class TestResultsCache:
    def test_results_stored(self) -> None:
        store = FakeStore(
            [
                _make_hit("other/dup.md", distance=0.05),
            ]
        )
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()
        detector.find_duplicates(note)

        cached = detector.get_results("test/note.md")
        assert len(cached) == 1

    def test_get_results_miss(self) -> None:
        detector = DuplicateDetector(
            FakeDuplicateConfig(),  # type: ignore[arg-type]
            FakeStore(),  # type: ignore[arg-type]
        )
        assert detector.get_results("nonexistent.md") == []

    def test_clear_specific(self) -> None:
        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        detector.find_duplicates(_make_note())

        detector.clear_results("test/note.md")
        assert detector.get_results("test/note.md") == []

    def test_clear_all(self) -> None:
        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        detector.find_duplicates(_make_note())

        detector.clear_results()
        assert detector.result_count == 0

    def test_result_count(self) -> None:
        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        assert detector.result_count == 0

        detector.find_duplicates(_make_note())
        assert detector.result_count == 1


# ---------------------------------------------------------------------------
# Tests — Event bus callback
# ---------------------------------------------------------------------------


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_on_note_changed_runs_detection(self) -> None:
        from vaultmind.vault.events import NoteCreatedEvent

        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]
        note = _make_note()

        event = NoteCreatedEvent(path=note.path, note=note, chunks_indexed=1)
        await detector.on_note_changed(event)

        assert detector.result_count == 1

    @pytest.mark.asyncio
    async def test_on_note_changed_disabled(self) -> None:
        from vaultmind.vault.events import NoteCreatedEvent

        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        config = FakeDuplicateConfig(enabled=False)
        detector = DuplicateDetector(config, store)  # type: ignore[arg-type]
        note = _make_note()

        event = NoteCreatedEvent(path=note.path, note=note, chunks_indexed=1)
        await detector.on_note_changed(event)

        assert detector.result_count == 0

    @pytest.mark.asyncio
    async def test_on_note_changed_no_note(self) -> None:
        from vaultmind.vault.events import NoteCreatedEvent

        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]

        from pathlib import Path

        event = NoteCreatedEvent(path=Path("ghost.md"), note=None, chunks_indexed=0)
        await detector.on_note_changed(event)

        assert detector.result_count == 0


# ---------------------------------------------------------------------------
# Tests — Batch scan
# ---------------------------------------------------------------------------


class TestScanVault:
    def test_batch_scan(self) -> None:
        store = FakeStore([_make_hit("other/dup.md", distance=0.05)])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]

        notes = [
            _make_note(path="a.md", title="A"),
            _make_note(path="b.md", title="B"),
        ]
        results = detector.scan_vault(notes)

        # Both notes should match "other/dup.md"
        assert len(results) == 2

    def test_batch_scan_empty(self) -> None:
        store = FakeStore([])
        detector = DuplicateDetector(FakeDuplicateConfig(), store)  # type: ignore[arg-type]

        results = detector.scan_vault([_make_note()])
        assert len(results) == 0
