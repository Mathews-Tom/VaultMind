"""Tests for tag synonym / merge detection (tag_analyzer module)."""

from __future__ import annotations

from pathlib import Path

from vaultmind.indexer.tag_analyzer import compute_tag_stats, find_synonyms
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(path: str = "note.md", tags: list[str] | None = None) -> Note:
    return Note(
        path=Path(path),
        title=path.replace(".md", ""),
        content="content",
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# compute_tag_stats
# ---------------------------------------------------------------------------


class TestComputeTagStats:
    def test_counts_tags(self) -> None:
        notes = [
            _make_note("a.md", ["ml", "ai"]),
            _make_note("b.md", ["ml"]),
        ]
        counts, co = compute_tag_stats(notes)
        assert counts["ml"] == 2
        assert counts["ai"] == 1

    def test_co_occurrence_recorded(self) -> None:
        notes = [
            _make_note("a.md", ["ml", "ai"]),
            _make_note("b.md", ["ml", "ai"]),
        ]
        counts, co = compute_tag_stats(notes)
        assert co[frozenset({"ml", "ai"})] == 2

    def test_no_self_pairs(self) -> None:
        notes = [_make_note("a.md", ["ml"])]
        counts, co = compute_tag_stats(notes)
        # No pair with a single unique tag
        assert all(len(pair) == 2 for pair in co)

    def test_empty_vault(self) -> None:
        counts, co = compute_tag_stats([])
        assert counts == {}
        assert co == {}

    def test_notes_with_no_tags(self) -> None:
        notes = [_make_note("a.md", []), _make_note("b.md", [])]
        counts, co = compute_tag_stats(notes)
        assert counts == {}
        assert co == {}


# ---------------------------------------------------------------------------
# find_synonyms — string similarity signal
# ---------------------------------------------------------------------------


class TestStringSimilarTagsDetected:
    def test_machine_learning_variants(self) -> None:
        """'machine-learning' and 'machinelearning' should be flagged."""
        tag_counts = {"machine-learning": 5, "machinelearning": 3}
        co: dict[frozenset[str], int] = {}
        results = find_synonyms(tag_counts, co, min_similarity=0.75, min_co_occurrence=1.0)
        assert len(results) == 1
        s = results[0]
        assert "machine-learning" in (s.tag_a, s.tag_b)
        assert "machinelearning" in (s.tag_a, s.tag_b)

    def test_plural_singular(self) -> None:
        """'note' / 'notes' should be flagged by similarity."""
        tag_counts = {"note": 10, "notes": 8}
        co: dict[frozenset[str], int] = {}
        results = find_synonyms(tag_counts, co, min_similarity=0.75, min_co_occurrence=1.0)
        assert any("note" in (s.tag_a, s.tag_b) and "notes" in (s.tag_a, s.tag_b) for s in results)

    def test_below_threshold_not_reported(self) -> None:
        """Completely dissimilar tags must not appear in results."""
        tag_counts = {"python": 5, "gardening": 5}
        co: dict[frozenset[str], int] = {}
        results = find_synonyms(tag_counts, co, min_similarity=0.75, min_co_occurrence=1.0)
        assert results == []


# ---------------------------------------------------------------------------
# find_synonyms — co-occurrence signal
# ---------------------------------------------------------------------------


class TestCoOccurringTagsDetected:
    def test_always_together_flagged(self) -> None:
        """Tags that always appear together should be flagged even if dissimilar strings."""
        tag_counts = {"python": 4, "programming": 4}
        # Both appear in every note together
        co: dict[frozenset[str], int] = {frozenset({"python", "programming"}): 4}
        results = find_synonyms(tag_counts, co, min_similarity=1.0, min_co_occurrence=0.5)
        assert len(results) == 1

    def test_low_co_occurrence_not_flagged(self) -> None:
        """Tags that occasionally co-occur should not be flagged."""
        tag_counts = {"python": 10, "programming": 10}
        co: dict[frozenset[str], int] = {frozenset({"python", "programming"}): 1}
        results = find_synonyms(tag_counts, co, min_similarity=1.0, min_co_occurrence=0.5)
        assert results == []


# ---------------------------------------------------------------------------
# find_synonyms — canonical selection
# ---------------------------------------------------------------------------


class TestCanonicalIsMoreFrequent:
    def test_higher_count_wins(self) -> None:
        tag_counts = {"ml": 10, "machine-learning": 3}
        co: dict[frozenset[str], int] = {}
        results = find_synonyms(
            tag_counts,
            co,
            min_similarity=0.0,  # force all pairs through
            min_co_occurrence=0.0,
        )
        # Find the pair we care about
        pair = next(
            s
            for s in results
            if "ml" in (s.tag_a, s.tag_b) and "machine-learning" in (s.tag_a, s.tag_b)
        )
        assert pair.suggested_canonical == "ml"

    def test_tie_resolved_alphabetically(self) -> None:
        tag_counts = {"beta": 5, "alpha": 5}
        co: dict[frozenset[str], int] = {frozenset({"beta", "alpha"}): 5}
        results = find_synonyms(tag_counts, co, min_similarity=0.0, min_co_occurrence=0.0)
        pair = next(
            s for s in results if "alpha" in (s.tag_a, s.tag_b) and "beta" in (s.tag_a, s.tag_b)
        )
        assert pair.suggested_canonical == "alpha"


# ---------------------------------------------------------------------------
# Integration — compute_tag_stats → find_synonyms pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_end_to_end(self) -> None:
        """Tags always appearing together in every note are surfaced as synonyms."""
        notes = [
            _make_note("a.md", ["ai", "ml"]),
            _make_note("b.md", ["ai", "ml"]),
            _make_note("c.md", ["ai", "ml"]),
        ]
        counts, co = compute_tag_stats(notes)
        results = find_synonyms(counts, co, min_similarity=1.0, min_co_occurrence=0.5)
        assert len(results) == 1
        s = results[0]
        assert frozenset({s.tag_a, s.tag_b}) == frozenset({"ai", "ml"})
        assert s.co_occurrence_ratio == 1.0
