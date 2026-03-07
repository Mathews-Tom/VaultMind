"""Tests for note-type-aware search ranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vaultmind.indexer.ranking import (
    ARCHIVED_MULTIPLIER,
    RankedResult,
    rank_results,
    score,
)


class TestScore:
    def test_permanent_no_decay(self) -> None:
        """Permanent notes get 1.3x multiplier, no temporal decay."""
        s = score(1.0, "permanent", "2020-01-01", "active")
        assert s == pytest.approx(1.3)

    def test_fleeting_decays_aggressively(self) -> None:
        """Fleeting notes have 7-day half-life."""
        old_date = (datetime.now(UTC) - timedelta(days=14)).isoformat()
        s = score(1.0, "fleeting", old_date, "active")
        # 0.8 * exp(-0.693 * 14 / 7) = 0.8 * exp(-1.386) = 0.8 * 0.25 = 0.2
        assert s < 0.3
        assert s > 0.0

    def test_archived_project_suppressed(self) -> None:
        """Archived projects get 0.4x status multiplier."""
        s = score(1.0, "project", "", "archived")
        assert s == pytest.approx(1.0 * ARCHIVED_MULTIPLIER)

    def test_completed_project_suppressed(self) -> None:
        """Completed projects also get 0.4x."""
        s = score(1.0, "project", "", "completed")
        assert s == pytest.approx(1.0 * ARCHIVED_MULTIPLIER)

    def test_untyped_note_uses_defaults(self) -> None:
        """Unknown note types get default config (1.0x, 30-day decay)."""
        s = score(1.0, "unknown_type", "", "active")
        assert s == pytest.approx(1.0)

    def test_very_old_fleeting_near_zero(self) -> None:
        """365+ day old fleeting note should be near zero but positive."""
        old_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        s = score(1.0, "fleeting", old_date, "active")
        assert s > 0.0
        assert s < 0.01

    def test_empty_created_at_skips_decay(self) -> None:
        """Notes without created metadata skip decay."""
        s = score(1.0, "fleeting", "", "active")
        assert s == pytest.approx(0.8)  # type multiplier only

    def test_invalid_created_at_skips_decay(self) -> None:
        """Malformed date string skips decay."""
        s = score(1.0, "daily", "not-a-date", "active")
        assert s == pytest.approx(0.9)  # type multiplier only

    def test_archived_applies_after_type(self) -> None:
        """Status multiplier stacks with type multiplier."""
        s = score(1.0, "permanent", "", "archived")
        assert s == pytest.approx(1.3 * ARCHIVED_MULTIPLIER)

    def test_concept_no_decay(self) -> None:
        """Concept notes get 1.2x, no decay."""
        old_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        s = score(1.0, "concept", old_date, "active")
        assert s == pytest.approx(1.2)


class TestRankResults:
    def _make_hit(
        self,
        chunk_id: str = "test::0",
        distance: float = 0.3,
        note_type: str = "fleeting",
        created: str = "",
        status: str = "active",
        content: str = "test content",
    ) -> dict:
        return {
            "chunk_id": chunk_id,
            "distance": distance,
            "content": content,
            "metadata": {
                "note_type": note_type,
                "created": created,
                "status": status,
            },
        }

    def test_score_ordering_integration(self) -> None:
        """Permanent note ranks above fleeting with same semantic distance."""
        hits = [
            self._make_hit("a::0", 0.3, "fleeting"),
            self._make_hit("b::0", 0.3, "permanent"),
        ]
        results = rank_results(hits, enabled=True)
        assert results[0].note_type == "permanent"
        assert results[1].note_type == "fleeting"

    def test_disabled_ranking_preserves_raw_order(self) -> None:
        """When disabled, final_score equals raw_score."""
        hits = [
            self._make_hit("a::0", 0.3, "fleeting"),
            self._make_hit("b::0", 0.5, "permanent"),
        ]
        results = rank_results(hits, enabled=False)
        # Lower distance = higher similarity = first
        assert results[0].chunk_id == "a::0"
        for r in results:
            assert r.final_score == r.raw_score

    def test_returns_ranked_result_dataclass(self) -> None:
        hits = [self._make_hit()]
        results = rank_results(hits)
        assert isinstance(results[0], RankedResult)

    def test_empty_hits_returns_empty(self) -> None:
        assert rank_results([]) == []
