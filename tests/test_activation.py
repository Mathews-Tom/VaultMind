"""Tests for activation-based note decay tracking."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.indexer.activation import ActivationScore, ActivationTracker


@pytest.fixture
def tracker(tmp_path: Path) -> ActivationTracker:
    t = ActivationTracker(tmp_path / "activations.db")
    yield t
    t.close()


class TestActivationTracker:
    def test_record_and_score(self, tracker: ActivationTracker) -> None:
        """Recording activations produces a positive score."""
        tracker.record("notes/example.md")
        tracker.record("notes/example.md")
        tracker.record("notes/example.md")

        result = tracker.get_score("notes/example.md")
        assert isinstance(result, ActivationScore)
        assert result.total_activations == 3
        assert result.score > 0.0

    def test_recent_activations_boost(self, tracker: ActivationTracker) -> None:
        """Notes accessed recently score higher than old accesses."""
        # Inject an old activation directly into the DB at 60 days ago
        old_ts = time.time() - 60 * 86400.0
        tracker._conn.execute(
            "INSERT INTO activations (note_path, timestamp, activation_type) VALUES (?, ?, ?)",
            ("notes/old.md", old_ts, "access"),
        )
        tracker._conn.commit()

        # Recent note — 3 activations just now
        tracker.record("notes/recent.md")
        tracker.record("notes/recent.md")
        tracker.record("notes/recent.md")

        old_score = tracker.get_score("notes/old.md")
        recent_score = tracker.get_score("notes/recent.md")

        assert recent_score.score > old_score.score

    def test_no_activations_near_sigmoid_at_zero(self, tracker: ActivationTracker) -> None:
        """Unknown path returns sigmoid(−3) ≈ 0.047."""
        result = tracker.get_score("notes/never-accessed.md")
        expected = 1.0 / (1.0 + math.exp(3.0))  # sigmoid(-3)
        assert result.total_activations == 0
        assert result.recent_activations == 0
        assert result.score == pytest.approx(expected, abs=1e-6)

    def test_activation_types(self, tracker: ActivationTracker) -> None:
        """Different activation types are recorded and counted together."""
        tracker.record("notes/multi.md", activation_type="access")
        tracker.record("notes/multi.md", activation_type="edit")
        tracker.record("notes/multi.md", activation_type="access")

        result = tracker.get_score("notes/multi.md")
        assert result.total_activations == 3
        # All are recent
        assert result.recent_activations == 3

    def test_recent_activations_count(self, tracker: ActivationTracker) -> None:
        """recent_activations counts only events within the last 30 days."""
        old_ts = time.time() - 45 * 86400.0
        tracker._conn.execute(
            "INSERT INTO activations (note_path, timestamp, activation_type) VALUES (?, ?, ?)",
            ("notes/mixed.md", old_ts, "access"),
        )
        tracker._conn.commit()
        tracker.record("notes/mixed.md")  # recent

        result = tracker.get_score("notes/mixed.md")
        assert result.total_activations == 2
        assert result.recent_activations == 1
