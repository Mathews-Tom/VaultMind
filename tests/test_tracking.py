"""Tests for tracking/preferences.py and tracking/analyzer.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.tracking.analyzer import analyze_preferences, generate_preference_report
from vaultmind.tracking.preferences import Interaction, InteractionType, PreferenceStore


@pytest.fixture
def store(tmp_path: Path) -> PreferenceStore:
    db = tmp_path / "test_prefs.db"
    return PreferenceStore(db_path=db)


class TestPreferenceStore:
    def test_record_and_query(self, store: PreferenceStore) -> None:
        interaction = Interaction(
            interaction_type=InteractionType.SEARCH,
            content="python async",
        )
        store.record(interaction)
        results = store.query(interaction_type=InteractionType.SEARCH)
        assert len(results) == 1
        assert results[0].content == "python async"

    def test_record_batch(self, store: PreferenceStore) -> None:
        interactions = [
            Interaction(interaction_type=InteractionType.CAPTURE, content=f"note {i}")
            for i in range(5)
        ]
        store.record_batch(interactions)
        results = store.query(interaction_type=InteractionType.CAPTURE)
        assert len(results) == 5

    def test_query_with_since_filter(self, store: PreferenceStore) -> None:
        old = Interaction(
            interaction_type=InteractionType.SEARCH,
            content="old query",
            timestamp=datetime.now() - timedelta(days=60),
        )
        new = Interaction(
            interaction_type=InteractionType.SEARCH,
            content="new query",
        )
        store.record(old)
        store.record(new)

        recent = store.query(
            interaction_type=InteractionType.SEARCH,
            since=datetime.now() - timedelta(days=30),
        )
        assert len(recent) == 1
        assert recent[0].content == "new query"

    def test_get_counts(self, store: PreferenceStore) -> None:
        store.record(Interaction(InteractionType.SEARCH, "q1"))
        store.record(Interaction(InteractionType.SEARCH, "q2"))
        store.record(Interaction(InteractionType.CAPTURE, "note"))
        counts = store.get_counts()
        assert counts[InteractionType.SEARCH] == 2
        assert counts[InteractionType.CAPTURE] == 1

    def test_get_top_searches(self, store: PreferenceStore) -> None:
        for _ in range(3):
            store.record(Interaction(InteractionType.SEARCH, "python"))
        store.record(Interaction(InteractionType.SEARCH, "rust"))

        top = store.get_top_searches(limit=5)
        assert top[0] == ("python", 3)
        assert top[1] == ("rust", 1)

    def test_get_top_tags(self, store: PreferenceStore) -> None:
        store.record(Interaction(InteractionType.TAG_APPROVED, "python"))
        store.record(Interaction(InteractionType.TAG_APPROVED, "python"))
        store.record(Interaction(InteractionType.TAG_REJECTED, "misc"))

        approved = store.get_top_tags(approved=True)
        assert approved[0] == ("python", 2)

        rejected = store.get_top_tags(approved=False)
        assert rejected[0] == ("misc", 1)

    def test_get_capture_topics(self, store: PreferenceStore) -> None:
        store.record(Interaction(InteractionType.CAPTURE, "learning python programming today"))
        store.record(Interaction(InteractionType.CAPTURE, "more python notes about async"))

        topics = store.get_capture_topics(limit=5)
        topic_words = [t[0] for t in topics]
        assert "python" in topic_words

    def test_get_active_hours(self, store: PreferenceStore) -> None:
        store.record(Interaction(InteractionType.SEARCH, "test"))
        hours = store.get_active_hours()
        assert len(hours) >= 1
        assert all(0 <= h <= 23 for h in hours)

    def test_empty_store(self, store: PreferenceStore) -> None:
        assert store.query() == []
        assert store.get_counts() == {}
        assert store.get_top_searches() == []
        assert store.get_active_hours() == []

    def test_close(self, store: PreferenceStore) -> None:
        store.close()  # Should not raise


class TestAnalyzer:
    def test_analyze_empty_store(self, store: PreferenceStore) -> None:
        insights = analyze_preferences(store, days=30)
        assert insights.total_interactions == 0
        assert insights.recommendations == []

    def test_analyze_with_data(self, store: PreferenceStore) -> None:
        for _ in range(5):
            store.record(Interaction(InteractionType.SEARCH, "python async"))
        store.record(Interaction(InteractionType.SUGGESTION_ACCEPTED, "note-a"))
        store.record(Interaction(InteractionType.SUGGESTION_REJECTED, "note-b"))

        insights = analyze_preferences(store, days=30)
        assert insights.total_interactions == 7
        assert insights.top_searches[0] == ("python async", 5)
        assert insights.suggestions_acceptance_rate == 0.5

    def test_generate_report_nonempty(self, store: PreferenceStore) -> None:
        store.record(Interaction(InteractionType.SEARCH, "test query"))
        store.record(Interaction(InteractionType.CAPTURE, "captured note content"))

        insights = analyze_preferences(store, days=30)
        report = generate_preference_report(insights)

        assert "# VaultMind Usage Insights" in report
        assert "test query" in report
        assert "Activity Distribution" in report

    def test_generate_report_empty(self, store: PreferenceStore) -> None:
        insights = analyze_preferences(store, days=30)
        report = generate_preference_report(insights)
        assert "# VaultMind Usage Insights" in report
        assert "0" in report
