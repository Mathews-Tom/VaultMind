"""Tests for memory horizon (short-term / long-term episode classification)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.memory.models import Episode, MemoryHorizon, OutcomeStatus
from vaultmind.memory.store import EpisodeStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


class TestMemoryHorizonModel:
    def test_enum_values(self) -> None:
        assert MemoryHorizon.SHORT_TERM == "short_term"
        assert MemoryHorizon.LONG_TERM == "long_term"

    def test_episode_defaults_to_short_term(self) -> None:
        ep = Episode(
            episode_id="test",
            decision="test decision",
            context="",
            outcome="",
            outcome_status=OutcomeStatus.PENDING,
            lessons=[],
            entities=[],
            source_notes=[],
            created=__import__("datetime").datetime.now(),
        )
        assert ep.memory_horizon == MemoryHorizon.SHORT_TERM

    def test_episode_accepts_long_term(self) -> None:
        ep = Episode(
            episode_id="test",
            decision="test",
            context="",
            outcome="",
            outcome_status=OutcomeStatus.SUCCESS,
            lessons=[],
            entities=[],
            source_notes=[],
            created=__import__("datetime").datetime.now(),
            memory_horizon=MemoryHorizon.LONG_TERM,
        )
        assert ep.memory_horizon == MemoryHorizon.LONG_TERM


class TestStoreHorizonPersistence:
    def test_created_episode_has_short_term_horizon(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Test decision")
        loaded = store.get(ep.episode_id)
        assert loaded is not None
        assert loaded.memory_horizon == MemoryHorizon.SHORT_TERM

    def test_horizon_survives_roundtrip(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Roundtrip test")
        loaded = store.get(ep.episode_id)
        assert loaded is not None
        assert loaded.memory_horizon == MemoryHorizon.SHORT_TERM


class TestPromoteToLongTerm:
    def test_promote_old_episodes(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Old decision")
        # Backdate the created timestamp to 60 days ago
        store._conn.execute(
            "UPDATE episodes SET created = datetime('now', '-60 days') WHERE episode_id = ?",
            (ep.episode_id,),
        )
        store._conn.commit()

        promoted = store.promote_to_long_term(age_days=30)
        assert promoted == 1

        loaded = store.get(ep.episode_id)
        assert loaded is not None
        assert loaded.memory_horizon == MemoryHorizon.LONG_TERM

    def test_recent_episodes_not_promoted(self, store: EpisodeStore) -> None:
        store.create(decision="Recent decision")
        promoted = store.promote_to_long_term(age_days=30)
        assert promoted == 0

    def test_already_long_term_not_double_promoted(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Already promoted")
        # Backdate and promote
        store._conn.execute(
            "UPDATE episodes SET created = datetime('now', '-60 days') WHERE episode_id = ?",
            (ep.episode_id,),
        )
        store._conn.commit()
        store.promote_to_long_term(age_days=30)

        # Promote again — should not count
        promoted = store.promote_to_long_term(age_days=30)
        assert promoted == 0

    def test_promote_multiple_episodes(self, store: EpisodeStore) -> None:
        for i in range(3):
            ep = store.create(decision=f"Decision {i}")
            store._conn.execute(
                "UPDATE episodes SET created = datetime('now', '-60 days') WHERE episode_id = ?",
                (ep.episode_id,),
            )
        store._conn.commit()

        promoted = store.promote_to_long_term(age_days=30)
        assert promoted == 3


class TestQueryByHorizon:
    def test_query_short_term_only(self, store: EpisodeStore) -> None:
        store.create(decision="Short-term A")
        store.create(decision="Short-term B")

        results = store.query_by_horizon(MemoryHorizon.SHORT_TERM)
        assert len(results) == 2
        assert all(r.memory_horizon == MemoryHorizon.SHORT_TERM for r in results)

    def test_query_long_term_only(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Will be promoted")
        store._conn.execute(
            "UPDATE episodes SET created = datetime('now', '-60 days') WHERE episode_id = ?",
            (ep.episode_id,),
        )
        store._conn.commit()
        store.promote_to_long_term(age_days=30)

        store.create(decision="Still short-term")

        long_term = store.query_by_horizon(MemoryHorizon.LONG_TERM)
        short_term = store.query_by_horizon(MemoryHorizon.SHORT_TERM)

        assert len(long_term) == 1
        assert long_term[0].decision == "Will be promoted"
        assert len(short_term) == 1
        assert short_term[0].decision == "Still short-term"

    def test_query_empty_horizon_returns_empty(self, store: EpisodeStore) -> None:
        results = store.query_by_horizon(MemoryHorizon.LONG_TERM)
        assert results == []

    def test_query_respects_limit(self, store: EpisodeStore) -> None:
        for i in range(5):
            store.create(decision=f"Decision {i}")

        results = store.query_by_horizon(MemoryHorizon.SHORT_TERM, limit=2)
        assert len(results) == 2


class TestEpisodicConfig:
    def test_short_term_days_default(self) -> None:
        from vaultmind.config import EpisodicConfig

        cfg = EpisodicConfig()
        assert cfg.short_term_days == 30


class TestMigration:
    def test_existing_db_without_horizon_column(self, tmp_path: Path) -> None:
        """Simulate an existing DB without memory_horizon column."""
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                outcome_status TEXT NOT NULL DEFAULT 'pending',
                lessons TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                source_notes TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                created TEXT NOT NULL,
                resolved TEXT
            );
        """)
        conn.execute(
            "INSERT INTO episodes (episode_id, decision, created) VALUES (?, ?, ?)",
            ("legacy1", "Legacy decision", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Opening with EpisodeStore should migrate
        store = EpisodeStore(db_path)
        ep = store.get("legacy1")
        assert ep is not None
        assert ep.memory_horizon == MemoryHorizon.SHORT_TERM
        assert ep.decision == "Legacy decision"
        store.close()
