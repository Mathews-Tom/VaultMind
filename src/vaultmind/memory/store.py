"""SQLite-backed episodic store for decision-outcome tracking."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime for mkdir/connect

from vaultmind.memory.models import Episode, MemoryHorizon, OutcomeStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
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
    resolved TEXT,
    memory_horizon TEXT NOT NULL DEFAULT 'short_term'
);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(outcome_status);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created);
CREATE INDEX IF NOT EXISTS idx_episodes_horizon ON episodes(memory_horizon);
"""

_MIGRATE_HORIZON = (
    "ALTER TABLE episodes ADD COLUMN memory_horizon TEXT NOT NULL DEFAULT 'short_term'"
)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _row_to_episode(row: sqlite3.Row) -> Episode:
    row_dict = dict(row)
    horizon_raw: str = row_dict.get("memory_horizon", "short_term")
    return Episode(
        episode_id=row["episode_id"],
        decision=row["decision"],
        context=row["context"],
        outcome=row["outcome"],
        outcome_status=OutcomeStatus(row["outcome_status"]),
        lessons=json.loads(row["lessons"]),
        entities=json.loads(row["entities"]),
        source_notes=json.loads(row["source_notes"]),
        tags=json.loads(row["tags"]),
        created=datetime.fromisoformat(row["created"]),
        resolved=datetime.fromisoformat(row["resolved"]) if row["resolved"] else None,
        memory_horizon=MemoryHorizon(horizon_raw),
    )


class EpisodeStore:
    """SQLite-backed store for episodic memory (decisions + outcomes)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        # Migrate existing DBs before running full schema (index on memory_horizon)
        import contextlib

        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute(_MIGRATE_HORIZON)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create(
        self,
        decision: str,
        context: str = "",
        entities: list[str] | None = None,
        source_notes: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Episode:
        """Create a new pending episode."""
        episode = Episode(
            episode_id=_new_id(),
            decision=decision,
            context=context,
            outcome="",
            outcome_status=OutcomeStatus.PENDING,
            lessons=[],
            entities=entities or [],
            source_notes=source_notes or [],
            tags=tags or [],
            created=datetime.now(),
        )
        self._conn.execute(
            """
            INSERT INTO episodes
                (episode_id, decision, context, outcome, outcome_status,
                 lessons, entities, source_notes, tags, created, resolved,
                 memory_horizon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.decision,
                episode.context,
                episode.outcome,
                episode.outcome_status.value,
                json.dumps(episode.lessons),
                json.dumps(episode.entities),
                json.dumps(episode.source_notes),
                json.dumps(episode.tags),
                episode.created.isoformat(),
                None,
                episode.memory_horizon.value,
            ),
        )
        self._conn.commit()
        return episode

    def resolve(
        self,
        episode_id: str,
        outcome: str,
        status: OutcomeStatus,
        lessons: list[str],
    ) -> None:
        """Resolve an episode with its outcome, status, and lessons."""
        resolved = datetime.now()
        self._conn.execute(
            """
            UPDATE episodes
            SET outcome = ?, outcome_status = ?, lessons = ?, resolved = ?
            WHERE episode_id = ?
            """,
            (
                outcome,
                status.value,
                json.dumps(lessons),
                resolved.isoformat(),
                episode_id,
            ),
        )
        self._conn.commit()

    def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return _row_to_episode(row) if row else None

    def query_pending(self, limit: int = 20) -> list[Episode]:
        """Return pending episodes ordered by created desc."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE outcome_status = 'pending' ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def query_resolved(self, limit: int = 100) -> list[Episode]:
        """Return resolved episodes (non-pending), ordered by created desc."""
        rows = self._conn.execute(
            "SELECT * FROM episodes"
            " WHERE outcome_status != 'pending'"
            " ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def search_by_entity(self, entity: str, limit: int = 10) -> list[Episode]:
        """Find episodes mentioning a specific entity (case-insensitive substring)."""
        rows = self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE lower(entities) LIKE ?
            ORDER BY created DESC
            LIMIT ?
            """,
            (f'%"{entity.lower()}"%', limit),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def promote_to_long_term(self, age_days: int = 30) -> int:
        """Promote episodes older than age_days from short_term to long_term.

        Returns the number of episodes promoted.
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=age_days)).isoformat()
        cursor = self._conn.execute(
            """
            UPDATE episodes
            SET memory_horizon = 'long_term'
            WHERE memory_horizon = 'short_term' AND created < ?
            """,
            (cutoff,),
        )
        self._conn.commit()
        promoted = cursor.rowcount
        if promoted > 0:
            logger.info("Promoted %d episodes to long_term", promoted)
        return promoted

    def query_by_horizon(
        self,
        horizon: MemoryHorizon,
        limit: int = 50,
    ) -> list[Episode]:
        """Return episodes filtered by memory horizon, ordered by created desc."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE memory_horizon = ? ORDER BY created DESC LIMIT ?",
            (horizon.value, limit),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
