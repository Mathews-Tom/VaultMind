"""SQLite-backed preference store for tracking user interactions."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "this",
        "that",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "not",
        "no",
        "yes",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "how",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "than",
        "then",
    }
)


class InteractionType(StrEnum):
    SEARCH = "search"
    CAPTURE = "capture"
    TAG_APPROVED = "tag_approved"
    TAG_REJECTED = "tag_rejected"
    SUGGESTION_ACCEPTED = "suggestion_accepted"
    SUGGESTION_REJECTED = "suggestion_rejected"
    DUPLICATE_MERGED = "duplicate_merged"
    DUPLICATE_DISMISSED = "duplicate_dismissed"
    BOOKMARK = "bookmark"
    RECALL = "recall"
    THINK = "think"
    URL_INGESTED = "url_ingested"


@dataclass(frozen=True, slots=True)
class Interaction:
    interaction_type: InteractionType
    content: str
    metadata: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class PreferenceStore:
    """SQLite-backed interaction tracking."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".vaultmind" / "data" / "preferences.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON interactions(interaction_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")
        self._conn.commit()

    def record(self, interaction: Interaction) -> None:
        """Record a single interaction."""
        self._conn.execute(
            "INSERT INTO interactions (interaction_type, content, metadata, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (
                interaction.interaction_type.value,
                interaction.content,
                interaction.metadata,
                interaction.timestamp.isoformat(),
            ),
        )
        self._conn.commit()

    def record_batch(self, interactions: list[Interaction]) -> None:
        """Record multiple interactions in a single transaction."""
        self._conn.executemany(
            "INSERT INTO interactions (interaction_type, content, metadata, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    i.interaction_type.value,
                    i.content,
                    i.metadata,
                    i.timestamp.isoformat(),
                )
                for i in interactions
            ],
        )
        self._conn.commit()

    def query(
        self,
        interaction_type: InteractionType | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Interaction]:
        """Query interactions with optional filters."""
        clauses: list[str] = []
        params: list[str | int] = []

        if interaction_type is not None:
            clauses.append("interaction_type = ?")
            params.append(interaction_type.value)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT interaction_type, content, metadata, timestamp "  # noqa: S608
            f"FROM interactions{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

        return [
            Interaction(
                interaction_type=InteractionType(row[0]),
                content=row[1],
                metadata=row[2],
                timestamp=datetime.fromisoformat(row[3]),
            )
            for row in rows
        ]

    def get_counts(self, since: datetime | None = None) -> dict[InteractionType, int]:
        """Get interaction counts by type."""
        if since is not None:
            rows = self._conn.execute(
                "SELECT interaction_type, COUNT(*) FROM interactions "
                "WHERE timestamp >= ? GROUP BY interaction_type",
                (since.isoformat(),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT interaction_type, COUNT(*) FROM interactions GROUP BY interaction_type"
            ).fetchall()

        return {InteractionType(row[0]): row[1] for row in rows}

    def get_top_searches(
        self, limit: int = 20, since: datetime | None = None
    ) -> list[tuple[str, int]]:
        """Most frequent search queries."""
        params: list[str | int] = [InteractionType.SEARCH.value]
        time_clause = ""
        if since is not None:
            time_clause = " AND timestamp >= ?"
            params.append(since.isoformat())
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT content, COUNT(*) as cnt FROM interactions "  # noqa: S608
            f"WHERE interaction_type = ?{time_clause} "
            f"GROUP BY content ORDER BY cnt DESC LIMIT ?",
            params,
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_top_tags(self, approved: bool = True, limit: int = 20) -> list[tuple[str, int]]:
        """Most frequently approved or rejected tags."""
        tag_type = InteractionType.TAG_APPROVED if approved else InteractionType.TAG_REJECTED
        rows = self._conn.execute(
            "SELECT content, COUNT(*) as cnt FROM interactions "
            "WHERE interaction_type = ? GROUP BY content ORDER BY cnt DESC LIMIT ?",
            (tag_type.value, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_capture_topics(
        self, limit: int = 20, since: datetime | None = None
    ) -> list[tuple[str, int]]:
        """Most common words in captured notes (basic frequency analysis)."""
        params: list[str | int] = [InteractionType.CAPTURE.value]
        time_clause = ""
        if since is not None:
            time_clause = " AND timestamp >= ?"
            params.append(since.isoformat())

        rows = self._conn.execute(
            f"SELECT content FROM interactions "  # noqa: S608
            f"WHERE interaction_type = ?{time_clause}",
            params,
        ).fetchall()

        counter: Counter[str] = Counter()
        for (content,) in rows:
            words = content.lower().split()
            counter.update(w for w in words if len(w) >= 4 and w not in _STOP_WORDS)

        return counter.most_common(limit)

    def get_active_hours(self, since: datetime | None = None) -> list[int]:
        """Hours of day (0-23) with most activity, sorted by frequency."""
        if since is not None:
            rows = self._conn.execute(
                "SELECT timestamp FROM interactions WHERE timestamp >= ?",
                (since.isoformat(),),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT timestamp FROM interactions").fetchall()

        counter: Counter[int] = Counter()
        for (ts_str,) in rows:
            dt = datetime.fromisoformat(ts_str)
            counter[dt.hour] += 1

        return [hour for hour, _ in counter.most_common()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
