"""SQLite-backed durable cursor + run-summary store for source connectors.

Mirrors `memory/gaps.py::GapStore`'s and `services/review_queue.py::ReviewQueue`'s
established pattern: `sqlite3.connect(check_same_thread=False)`,
`row_factory=sqlite3.Row`, `executescript` DDL, dataclass row models.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vaultmind.sources.models import ConnectorState, RunSummary, _row_to_run_summary

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TC003 — used at runtime for mkdir/connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connector_state (
    instance_name TEXT PRIMARY KEY,
    last_seen_id TEXT NOT NULL DEFAULT '',
    etag TEXT NOT NULL DEFAULT '',
    last_run TEXT,
    run_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_name TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT NOT NULL,
    items_fetched INTEGER NOT NULL,
    items_ingested INTEGER NOT NULL,
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_run_summaries_instance ON run_summaries(instance_name);

-- Bounded run history per instance — a trigger-free application-level
-- cap (see `record_run`'s own trim query) keeps this table small without
-- a scheduled cleanup job, matching the repo's lazy-reap idiom
-- (GapStore.list_open(), SessionStore.cleanup_expired()).
"""

_MAX_RUNS_PER_INSTANCE = 20


def _row_to_state(row: sqlite3.Row) -> ConnectorState:
    return ConnectorState(
        instance_name=row["instance_name"],
        last_seen_id=row["last_seen_id"],
        etag=row["etag"],
        last_run=datetime.fromisoformat(row["last_run"]) if row["last_run"] else None,
        run_count=row["run_count"],
    )


class SourceStore:
    """SQLite-backed cursor + bounded run-summary store for source connectors."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_state(self, instance_name: str) -> ConnectorState:
        """Return the durable cursor for `instance_name`, or a fresh (empty)
        cursor if this instance has never run — never raises for an
        unknown instance, matching `run`'s "first run ingests everything"
        contract."""
        row = self._conn.execute(
            "SELECT * FROM connector_state WHERE instance_name = ?", (instance_name,)
        ).fetchone()
        return _row_to_state(row) if row else ConnectorState(instance_name=instance_name)

    def save_state(self, state: ConnectorState) -> None:
        """Upsert the durable cursor for `state.instance_name`."""
        self._conn.execute(
            """
            INSERT INTO connector_state
                (instance_name, last_seen_id, etag, last_run, run_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(instance_name) DO UPDATE SET
                last_seen_id = excluded.last_seen_id,
                etag = excluded.etag,
                last_run = excluded.last_run,
                run_count = excluded.run_count
            """,
            (
                state.instance_name,
                state.last_seen_id,
                state.etag,
                state.last_run.isoformat() if state.last_run else None,
                state.run_count,
            ),
        )
        self._conn.commit()

    def record_run(self, summary: RunSummary) -> None:
        """Append one run summary, trimming to the most recent
        `_MAX_RUNS_PER_INSTANCE` rows for that instance."""
        self._conn.execute(
            """
            INSERT INTO run_summaries
                (instance_name, started, finished, items_fetched, items_ingested, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary.instance_name,
                summary.started.isoformat(),
                summary.finished.isoformat(),
                summary.items_fetched,
                summary.items_ingested,
                summary.error,
            ),
        )
        self._conn.execute(
            """
            DELETE FROM run_summaries
            WHERE instance_name = ? AND id NOT IN (
                SELECT id FROM run_summaries WHERE instance_name = ?
                ORDER BY id DESC LIMIT ?
            )
            """,
            (summary.instance_name, summary.instance_name, _MAX_RUNS_PER_INSTANCE),
        )
        self._conn.commit()

    def list_runs(self, instance_name: str, limit: int = 10) -> list[RunSummary]:
        """Most-recent-first run history for `instance_name`."""
        rows = self._conn.execute(
            "SELECT * FROM run_summaries WHERE instance_name = ? ORDER BY id DESC LIMIT ?",
            (instance_name, limit),
        ).fetchall()
        return [_row_to_run_summary(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def advance_cursor(
    store: SourceStore,
    instance_name: str,
    *,
    next_cursor_id: str | None,
    next_etag: str | None,
) -> None:
    """Persist the post-run cursor for `instance_name`, bumping `run_count`
    and `last_run`. `None` for either field leaves that part of the cursor
    unchanged (e.g. a run that found zero new items keeps the prior
    `last_seen_id`)."""
    prior = store.get_state(instance_name)
    store.save_state(
        ConnectorState(
            instance_name=instance_name,
            last_seen_id=next_cursor_id if next_cursor_id is not None else prior.last_seen_id,
            etag=next_etag if next_etag is not None else prior.etag,
            last_run=datetime.now(UTC),
            run_count=prior.run_count + 1,
        )
    )


__all__ = ["SourceStore", "advance_cursor"]
