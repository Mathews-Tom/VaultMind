"""SQLite-backed persistence for thinking sessions."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class SessionStore:
    """Persists thinking sessions to SQLite."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thinking_sessions (
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                history TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_active REAL NOT NULL,
                PRIMARY KEY (user_id, session_name)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thinking_session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                batch_number INTEGER NOT NULL,
                start_turn_index INTEGER NOT NULL,
                end_turn_index INTEGER NOT NULL,
                summary_text TEXT NOT NULL,
                key_topics TEXT NOT NULL,
                open_questions TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE (user_id, session_name, batch_number)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thinking_session_archives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                history TEXT NOT NULL,
                summaries TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_active REAL NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                error TEXT
            )
            """
        )
        self._conn.commit()

    def load(self, user_id: int, session_name: str = "default") -> list[dict[str, str]] | None:
        row = self._conn.execute(
            "SELECT history FROM thinking_sessions WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        ).fetchone()
        if row is None:
            return None
        result: list[dict[str, str]] = json.loads(row[0])
        return result

    def load_active(self, user_id: int, ttl: int) -> dict[str, Any] | None:
        """Select the current conversation without touching expired generations."""
        row = self._conn.execute(
            "SELECT session_name, history, last_active FROM thinking_sessions "
            "WHERE user_id = ? AND (session_name = 'default' OR session_name LIKE 'thinking:%') "
            "AND last_active >= ? ORDER BY last_active DESC LIMIT 1",
            (user_id, time.time() - ttl),
        ).fetchone()
        if row is None:
            return None
        return dict(session_name=row[0], history=json.loads(row[1]), last_active=row[2])

    def snapshot(self, user_id: int, session_name: str) -> dict[str, Any] | None:
        """Read a complete stored revision without changing its lifecycle."""
        row = self._conn.execute(
            "SELECT history, created_at, last_active FROM thinking_sessions "
            "WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        ).fetchone()
        if row is None:
            return None
        cursor = self._conn.execute(
            "SELECT * FROM thinking_session_summaries "
            "WHERE user_id = ? AND session_name = ? ORDER BY batch_number",
            (user_id, session_name),
        )
        assert cursor.description is not None
        columns = [column[0] for column in cursor.description]
        return dict(
            user_id=user_id,
            session_name=session_name,
            history=json.loads(row[0]),
            created_at=row[1],
            last_active=row[2],
            summaries=[dict(zip(columns, item, strict=True)) for item in cursor.fetchall()],
        )

    def save(
        self,
        user_id: int,
        history: list[dict[str, str]],
        session_name: str = "default",
    ) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO thinking_sessions (user_id, session_name, history, created_at, last_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id, session_name) DO UPDATE SET
                history = excluded.history,
                last_active = excluded.last_active
            """,
            (user_id, session_name, json.dumps(history), now, now),
        )
        self._conn.commit()

    def delete(self, user_id: int, session_name: str = "default") -> None:
        self._conn.execute(
            "DELETE FROM thinking_sessions WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        )
        self._conn.execute(
            "DELETE FROM thinking_session_summaries WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        )
        self._conn.execute(
            "DELETE FROM thinking_session_archives WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        )
        self._conn.commit()

    def delete_conversations(self, user_id: int) -> None:
        """Explicit user clear, including every retained conversation generation."""
        with self._conn:
            for table in (
                "thinking_sessions",
                "thinking_session_summaries",
                "thinking_session_archives",
            ):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE user_id = ? "
                    "AND (session_name = 'default' OR session_name LIKE 'thinking:%')",
                    (user_id,),
                )

    def _archive_before(self, cutoff: float, excluded_user_ids: frozenset[int]) -> None:
        rows = self._conn.execute(
            "SELECT user_id, session_name, history, created_at, last_active "
            "FROM thinking_sessions WHERE last_active < ?",
            (cutoff,),
        ).fetchall()
        for uid, name, history, created, active in rows:
            if uid in excluded_user_ids:
                continue
            cursor = self._conn.execute(
                "SELECT * FROM thinking_session_summaries "
                "WHERE user_id = ? AND session_name = ? ORDER BY batch_number",
                (uid, name),
            )
            assert cursor.description is not None
            columns = [column[0] for column in cursor.description]
            summaries = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
            self._conn.execute(
                "INSERT INTO thinking_session_archives "
                "(user_id, session_name, history, summaries, created_at, last_active) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uid, name, history, json.dumps(summaries), created, active),
            )
            self._conn.execute(
                "DELETE FROM thinking_sessions WHERE user_id = ? AND session_name = ?",
                (uid, name),
            )
            self._conn.execute(
                "DELETE FROM thinking_session_summaries WHERE user_id = ? AND session_name = ?",
                (uid, name),
            )

    def claim_expired(
        self, ttl: int, *, excluded_user_ids: frozenset[int] = frozenset()
    ) -> list[dict[str, Any]]:
        """Atomically archive and claim pending work before any provider call.

        Claimed records survive interruption but are not automatically retried:
        output/index side effects may already exist. Failed records stay failed.
        Active users are excluded without an unbounded SQL parameter list.
        """
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._archive_before(time.time() - ttl, excluded_user_ids)
            rows = [
                row
                for row in self._conn.execute(
                    "SELECT id, user_id, history, last_active, session_name, created_at, summaries "
                    "FROM thinking_session_archives WHERE state = 'pending' ORDER BY id"
                )
                if row[1] not in excluded_user_ids
            ]
            self._conn.executemany(
                "UPDATE thinking_session_archives SET state = 'claimed' "
                "WHERE id = ? AND state = 'pending'",
                ((row[0],) for row in rows),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return [
            dict(
                id=rid,
                user_id=uid,
                history=json.loads(history),
                last_active=active,
                session_name=name,
                created_at=created,
                summaries=json.loads(summaries),
            )
            for rid, uid, history, active, name, created, summaries in rows
        ]

    def finish_finalization(self, record_id: int, state: str, error: str | None = None) -> None:
        """Record actual producer disposition while retaining all original data."""
        if state not in {"completed", "failed", "skipped"}:
            raise ValueError("Invalid session finalization state")
        with self._conn:
            self._conn.execute(
                "UPDATE thinking_session_archives SET state = ?, error = ? "
                "WHERE id = ? AND state = 'claimed'",
                (state, error, record_id),
            )

    def has_active_session(self, user_id: int, ttl: int) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM thinking_sessions WHERE user_id = ? "
                "AND (session_name = 'default' OR session_name LIKE 'thinking:%') "
                "AND last_active >= ? LIMIT 1",
                (user_id, time.time() - ttl),
            ).fetchone()
            is not None
        )

    def has_session(self, user_id: int, session_name: str = "default") -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM thinking_sessions WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        ).fetchone()
        return row is not None

    def get_summaries(self, user_id: int, session_name: str = "default") -> list[dict[str, Any]]:
        """Retrieve all summaries for a session, ordered by batch_number ascending."""
        rows = self._conn.execute(
            """
            SELECT batch_number, start_turn_index, end_turn_index,
                   summary_text, key_topics, open_questions
            FROM thinking_session_summaries
            WHERE user_id = ? AND session_name = ?
            ORDER BY batch_number ASC
            """,
            (user_id, session_name),
        ).fetchall()
        return [
            {
                "batch_number": r[0],
                "start_turn": r[1],
                "end_turn": r[2],
                "summary": r[3],
                "topics": json.loads(r[4]),
                "questions": json.loads(r[5]),
            }
            for r in rows
        ]

    def save_summary(
        self,
        user_id: int,
        session_name: str,
        batch_number: int,
        start_turn_index: int,
        end_turn_index: int,
        summary_text: str,
        key_topics: list[str],
        open_questions: list[str],
    ) -> None:
        """Save a batch summary. Uses INSERT OR REPLACE for idempotency."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO thinking_session_summaries
            (user_id, session_name, batch_number, start_turn_index, end_turn_index,
             summary_text, key_topics, open_questions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_name,
                batch_number,
                start_turn_index,
                end_turn_index,
                summary_text,
                json.dumps(key_topics),
                json.dumps(open_questions),
                time.time(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
