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
        self._conn.commit()

    def cleanup_expired(self, ttl: int) -> int:
        cutoff = time.time() - ttl
        cursor = self._conn.execute(
            "DELETE FROM thinking_sessions WHERE last_active < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount

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

    def count_turns(self, user_id: int, session_name: str = "default") -> int:
        """Count total turns in the session history. Returns 0 if no session."""
        history = self.load(user_id, session_name)
        if history is None:
            return 0
        return len(history)

    def get_unsummarized_batch(
        self,
        user_id: int,
        session_name: str,
        recent_turns_to_keep: int,
        batch_size: int,
    ) -> tuple[list[dict[str, str]], int, int] | None:
        """Get the oldest batch of turns eligible for summarization.

        Returns (turns_to_summarize, start_turn_idx, end_turn_idx) or None if
        no summarization is needed.
        """
        history = self.load(user_id, session_name)
        if history is None or len(history) <= recent_turns_to_keep:
            return None

        summaries = self.get_summaries(user_id, session_name)
        next_start = summaries[-1]["end_turn"] + 1 if summaries else 0

        next_end = next_start + batch_size - 1

        # Don't eat into the recent turns we want to keep in full
        if next_end >= len(history) - recent_turns_to_keep:
            return None

        return history[next_start : next_end + 1], next_start, next_end

    def close(self) -> None:
        self._conn.close()
