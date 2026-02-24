"""SQLite-backed persistence for thinking sessions."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING

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

    def close(self) -> None:
        self._conn.close()
