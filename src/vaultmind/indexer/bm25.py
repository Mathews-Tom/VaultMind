"""SQLite FTS5-backed BM25 keyword index for vault chunks.

No external dependencies — FTS5 is bundled with Python's sqlite3.
"""

from __future__ import annotations

import logging
import sqlite3
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id,
    note_path,
    note_title,
    content,
    tokenize='porter unicode61'
);
"""

_CREATE_META = """
CREATE TABLE IF NOT EXISTS bm25_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_REPLACEMENT_IDS = """
CREATE TEMP TABLE IF NOT EXISTS bm25_replacement_ids (
    chunk_id TEXT PRIMARY KEY
) WITHOUT ROWID;
"""

_CREATE_PERMITTED_PATHS = """
CREATE TEMP TABLE IF NOT EXISTS bm25_permitted_paths (
    note_path TEXT PRIMARY KEY
) WITHOUT ROWID;
"""


class BM25Index:
    """FTS5-backed keyword index.

    Stores chunk text in an FTS5 virtual table and uses SQLite's built-in
    BM25 ranking (accessed via the ``rank`` hidden column) for retrieval.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._write_lock = Lock()
        self._init_schema()
        logger.info("BM25Index opened at %s (%d chunks)", db_path, self.count)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._con:
            self._con.executescript(
                _CREATE_FTS + _CREATE_META + _CREATE_REPLACEMENT_IDS + _CREATE_PERMITTED_PATHS
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunk_id: str, note_path: str, note_title: str, content: str) -> None:
        """Insert or replace a single chunk."""
        self.upsert_batch([(chunk_id, note_path, note_title, content)])

    def upsert_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Insert or replace a batch of (chunk_id, note_path, note_title, content) tuples.

        FTS5 does not support ``INSERT OR REPLACE`` directly, so we delete
        existing rows for the given chunk_ids first.
        """
        if not rows:
            return
        with self._write_lock, self._con:
            self._con.executemany(
                "INSERT OR IGNORE INTO bm25_replacement_ids(chunk_id) VALUES (?)",
                ((row[0],) for row in rows),
            )
            # FTS5 cannot index chunk_id: scan once, not once per ID batch.
            self._con.execute(
                "DELETE FROM fts_chunks WHERE chunk_id IN"
                " (SELECT chunk_id FROM bm25_replacement_ids)"
            )
            self._con.executemany(
                "INSERT INTO fts_chunks(chunk_id, note_path, note_title, content)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
            self._con.execute("DELETE FROM bm25_replacement_ids")
        logger.debug("BM25 upserted %d chunks", len(rows))

    def delete_note(self, note_path: str) -> None:
        """Remove all chunks belonging to a note."""
        with self._write_lock, self._con:
            cur = self._con.execute("DELETE FROM fts_chunks WHERE note_path = ?", (note_path,))
        logger.debug("BM25 deleted %d chunks for %s", cur.rowcount, note_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self, query: str, n_results: int = 20, *, note_paths: list[str] | None = None
    ) -> list[dict[str, float | str]]:
        """Full-text search using FTS5 BM25 ranking.

        Returns list of dicts with keys: chunk_id, note_path, note_title, bm25_score.
        FTS5 ``rank`` is negative (lower = better), so we negate it to produce
        a positive score where higher means more relevant.
        """
        query = query.strip()
        if not query:
            return []

        # Escape FTS5 special characters to avoid syntax errors on bare queries
        safe_query = _fts5_escape(query)

        # Serialize the temporary permitted relation with writes and other scoped
        # reads. executemany uses one host parameter regardless of vault size.
        # Finish the temporary writes and read snapshot before releasing the lock.
        with self._write_lock, self._con:
            if note_paths is not None:
                self._con.execute("DELETE FROM bm25_permitted_paths")
                self._con.executemany(
                    "INSERT OR IGNORE INTO bm25_permitted_paths(note_path) VALUES (?)",
                    ((path,) for path in note_paths),
                )
            scope = (
                " AND note_path IN (SELECT note_path FROM bm25_permitted_paths)"
                if note_paths is not None
                else ""
            )
            try:
                rows = self._con.execute(
                    "SELECT chunk_id, note_path, note_title, -rank AS bm25_score"
                    " FROM fts_chunks WHERE fts_chunks MATCH ?"
                    + scope
                    + " ORDER BY rank, note_path, chunk_id LIMIT ?",
                    (safe_query, n_results),
                ).fetchall()
            except sqlite3.OperationalError:
                logger.warning("BM25 search failed")
                return []
            finally:
                if note_paths is not None:
                    self._con.execute("DELETE FROM bm25_permitted_paths")

        return [
            {
                "chunk_id": row["chunk_id"],
                "note_path": row["note_path"],
                "note_title": row["note_title"],
                "bm25_score": float(row["bm25_score"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Properties / lifecycle
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of indexed chunks."""
        row = self._con.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._con.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fts5_escape(query: str) -> str:
    """Wrap each token in double-quotes for safe FTS5 MATCH.

    This prevents special characters (``AND``, ``OR``, ``*``, etc.) from
    being interpreted as FTS5 syntax. Porter stemming still applies through
    the tokenizer on each quoted token.
    """
    tokens = query.split()
    if not tokens:
        return '""'
    escaped: list[str] = []
    for tok in tokens:
        # Remove embedded double-quotes; wrap in double-quotes
        safe = tok.replace('"', "")
        if safe:
            escaped.append(f'"{safe}"')
    return " ".join(escaped) if escaped else '""'
