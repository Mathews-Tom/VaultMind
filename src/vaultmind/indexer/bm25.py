"""SQLite FTS5-backed BM25 keyword index for vault chunks.

No external dependencies — FTS5 is bundled with Python's sqlite3.
"""

from __future__ import annotations

import logging
import sqlite3
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


class BM25Index:
    """FTS5-backed keyword index.

    Stores chunk text in an FTS5 virtual table and uses SQLite's built-in
    BM25 ranking (accessed via the ``rank`` hidden column) for retrieval.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("BM25Index opened at %s (%d chunks)", db_path, self.count)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._con:
            self._con.executescript(_CREATE_FTS + _CREATE_META)

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
        chunk_ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(chunk_ids))
        with self._con:
            self._con.execute(
                f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._con.executemany(
                "INSERT INTO fts_chunks(chunk_id, note_path, note_title, content)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.debug("BM25 upserted %d chunks", len(rows))

    def delete_note(self, note_path: str) -> None:
        """Remove all chunks belonging to a note."""
        with self._con:
            cur = self._con.execute("DELETE FROM fts_chunks WHERE note_path = ?", (note_path,))
        logger.debug("BM25 deleted %d chunks for %s", cur.rowcount, note_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 20) -> list[dict[str, float | str]]:
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

        try:
            rows = self._con.execute(
                "SELECT chunk_id, note_path, note_title, -rank AS bm25_score"
                " FROM fts_chunks"
                " WHERE fts_chunks MATCH ?"
                " ORDER BY rank"  # rank is negative; ORDER BY rank = best first
                " LIMIT ?",
                (safe_query, n_results),
            ).fetchall()
        except sqlite3.OperationalError:
            # If the query is malformed after escaping, return empty rather than crash
            logger.warning("BM25 search failed for query %r", query)
            return []

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
