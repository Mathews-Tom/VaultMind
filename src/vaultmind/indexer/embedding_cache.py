"""SQLite-backed embedding cache — eliminates redundant API calls during re-indexing."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    PRIMARY KEY (content_hash, provider, model)
);
"""


def content_hash(text: str) -> str:
    """SHA-256 hex digest of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Pack a float list into a compact binary blob (little-endian float32)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float list."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class EmbeddingCache:
    """SQLite-backed cache for embedding vectors.

    Keyed by (content_hash, provider, model) so provider/model switches
    produce cache misses. Uses WAL mode for concurrent read performance.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        logger.info("Embedding cache opened at %s", db_path)

    def get(self, content_hash: str, provider: str, model: str) -> list[float] | None:
        """Retrieve a cached embedding. Returns None on miss."""
        row = self._conn.execute(
            "SELECT embedding FROM embedding_cache "
            "WHERE content_hash = ? AND provider = ? AND model = ?",
            (content_hash, provider, model),
        ).fetchone()
        if row is None:
            return None
        # Update last_accessed
        self._conn.execute(
            "UPDATE embedding_cache SET last_accessed = ? "
            "WHERE content_hash = ? AND provider = ? AND model = ?",
            (time.time(), content_hash, provider, model),
        )
        self._conn.commit()
        return _blob_to_embedding(row[0])

    def put(
        self,
        content_hash: str,
        provider: str,
        model: str,
        dimensions: int,
        embedding: list[float],
    ) -> None:
        """Store an embedding in the cache."""
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache "
            "(content_hash, provider, model, dimensions, embedding, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content_hash, provider, model, dimensions, _embedding_to_blob(embedding), now, now),
        )
        self._conn.commit()

    def get_batch(
        self, content_hashes: list[str], provider: str, model: str
    ) -> dict[str, list[float]]:
        """Retrieve multiple cached embeddings. Returns dict mapping hash -> embedding."""
        if not content_hashes:
            return {}

        placeholders = ",".join("?" for _ in content_hashes)
        rows = self._conn.execute(
            f"SELECT content_hash, embedding FROM embedding_cache "  # noqa: S608
            f"WHERE provider = ? AND model = ? AND content_hash IN ({placeholders})",
            [provider, model, *content_hashes],
        ).fetchall()

        result: dict[str, list[float]] = {}
        hit_hashes: list[str] = []
        for row in rows:
            result[row[0]] = _blob_to_embedding(row[1])
            hit_hashes.append(row[0])

        # Batch update last_accessed for hits
        if hit_hashes:
            now = time.time()
            hit_placeholders = ",".join("?" for _ in hit_hashes)
            self._conn.execute(
                f"UPDATE embedding_cache SET last_accessed = ? "  # noqa: S608
                f"WHERE provider = ? AND model = ? AND content_hash IN ({hit_placeholders})",
                [now, provider, model, *hit_hashes],
            )
            self._conn.commit()

        return result

    def put_batch(
        self,
        entries: list[tuple[str, int, list[float]]],
        provider: str,
        model: str,
    ) -> None:
        """Store multiple embeddings. Each entry is (content_hash, dimensions, embedding)."""
        if not entries:
            return
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache "
            "(content_hash, provider, model, dimensions, embedding, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (h, provider, model, dims, _embedding_to_blob(emb), now, now)
                for h, dims, emb in entries
            ],
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        """Return cache statistics: total_entries and total_size_bytes."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(embedding)), 0) FROM embedding_cache"
        ).fetchone()
        assert row is not None
        return {"total_entries": row[0], "total_size_bytes": row[1]}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
