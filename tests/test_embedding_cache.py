"""Tests for EmbeddingCache — SQLite-backed embedding vector cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.indexer.embedding_cache import (
    EmbeddingCache,
    _blob_to_embedding,
    _embedding_to_blob,
    content_hash,
)


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "cache.db")


PROVIDER = "openai"
MODEL = "text-embedding-3-small"
DIMS = 4
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


class TestContentHash:
    def test_deterministic(self) -> None:
        assert content_hash("hello") == content_hash("hello")

    def test_different_input_different_hash(self) -> None:
        assert content_hash("hello") != content_hash("world")

    def test_sha256_length(self) -> None:
        assert len(content_hash("test")) == 64


class TestBlobSerialization:
    def test_roundtrip(self) -> None:
        original = [1.0, 2.5, -3.7, 0.0]
        blob = _embedding_to_blob(original)
        recovered = _blob_to_embedding(blob)
        assert len(recovered) == len(original)
        for a, b in zip(original, recovered, strict=False):
            assert abs(a - b) < 1e-6

    def test_blob_size(self) -> None:
        emb = [0.0] * 1536
        blob = _embedding_to_blob(emb)
        assert len(blob) == 1536 * 4


class TestPutGet:
    def test_roundtrip(self, cache: EmbeddingCache) -> None:
        h = content_hash("test text")
        cache.put(h, PROVIDER, MODEL, DIMS, SAMPLE_EMBEDDING)
        result = cache.get(h, PROVIDER, MODEL)
        assert result is not None
        assert len(result) == DIMS
        for a, b in zip(result, SAMPLE_EMBEDDING, strict=False):
            assert abs(a - b) < 1e-6

    def test_miss_returns_none(self, cache: EmbeddingCache) -> None:
        assert cache.get("nonexistent", PROVIDER, MODEL) is None

    def test_provider_isolation(self, cache: EmbeddingCache) -> None:
        h = content_hash("same content")
        cache.put(h, "openai", MODEL, DIMS, SAMPLE_EMBEDDING)
        assert cache.get(h, "voyage", MODEL) is None
        assert cache.get(h, "openai", MODEL) is not None

    def test_model_isolation(self, cache: EmbeddingCache) -> None:
        h = content_hash("same content")
        cache.put(h, PROVIDER, "model-a", DIMS, SAMPLE_EMBEDDING)
        assert cache.get(h, PROVIDER, "model-b") is None
        assert cache.get(h, PROVIDER, "model-a") is not None

    def test_overwrite(self, cache: EmbeddingCache) -> None:
        h = content_hash("text")
        cache.put(h, PROVIDER, MODEL, DIMS, [1.0, 2.0, 3.0, 4.0])
        cache.put(h, PROVIDER, MODEL, DIMS, [5.0, 6.0, 7.0, 8.0])
        result = cache.get(h, PROVIDER, MODEL)
        assert result is not None
        assert abs(result[0] - 5.0) < 1e-6


class TestBatchOperations:
    def test_get_batch_mixed(self, cache: EmbeddingCache) -> None:
        h1 = content_hash("text1")
        h2 = content_hash("text2")
        h3 = content_hash("text3")
        cache.put(h1, PROVIDER, MODEL, DIMS, [1.0, 0.0, 0.0, 0.0])
        cache.put(h3, PROVIDER, MODEL, DIMS, [0.0, 0.0, 0.0, 1.0])

        result = cache.get_batch([h1, h2, h3], PROVIDER, MODEL)
        assert h1 in result
        assert h2 not in result
        assert h3 in result

    def test_get_batch_empty(self, cache: EmbeddingCache) -> None:
        assert cache.get_batch([], PROVIDER, MODEL) == {}

    def test_put_batch(self, cache: EmbeddingCache) -> None:
        entries: list[tuple[str, int, list[float]]] = [
            (content_hash("a"), DIMS, [1.0, 0.0, 0.0, 0.0]),
            (content_hash("b"), DIMS, [0.0, 1.0, 0.0, 0.0]),
            (content_hash("c"), DIMS, [0.0, 0.0, 1.0, 0.0]),
        ]
        cache.put_batch(entries, PROVIDER, MODEL)

        for h, _, expected in entries:
            result = cache.get(h, PROVIDER, MODEL)
            assert result is not None
            assert abs(result[0] - expected[0]) < 1e-6

    def test_put_batch_empty(self, cache: EmbeddingCache) -> None:
        cache.put_batch([], PROVIDER, MODEL)  # should not raise


class TestStats:
    def test_empty_cache(self, cache: EmbeddingCache) -> None:
        s = cache.stats()
        assert s["total_entries"] == 0
        assert s["total_size_bytes"] == 0

    def test_after_inserts(self, cache: EmbeddingCache) -> None:
        cache.put(content_hash("a"), PROVIDER, MODEL, DIMS, SAMPLE_EMBEDDING)
        cache.put(content_hash("b"), PROVIDER, MODEL, DIMS, SAMPLE_EMBEDDING)
        s = cache.stats()
        assert s["total_entries"] == 2
        assert s["total_size_bytes"] == DIMS * 4 * 2  # 4 bytes per float32


class TestWALMode:
    def test_wal_enabled(self, cache: EmbeddingCache) -> None:
        row = cache._conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] == "wal"


class TestClose:
    def test_close(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "close_test.db")
        cache.put(content_hash("x"), PROVIDER, MODEL, DIMS, SAMPLE_EMBEDDING)
        cache.close()
        # Re-open and verify data persisted
        cache2 = EmbeddingCache(tmp_path / "close_test.db")
        assert cache2.get(content_hash("x"), PROVIDER, MODEL) is not None
        cache2.close()
