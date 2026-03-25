"""Tests for semantic search result cache."""

from __future__ import annotations

import pytest

from vaultmind.indexer.search_cache import SearchResultCache, _cosine_similarity


def _make_result(chunk_id: str = "test::0", note_path: str = "test.md") -> dict:
    return {
        "chunk_id": chunk_id,
        "content": "test content",
        "metadata": {"note_path": note_path},
        "distance": 0.3,
    }


def _make_embedding(value: float = 1.0, dim: int = 8) -> list[float]:
    return [value] * dim


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_similar_vectors_high_score(self) -> None:
        a = [1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.1]
        assert _cosine_similarity(a, b) > 0.99


class TestCacheGetPut:
    def test_exact_match_returns_cached(self) -> None:
        cache = SearchResultCache(max_entries=10)
        emb = _make_embedding()
        results = [_make_result()]
        cache.put("test query", emb, results, n_requested=5)

        cached = cache.get("test query", emb, n_results=5)
        assert cached is not None
        assert len(cached) == 1

    def test_miss_returns_none(self) -> None:
        cache = SearchResultCache(max_entries=10)
        cached = cache.get("unknown", _make_embedding(), n_results=5)
        assert cached is None

    def test_semantic_match_returns_cached(self) -> None:
        cache = SearchResultCache(max_entries=10, similarity_threshold=0.9)
        emb_a = [1.0, 1.0, 1.0, 1.0]
        emb_b = [1.0, 1.0, 1.0, 1.01]  # very similar
        results = [_make_result()]
        cache.put("query a", emb_a, results, n_requested=5)

        cached = cache.get("query b", emb_b, n_results=5)
        assert cached is not None

    def test_dissimilar_query_misses(self) -> None:
        cache = SearchResultCache(max_entries=10, similarity_threshold=0.9)
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 0.0, 0.0, 1.0]  # orthogonal
        cache.put("query a", emb_a, [_make_result()], n_requested=5)

        cached = cache.get("query b", emb_b, n_results=5)
        assert cached is None

    def test_results_sliced_to_n_results(self) -> None:
        cache = SearchResultCache(max_entries=10)
        emb = _make_embedding()
        results = [_make_result(f"t::{i}") for i in range(10)]
        cache.put("query", emb, results, n_requested=10)

        cached = cache.get("query", emb, n_results=3)
        assert cached is not None
        assert len(cached) == 3


class TestLRUEviction:
    def test_evicts_oldest_at_capacity(self) -> None:
        cache = SearchResultCache(max_entries=2)
        cache.put("q1", [1.0, 0.0], [_make_result("a::0")], n_requested=5)
        cache.put("q2", [0.0, 1.0], [_make_result("b::0")], n_requested=5)
        cache.put("q3", [1.0, 1.0], [_make_result("c::0")], n_requested=5)

        assert cache.size == 2
        # q1 should have been evicted
        assert cache.get("q1", [1.0, 0.0], n_results=5) is None
        assert cache.get("q3", [1.0, 1.0], n_results=5) is not None

    def test_access_refreshes_lru(self) -> None:
        cache = SearchResultCache(max_entries=2)
        cache.put("q1", [1.0, 0.0], [_make_result()], n_requested=5)
        cache.put("q2", [0.0, 1.0], [_make_result()], n_requested=5)

        # Access q1 to make it most recent
        cache.get("q1", [1.0, 0.0], n_results=5)

        # Add q3 — should evict q2 (least recently used), not q1
        cache.put("q3", [1.0, 1.0], [_make_result()], n_requested=5)

        assert cache.get("q1", [1.0, 0.0], n_results=5) is not None
        assert cache.get("q2", [0.0, 1.0], n_results=5) is None


class TestInvalidation:
    def test_invalidate_removes_matching_entries(self) -> None:
        cache = SearchResultCache(max_entries=10)
        cache.put("q1", [1.0], [_make_result(note_path="notes/a.md")], n_requested=5)
        cache.put("q2", [0.0], [_make_result(note_path="notes/b.md")], n_requested=5)

        removed = cache.invalidate("notes/a.md")
        assert removed == 1
        assert cache.size == 1

    def test_invalidate_no_match_keeps_all(self) -> None:
        cache = SearchResultCache(max_entries=10)
        cache.put("q1", [1.0], [_make_result(note_path="notes/a.md")], n_requested=5)

        removed = cache.invalidate("notes/z.md")
        assert removed == 0
        assert cache.size == 1


class TestDBExhaustion:
    def test_exhausted_db_serves_smaller_request(self) -> None:
        cache = SearchResultCache(max_entries=10)
        emb = _make_embedding()
        # DB returned 3 results when 10 were requested → DB exhausted
        results = [_make_result(f"t::{i}") for i in range(3)]
        cache.put("query", emb, results, n_requested=10)

        # Request 5, but DB only had 3 total — cache should serve
        cached = cache.get("query", emb, n_results=5)
        assert cached is not None
        assert len(cached) == 3

    def test_non_exhausted_db_requires_enough_results(self) -> None:
        cache = SearchResultCache(max_entries=10)
        emb = _make_embedding()
        # DB returned 3 results when 3 were requested → NOT exhausted
        results = [_make_result(f"t::{i}") for i in range(3)]
        cache.put("query", emb, results, n_requested=3)

        # Request 5 — cache has only 3, DB was NOT exhausted → miss
        cached = cache.get("query", emb, n_results=5)
        assert cached is None


class TestClearAndStats:
    def test_clear_empties_cache(self) -> None:
        cache = SearchResultCache(max_entries=10)
        cache.put("q1", [1.0], [_make_result()], n_requested=5)
        cache.clear()
        assert cache.size == 0

    def test_stats_track_hits_and_misses(self) -> None:
        cache = SearchResultCache(max_entries=10)
        emb = _make_embedding()
        cache.put("q", emb, [_make_result()], n_requested=5)

        cache.get("q", emb, n_results=5)  # hit
        cache.get("miss", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], n_results=5)  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestSearchConfig:
    def test_cache_config_defaults(self) -> None:
        from vaultmind.config import SearchConfig

        cfg = SearchConfig()
        assert cfg.cache_enabled is True
        assert cfg.cache_max_entries == 50
        assert cfg.cache_similarity_threshold == pytest.approx(0.85)
