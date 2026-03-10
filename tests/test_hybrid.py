"""Tests for Reciprocal Rank Fusion hybrid search combiner."""

from __future__ import annotations

from typing import Any

from vaultmind.indexer.hybrid import HybridResult, reciprocal_rank_fusion


def _make_vector_hit(chunk_id: str, distance: float = 0.2, content: str = "") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content": content or f"content for {chunk_id}",
        "metadata": {"note_path": f"{chunk_id}.md", "note_title": chunk_id},
        "distance": distance,
    }


def _make_bm25_hit(chunk_id: str, score: float = 1.0) -> dict[str, float | str]:
    return {
        "chunk_id": chunk_id,
        "note_path": f"{chunk_id}.md",
        "note_title": chunk_id,
        "bm25_score": score,
    }


class TestRRFMergesDisjoint:
    def test_rrf_merges_disjoint(self) -> None:
        """Two non-overlapping lists produce their union."""
        vector_hits = [_make_vector_hit("a"), _make_vector_hit("b")]
        bm25_hits = [_make_bm25_hit("c"), _make_bm25_hit("d")]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        result_ids = {r.chunk_id for r in results}

        assert result_ids == {"a", "b", "c", "d"}

    def test_all_items_present_from_each_list(self) -> None:
        vector_hits = [_make_vector_hit(f"v{i}") for i in range(5)]
        bm25_hits = [_make_bm25_hit(f"b{i}") for i in range(5)]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        assert len(results) == 10


class TestRRFBoostsShared:
    def test_rrf_boosts_shared(self) -> None:
        """Items appearing in both lists rank above items in only one list."""
        # "shared" appears in both at rank 2
        vector_hits = [
            _make_vector_hit("unique_v"),
            _make_vector_hit("shared"),
        ]
        bm25_hits = [
            _make_bm25_hit("unique_b"),
            _make_bm25_hit("shared"),
        ]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        result_ids = [r.chunk_id for r in results]

        # shared should rank first because it gets contributions from both lists
        assert result_ids[0] == "shared"

    def test_top_rank_in_both_is_highest_score(self) -> None:
        """Top-ranked item in both lists gets the highest RRF score."""
        vector_hits = [_make_vector_hit("best"), _make_vector_hit("ok")]
        bm25_hits = [_make_bm25_hit("best"), _make_bm25_hit("other")]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        assert results[0].chunk_id == "best"
        assert results[0].vector_rank == 1
        assert results[0].bm25_rank == 1


class TestRRFScoreOrdering:
    def test_rrf_score_ordering(self) -> None:
        """Results are sorted by rrf_score descending."""
        vector_hits = [_make_vector_hit(f"v{i}") for i in range(10)]
        bm25_hits = [_make_bm25_hit(f"v{i}") for i in range(10)]  # complete overlap

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)

        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_scores_positive(self) -> None:
        """All RRF scores must be positive."""
        vector_hits = [_make_vector_hit("a"), _make_vector_hit("b")]
        bm25_hits = [_make_bm25_hit("c")]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        for r in results:
            assert r.rrf_score > 0.0

    def test_rank_metadata_set_correctly(self) -> None:
        """vector_rank and bm25_rank are set accurately."""
        vector_hits = [_make_vector_hit("a"), _make_vector_hit("b")]
        bm25_hits = [_make_bm25_hit("b"), _make_bm25_hit("c")]

        results = reciprocal_rank_fusion(vector_hits, bm25_hits)
        by_id = {r.chunk_id: r for r in results}

        assert by_id["a"].vector_rank == 1
        assert by_id["a"].bm25_rank is None

        assert by_id["b"].vector_rank == 2
        assert by_id["b"].bm25_rank == 1

        assert by_id["c"].vector_rank is None
        assert by_id["c"].bm25_rank == 2


class TestVectorOnlyFallback:
    def test_vector_only_fallback(self) -> None:
        """When bm25_hits is empty, vector order is preserved."""
        vector_hits = [
            _make_vector_hit("first", distance=0.1),
            _make_vector_hit("second", distance=0.3),
            _make_vector_hit("third", distance=0.5),
        ]

        results = reciprocal_rank_fusion(vector_hits, [])

        # All items present
        assert {r.chunk_id for r in results} == {"first", "second", "third"}

        # Order should match vector ranking (rank 1 > rank 2 > rank 3 in RRF terms)
        result_ids = [r.chunk_id for r in results]
        assert result_ids[0] == "first"
        assert result_ids[1] == "second"
        assert result_ids[2] == "third"

    def test_bm25_only_fallback(self) -> None:
        """When vector_hits is empty, bm25 order is preserved."""
        bm25_hits = [
            _make_bm25_hit("top", score=5.0),
            _make_bm25_hit("mid", score=3.0),
        ]

        results = reciprocal_rank_fusion([], bm25_hits)
        assert {r.chunk_id for r in results} == {"top", "mid"}
        assert results[0].chunk_id == "top"

    def test_both_empty_returns_empty(self) -> None:
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_returns_hybrid_result_dataclass(self) -> None:
        vector_hits = [_make_vector_hit("x")]
        results = reciprocal_rank_fusion(vector_hits, [])
        assert isinstance(results[0], HybridResult)

    def test_content_backfilled_from_bm25(self) -> None:
        """BM25-only hits get note_path/note_title in metadata."""
        bm25_hits = [_make_bm25_hit("bm25only")]
        results = reciprocal_rank_fusion([], bm25_hits)
        r = results[0]
        assert r.chunk_id == "bm25only"
        assert r.metadata["note_path"] == "bm25only.md"
