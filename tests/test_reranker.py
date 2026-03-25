"""Tests for cross-encoder reranking."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vaultmind.config import RankingConfig
from vaultmind.indexer.ranking import RankedResult


def _make_reranker(scores: list[float]) -> Any:
    """Create a CrossEncoderReranker with a mocked model returning given scores."""
    from vaultmind.indexer.reranker import CrossEncoderReranker

    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = mock_model
    return reranker


def _doc(content: str, **extra: Any) -> dict[str, Any]:
    """Build a minimal document dict."""
    d: dict[str, Any] = {"content": content}
    d.update(extra)
    return d


class TestCrossEncoderReranker:
    """Unit tests for CrossEncoderReranker with mocked model."""

    def test_rerank_sorts_by_score_descending(self) -> None:
        reranker = _make_reranker([0.1, 0.9, 0.5])
        docs = [_doc("a"), _doc("b"), _doc("c")]
        results = reranker.rerank("query", docs)
        scores = [s for _, s in results]
        assert scores == [0.9, 0.5, 0.1]
        assert results[0][0]["content"] == "b"

    def test_rerank_top_k_limits_results(self) -> None:
        reranker = _make_reranker([0.1, 0.9, 0.5, 0.3, 0.7])
        docs = [_doc(f"doc{i}") for i in range(5)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2

    def test_rerank_empty_documents_returns_empty(self) -> None:
        reranker = _make_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_missing_content_key_skipped(self) -> None:
        reranker = _make_reranker([0.8])
        docs = [{"title": "no content"}, _doc("has content")]
        results = reranker.rerank("query", docs)
        assert len(results) == 1
        assert results[0][0]["content"] == "has content"

    def test_rerank_all_docs_missing_content(self) -> None:
        reranker = _make_reranker([])
        docs = [{"title": "a"}, {"title": "b"}]
        results = reranker.rerank("query", docs)
        # All docs lack content -> returns defaults with score 0.0
        assert len(results) == 2
        for _doc_item, score in results:
            assert score == 0.0

    def test_rerank_returns_tuples_of_doc_and_score(self) -> None:
        reranker = _make_reranker([0.42])
        docs = [_doc("text")]
        results = reranker.rerank("query", docs)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, dict)
        assert isinstance(score, float)

    def test_rerank_preserves_document_metadata(self) -> None:
        reranker = _make_reranker([0.7])
        docs = [_doc("text", note_type="permanent", path="/a.md")]
        results = reranker.rerank("query", docs)
        doc, _ = results[0]
        assert doc["note_type"] == "permanent"
        assert doc["path"] == "/a.md"
        assert doc["content"] == "text"


class TestRankedResultRerankerField:
    """Tests for reranker_score field on RankedResult."""

    def test_ranked_result_has_reranker_score_field(self) -> None:
        r = RankedResult(
            chunk_id="c1",
            raw_score=0.8,
            final_score=0.7,
            note_type="fleeting",
            metadata={},
            content="test",
        )
        assert r.reranker_score == 0.0

    def test_ranked_result_reranker_score_populated(self) -> None:
        r = RankedResult(
            chunk_id="c1",
            raw_score=0.8,
            final_score=0.7,
            note_type="fleeting",
            metadata={},
            content="test",
            reranker_score=0.95,
        )
        assert r.reranker_score == 0.95


class TestRankingConfigReranker:
    """Tests for reranker-related fields on RankingConfig."""

    def test_default_reranker_disabled(self) -> None:
        cfg = RankingConfig()
        assert cfg.reranker_enabled is False

    def test_reranker_model_default(self) -> None:
        cfg = RankingConfig()
        assert cfg.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_reranker_top_k_default(self) -> None:
        cfg = RankingConfig()
        assert cfg.reranker_top_k == 20


class TestRankedSearchWithReranker:
    """Tests for reranker integration with ranked_search plumbing."""

    def test_ranked_search_without_reranker_backward_compat(self) -> None:
        from vaultmind.indexer.ranking import rank_results

        hits = [
            {
                "chunk_id": "c1",
                "content": "some text",
                "metadata": {"note_type": "permanent", "created": "", "status": ""},
                "distance": 0.2,
            }
        ]
        results = rank_results(hits, enabled=True)
        assert len(results) == 1
        assert results[0].reranker_score == 0.0

    def test_reranker_score_in_result_dict(self) -> None:
        r = RankedResult(
            chunk_id="c1",
            raw_score=0.9,
            final_score=0.85,
            note_type="permanent",
            metadata={"note_type": "permanent"},
            content="text",
            reranker_score=0.77,
        )
        assert r.reranker_score == 0.77
        assert r.chunk_id == "c1"
