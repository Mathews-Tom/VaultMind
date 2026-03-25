"""Cross-encoder reranking for two-stage search refinement."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Two-stage reranker using a cross-encoder model.

    Scores (query, document) pairs jointly for higher-quality relevance
    ranking than bi-encoder cosine similarity alone. Default model:
    cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~44MB).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        # Lazy import to avoid loading torch at module import time
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_length)
        logger.info("Loaded cross-encoder model: %s", model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        content_key: str = "content",
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank documents by cross-encoder relevance score.

        Args:
            query: Search query text.
            documents: List of search result dicts (must contain content_key).
            content_key: Key to extract document text from each dict.
            top_k: Return only top-K results. None = return all, re-sorted.

        Returns:
            List of (document, cross_encoder_score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        pairs: list[tuple[str, str]] = []
        valid_docs: list[dict[str, Any]] = []
        for doc in documents:
            text = doc.get(content_key, "")
            if not text:
                continue
            pairs.append((query, str(text)))
            valid_docs.append(doc)

        if not pairs:
            return [(doc, 0.0) for doc in documents[:top_k]]

        scores = self._model.predict(pairs)

        scored = list(zip(valid_docs, [float(s) for s in scores], strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
