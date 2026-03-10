"""Reciprocal Rank Fusion (RRF) combiner for hybrid vector + BM25 search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HybridResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    vector_rank: int | None
    bm25_rank: int | None
    rrf_score: float


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, float | str]],
    k: int = 60,
) -> list[HybridResult]:
    """Merge vector and BM25 ranked lists via Reciprocal Rank Fusion.

    RRF score for document d: sum over each list R of 1 / (k + rank_in_R(d)).
    Documents appearing in both lists get contributions from each, boosting them
    above documents that appear in only one list.

    Args:
        vector_hits: ChromaDB search results — dicts with keys chunk_id, content,
            metadata, distance (sorted best-first, index 0 = rank 1).
        bm25_hits: BM25 search results — dicts with keys chunk_id, note_path,
            note_title, bm25_score (sorted best-first, index 0 = rank 1).
        k: RRF constant (default 60, as in the original Cormack et al. paper).

    Returns:
        List of HybridResult sorted by rrf_score descending.
    """
    # Build lookup maps
    vector_by_id: dict[str, dict[str, Any]] = {h["chunk_id"]: h for h in vector_hits}
    bm25_by_id: dict[str, dict[str, float | str]] = {str(h["chunk_id"]): h for h in bm25_hits}

    all_ids: set[str] = set(vector_by_id) | set(bm25_by_id)

    # Rank maps (1-based)
    vector_rank_map: dict[str, int] = {h["chunk_id"]: i + 1 for i, h in enumerate(vector_hits)}
    bm25_rank_map: dict[str, int] = {str(h["chunk_id"]): i + 1 for i, h in enumerate(bm25_hits)}

    results: list[HybridResult] = []
    for chunk_id in all_ids:
        v_rank = vector_rank_map.get(chunk_id)
        b_rank = bm25_rank_map.get(chunk_id)

        rrf = 0.0
        if v_rank is not None:
            rrf += 1.0 / (k + v_rank)
        if b_rank is not None:
            rrf += 1.0 / (k + b_rank)

        # Content and metadata come from vector hit; fall back to BM25 fields
        if chunk_id in vector_by_id:
            hit = vector_by_id[chunk_id]
            content: str = hit.get("content", "")
            metadata: dict[str, Any] = hit.get("metadata", {})
        else:
            bm25_hit = bm25_by_id[chunk_id]
            content = ""
            metadata = {
                "note_path": bm25_hit.get("note_path", ""),
                "note_title": bm25_hit.get("note_title", ""),
            }

        results.append(
            HybridResult(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
                vector_rank=v_rank,
                bm25_rank=b_rank,
                rrf_score=rrf,
            )
        )

    results.sort(key=lambda r: r.rrf_score, reverse=True)
    return results
