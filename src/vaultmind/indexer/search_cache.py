"""In-memory LRU search result cache with semantic similarity matching."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """A cached search result with its query embedding."""

    query: str
    embedding: list[float]
    results: list[dict[str, Any]]
    n_requested: int
    n_returned: int


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SearchResultCache:
    """LRU cache with semantic similarity matching for search results.

    A cached result for query A satisfies query B if their embedding
    cosine similarity exceeds the threshold. Tracks DB exhaustion:
    if the original search returned fewer results than requested,
    the cached result IS the complete result set.
    """

    def __init__(
        self,
        max_entries: int = 50,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._max_entries = max_entries
        self._threshold = similarity_threshold
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0

    def get(
        self,
        query: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Check cache for a matching result.

        Checks exact match first, then scans for semantically similar
        cached queries above the similarity threshold.

        Returns cached results if found (sliced to n_results), else None.
        """
        # Exact match (fast path)
        if query in self._cache:
            entry = self._cache[query]
            self._cache.move_to_end(query)
            if self._can_serve(entry, n_results):
                self._hit_count += 1
                return entry.results[:n_results]

        # Semantic similarity scan
        for key, entry in self._cache.items():
            sim = _cosine_similarity(query_embedding, entry.embedding)
            if sim >= self._threshold and self._can_serve(entry, n_results):
                self._cache.move_to_end(key)
                self._hit_count += 1
                return entry.results[:n_results]

        self._miss_count += 1
        return None

    def put(
        self,
        query: str,
        query_embedding: list[float],
        results: list[dict[str, Any]],
        n_requested: int,
    ) -> None:
        """Store search results in cache.

        Args:
            query: The search query text.
            query_embedding: The query's embedding vector.
            results: Search results to cache.
            n_requested: Number of results originally requested (for DB exhaustion detection).
        """
        if query in self._cache:
            self._cache.move_to_end(query)
            self._cache[query] = _CacheEntry(
                query=query,
                embedding=query_embedding,
                results=results,
                n_requested=n_requested,
                n_returned=len(results),
            )
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)

        self._cache[query] = _CacheEntry(
            query=query,
            embedding=query_embedding,
            results=results,
            n_requested=n_requested,
            n_returned=len(results),
        )

    def invalidate(self, note_path: str) -> int:
        """Remove cache entries containing results from a modified note.

        Returns number of entries invalidated.
        """
        to_remove: list[str] = []
        for key, entry in self._cache.items():
            for result in entry.results:
                meta = result.get("metadata", {})
                if meta.get("note_path") == note_path:
                    to_remove.append(key)
                    break

        for key in to_remove:
            del self._cache[key]

        if to_remove:
            logger.debug("Invalidated %d cache entries for %s", len(to_remove), note_path)
        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics."""
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "size": len(self._cache),
            "max_entries": self._max_entries,
        }

    @staticmethod
    def _can_serve(entry: _CacheEntry, n_results: int) -> bool:
        """Check if a cache entry can serve a request for n_results.

        A cached entry can serve if:
        1. It has enough results (n_returned >= n_results), OR
        2. The DB was exhausted (n_returned < n_requested), meaning the
           cached result IS the complete result set.
        """
        if entry.n_returned >= n_results:
            return True
        # DB exhaustion: original search returned fewer than requested
        return entry.n_returned < entry.n_requested
