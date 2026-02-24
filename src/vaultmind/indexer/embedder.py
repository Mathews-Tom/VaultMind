"""Embedding pipeline — batch-embeds text chunks via OpenAI or Voyage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from vaultmind.config import EmbeddingConfig
    from vaultmind.indexer.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text chunks.

    Supports OpenAI (text-embedding-3-small/large) and Voyage (voyage-3-lite)
    via the OpenAI-compatible API format. Optionally backed by an SQLite
    embedding cache to skip redundant API calls on re-indexing.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        api_key: str,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.config = config
        self._cache = cache
        if config.provider == "voyage":
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.voyageai.com/v1",
            )
        else:
            self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, using cache when available."""
        if not texts:
            return []

        if self._cache is None:
            return self._embed_texts_uncached(texts)

        from vaultmind.indexer.embedding_cache import content_hash

        provider = self.config.provider
        model = self.config.model

        # Hash all texts
        hashes = [content_hash(t) for t in texts]

        # Batch lookup
        cached = self._cache.get_batch(hashes, provider, model)

        # Identify uncached indices
        uncached_indices: list[int] = []
        for i, h in enumerate(hashes):
            if h not in cached:
                uncached_indices.append(i)

        cache_hits = len(texts) - len(uncached_indices)
        if cache_hits > 0:
            logger.info(
                "Embedding cache: %d hits, %d misses out of %d texts",
                cache_hits,
                len(uncached_indices),
                len(texts),
            )

        # Embed uncached texts via API
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            fresh_embeddings = self._embed_texts_uncached(uncached_texts)

            # Store in cache
            entries: list[tuple[str, int, list[float]]] = []
            for idx, emb in zip(uncached_indices, fresh_embeddings, strict=False):
                h = hashes[idx]
                cached[h] = emb
                entries.append((h, self.config.dimensions, emb))
            self._cache.put_batch(entries, provider, model)

        # Reassemble in original order
        return [cached[h] for h in hashes]

    def _embed_texts_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via API without cache, handling batching internally."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            response = self._client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                "Embedded batch %d-%d of %d",
                i,
                min(i + self.config.batch_size, len(texts)),
                len(texts),
            )

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]
