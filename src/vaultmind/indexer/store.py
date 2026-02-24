"""Vector store — ChromaDB-backed storage for note chunks with incremental indexing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

if TYPE_CHECKING:
    from vaultmind.config import ChromaConfig
    from vaultmind.indexer.embedder import Embedder
    from vaultmind.vault.models import Note, NoteChunk
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)


class VaultStore:
    """Manages the vector index over vault notes.

    Supports full and incremental indexing, semantic search with metadata
    filtering, and chunk-level retrieval.
    """

    def __init__(
        self,
        chroma_config: ChromaConfig,
        embedder: Embedder,
    ) -> None:
        self.config = chroma_config
        self.embedder = embedder

        self._client = chromadb.PersistentClient(
            path=str(chroma_config.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=chroma_config.collection_name,
            metadata={"hnsw:space": chroma_config.distance_fn},
        )
        logger.info(
            "ChromaDB collection '%s' loaded (%d documents)",
            chroma_config.collection_name,
            self._collection.count(),
        )

    def index_notes(self, notes: list[Note], parser: VaultParser) -> int:
        """Full index: chunk and embed all notes. Returns number of chunks indexed."""
        all_chunks: list[NoteChunk] = []
        for note in notes:
            chunks = parser.chunk_note(note)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks to index")
            return 0

        # Embed all chunks
        texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=[c.chunk_id for c in all_chunks],
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[c.to_chroma_metadata() for c in all_chunks],
        )

        logger.info("Indexed %d chunks from %d notes", len(all_chunks), len(notes))
        return len(all_chunks)

    def index_single_note(self, note: Note, parser: VaultParser) -> int:
        """Incremental index: re-index a single note (delete old chunks, add new)."""
        # Delete existing chunks for this note
        self.delete_note(str(note.path))

        # Chunk and embed
        chunks = parser.chunk_note(note)
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[c.to_chroma_metadata() for c in chunks],
        )

        logger.info("Re-indexed %d chunks for %s", len(chunks), note.path)
        return len(chunks)

    def delete_note(self, note_path: str) -> None:
        """Remove all chunks for a given note path."""
        # ChromaDB where filter on metadata
        existing = self._collection.get(
            where={"note_path": note_path},
        )
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
            logger.debug("Deleted %d chunks for %s", len(existing["ids"]), note_path)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over indexed chunks.

        Args:
            query: Natural language search query.
            n_results: Number of results to return.
            where: Optional ChromaDB metadata filter (e.g., {"note_type": "project"}).

        Returns:
            List of dicts with keys: chunk_id, content, metadata, distance.
        """
        query_embedding = self.embedder.embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        # Flatten ChromaDB's nested result format
        hits: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            docs = results["documents"]
            metas = results["metadatas"]
            dists = results["distances"]
            for i, chunk_id in enumerate(results["ids"][0]):
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "content": docs[0][i] if docs else "",
                        "metadata": metas[0][i] if metas else {},
                        "distance": dists[0][i] if dists else 0.0,
                    }
                )

        return hits

    @property
    def count(self) -> int:
        """Total number of indexed chunks."""
        return self._collection.count()
