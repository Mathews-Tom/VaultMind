"""Vector store — ChromaDB-backed storage for note chunks with incremental indexing.

Index mutations follow one discipline: everything that can fail without
touching an index (chunking through the shared current-content projection,
embedding, batch validation) is finished *before* the first write. Writes then
go vector index first, keyword index second. The two indexes are separate
databases, so there is no cross-index atomicity: a failure raises
:class:`IndexUpdateError` naming the failing stage and the stages already
applied, and every operation is idempotent so repeating it converges.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from vaultmind.errors import VaultMindError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vaultmind.config import ChromaConfig
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.embedder import Embedder
    from vaultmind.vault.models import Note, NoteChunk
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)

# Chroma deletes by id with an ``IN (...)`` predicate; keep each call well below
# SQLite's bound-parameter limit so clearing a large collection cannot fail on size.
_DELETE_BATCH = 500


class IndexStage(StrEnum):
    """Stage of an index mutation, as reported by :class:`IndexUpdateError`."""

    EMBED = "embed"
    VECTOR_LOOKUP = "vector_lookup"
    VECTOR_UPSERT = "vector_upsert"
    VECTOR_DELETE = "vector_delete"
    KEYWORD_REPLACE = "keyword_replace"
    KEYWORD_DELETE = "keyword_delete"

    @property
    def writes(self) -> bool:
        """Whether the stage changes an index (EMBED and VECTOR_LOOKUP only read)."""
        return self not in (IndexStage.EMBED, IndexStage.VECTOR_LOOKUP)


class IndexUpdateError(VaultMindError):
    """An index mutation failed; the vector and keyword indexes may now disagree.

    ``stage`` is the failing stage, ``completed_stages`` the write stages that
    had fully applied before it (empty when nothing was changed). The original
    exception is chained as ``__cause__``. Content errors raised before any
    stage runs (``ValueError`` from chunking an ambiguous note, duplicate chunk
    ids) are not wrapped: they leave both indexes untouched.
    """

    def __init__(
        self,
        operation: str,
        scope: str,
        stage: IndexStage,
        completed_stages: tuple[IndexStage, ...],
        cause: BaseException,
    ) -> None:
        self.operation = operation
        self.scope = scope
        self.stage = stage
        self.completed_stages = completed_stages
        if completed_stages:
            progress = (
                "stages applied before the failure: "
                + ", ".join(s.value for s in completed_stages)
                + "; vector and keyword indexes may disagree until the operation is repeated"
            )
        elif stage.writes:
            progress = "the failing stage may be partially applied; repeat the operation"
        else:
            progress = "no index changes were applied"
        super().__init__(
            f"{operation} failed for {scope} at stage '{stage.value}' "
            f"({type(cause).__name__}: {cause}); {progress}"
        )


class _Mutation:
    """Runs the stages of one index mutation and records which writes applied."""

    def __init__(self, operation: str, scope: str) -> None:
        self._operation = operation
        self._scope = scope
        self._completed: list[IndexStage] = []

    def run[**P, T](
        self,
        stage: IndexStage,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            raise IndexUpdateError(
                self._operation, self._scope, stage, tuple(self._completed), exc
            ) from exc
        if stage.writes:
            self._completed.append(stage)
        return result


@dataclass(frozen=True, slots=True)
class _PreparedChunks:
    """Chunks with embeddings and every write payload, built before any write."""

    ids: list[str]
    documents: list[str]
    embeddings: list[list[float]]
    metadatas: list[Mapping[str, str | int | float | bool]]
    keyword_rows: list[tuple[str, str, str, str]]


def _validate_embeddings(embeddings: Sequence[Sequence[float]], expected: int) -> None:
    """Reject an embedder batch the vector index could not accept, before any write."""
    if len(embeddings) != expected:
        raise ValueError(f"Embedder returned {len(embeddings)} vectors for {expected} chunks")
    dimensions: set[int] = set()
    for position, vector in enumerate(embeddings):
        if len(vector) == 0:
            raise ValueError(f"Embedder returned an empty vector for chunk {position}")
        dimensions.add(len(vector))
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Embedder returned a non-finite value for chunk {position}")
    if len(dimensions) > 1:
        raise ValueError(f"Embedder returned differing vector dimensions: {sorted(dimensions)}")


class VaultStore:
    """Manages the vector index over vault notes.

    Supports full and incremental indexing, semantic search with metadata
    filtering, and chunk-level retrieval. Indexed text is whatever
    ``parser.chunk_note`` yields, i.e. the shared current-content projection.
    """

    def __init__(
        self,
        chroma_config: ChromaConfig,
        embedder: Embedder,
        bm25: BM25Index | None = None,
        search_cache: Any = None,
    ) -> None:
        self.config = chroma_config
        self.embedder = embedder
        self._bm25 = bm25
        self._search_cache = search_cache

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
        """Full index: reconcile the configured indexes to exactly *notes*.

        *notes* is the canonical snapshot. Chunks for every note are embedded
        first; only then are the configured Chroma collection and BM25 index
        made to contain precisely those chunks, removing rows for notes and
        chunks absent from the snapshot. An empty snapshot (or one with no
        indexable content) clears both indexes. Other Chroma collections and
        the notes themselves are never touched. Returns the chunk count.

        Raises ``ValueError`` before any write for unchunkable notes or
        duplicate chunk ids, and :class:`IndexUpdateError` for embedding or
        index failures.
        """
        mutation = _Mutation("index_notes", f"snapshot of {len(notes)} notes")
        chunks: list[NoteChunk] = []
        for note in notes:
            try:
                chunks.extend(parser.chunk_note(note))
            except ValueError as exc:
                raise ValueError(f"Cannot index {note.path}: {exc}") from exc
        prepared = self._prepare(chunks, mutation)
        keep = set(prepared.ids)

        try:
            existing = mutation.run(IndexStage.VECTOR_LOOKUP, self._vector_ids, None)
            if prepared.ids:
                mutation.run(IndexStage.VECTOR_UPSERT, self._vector_upsert, prepared)
            stale = [chunk_id for chunk_id in existing if chunk_id not in keep]
            if stale:
                mutation.run(IndexStage.VECTOR_DELETE, self._vector_delete, stale)
            if self._bm25 is not None:
                mutation.run(
                    IndexStage.KEYWORD_REPLACE, self._bm25.replace_all, prepared.keyword_rows
                )
        finally:
            self._clear_search_cache()

        if not prepared.ids:
            logger.warning("Full index snapshot has no indexable chunks; indexes cleared")
        logger.info(
            "Indexed %d chunks from %d notes (%d stale vector chunks removed)",
            len(prepared.ids),
            len(notes),
            len(stale),
        )
        return len(prepared.ids)

    def index_single_note(self, note: Note, parser: VaultParser) -> int:
        """Incremental index: make the indexes hold exactly the note's current chunks.

        Embeds before writing, upserts the new chunks, removes chunk ids the
        note no longer produces (vector and keyword), and retires the note from
        both indexes when its current content yields no chunks. Returns the
        chunk count. Errors follow :meth:`index_notes`.
        """
        note_path = str(note.path)
        mutation = _Mutation("index_single_note", note_path)
        chunks = parser.chunk_note(note)
        if not chunks:
            self._retire(note_path, mutation)
            logger.info("No current content for %s; retired from both indexes", note_path)
            return 0

        foreign = {chunk.note_path for chunk in chunks} - {note_path}
        if foreign:
            raise ValueError(f"Chunks for {note_path} carry other note paths: {sorted(foreign)}")
        prepared = self._prepare(chunks, mutation)
        keep = set(prepared.ids)

        try:
            existing = mutation.run(IndexStage.VECTOR_LOOKUP, self._vector_ids, note_path)
            mutation.run(IndexStage.VECTOR_UPSERT, self._vector_upsert, prepared)
            stale = [chunk_id for chunk_id in existing if chunk_id not in keep]
            if stale:
                mutation.run(IndexStage.VECTOR_DELETE, self._vector_delete, stale)
            if self._bm25 is not None:
                mutation.run(
                    IndexStage.KEYWORD_REPLACE,
                    self._bm25.replace_note,
                    note_path,
                    prepared.keyword_rows,
                )
        finally:
            self._clear_search_cache()

        logger.info("Re-indexed %d chunks for %s", len(chunks), note.path)
        return len(chunks)

    def delete_note(self, note_path: str) -> None:
        """Remove all chunks for a given note path from the vector and keyword index."""
        self._retire(note_path, _Mutation("delete_note", note_path))

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def _prepare(self, chunks: list[NoteChunk], mutation: _Mutation) -> _PreparedChunks:
        """Build every write payload; nothing here changes an index."""
        if not chunks:
            return _PreparedChunks([], [], [], [], [])

        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
            raise ValueError(f"Duplicate chunk ids cannot be indexed: {duplicates}")

        documents = [chunk.content for chunk in chunks]
        metadatas: list[Mapping[str, str | int | float | bool]] = [
            chunk.to_chroma_metadata() for chunk in chunks
        ]
        keyword_rows = [
            (chunk.chunk_id, str(chunk.note_path), chunk.note_title, chunk.content)
            for chunk in chunks
        ]
        embeddings = mutation.run(IndexStage.EMBED, self._embed, documents)
        return _PreparedChunks(ids, documents, embeddings, metadatas, keyword_rows)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedder.embed_texts(texts)
        _validate_embeddings(embeddings, expected=len(texts))
        return embeddings

    def _retire(self, note_path: str, mutation: _Mutation) -> None:
        try:
            ids = mutation.run(IndexStage.VECTOR_LOOKUP, self._vector_ids, note_path)
            if ids:
                mutation.run(IndexStage.VECTOR_DELETE, self._vector_delete, ids)
                logger.debug("Deleted %d chunks for %s", len(ids), note_path)
            if self._bm25 is not None:
                mutation.run(IndexStage.KEYWORD_DELETE, self._bm25.delete_note, note_path)
        finally:
            self._clear_search_cache()

    def _vector_ids(self, note_path: str | None) -> list[str]:
        """Chunk ids in the configured collection, for one note or all notes."""
        if note_path is None:
            found = self._collection.get(include=[])
        else:
            found = self._collection.get(where={"note_path": note_path}, include=[])
        return list(found["ids"])

    def _vector_upsert(self, prepared: _PreparedChunks) -> None:
        self._collection.upsert(
            ids=prepared.ids,
            documents=prepared.documents,
            embeddings=prepared.embeddings,  # type: ignore[arg-type]
            metadatas=prepared.metadatas,
        )

    def _vector_delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), _DELETE_BATCH):
            self._collection.delete(ids=ids[start : start + _DELETE_BATCH])

    def _clear_search_cache(self) -> None:
        """Drop cached results so a mutation can never be masked by retired hits."""
        if self._search_cache is not None:
            self._search_cache.clear()

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

        # Check search cache (only for unfiltered queries)
        if self._search_cache is not None and where is None:
            cached = self._search_cache.get(query, query_embedding, n_results)
            if cached is not None:
                return list(cached)

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
                        "distance": dists[0][i] if dists else None,
                    }
                )

        # Store in cache (only unfiltered queries)
        if self._search_cache is not None and where is None:
            self._search_cache.put(query, query_embedding, hits, n_results)

        return hits

    def ranked_search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        ranking_enabled: bool = True,
        knowledge_graph: Any = None,
        ranking_config: Any = None,
        reranker: Any = None,
    ) -> list[dict[str, Any]]:
        """Semantic search with composite ranking.

        Args:
            query: Natural language search query.
            n_results: Number of results to return.
            where: Optional ChromaDB metadata filter.
            ranking_enabled: If False, return results with raw scores only.
            knowledge_graph: Optional KnowledgeGraph for connection density scoring.
            ranking_config: Optional RankingConfig with weight overrides.
            reranker: Optional CrossEncoderReranker for two-stage refinement.

        Returns results sorted by ranked score instead of raw distance.
        """
        from vaultmind.indexer.ranking import rank_results

        fetch_n = n_results * 4 if reranker is not None else n_results
        hits = self.search(query, n_results=fetch_n, where=where)

        if reranker is not None:
            from vaultmind.indexer.reranker import CrossEncoderReranker

            assert isinstance(reranker, CrossEncoderReranker)
            reranked = reranker.rerank(query, hits, content_key="content", top_k=n_results * 2)
            hits = []
            for doc, ce_score in reranked:
                doc = dict(doc)  # copy to avoid mutation
                doc["reranker_score"] = ce_score
                # Normalize CE score to [0, 1] via sigmoid
                normalized = 1.0 / (1.0 + math.exp(-ce_score))
                doc["distance"] = 1.0 - normalized  # lower distance = better
                hits.append(doc)

        ranked = rank_results(
            hits,
            enabled=ranking_enabled,
            knowledge_graph=knowledge_graph,
            ranking_config=ranking_config,
        )
        return [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "metadata": r.metadata,
                "distance": 1.0 - r.raw_score,
                "raw_score": r.raw_score,
                "final_score": r.final_score,
                "reranker_score": r.reranker_score,
            }
            for r in ranked
        ]

    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + BM25 search with RRF fusion.

        Falls back to pure vector search when BM25 index is not configured.

        ``distance`` preserves raw vector evidence (None for lexical-only hits);
        ``rrf_score`` determines fusion ordering, never relevance confidence.
        """
        from vaultmind.indexer.hybrid import reciprocal_rank_fusion

        # Fetch more candidates from each source so RRF has a deep pool to rank
        fetch_n = max(n_results * 4, 20)

        vector_hits = self.search(query, n_results=fetch_n, where=where)

        if self._bm25 is None:
            return vector_hits[:n_results]

        bm25_hits = self._bm25.search(query, n_results=fetch_n)
        fused = reciprocal_rank_fusion(vector_hits, bm25_hits)

        # Reformat to match the standard search() output shape
        results: list[dict[str, Any]] = []
        for r in fused[:n_results]:
            results.append(
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "distance": r.distance,
                    "rrf_score": r.rrf_score,
                    "vector_rank": r.vector_rank,
                    "bm25_rank": r.bm25_rank,
                }
            )
        return results

    @property
    def count(self) -> int:
        """Total number of indexed chunks."""
        return self._collection.count()
