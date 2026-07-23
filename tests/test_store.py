"""Tests for VaultStore — ChromaDB-backed vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaultmind.config import ChromaConfig
from vaultmind.indexer.store import VaultStore
from vaultmind.vault.models import Note, NoteChunk


class FakeEmbedder:
    """Deterministic embedder — no API calls."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeParser:
    """Chunker that splits a note into one chunk per line."""

    def chunk_note(self, note: Note, max_tokens: int = 500) -> list[NoteChunk]:
        return [
            NoteChunk(
                note_path=str(note.path),
                note_title=note.title,
                chunk_idx=i,
                content=line,
            )
            for i, line in enumerate(note.content.splitlines())
        ]


@pytest.fixture
def store(tmp_path: Path) -> VaultStore:
    config = ChromaConfig(persist_dir=tmp_path / "chroma", collection_name="test_chunks")
    return VaultStore(config, embedder=FakeEmbedder())  # type: ignore[arg-type]


def _make_notes(n_notes: int, chunks_per_note: int) -> list[Note]:
    return [
        Note(
            path=Path(f"notes/note{i}.md"),
            title=f"Note {i}",
            content="\n".join(f"note {i} line {j}" for j in range(chunks_per_note)),
        )
        for i in range(n_notes)
    ]


class TestUpsertBatching:
    """Regression: a full index with more chunks than Chroma's max batch size.

    Chroma caps the number of records in a single upsert call (~5461 on
    default builds) and raises ValueError above it, so one flat upsert of a
    large vault aborts the index run.
    """

    def test_index_notes_exceeding_max_batch_size(self, store: VaultStore) -> None:
        calls: list[int] = []
        original_upsert = store._collection.upsert

        def recording_upsert(*, ids, documents, embeddings, metadatas):  # type: ignore[no-untyped-def]
            calls.append(len(ids))
            return original_upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

        store._collection.upsert = recording_upsert  # type: ignore[method-assign]
        # Shrink the cap so the test stays fast; the slicing logic is the same
        store._max_batch_size = 10

        notes = _make_notes(n_notes=5, chunks_per_note=7)  # 35 chunks > 3 batches
        indexed = store.index_notes(notes, FakeParser())  # type: ignore[arg-type]

        assert indexed == 35
        assert len(calls) == 4  # ceil(35 / 10)
        assert all(size <= 10 for size in calls)
        assert sum(calls) == 35
        assert store._collection.count() == 35

    def test_index_single_note_uses_batched_upsert(self, store: VaultStore) -> None:
        store._max_batch_size = 4
        note = _make_notes(n_notes=1, chunks_per_note=9)[0]
        indexed = store.index_single_note(note, FakeParser())  # type: ignore[arg-type]

        assert indexed == 9
        assert store._collection.count() == 9

    def test_max_batch_size_from_client(self, store: VaultStore) -> None:
        # PersistentClient exposes get_max_batch_size(); the store should
        # pick it up rather than hardcoding a guess.
        assert store._max_batch_size == store._client.get_max_batch_size()
        assert store._max_batch_size > 0
