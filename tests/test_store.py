"""Tests for VaultStore — ChromaDB-backed vector store."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from chromadb.db.impl.sqlite import SqliteDB

from vaultmind.config import ChromaConfig
from vaultmind.indexer.store import VaultStore
from vaultmind.vault.models import Note, NoteChunk

if TYPE_CHECKING:
    from chromadb.api.client import Client


class FakeEmbedder:
    """Deterministic embedder — no API calls."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1), 0.2, 0.3, 0.4] for i, _ in enumerate(texts)]

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

    @pytest.mark.parametrize("legacy", [False, True], ids=["method", "property"])
    def test_client_limit_preserves_record_associations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
    ) -> None:
        config = ChromaConfig(persist_dir=tmp_path / "chroma", collection_name="bounded_chunks")
        client = chromadb.PersistentClient(
            path=str(config.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        limited_client = SimpleNamespace(get_or_create_collection=client.get_or_create_collection)
        if legacy:
            limited_client.max_batch_size = 3
        else:
            limited_client.get_max_batch_size = lambda: 3
        monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: limited_client)
        store = VaultStore(config, embedder=FakeEmbedder())  # type: ignore[arg-type]
        original_upsert = store._collection.upsert

        def bounded_upsert(
            *,
            ids: list[str],
            documents: list[str],
            embeddings: list[list[float]],
            metadatas: list[dict[str, Any]],
        ) -> None:
            if len(ids) > 3:
                raise ValueError("client batch limit exceeded")
            original_upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

        monkeypatch.setattr(store._collection, "upsert", bounded_upsert)
        notes = _make_notes(n_notes=2, chunks_per_note=4)
        parser = FakeParser()
        chunks = [chunk for note in notes for chunk in parser.chunk_note(note)]
        vectors = FakeEmbedder().embed_texts([chunk.content for chunk in chunks])
        store.index_notes(notes, parser)  # type: ignore[arg-type]
        records = store._collection.get(include=["documents", "embeddings", "metadatas"])
        positions = {chunk_id: i for i, chunk_id in enumerate(records["ids"])}
        assert set(positions) == {chunk.chunk_id for chunk in chunks}
        documents = records["documents"]
        metadatas = records["metadatas"]
        embeddings = records["embeddings"]
        assert documents is not None
        assert metadatas is not None
        assert embeddings is not None
        for chunk, vector in zip(chunks, vectors, strict=True):
            position = positions[chunk.chunk_id]
            assert documents[position] == chunk.content
            assert metadatas[position] == chunk.to_chroma_metadata()
            assert embeddings[position] == pytest.approx(vector)

    def test_large_note_can_be_replaced_and_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = ChromaConfig(persist_dir=tmp_path / "chroma", collection_name="bounded_deletes")
        client = cast(
            "Client",
            chromadb.PersistentClient(
                path=str(config.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
            ),
        )
        monkeypatch.setattr(client._system.instance(SqliteDB), "_max_batch_size", 3)
        store = VaultStore(config, embedder=FakeEmbedder())  # type: ignore[arg-type]
        parser = FakeParser()
        target, unrelated = _make_notes(n_notes=2, chunks_per_note=8)
        store.index_notes([target, unrelated], parser)  # type: ignore[arg-type]
        replacement = Note(
            path=target.path,
            title=target.title,
            content="\n".join(f"replacement line {i}" for i in range(5)),
        )
        store.index_single_note(replacement, parser)  # type: ignore[arg-type]
        records = store._collection.get(include=["documents"])
        documents = records["documents"]
        assert documents is not None
        assert dict(zip(records["ids"], documents, strict=True)) == {
            chunk.chunk_id: chunk.content
            for note in (replacement, unrelated)
            for chunk in parser.chunk_note(note)
        }

        store.delete_note(str(target.path))
        remaining = store._collection.get(include=["documents"])
        remaining_documents = remaining["documents"]
        assert remaining_documents is not None
        assert dict(zip(remaining["ids"], remaining_documents, strict=True)) == {
            chunk.chunk_id: chunk.content for chunk in parser.chunk_note(unrelated)
        }
