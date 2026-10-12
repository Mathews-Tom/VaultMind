"""Real-index test harness: actual parser, Chroma collection and SQLite BM25 index.

Only the embedding provider boundary is synthetic. Failures are injected at the
embedder and at the keyword-index mutators, never by patching the code under test.
"""

from __future__ import annotations

import re
import sqlite3
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vaultmind.config import ChromaConfig, load_settings
from vaultmind.indexer.bm25 import BM25Index
from vaultmind.indexer.store import VaultStore
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vaultmind.indexer.embedder import Embedder
    from vaultmind.vault.models import Note

ALL_SURFACES = {"vector", "keyword", "hybrid"}


class BagOfWordsEmbedder:
    """Deterministic synthetic embeddings at the provider boundary; no provider calls."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions
        self.calls = 0
        self.delay_seconds = 0.0
        self.failure: Exception | None = None
        self.corrupt: Callable[[list[list[float]]], list[list[float]]] | None = None

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.failure is not None:
            raise self.failure
        vectors = [self._vector(text) for text in texts]
        return self.corrupt(vectors) if self.corrupt is not None else vectors

    def _vector(self, text: str) -> list[float]:
        vector = [0.001] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dimensions] += 1.0
        return vector


class FaultyBM25(BM25Index):
    """Real SQLite BM25 index whose named mutators can be made to fail."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.fail_on: set[str] = set()

    def _check(self, mutator: str) -> None:
        if mutator in self.fail_on:
            raise sqlite3.OperationalError(f"database is locked ({mutator})")

    def replace_note(self, note_path: str, rows: list[tuple[str, str, str, str]]) -> None:
        self._check("replace_note")
        super().replace_note(note_path, rows)

    def replace_all(self, rows: list[tuple[str, str, str, str]]) -> None:
        self._check("replace_all")
        super().replace_all(rows)

    def delete_note(self, note_path: str) -> None:
        self._check("delete_note")
        super().delete_note(note_path)


@dataclass
class IndexEnv:
    """A vault plus its real vector and keyword indexes in an isolated directory."""

    vault: Path
    data: Path
    parser: VaultParser
    embedder: BagOfWordsEmbedder
    bm25: FaultyBM25
    store: VaultStore

    def write(self, name: str, body: str) -> Path:
        path = self.vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {path.stem}\ntype: permanent\n---\n\n{body}\n", encoding="utf-8"
        )
        return path

    def note(self, name: str) -> Note:
        return self.parser.parse_file(self.vault / name)

    def index(self, name: str) -> int:
        return self.store.index_single_note(self.note(name), self.parser)

    def surfaces(self, token: str) -> set[str]:
        """Retrieval paths (vector, keyword, hybrid) on which *token* is currently returned."""
        found: set[str] = set()
        vector = self.store.search(token, n_results=50)
        if any(token in hit["content"] for hit in vector):
            found.add("vector")
        keyword_ids = {str(hit["chunk_id"]) for hit in self.bm25.search(token, n_results=50)}
        if keyword_ids:
            found.add("keyword")
        hybrid = self.store.hybrid_search(token, n_results=50)
        if any(token in hit["content"] or hit["chunk_id"] in keyword_ids for hit in hybrid):
            found.add("hybrid")
        return found

    def close(self) -> None:
        self.bm25.close()


def make_env(root: Path) -> IndexEnv:
    """Build an isolated environment under *root* (use a resolved temporary path)."""
    vault = root / "vault"
    data = root / "data"
    vault.mkdir(parents=True)
    data.mkdir()
    config = root / "config.toml"
    config.write_text(
        f'[vault]\npath = "{vault}"\n'
        f'[chroma]\npersist_dir = "{data / "chroma"}"\ncollection_name = "vault_chunks"\n'
        f'[graph]\npersist_path = "{data / "graph.json"}"\n'
        f'[search]\nbm25_db_path = "{data / "bm25.db"}"\ncache_enabled = false\n'
        "[embedding]\ncache_enabled = false\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    embedder = BagOfWordsEmbedder()
    bm25 = FaultyBM25(data / "bm25.db")
    store = VaultStore(settings.chroma, cast("Embedder", embedder), bm25)
    parser = VaultParser(settings.vault)
    return IndexEnv(vault, data, parser, embedder, bm25, store)


def make_neighbour_store(env: IndexEnv, collection_name: str) -> tuple[VaultStore, BM25Index]:
    """A second collection and BM25 file next to the environment's own, sharing nothing."""
    bm25 = BM25Index(env.data / f"{collection_name}.db")
    chroma = ChromaConfig(persist_dir=env.data / "chroma", collection_name=collection_name)
    return VaultStore(chroma, cast("Embedder", env.embedder), bm25), bm25
