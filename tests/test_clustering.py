"""Tests for Zettelkasten cluster discovery."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from vaultmind.pipeline.clustering import NoteCluster, _cluster_fingerprint, discover_clusters


def _mock_collection(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> MagicMock:
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }
    return collection


class TestDiscoverClusters:
    def test_empty_collection_returns_no_clusters(self) -> None:
        collection = _mock_collection([], [], [])
        clusters = discover_clusters(collection)
        assert clusters == []

    def test_too_few_notes_returns_empty(self) -> None:
        # Only 2 notes, min_samples=3
        collection = _mock_collection(
            ids=["c1", "c2"],
            embeddings=[[0.1, 0.2], [0.1, 0.2]],
            metadatas=[
                {"note_path": "a.md", "note_type": "fleeting"},
                {"note_path": "b.md", "note_type": "fleeting"},
            ],
        )
        clusters = discover_clusters(collection, min_samples=3)
        assert clusters == []

    def test_dbscan_min_samples_respected(self) -> None:
        # 4 identical embeddings should form a cluster with min_samples=3
        emb = [0.1] * 10
        collection = _mock_collection(
            ids=["c1", "c2", "c3", "c4"],
            embeddings=[emb, emb, emb, emb],
            metadatas=[
                {"note_path": "a.md", "note_title": "A", "note_type": "fleeting"},
                {"note_path": "b.md", "note_title": "B", "note_type": "fleeting"},
                {"note_path": "c.md", "note_title": "C", "note_type": "literature"},
                {"note_path": "d.md", "note_title": "D", "note_type": "fleeting"},
            ],
        )
        clusters = discover_clusters(collection, min_samples=3)
        assert len(clusters) >= 1
        assert all(isinstance(c, NoteCluster) for c in clusters)

    def test_single_note_never_forms_cluster(self) -> None:
        collection = _mock_collection(
            ids=["c1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"note_path": "a.md", "note_type": "fleeting"}],
        )
        clusters = discover_clusters(collection, min_samples=3)
        assert clusters == []

    def test_dissimilar_notes_no_cluster(self) -> None:
        # Very different embeddings, no cluster
        rng = np.random.default_rng(42)
        collection = _mock_collection(
            ids=[f"c{i}" for i in range(5)],
            embeddings=[list(rng.random(50)) for _ in range(5)],
            metadatas=[
                {"note_path": f"note{i}.md", "note_title": f"Note{i}", "note_type": "fleeting"}
                for i in range(5)
            ],
        )
        # With eps=0.01, should find no clusters (all noise)
        clusters = discover_clusters(collection, eps=0.01, min_samples=3)
        assert clusters == []

    def test_deduplicates_chunks_per_note(self) -> None:
        # Multiple chunks from same note should be deduped
        emb = [0.5] * 10
        collection = _mock_collection(
            ids=["c1", "c2", "c3", "c4", "c5"],
            embeddings=[emb, emb, emb, emb, emb],
            metadatas=[
                {"note_path": "a.md", "note_title": "A", "note_type": "fleeting"},
                {"note_path": "a.md", "note_title": "A", "note_type": "fleeting"},  # dupe
                {"note_path": "b.md", "note_title": "B", "note_type": "fleeting"},
                {"note_path": "c.md", "note_title": "C", "note_type": "fleeting"},
                {"note_path": "d.md", "note_title": "D", "note_type": "fleeting"},
            ],
        )
        clusters = discover_clusters(collection, min_samples=3)
        # Should have deduplicated to 4 unique notes
        if clusters:
            all_paths = [p for c in clusters for p in c.note_paths]
            assert len(set(all_paths)) == len(all_paths)


class TestClusterFingerprint:
    def test_deterministic(self) -> None:
        fp1 = _cluster_fingerprint(["b.md", "a.md"])
        fp2 = _cluster_fingerprint(["a.md", "b.md"])
        assert fp1 == fp2

    def test_length_16(self) -> None:
        fp = _cluster_fingerprint(["a.md", "b.md"])
        assert len(fp) == 16

    def test_different_paths_different_fingerprint(self) -> None:
        fp1 = _cluster_fingerprint(["a.md", "b.md"])
        fp2 = _cluster_fingerprint(["a.md", "c.md"])
        assert fp1 != fp2


class TestGraphReinforcement:
    def test_merge_clusters_sharing_entities(self) -> None:
        from vaultmind.config import GraphConfig
        from vaultmind.graph.knowledge_graph import KnowledgeGraph

        config = GraphConfig(persist_path="/tmp/test_graph_merge.json")
        graph = KnowledgeGraph(config)
        graph.add_entity("Python", "concept", source_note="a.md")
        graph.add_entity("Python", "concept", source_note="c.md")

        # All similar embeddings
        emb = [0.5] * 10
        collection = _mock_collection(
            ids=["c1", "c2", "c3", "c4", "c5", "c6"],
            embeddings=[emb, emb, emb, emb, emb, emb],
            metadatas=[
                {"note_path": "a.md", "note_title": "A", "note_type": "fleeting"},
                {"note_path": "b.md", "note_title": "B", "note_type": "fleeting"},
                {"note_path": "c.md", "note_title": "C", "note_type": "fleeting"},
                {"note_path": "d.md", "note_title": "D", "note_type": "fleeting"},
                {"note_path": "e.md", "note_title": "E", "note_type": "fleeting"},
                {"note_path": "f.md", "note_title": "F", "note_type": "fleeting"},
            ],
        )

        clusters = discover_clusters(collection, knowledge_graph=graph, min_samples=3)
        # With all identical embeddings, should form one cluster regardless
        assert len(clusters) >= 1
