"""Tests for KnowledgeGraph ego_subgraph and shortest_paths methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest

from vaultmind.graph.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    """Create a test graph with some entities and relationships."""
    from vaultmind.config import GraphConfig

    config = GraphConfig(persist_path=tmp_path / "graph.json")
    g = KnowledgeGraph(config)

    # Build a small test graph: A -> B -> C -> D, A -> D (shortcut)
    g.add_entity("Alpha", "project", confidence=0.9)
    g.add_entity("Beta", "concept", confidence=0.8)
    g.add_entity("Gamma", "tool", confidence=0.7)
    g.add_entity("Delta", "person", confidence=0.85)
    g.add_entity("Epsilon", "concept", confidence=0.6)  # isolated

    g.add_relationship("Alpha", "Beta", "depends_on", confidence=0.9)
    g.add_relationship("Beta", "Gamma", "influences", confidence=0.7)
    g.add_relationship("Gamma", "Delta", "created_by", confidence=0.8)
    g.add_relationship("Alpha", "Delta", "related_to", confidence=0.5)

    return g


class TestEgoSubgraph:
    def test_depth_1_returns_direct_neighbors(self, graph: KnowledgeGraph) -> None:
        sub = graph.ego_subgraph("Alpha", depth=1)
        labels = {data.get("label") for _, data in sub.nodes(data=True)}
        assert "Alpha" in labels
        assert "Beta" in labels
        assert "Delta" in labels
        # Gamma is 2 hops away
        assert "Gamma" not in labels

    def test_depth_2_returns_2hop(self, graph: KnowledgeGraph) -> None:
        sub = graph.ego_subgraph("Alpha", depth=2)
        labels = {data.get("label") for _, data in sub.nodes(data=True)}
        assert "Gamma" in labels

    def test_nonexistent_entity_returns_empty(self, graph: KnowledgeGraph) -> None:
        sub = graph.ego_subgraph("NonExistent", depth=2)
        assert sub.number_of_nodes() == 0

    def test_returns_digraph(self, graph: KnowledgeGraph) -> None:
        sub = graph.ego_subgraph("Alpha", depth=1)
        assert isinstance(sub, nx.DiGraph)

    def test_preserves_edge_data(self, graph: KnowledgeGraph) -> None:
        sub = graph.ego_subgraph("Alpha", depth=1)
        # Check that the edge from Alpha to Beta has relation data
        alpha_id = graph._normalize("Alpha")
        beta_id = graph._normalize("Beta")
        assert sub.has_edge(alpha_id, beta_id)
        assert sub.edges[alpha_id, beta_id]["relation"] == "depends_on"


class TestShortestPaths:
    def test_direct_path(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Delta")
        assert len(paths) >= 1
        # Direct path A -> D should be first (shorter)
        shortest = paths[0]
        assert shortest[0]["entity"] == "Alpha"
        assert shortest[-1]["entity"] == "Delta"

    def test_max_length_respected(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Delta", max_length=1)
        # Only the direct A -> D path should fit
        for p in paths:
            assert len(p) <= 2  # max_length=1 means 2 nodes

    def test_nonexistent_entity_returns_empty(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "NonExistent")
        assert paths == []

    def test_no_path_returns_empty(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Epsilon")
        assert paths == []

    def test_paths_sorted_by_min_confidence(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Delta", max_length=4)
        if len(paths) >= 2:

            def min_conf(path):
                confs = [float(s.get("confidence", 1.0)) for s in path if "confidence" in s]
                return min(confs) if confs else 0.0

            for i in range(len(paths) - 1):
                assert min_conf(paths[i]) >= min_conf(paths[i + 1])

    def test_enriched_with_relation_data(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Beta", max_length=1)
        assert len(paths) == 1
        step = paths[0][0]
        assert "relation" in step
        assert step["relation"] == "depends_on"

    def test_top_k_limits_results(self, graph: KnowledgeGraph) -> None:
        paths = graph.shortest_paths("Alpha", "Delta", max_length=4, top_k=1)
        assert len(paths) <= 1
