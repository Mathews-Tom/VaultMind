"""Tests for the knowledge graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.config import GraphConfig
from vaultmind.graph.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    config = GraphConfig(persist_path=tmp_path / "test_graph.json")
    return KnowledgeGraph(config)


def test_add_and_retrieve_entity(graph: KnowledgeGraph) -> None:
    graph.add_entity("Python", "tool", source_note="test.md")
    entity = graph.get_entity("Python")
    assert entity is not None
    assert entity["type"] == "tool"
    assert "test.md" in entity["source_notes"]


def test_entity_name_normalization(graph: KnowledgeGraph) -> None:
    graph.add_entity("Machine Learning", "concept")
    entity = graph.get_entity("machine learning")
    assert entity is not None
    assert entity["label"] == "Machine Learning"


def test_add_relationship(graph: KnowledgeGraph) -> None:
    graph.add_entity("CAIRN", "project")
    graph.add_entity("MCP", "tool")
    graph.add_relationship("CAIRN", "MCP", "depends_on", source_note="arch.md")

    neighbors = graph.get_neighbors("CAIRN")
    assert len(neighbors["outgoing"]) == 1
    assert neighbors["outgoing"][0]["target"] == "MCP"
    assert neighbors["outgoing"][0]["relation"] == "depends_on"


def test_find_path(graph: KnowledgeGraph) -> None:
    graph.add_entity("A", "concept")
    graph.add_entity("B", "concept")
    graph.add_entity("C", "concept")
    graph.add_relationship("A", "B", "related_to")
    graph.add_relationship("B", "C", "related_to")

    path = graph.find_path("A", "C")
    assert path is not None
    assert len(path) == 3


def test_no_path_returns_none(graph: KnowledgeGraph) -> None:
    graph.add_entity("X", "concept")
    graph.add_entity("Y", "concept")
    path = graph.find_path("X", "Y")
    assert path is None


def test_bridge_entities(graph: KnowledgeGraph) -> None:
    # Create a graph where B connects two clusters
    for name in ["A1", "A2", "B", "C1", "C2"]:
        graph.add_entity(name, "concept")
    graph.add_relationship("A1", "B", "related_to")
    graph.add_relationship("A2", "B", "related_to")
    graph.add_relationship("B", "C1", "related_to")
    graph.add_relationship("B", "C2", "related_to")

    bridges = graph.get_bridge_entities(3)
    assert len(bridges) > 0
    assert bridges[0]["entity"] == "B"


def test_orphan_entities(graph: KnowledgeGraph) -> None:
    graph.add_entity("Connected", "concept")
    graph.add_entity("Orphan", "concept")
    graph.add_relationship("Connected", "Other", "related_to")

    orphans = graph.get_orphan_entities()
    orphan_labels = [o.get("label") for o in orphans]
    assert "Orphan" in orphan_labels


def test_persistence(tmp_path: Path) -> None:
    config = GraphConfig(persist_path=tmp_path / "persist_graph.json")

    # Create and save
    g1 = KnowledgeGraph(config)
    g1.add_entity("Persistent", "concept")
    g1.add_relationship("Persistent", "Other", "related_to")
    g1.save()

    # Load fresh
    g2 = KnowledgeGraph(config)
    entity = g2.get_entity("Persistent")
    assert entity is not None
    assert g2.stats["nodes"] == 2
    assert g2.stats["edges"] == 1


def test_merge_duplicate_entities(graph: KnowledgeGraph) -> None:
    graph.add_entity("Python", "tool", source_note="a.md", confidence=0.7)
    graph.add_entity("Python", "tool", source_note="b.md", confidence=0.9)

    entity = graph.get_entity("Python")
    assert entity is not None
    assert entity["confidence"] == 0.9  # Max confidence
    assert len(entity["source_notes"]) == 2


def test_markdown_summary(graph: KnowledgeGraph) -> None:
    graph.add_entity("A", "project")
    graph.add_entity("B", "tool")
    graph.add_relationship("A", "B", "depends_on")

    report = graph.to_markdown_summary()
    assert "Knowledge Graph Report" in report
    assert "Nodes" in report
