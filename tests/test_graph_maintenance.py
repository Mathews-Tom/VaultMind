"""Tests for GraphMaintainer — orphan cleanup, stale source pruning, event integration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.graph.maintenance import GraphMaintainer
from vaultmind.vault.events import NoteDeletedEvent


class FakeGraphConfig:
    """Minimal GraphConfig stand-in for tests."""

    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.min_confidence = 0.7


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    config = FakeGraphConfig(tmp_path / "graph.json")
    return KnowledgeGraph(config)  # type: ignore[arg-type]


@pytest.fixture
def maintainer(graph: KnowledgeGraph) -> GraphMaintainer:
    return GraphMaintainer(graph)


class TestPruneStaleSources:
    def test_removes_references_to_deleted_notes(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Python", "concept", source_note="notes/a.md")
        graph.add_entity("Python", "concept", source_note="notes/b.md")

        stats = maintainer.prune_stale_sources({"notes/a.md"})
        data = graph._graph.nodes[graph._normalize("Python")]
        assert "notes/b.md" not in data["source_notes"]
        assert "notes/a.md" in data["source_notes"]
        assert stats["nodes_pruned"] == 1

    def test_prunes_edge_sources(self, graph: KnowledgeGraph, maintainer: GraphMaintainer) -> None:
        graph.add_entity("A", "concept", source_note="x.md")
        graph.add_entity("B", "concept", source_note="x.md")
        graph.add_relationship("A", "B", source_note="x.md")
        graph.add_relationship("A", "B", source_note="y.md")

        stats = maintainer.prune_stale_sources({"x.md"})
        edge_data = graph._graph.edges[graph._normalize("A"), graph._normalize("B")]
        assert "y.md" not in edge_data["source_notes"]
        assert stats["edges_pruned"] == 1

    def test_noop_when_all_sources_exist(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("X", "tool", source_note="a.md")
        stats = maintainer.prune_stale_sources({"a.md"})
        assert stats["nodes_pruned"] == 0
        assert stats["edges_pruned"] == 0


class TestRemoveOrphans:
    def test_removes_orphan_with_no_sources(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Orphan", "concept")
        assert graph._graph.has_node(graph._normalize("Orphan"))

        removed = maintainer.remove_orphans(require_no_sources=True)
        assert removed == 1
        assert not graph._graph.has_node(graph._normalize("Orphan"))

    def test_keeps_orphan_with_sources(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Referenced", "concept", source_note="note.md")

        removed = maintainer.remove_orphans(require_no_sources=True)
        assert removed == 0
        assert graph._graph.has_node(graph._normalize("Referenced"))

    def test_removes_orphan_with_sources_when_not_required(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("HasSource", "concept", source_note="note.md")

        removed = maintainer.remove_orphans(require_no_sources=False)
        assert removed == 1

    def test_keeps_connected_entities(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("A", "concept")
        graph.add_entity("B", "concept")
        graph.add_relationship("A", "B")

        removed = maintainer.remove_orphans(require_no_sources=False)
        assert removed == 0


class TestRemoveEdgesForNote:
    def test_removes_single_source_edge(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("A", "concept")
        graph.add_entity("B", "concept")
        graph.add_relationship("A", "B", source_note="deleted.md")

        removed = maintainer.remove_edges_for_note("deleted.md")
        assert removed == 1
        assert not graph._graph.has_edge(graph._normalize("A"), graph._normalize("B"))

    def test_preserves_multi_source_edge(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("A", "concept")
        graph.add_entity("B", "concept")
        graph.add_relationship("A", "B", source_note="keep.md")
        graph.add_relationship("A", "B", source_note="delete.md")

        removed = maintainer.remove_edges_for_note("delete.md")
        assert removed == 0
        edge = graph._graph.edges[graph._normalize("A"), graph._normalize("B")]
        assert "delete.md" not in edge["source_notes"]
        assert "keep.md" in edge["source_notes"]


class TestRemoveEntitiesForNote:
    def test_removes_single_source_entity(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Lonely", "concept", source_note="gone.md")
        removed = maintainer.remove_entities_for_note("gone.md")
        assert removed == 1

    def test_preserves_multi_source_entity(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Shared", "concept", source_note="a.md")
        graph.add_entity("Shared", "concept", source_note="b.md")

        removed = maintainer.remove_entities_for_note("a.md")
        assert removed == 0
        data = graph._graph.nodes[graph._normalize("Shared")]
        assert data["source_notes"] == ["b.md"]


class TestCleanupDeletedNote:
    def test_full_cleanup(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer, tmp_path: Path
    ) -> None:
        graph.add_entity("E1", "concept", source_note="target.md")
        graph.add_entity("E2", "concept", source_note="target.md")
        graph.add_relationship("E1", "E2", source_note="target.md")

        stats = maintainer.cleanup_deleted_note("target.md")
        assert stats["edges_removed"] == 1
        assert stats["entities_removed"] == 2
        assert graph._graph.number_of_nodes() == 0


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_on_note_deleted(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer
    ) -> None:
        graph.add_entity("Temp", "concept", source_note="deleted.md")

        event = NoteDeletedEvent(path=Path("deleted.md"))
        await maintainer.on_note_deleted(event)

        # Allow thread to complete
        await asyncio.sleep(0.05)

        assert not graph._graph.has_node(graph._normalize("Temp"))


class TestFullMaintenance:
    def test_full_maintenance(
        self, graph: KnowledgeGraph, maintainer: GraphMaintainer, tmp_path: Path
    ) -> None:
        graph.add_entity("Alive", "concept", source_note="exists.md")
        graph.add_entity("Dead", "concept", source_note="gone.md")
        graph.add_entity("Orphan", "concept")

        stats = maintainer.full_maintenance({"exists.md"})

        assert stats["nodes_pruned"] == 1  # Dead lost its source
        assert stats["orphans_removed"] >= 1  # Both Dead (now sourceless) and Orphan
        assert graph._graph.has_node(graph._normalize("Alive"))
