"""Tests for belief evolution detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

from vaultmind.graph.evolution import EvolutionDetector, LineageEdge, LineageStore
from vaultmind.graph.knowledge_graph import KnowledgeGraph


def _add_edge(
    g: nx.MultiDiGraph,
    src: str,
    tgt: str,
    *,
    key: str,
    relation: str = "depends_on",
    confidence: float = 0.5,
    source_notes: list[str] | None = None,
) -> None:
    """Helper to add an edge with less boilerplate."""
    g.add_edge(
        src,
        tgt,
        key=key,
        relation=relation,
        confidence=confidence,
        source_notes=source_notes or [],
    )


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    from vaultmind.config import GraphConfig

    config = GraphConfig(persist_path=tmp_path / "graph.json")
    g = KnowledgeGraph(config)
    return g


@pytest.fixture
def multi_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Swap the internal DiGraph for a MultiDiGraph to test multi-edge scenarios."""
    graph._graph = nx.MultiDiGraph()
    return graph


@pytest.fixture
def detector(graph: KnowledgeGraph, tmp_path: Path) -> EvolutionDetector:
    return EvolutionDetector(
        knowledge_graph=graph,
        confidence_drift_threshold=0.3,
        stale_days=180,
        min_confidence_for_stale=0.8,
        store_path=tmp_path / "dismissed.json",
    )


@pytest.fixture
def multi_detector(multi_graph: KnowledgeGraph, tmp_path: Path) -> EvolutionDetector:
    return EvolutionDetector(
        knowledge_graph=multi_graph,
        confidence_drift_threshold=0.3,
        stale_days=180,
        min_confidence_for_stale=0.8,
        store_path=tmp_path / "dismissed.json",
    )


class TestConfidenceDrift:
    def test_detects_drift(
        self, multi_graph: KnowledgeGraph, multi_detector: EvolutionDetector
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="Alpha")
        g.add_node("b", label="Beta")
        g.add_edge(
            "a",
            "b",
            key="edge1",
            relation="depends_on",
            confidence=0.9,
            source_notes=["note1.md"],
        )
        g.add_edge(
            "a",
            "b",
            key="edge2",
            relation="depends_on",
            confidence=0.4,
            source_notes=["note2.md"],
        )

        signals = multi_detector.scan()
        drift_signals = [s for s in signals if s.signal_type == "confidence_drift"]
        assert len(drift_signals) >= 1

    def test_ignores_drift_below_threshold(
        self, multi_graph: KnowledgeGraph, multi_detector: EvolutionDetector
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="Alpha")
        g.add_node("b", label="Beta")
        g.add_edge(
            "a",
            "b",
            key="edge1",
            relation="depends_on",
            confidence=0.8,
            source_notes=["note1.md"],
        )
        g.add_edge(
            "a",
            "b",
            key="edge2",
            relation="depends_on",
            confidence=0.7,
            source_notes=["note2.md"],
        )

        signals = multi_detector.scan()
        drift_signals = [s for s in signals if s.signal_type == "confidence_drift"]
        assert len(drift_signals) == 0

    def test_single_source_no_drift(
        self, graph: KnowledgeGraph, detector: EvolutionDetector
    ) -> None:
        graph.add_entity("Alpha", "concept")
        graph.add_entity("Beta", "concept")
        graph.add_relationship("Alpha", "Beta", "depends_on", confidence=0.9)
        signals = detector.scan()
        drift_signals = [s for s in signals if s.signal_type == "confidence_drift"]
        assert len(drift_signals) == 0


class TestRelationshipShift:
    def test_detects_shift(
        self, multi_graph: KnowledgeGraph, multi_detector: EvolutionDetector
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="Alpha")
        g.add_node("b", label="Beta")
        g.add_edge(
            "a",
            "b",
            key="edge1",
            relation="depends_on",
            confidence=0.8,
            source_notes=["early.md"],
        )
        g.add_edge(
            "a",
            "b",
            key="edge2",
            relation="competes_with",
            confidence=0.7,
            source_notes=["revised.md"],
        )

        signals = multi_detector.scan()
        shift_signals = [s for s in signals if s.signal_type == "relationship_shift"]
        assert len(shift_signals) >= 1


class TestStaleClaims:
    def test_detects_stale(self, graph: KnowledgeGraph, detector: EvolutionDetector) -> None:
        graph.add_entity("Alpha", "concept")
        graph.add_entity("Beta", "concept")
        graph.add_relationship(
            "Alpha",
            "Beta",
            "influences",
            source_note="old-paper.md",
            confidence=0.95,
        )

        old_date = datetime.now(UTC) - timedelta(days=280)
        file_ages = {"old-paper.md": old_date}

        signals = detector.scan_with_file_ages(file_ages)
        stale = [s for s in signals if s.signal_type == "stale_claim"]
        assert len(stale) >= 1
        assert "280" in stale[0].detail

    def test_ignores_low_confidence_stale(
        self, graph: KnowledgeGraph, detector: EvolutionDetector
    ) -> None:
        graph.add_entity("Alpha", "concept")
        graph.add_entity("Beta", "concept")
        graph.add_relationship(
            "Alpha",
            "Beta",
            "related_to",
            source_note="old.md",
            confidence=0.5,
        )

        old_date = datetime.now(UTC) - timedelta(days=365)
        signals = detector.scan_with_file_ages({"old.md": old_date})
        stale = [s for s in signals if s.signal_type == "stale_claim"]
        assert len(stale) == 0

    def test_ignores_recent_notes(self, graph: KnowledgeGraph, detector: EvolutionDetector) -> None:
        graph.add_entity("Alpha", "concept")
        graph.add_entity("Beta", "concept")
        graph.add_relationship(
            "Alpha",
            "Beta",
            "influences",
            source_note="recent.md",
            confidence=0.95,
        )

        recent_date = datetime.now(UTC) - timedelta(days=30)
        signals = detector.scan_with_file_ages({"recent.md": recent_date})
        stale = [s for s in signals if s.signal_type == "stale_claim"]
        assert len(stale) == 0

    def test_handles_naive_datetime(
        self, graph: KnowledgeGraph, detector: EvolutionDetector
    ) -> None:
        graph.add_entity("Alpha", "concept")
        graph.add_entity("Beta", "concept")
        graph.add_relationship(
            "Alpha",
            "Beta",
            "influences",
            source_note="old.md",
            confidence=0.9,
        )

        # Naive datetime (no tzinfo) should be treated as UTC
        old_date = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=200)
        signals = detector.scan_with_file_ages({"old.md": old_date})
        stale = [s for s in signals if s.signal_type == "stale_claim"]
        assert len(stale) >= 1


class TestDismiss:
    def test_dismiss_persists(
        self,
        multi_graph: KnowledgeGraph,
        multi_detector: EvolutionDetector,
        tmp_path: Path,
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        _add_edge(g, "a", "b", key="e1", confidence=0.9, source_notes=["n1.md"])
        _add_edge(g, "a", "b", key="e2", confidence=0.4, source_notes=["n2.md"])

        signals = multi_detector.scan()
        assert len(signals) > 0
        evo_id = signals[0].evolution_id

        multi_detector.dismiss_by_id(evo_id)

        # Re-scan should exclude dismissed
        signals2 = multi_detector.scan()
        dismissed_ids = [s.evolution_id for s in signals2]
        assert evo_id not in dismissed_ids

    def test_dismissed_excluded_from_scan(
        self,
        multi_graph: KnowledgeGraph,
        multi_detector: EvolutionDetector,
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        _add_edge(g, "a", "b", key="e1", confidence=0.9, source_notes=["n1.md"])
        _add_edge(g, "a", "b", key="e2", confidence=0.3, source_notes=["n2.md"])

        signals = multi_detector.scan()
        for s in signals:
            multi_detector.dismiss_by_id(s.evolution_id)

        signals2 = multi_detector.scan()
        assert len(signals2) == 0

    def test_dismiss_by_prefix(
        self,
        multi_graph: KnowledgeGraph,
        multi_detector: EvolutionDetector,
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        _add_edge(g, "a", "b", key="e1", confidence=0.9, source_notes=["n1.md"])
        _add_edge(g, "a", "b", key="e2", confidence=0.4, source_notes=["n2.md"])

        signals = multi_detector.scan()
        assert len(signals) > 0
        evo_id = signals[0].evolution_id
        prefix = evo_id[:8]

        found = multi_detector.dismiss(prefix)
        assert found is True

        signals2 = multi_detector.scan()
        assert all(s.evolution_id != evo_id for s in signals2)

    def test_dismiss_nonexistent_returns_false(self, detector: EvolutionDetector) -> None:
        found = detector.dismiss("nonexistent")
        assert found is False


class TestEvolutionId:
    def test_id_is_stable(self) -> None:
        id1 = EvolutionDetector._make_id("A", "B", "drift", ["n1.md", "n2.md"])
        id2 = EvolutionDetector._make_id("A", "B", "drift", ["n2.md", "n1.md"])
        assert id1 == id2  # sorted source_notes

    def test_id_is_16_chars(self) -> None:
        eid = EvolutionDetector._make_id("X", "Y", "shift", ["a.md"])
        assert len(eid) == 16

    def test_different_types_different_ids(self) -> None:
        id1 = EvolutionDetector._make_id("A", "B", "drift", ["n.md"])
        id2 = EvolutionDetector._make_id("A", "B", "shift", ["n.md"])
        assert id1 != id2


class TestEmptyGraph:
    def test_empty_graph_returns_empty(self, detector: EvolutionDetector) -> None:
        signals = detector.scan()
        assert signals == []

    def test_empty_graph_with_file_ages(self, detector: EvolutionDetector) -> None:
        signals = detector.scan_with_file_ages({})
        assert signals == []


class TestSeveritySorting:
    def test_signals_sorted_by_severity_descending(
        self,
        multi_graph: KnowledgeGraph,
        multi_detector: EvolutionDetector,
    ) -> None:
        g = multi_graph._graph
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        g.add_node("c", label="C")
        g.add_node("d", label="D")

        # Small drift (severity ~0.6)
        _add_edge(g, "a", "b", key="e1", relation="r", confidence=0.9, source_notes=["n1.md"])
        _add_edge(g, "a", "b", key="e2", relation="r", confidence=0.6, source_notes=["n2.md"])

        # Large drift (severity 1.0)
        _add_edge(g, "c", "d", key="e3", relation="r", confidence=1.0, source_notes=["n3.md"])
        _add_edge(g, "c", "d", key="e4", relation="r", confidence=0.1, source_notes=["n4.md"])

        signals = multi_detector.scan()
        drift_signals = [s for s in signals if s.signal_type == "confidence_drift"]
        assert len(drift_signals) == 2
        assert drift_signals[0].severity >= drift_signals[1].severity


class TestLineageStoreRecord:
    def test_record_returns_lineage_edge(self, tmp_path: Path) -> None:
        store = LineageStore(store_path=tmp_path / "lineage.json")

        edge = store.record("00-inbox/note.md", "deleted", "Requested via /delete")

        assert isinstance(edge, LineageEdge)
        assert edge.note_path == "00-inbox/note.md"
        assert edge.event == "deleted"
        assert edge.detail == "Requested via /delete"

    def test_record_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "lineage.json"
        store = LineageStore(store_path=path)
        store.record("note.md", "edited", "Instruction: add a summary")

        reloaded = LineageStore(store_path=path)

        assert len(reloaded.get_lineage("note.md")) == 1
        assert reloaded.get_lineage("note.md")[0].detail == "Instruction: add a summary"

    def test_default_store_path_under_vaultmind_home(self) -> None:
        store = LineageStore()

        assert store._store_path == Path.home() / ".vaultmind" / "data" / "lineage.json"


class TestLineageStoreGetLineage:
    def test_returns_only_matching_note(self, tmp_path: Path) -> None:
        store = LineageStore(store_path=tmp_path / "lineage.json")
        store.record("a.md", "deleted", "delete a")
        store.record("b.md", "edited", "edit b")

        lineage = store.get_lineage("a.md")

        assert len(lineage) == 1
        assert lineage[0].note_path == "a.md"

    def test_returns_oldest_first(self, tmp_path: Path) -> None:
        store = LineageStore(store_path=tmp_path / "lineage.json")
        store.record("a.md", "edited", "first edit")
        store.record("a.md", "edited", "second edit")
        store.record("a.md", "deleted", "then deleted")

        lineage = store.get_lineage("a.md")

        assert [e.detail for e in lineage] == ["first edit", "second edit", "then deleted"]

    def test_unknown_note_returns_empty(self, tmp_path: Path) -> None:
        store = LineageStore(store_path=tmp_path / "lineage.json")

        assert store.get_lineage("never-recorded.md") == []


class TestLineageStoreDecoupledFromGraph:
    def test_constructs_without_knowledge_graph_or_evolution_detector(self, tmp_path: Path) -> None:
        """Lineage recording must work even when belief-evolution scanning is
        disabled (`[evolution].enabled = false`) — it is a safety/audit
        invariant, not a feature of `EvolutionDetector`."""
        store = LineageStore(store_path=tmp_path / "lineage.json")

        edge = store.record("note.md", "deleted", "no graph required")

        assert edge.edge_id
