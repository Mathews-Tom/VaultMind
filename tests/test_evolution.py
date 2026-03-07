"""Tests for belief evolution detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from vaultmind.graph.evolution import EvolutionDetector
from vaultmind.graph.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from pathlib import Path


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
