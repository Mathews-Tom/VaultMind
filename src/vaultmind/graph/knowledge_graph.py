"""Knowledge graph — NetworkX-backed graph with JSON persistence.

The graph stores entities (nodes) and relationships (edges) extracted from vault notes.
Designed for single-user scale with upgrade path to Neo4j if needed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from vaultmind.config import GraphConfig

logger = logging.getLogger(__name__)

# Canonical node and edge types
NODE_TYPES = {"person", "project", "concept", "tool", "organization", "event", "location"}
EDGE_TYPES = {
    "related_to",
    "part_of",
    "depends_on",
    "created_by",
    "influences",
    "mentioned_in",
    "competes_with",
    "preceded_by",
}


class KnowledgeGraph:
    """Persistent knowledge graph over vault entities."""

    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self._graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        """Load graph from JSON file if it exists."""
        if self.config.persist_path.exists():
            with open(self.config.persist_path) as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(data, edges="edges")
            logger.info(
                "Loaded knowledge graph: %d nodes, %d edges",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        else:
            logger.info("No existing graph found, starting fresh")

    def save(self) -> None:
        """Persist graph to JSON."""
        self.config.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph, edges="edges")
        with open(self.config.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Saved graph to %s", self.config.persist_path)

    # --- Node operations ---

    def add_entity(
        self,
        name: str,
        entity_type: str,
        source_note: str = "",
        confidence: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add or update an entity node."""
        node_id = self._normalize(name)

        if self._graph.has_node(node_id):
            # Merge: keep highest confidence, accumulate source notes
            existing = self._graph.nodes[node_id]
            sources = set(existing.get("source_notes", []))
            if source_note:
                sources.add(source_note)
            existing["source_notes"] = list(sources)
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            existing.update(attrs)
        else:
            self._graph.add_node(
                node_id,
                label=name,
                type=entity_type,
                source_notes=[source_note] if source_note else [],
                confidence=confidence,
                **attrs,
            )

    def add_relationship(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        source_note: str = "",
        confidence: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add or update a directed relationship between entities."""
        src = self._normalize(source)
        tgt = self._normalize(target)

        # Ensure both nodes exist
        if not self._graph.has_node(src):
            self.add_entity(source, "concept", source_note=source_note)
        if not self._graph.has_node(tgt):
            self.add_entity(target, "concept", source_note=source_note)

        if self._graph.has_edge(src, tgt):
            existing = self._graph.edges[src, tgt]
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            sources = set(existing.get("source_notes", []))
            if source_note:
                sources.add(source_note)
            existing["source_notes"] = list(sources)
        else:
            self._graph.add_edge(
                src,
                tgt,
                relation=relation,
                source_notes=[source_note] if source_note else [],
                confidence=confidence,
                **attrs,
            )

    # --- Query operations ---

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get entity node data by name."""
        node_id = self._normalize(name)
        if self._graph.has_node(node_id):
            return {"id": node_id, **dict(self._graph.nodes[node_id])}
        return None

    def get_neighbors(self, name: str, depth: int = 1) -> dict[str, Any]:
        """Get an entity and its neighborhood subgraph.

        Returns:
            Dict with 'entity', 'outgoing', 'incoming', and 'neighbors' lists.
        """
        node_id = self._normalize(name)
        if not self._graph.has_node(node_id):
            return {"entity": None, "outgoing": [], "incoming": [], "neighbors": []}

        entity = self.get_entity(name)

        outgoing = []
        for _, tgt, data in self._graph.out_edges(node_id, data=True):
            outgoing.append(
                {
                    "target": self._graph.nodes[tgt].get("label", tgt),
                    "relation": data.get("relation", "related_to"),
                    "confidence": data.get("confidence", 1.0),
                }
            )

        incoming = []
        for src, _, data in self._graph.in_edges(node_id, data=True):
            incoming.append(
                {
                    "source": self._graph.nodes[src].get("label", src),
                    "relation": data.get("relation", "related_to"),
                    "confidence": data.get("confidence", 1.0),
                }
            )

        # Get N-hop neighborhood if depth > 1
        neighbors = []
        if depth > 1:
            ego = nx.ego_graph(self._graph, node_id, radius=depth, undirected=True)
            for n in ego.nodes:
                if n != node_id:
                    neighbors.append({"id": n, **dict(self._graph.nodes[n])})

        return {
            "entity": entity,
            "outgoing": outgoing,
            "incoming": incoming,
            "neighbors": neighbors,
        }

    def find_path(self, source: str, target: str) -> list[str] | None:
        """Find shortest path between two entities."""
        src = self._normalize(source)
        tgt = self._normalize(target)
        try:
            path = nx.shortest_path(self._graph.to_undirected(), src, tgt)
            return [self._graph.nodes[n].get("label", n) for n in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_clusters(self, min_size: int = 3) -> list[list[str]]:
        """Find connected communities in the graph."""
        undirected = self._graph.to_undirected()
        communities = list(nx.connected_components(undirected))
        clusters = [
            [self._graph.nodes[n].get("label", n) for n in comm]
            for comm in communities
            if len(comm) >= min_size
        ]
        return sorted(clusters, key=len, reverse=True)

    def get_bridge_entities(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Find entities with highest betweenness centrality (bridge nodes)."""
        if self._graph.number_of_nodes() < 3:
            return []

        centrality = nx.betweenness_centrality(self._graph.to_undirected())
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {
                "entity": self._graph.nodes[node_id].get("label", node_id),
                "centrality": score,
                "type": self._graph.nodes[node_id].get("type", "unknown"),
            }
            for node_id, score in sorted_nodes
            if score > 0
        ]

    def get_orphan_entities(self) -> list[dict[str, Any]]:
        """Find entities with no connections (degree 0)."""
        return [{"id": n, **dict(self._graph.nodes[n])} for n in nx.isolates(self._graph)]

    # --- Stats ---

    @property
    def stats(self) -> dict[str, Any]:
        """Graph statistics summary."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph) if self._graph.number_of_nodes() > 1 else 0,
            "components": nx.number_weakly_connected_components(self._graph),
            "orphans": len(list(nx.isolates(self._graph))),
        }

    # --- Utilities ---

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize entity name to a consistent node ID."""
        return name.strip().lower().replace(" ", "_")

    def to_markdown_summary(self) -> str:
        """Generate a markdown summary for writing to _meta/graph-report.md."""
        stats = self.stats
        bridges = self.get_bridge_entities(5)
        orphans = self.get_orphan_entities()

        lines = [
            "# Knowledge Graph Report",
            "",
            f"**Nodes:** {stats['nodes']} | **Edges:** {stats['edges']} | "
            f"**Density:** {stats['density']:.3f} | **Components:** {stats['components']}",
            "",
        ]

        if bridges:
            lines.append("## Bridge Entities (High Connectivity)")
            lines.append("")
            for b in bridges:
                centrality = b["centrality"]
                lines.append(f"- **{b['entity']}** ({b['type']}) — centrality: {centrality:.3f}")
            lines.append("")

        if orphans:
            lines.append("## Orphan Entities (No Connections)")
            lines.append("")
            for o in orphans:
                lines.append(f"- {o.get('label', o['id'])} ({o.get('type', 'unknown')})")
            lines.append("")

        return "\n".join(lines)
