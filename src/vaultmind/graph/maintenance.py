"""Graph maintenance — orphan cleanup and stale source-note pruning.

Provides:
1. **Stale source-note pruning** — Removes references to deleted vault files
   from entity ``source_notes`` lists and edge ``source_notes`` lists.
2. **Orphan entity cleanup** — Removes entities with no connections AND no
   remaining source notes (i.e., the note that introduced them was deleted
   and no other note references them).
3. **Event-bus integration** — Subscribes to ``NoteDeletedEvent`` to perform
   incremental graph cleanup on note deletion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.vault.events import NoteDeletedEvent

logger = logging.getLogger(__name__)


class GraphMaintainer:
    """Maintains graph hygiene by pruning stale references and orphan entities."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def prune_stale_sources(self, existing_note_paths: set[str]) -> dict[str, int]:
        """Remove source_notes references to files no longer in the vault.

        Args:
            existing_note_paths: Set of relative note paths that exist in the vault.

        Returns:
            Stats dict with counts of pruned node and edge source references.
        """
        nodes_pruned = 0
        edges_pruned = 0

        # Prune stale source_notes from nodes
        for node_id in list(self._graph._graph.nodes):
            data = self._graph._graph.nodes[node_id]
            sources = data.get("source_notes", [])
            if not sources:
                continue
            cleaned = [s for s in sources if s in existing_note_paths]
            if len(cleaned) < len(sources):
                data["source_notes"] = cleaned
                nodes_pruned += len(sources) - len(cleaned)

        # Prune stale source_notes from edges
        for u, v in list(self._graph._graph.edges):
            data = self._graph._graph.edges[u, v]
            sources = data.get("source_notes", [])
            if not sources:
                continue
            cleaned = [s for s in sources if s in existing_note_paths]
            if len(cleaned) < len(sources):
                data["source_notes"] = cleaned
                edges_pruned += len(sources) - len(cleaned)

        if nodes_pruned or edges_pruned:
            logger.info(
                "Pruned stale sources: %d node refs, %d edge refs",
                nodes_pruned,
                edges_pruned,
            )

        return {"nodes_pruned": nodes_pruned, "edges_pruned": edges_pruned}

    def remove_orphans(self, *, require_no_sources: bool = True) -> int:
        """Remove orphan entities (degree 0) from the graph.

        Args:
            require_no_sources: If True (default), only remove orphans that also
                have no remaining source_notes. This prevents removing entities
                that were explicitly defined in a note's frontmatter but happen
                to have no graph edges yet.

        Returns:
            Count of removed nodes.
        """
        import networkx as nx

        orphans = list(nx.isolates(self._graph._graph))
        removed = 0

        for node_id in orphans:
            if require_no_sources:
                data = self._graph._graph.nodes[node_id]
                sources = data.get("source_notes", [])
                if sources:
                    continue

            self._graph._graph.remove_node(node_id)
            removed += 1

        if removed:
            logger.info("Removed %d orphan entities", removed)

        return removed

    def remove_edges_for_note(self, note_path: str) -> int:
        """Remove edges whose only source is the given note path.

        Edges with multiple source_notes just get the path removed from the list.

        Returns:
            Count of fully removed edges.
        """
        removed = 0

        for u, v in list(self._graph._graph.edges):
            data = self._graph._graph.edges[u, v]
            sources = data.get("source_notes", [])
            if note_path not in sources:
                continue

            if len(sources) == 1:
                self._graph._graph.remove_edge(u, v)
                removed += 1
            else:
                data["source_notes"] = [s for s in sources if s != note_path]

        return removed

    def remove_entities_for_note(self, note_path: str) -> int:
        """Remove entities whose only source is the given note path.

        Entities with multiple source_notes just get the path removed from the list.

        Returns:
            Count of fully removed entities.
        """
        removed = 0

        for node_id in list(self._graph._graph.nodes):
            data = self._graph._graph.nodes[node_id]
            sources = data.get("source_notes", [])
            if note_path not in sources:
                continue

            if len(sources) == 1:
                self._graph._graph.remove_node(node_id)
                removed += 1
            else:
                data["source_notes"] = [s for s in sources if s != note_path]

        return removed

    def cleanup_deleted_note(self, note_path: str) -> dict[str, int]:
        """Full cleanup for a deleted note: remove edges, entities, then orphans.

        Returns:
            Stats dict with counts.
        """
        edges = self.remove_edges_for_note(note_path)
        entities = self.remove_entities_for_note(note_path)
        orphans = self.remove_orphans(require_no_sources=True)

        total = edges + entities + orphans
        if total:
            self._graph.save()
            logger.info(
                "Graph cleanup for %s: %d edges, %d entities, %d orphans removed",
                note_path,
                edges,
                entities,
                orphans,
            )

        return {
            "edges_removed": edges,
            "entities_removed": entities,
            "orphans_removed": orphans,
        }

    def full_maintenance(self, existing_note_paths: set[str]) -> dict[str, int]:
        """Run full maintenance: prune stale sources, remove orphans, save.

        Args:
            existing_note_paths: Set of note paths currently in the vault.

        Returns:
            Combined stats dict.
        """
        prune_stats = self.prune_stale_sources(existing_note_paths)
        orphan_count = self.remove_orphans(require_no_sources=True)

        self._graph.save()

        return {
            **prune_stats,
            "orphans_removed": orphan_count,
        }

    # ------------------------------------------------------------------
    # Event bus integration
    # ------------------------------------------------------------------

    async def on_note_deleted(self, event: NoteDeletedEvent) -> None:
        """Event bus callback — cleanup graph on note deletion."""
        path: Path = event.path
        # Try to compute relative path (same logic as watch_handler)
        note_path = str(path)

        await asyncio.to_thread(self.cleanup_deleted_note, note_path)
