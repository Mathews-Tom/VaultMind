"""Connection density scoring for knowledge graph-aware ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConnectionDensityScore:
    """Density score for a single note based on its graph connectivity."""

    note_path: str
    entity_count: int
    connected_entities: int
    total_neighbors: int
    density_score: float  # normalized [0.0, 1.0]


class ConnectionDensityCalculator:
    """Calculate connection density scores for notes using the knowledge graph."""

    def __init__(self, knowledge_graph: Any, config: Any) -> None:
        self._graph = knowledge_graph
        self._config = config

    def score_note(
        self, note_path: str, entities: list[str] | None = None
    ) -> ConnectionDensityScore:
        """Calculate connection density for a note based on its entities in the graph.

        For each entity:
        1. Look up in graph via get_entity()
        2. Skip if confidence < config.entity_confidence_threshold
        3. Build ego_subgraph(entity, depth=config.connection_max_hops)
        4. Collect unique neighbor node IDs (excluding the entity itself)

        Normalize: density = total_neighbors / (total_neighbors + 10)
        This gives sigmoid-like behavior: 10 neighbors -> 0.5, 30 -> 0.75, 100 -> 0.91
        """
        zero = ConnectionDensityScore(
            note_path=note_path,
            entity_count=0,
            connected_entities=0,
            total_neighbors=0,
            density_score=0.0,
        )

        if not entities:
            return zero

        if self._graph.stats["nodes"] == 0:
            return zero

        entity_count = len(entities)
        connected_entities = 0
        all_neighbors: set[str] = set()

        for entity_name in entities:
            entity_data = self._graph.get_entity(entity_name)
            if entity_data is None:
                continue

            confidence = entity_data.get("confidence", 1.0)
            if confidence < self._config.entity_confidence_threshold:
                continue

            connected_entities += 1
            subgraph = self._graph.ego_subgraph(entity_name, depth=self._config.connection_max_hops)
            normalized_id = entity_name.strip().lower().replace(" ", "_")
            for node_id in subgraph.nodes():
                if node_id != normalized_id:
                    all_neighbors.add(node_id)

        total = len(all_neighbors)
        density = total / (total + 10) if total > 0 else 0.0

        return ConnectionDensityScore(
            note_path=note_path,
            entity_count=entity_count,
            connected_entities=connected_entities,
            total_neighbors=total,
            density_score=density,
        )
