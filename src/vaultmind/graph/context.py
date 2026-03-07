"""Graph context builder for thinking partner sessions.

Extracts named entities from queries via LLM, builds ego subgraphs,
and serializes graph context for injection into thinking prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaultmind.llm.client import LLMError, Message

if TYPE_CHECKING:
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.llm.client import LLMClient

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """\
Extract named entities from this query. Return ONLY a JSON array of strings.
Named entities are: people, projects, concepts, tools, organizations.
Do not include generic words. Only include specific names and terms.

Query: {query}
"""


@dataclass
class GraphContextBlock:
    """Structured graph context ready for prompt injection."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)

    def render(self, min_confidence: float = 0.6) -> str:
        """Serialize to markdown for system prompt injection."""
        if not self.entities and not self.relationships:
            return ""

        lines = ["## Knowledge Graph Context", ""]

        for entity in self.entities:
            label = entity.get("label", entity.get("id", "unknown"))
            etype = entity.get("type", "unknown")
            lines.append(f"### Entity: {label} ({etype})")

            # Filter relationships for this entity
            ent_rels = [
                r
                for r in self.relationships
                if r.get("source") == label or r.get("target") == label
            ]
            if ent_rels:
                lines.append(f"Relationships (confidence >= {min_confidence}):")
                for rel in ent_rels:
                    conf = rel.get("confidence", 1.0)
                    if conf >= min_confidence:
                        lines.append(
                            f"- {rel.get('source', '?')} -> {rel.get('relation', 'related_to')}"
                            f" -> {rel.get('target', '?')} ({conf:.2f})"
                        )
            lines.append("")

        if self.paths:
            lines.append("### Cross-entity paths")
            for path_info in self.paths:
                path_str = " -> ".join(
                    f"{step.get('entity', '?')}" for step in path_info.get("steps", [])
                )
                min_conf = path_info.get("min_confidence", 0.0)
                lines.append(f"- {path_str} (min confidence: {min_conf:.2f})")
            lines.append("")

        return "\n".join(lines)


class GraphContextBuilder:
    """Builds graph context for thinking sessions via entity extraction."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_client: LLMClient,
        fast_model: str,
    ) -> None:
        self._graph = knowledge_graph
        self._llm = llm_client
        self._fast_model = fast_model
        self._session_cache: dict[str, list[str]] = {}

    async def extract_entities(self, query: str, session_id: str) -> list[str]:
        """Extract named entities from query with session-scoped cache."""
        if session_id in self._session_cache:
            # Merge: extract new entities and add to existing cache
            cached = self._session_cache[session_id]
            new_entities = await self._llm_extract(query)
            merged = list(set(cached + new_entities))
            self._session_cache[session_id] = merged
            return merged

        entities = await self._llm_extract(query)
        self._session_cache[session_id] = entities
        return entities

    async def _llm_extract(self, query: str) -> list[str]:
        """Call LLM to extract entities from query text."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        try:
            response = await asyncio.to_thread(
                self._llm.complete,
                messages=[Message(role="user", content=prompt)],
                model=self._fast_model,
                max_tokens=256,
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            entities: list[str] = json.loads(raw)
            if not isinstance(entities, list):
                return []
            return [str(e) for e in entities if isinstance(e, str) and len(e) > 1]
        except (json.JSONDecodeError, LLMError, IndexError, KeyError) as e:
            logger.warning("Entity extraction failed: %s", e)
            return []

    def clear_session(self, session_id: str) -> None:
        """Remove cached entities for a session."""
        self._session_cache.pop(session_id, None)

    async def build(
        self,
        query: str,
        session_id: str,
        hop_depth: int = 2,
        min_confidence: float = 0.6,
        max_relationships: int = 20,
    ) -> GraphContextBlock | None:
        """Full pipeline: extract entities -> build subgraphs -> serialize.

        Returns None if graph is empty or no entities found.
        """
        graph_stats = self._graph.stats
        if graph_stats["nodes"] == 0:
            logger.warning("Graph has no entities; skipping graph context")
            return None

        entities = await self.extract_entities(query, session_id)
        if not entities:
            logger.debug("No entities extracted from query: %s", query[:100])
            return None

        block = GraphContextBlock()
        seen_rels: set[tuple[str, str, str]] = set()
        rel_count = 0

        for entity_name in entities:
            # Look up entity in graph
            entity_data = self._graph.get_entity(entity_name)
            if entity_data is None:
                continue

            block.entities.append(entity_data)

            # Get ego subgraph for relationships
            sub = self._graph.ego_subgraph(entity_name, depth=hop_depth)
            for src, tgt, data in sub.edges(data=True):
                if rel_count >= max_relationships:
                    break
                rel_key = (src, tgt, data.get("relation", "related_to"))
                if rel_key in seen_rels:
                    continue
                seen_rels.add(rel_key)

                conf = data.get("confidence", 1.0)
                if conf < min_confidence:
                    continue

                src_label = sub.nodes[src].get("label", src)
                tgt_label = sub.nodes[tgt].get("label", tgt)
                block.relationships.append(
                    {
                        "source": src_label,
                        "target": tgt_label,
                        "relation": data.get("relation", "related_to"),
                        "confidence": conf,
                    }
                )
                rel_count += 1

        # Find cross-entity paths
        entity_names_in_graph = [e.get("label", e.get("id", "")) for e in block.entities]
        if len(entity_names_in_graph) >= 2:
            for i in range(len(entity_names_in_graph)):
                for j in range(i + 1, len(entity_names_in_graph)):
                    paths = self._graph.shortest_paths(
                        entity_names_in_graph[i],
                        entity_names_in_graph[j],
                        max_length=3,
                        top_k=2,
                    )
                    for path in paths:
                        confidences = [
                            float(step.get("confidence", 1.0))
                            for step in path
                            if "confidence" in step
                        ]
                        block.paths.append(
                            {
                                "steps": path,
                                "min_confidence": min(confidences) if confidences else 0.0,
                            }
                        )

        if not block.entities:
            return None

        return block
