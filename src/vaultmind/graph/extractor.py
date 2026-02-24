"""Entity extraction — uses LLM to extract structured entities and relationships from notes."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from vaultmind.graph.knowledge_graph import EDGE_TYPES, NODE_TYPES, KnowledgeGraph
from vaultmind.llm.client import LLMError, Message

if TYPE_CHECKING:
    from vaultmind.config import GraphConfig
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extract entities and relationships from the following note.

Rules:
- Entity types: {node_types}
- Relationship types: {edge_types}
- Only extract entities that are meaningful and specific (not generic words)
- Assign a confidence score (0.0-1.0) to each extraction
- If the note has explicit `entities` in frontmatter, include those with confidence 1.0
- Prefer specific entity types over "concept" when possible

Respond with ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "project", "confidence": 0.9}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "relation": "depends_on", "confidence": 0.8}}
  ]
}}

Note title: {title}
Note type: {note_type}
Frontmatter entities: {frontmatter_entities}
Tags: {tags}

Content:
{content}
"""


class EntityExtractor:
    """Extracts entities and relationships from notes using an LLM."""

    def __init__(
        self,
        config: GraphConfig,
        llm_client: LLMClient,
        model: str,
    ) -> None:
        self.config = config
        self._client = llm_client
        self._model = model

    def extract_from_note(self, note: Note) -> dict[str, Any]:
        """Extract entities and relationships from a single note.

        Returns:
            Dict with 'entities' and 'relationships' lists.
        """
        content = note.body_without_frontmatter()
        if len(content.strip()) < 50:
            return {"entities": [], "relationships": []}

        # Truncate very long notes to avoid token waste
        if len(content) > 8000:
            content = content[:8000] + "\n\n[truncated]"

        prompt = EXTRACTION_PROMPT.format(
            node_types=", ".join(sorted(NODE_TYPES)),
            edge_types=", ".join(sorted(EDGE_TYPES)),
            title=note.title,
            note_type=note.note_type.value,
            frontmatter_entities=(", ".join(note.entities) if note.entities else "none"),
            tags=", ".join(note.tags) if note.tags else "none",
            content=content,
        )

        try:
            response = self._client.complete(
                messages=[Message(role="user", content=prompt)],
                model=self._model,
                max_tokens=2048,
            )

            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            result: dict[str, Any] = json.loads(raw)
            logger.debug(
                "Extracted %d entities, %d relationships from %s",
                len(result.get("entities", [])),
                len(result.get("relationships", [])),
                note.title,
            )
            return result

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(
                "Failed to parse extraction result for %s: %s",
                note.title,
                e,
            )
            return {"entities": [], "relationships": []}
        except LLMError as e:
            logger.error(
                "LLM error during extraction for %s (%s): %s",
                note.title,
                e.provider,
                e,
            )
            return {"entities": [], "relationships": []}

    def extract_and_update_graph(
        self,
        notes: list[Note],
        graph: KnowledgeGraph,
    ) -> dict[str, int]:
        """Extract entities from multiple notes and update the knowledge graph.

        Returns:
            Stats dict with counts of entities and relationships added.
        """
        total_entities = 0
        total_relationships = 0

        for note in notes:
            result = self.extract_from_note(note)

            for entity in result.get("entities", []):
                confidence = entity.get("confidence", 0.5)
                if confidence < self.config.min_confidence:
                    continue

                graph.add_entity(
                    name=entity["name"],
                    entity_type=entity.get("type", "concept"),
                    source_note=str(note.path),
                    confidence=confidence,
                )
                total_entities += 1

            for rel in result.get("relationships", []):
                confidence = rel.get("confidence", 0.5)
                if confidence < self.config.min_confidence:
                    continue

                graph.add_relationship(
                    source=rel["source"],
                    target=rel["target"],
                    relation=rel.get("relation", "related_to"),
                    source_note=str(note.path),
                    confidence=confidence,
                )
                total_relationships += 1

            # Also add wikilinks as relationships
            for link in note.wikilinks:
                graph.add_relationship(
                    source=note.title,
                    target=link,
                    relation="related_to",
                    source_note=str(note.path),
                    confidence=1.0,
                )
                total_relationships += 1

        graph.save()
        logger.info(
            "Graph updated: +%d entities, +%d relationships",
            total_entities,
            total_relationships,
        )
        return {
            "entities_added": total_entities,
            "relationships_added": total_relationships,
        }
