"""Context-aware note suggestions — composite scoring across three signals.

Finds notes that should be linked based on semantic similarity, shared graph
entities, and graph distance.  Operates in the 0.70–0.80 similarity band,
below the duplicate detector's merge threshold.

Composite score:
    score = similarity + entity_weight × shared_entities + graph_weight × (1 / path_length)

Where ``1 / path_length`` is 0 when no graph path exists (disconnected components),
making the formula degrade gracefully to similarity + entity overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaultmind.config import NoteSuggestionsConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.events import NoteCreatedEvent, NoteModifiedEvent
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)

# Similarity band for link suggestions (below duplicate/merge thresholds)
_SUGGESTION_MIN_DISTANCE = 0.20  # similarity < 0.80 (above = merge territory)
_SUGGESTION_MAX_DISTANCE = 0.30  # similarity ≥ 0.70


@dataclass(frozen=True, slots=True)
class NoteSuggestion:
    """A suggested link between two notes."""

    source_path: str
    source_title: str
    target_path: str
    target_title: str
    similarity: float
    shared_entities: list[str]
    graph_distance: int | None
    composite_score: float


class NoteSuggester:
    """Finds notes that should be linked using composite scoring.

    Three scoring signals:
    1. Vector similarity (ChromaDB cosine distance)
    2. Shared graph entities (entity overlap count)
    3. Graph path distance (shortest path length)
    """

    def __init__(
        self,
        config: NoteSuggestionsConfig,
        store: VaultStore,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._graph = graph

        # In-memory results buffer
        self._results: dict[str, list[NoteSuggestion]] = {}

    # ------------------------------------------------------------------
    # Core suggestion logic
    # ------------------------------------------------------------------

    def suggest_links(
        self,
        note: Note,
        *,
        max_results: int = 5,
    ) -> list[NoteSuggestion]:
        """Find notes that should be linked to *note*.

        Returns suggestions sorted by composite score (descending).
        """
        body = note.body_without_frontmatter().strip()
        if len(body) < self._config.min_content_length:
            return []

        # Query ChromaDB for candidates in the suggestion band + above
        # We fetch a wide range and filter locally to capture both
        # in-band (link suggestion) and near-band results
        raw_results = self._store.search(
            query=body[:2000],
            n_results=max_results + 20,
        )

        note_path = str(note.path)
        note_entities = set(note.entities)

        # Also extract entity names from graph nodes mentioned in content
        if self._graph is not None:
            note_entities |= self._entities_from_graph(note)

        seen_paths: set[str] = set()
        suggestions: list[NoteSuggestion] = []

        for hit in raw_results:
            meta = hit.get("metadata", {})
            hit_path = meta.get("note_path", "")

            if hit_path == note_path or hit_path in seen_paths:
                continue
            seen_paths.add(hit_path)

            distance = hit.get("distance", 1.0)

            # Skip above merge threshold (those are duplicate detector's domain)
            if distance < _SUGGESTION_MIN_DISTANCE:
                continue

            # Skip below suggestion threshold
            if distance > _SUGGESTION_MAX_DISTANCE:
                continue

            similarity = 1.0 - distance

            # Signal 2: shared entities
            hit_entities_raw = meta.get("entities", "")
            hit_entities = (
                set(hit_entities_raw.split(",")) if hit_entities_raw else set()
            )
            shared = note_entities & hit_entities
            shared_count = len(shared)

            # Signal 3: graph distance
            graph_dist = self._graph_distance(note.title, meta.get("note_title", ""))

            # Composite score
            entity_boost = self._config.entity_weight * shared_count
            graph_boost = (
                self._config.graph_weight * (1.0 / graph_dist)
                if graph_dist is not None and graph_dist > 0
                else 0.0
            )
            composite = similarity + entity_boost + graph_boost

            suggestions.append(
                NoteSuggestion(
                    source_path=note_path,
                    source_title=note.title,
                    target_path=hit_path,
                    target_title=meta.get("note_title", "Untitled"),
                    similarity=round(similarity, 4),
                    shared_entities=sorted(shared),
                    graph_distance=graph_dist,
                    composite_score=round(composite, 4),
                )
            )

        # Sort by composite score descending
        suggestions.sort(key=lambda s: s.composite_score, reverse=True)
        suggestions = suggestions[:max_results]

        if suggestions:
            self._results[note_path] = suggestions
            logger.info(
                "Suggestions for %s: %d candidates (top score: %.3f)",
                note.title,
                len(suggestions),
                suggestions[0].composite_score,
            )

        return suggestions

    # ------------------------------------------------------------------
    # Event bus integration
    # ------------------------------------------------------------------

    async def on_note_changed(
        self,
        event: NoteCreatedEvent | NoteModifiedEvent,
    ) -> None:
        """Event bus callback — fire-and-forget suggestion computation."""
        if not self._config.enabled:
            return

        note = event.note
        if note is None:
            return

        self.suggest_links(note)

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_results(self, note_path: str) -> list[NoteSuggestion]:
        """Retrieve cached suggestions for a note path."""
        return self._results.get(note_path, [])

    def get_all_results(self) -> dict[str, list[NoteSuggestion]]:
        """All cached results, keyed by source note path."""
        return dict(self._results)

    def clear_results(self, note_path: str | None = None) -> None:
        """Clear cached results."""
        if note_path is None:
            self._results.clear()
        else:
            self._results.pop(note_path, None)

    @property
    def result_count(self) -> int:
        """Number of notes with active suggestions."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Batch scan
    # ------------------------------------------------------------------

    def scan_vault(
        self,
        notes: list[Note],
    ) -> dict[str, list[NoteSuggestion]]:
        """Run suggestion computation across all provided notes.

        Returns a dict of note_path → suggestions for notes that
        have at least one link suggestion.
        """
        all_suggestions: dict[str, list[NoteSuggestion]] = {}

        for note in notes:
            results = self.suggest_links(note)
            if results:
                all_suggestions[str(note.path)] = results

        logger.info(
            "Vault scan: %d/%d notes have link suggestions",
            len(all_suggestions),
            len(notes),
        )
        return all_suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entities_from_graph(self, note: Note) -> set[str]:
        """Extract entity names that appear in the graph for this note."""
        if self._graph is None:
            return set()

        entities: set[str] = set()
        # Check frontmatter entities
        for entity in note.entities:
            if self._graph.get_entity(entity) is not None:
                entities.add(entity.lower())

        # Check wikilinks as potential entity references
        for link in note.wikilinks:
            if self._graph.get_entity(link) is not None:
                entities.add(link.lower())

        return entities

    def _graph_distance(
        self,
        source_title: str,
        target_title: str,
    ) -> int | None:
        """Shortest path length between two note titles in the graph.

        Returns None if no graph is configured or no path exists.
        """
        if self._graph is None:
            return None

        path = self._graph.find_path(source_title, target_title)
        if path is None:
            return None

        return len(path) - 1  # path includes both endpoints
