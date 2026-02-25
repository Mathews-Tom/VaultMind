"""Semantic duplicate detection — finds near-duplicate and merge-candidate notes.

Uses existing ChromaDB embeddings (zero additional embedding cost) to identify
notes with high cosine similarity.  Results are classified into bands:

* **duplicate** (similarity ≥ 0.92) — near-identical content, likely accidental
* **merge**     (0.80 ≤ similarity < 0.92) — similar enough to consider merging

The detector exposes two integration surfaces:

1. **Event-driven** — subscribe to ``NoteCreatedEvent`` / ``NoteModifiedEvent``
   via the ``VaultEventBus`` for automatic, fire-and-forget detection on every
   index update.
2. **On-demand** — call ``find_duplicates()`` directly from a Telegram command,
   MCP tool, or CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaultmind.config import DuplicateDetectionConfig
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.events import NoteCreatedEvent, NoteModifiedEvent
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity bands
# ---------------------------------------------------------------------------

# ChromaDB cosine distance → similarity = 1 - distance
_DUPLICATE_MAX_DISTANCE = 0.08  # similarity ≥ 0.92
_MERGE_MAX_DISTANCE = 0.20  # similarity ≥ 0.80


class MatchType(StrEnum):
    """Classification of a similarity match."""

    DUPLICATE = "duplicate"
    MERGE = "merge"


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """A single duplicate/merge candidate for a source note."""

    source_path: str
    source_title: str
    match_path: str
    match_title: str
    similarity: float
    match_type: MatchType
    shared_heading: str = ""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DuplicateDetector:
    """Finds semantically similar notes using ChromaDB vector distances.

    Reuses the existing embedding index — no additional API calls.
    """

    def __init__(
        self,
        config: DuplicateDetectionConfig,
        store: VaultStore,
    ) -> None:
        self._config = config
        self._store = store

        # In-memory results buffer — most recent per source path
        self._results: dict[str, list[DuplicateMatch]] = {}

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def find_duplicates(
        self,
        note: Note,
        *,
        max_results: int = 10,
    ) -> list[DuplicateMatch]:
        """Find duplicate/merge candidates for a single note.

        Queries ChromaDB with the note's body text and filters out
        self-matches (same note_path). Returns matches sorted by
        descending similarity.
        """
        body = note.body_without_frontmatter().strip()
        if len(body) < self._config.min_content_length:
            return []

        # Query more results than needed to account for self-chunks
        raw_results = self._store.search(
            query=body[:2000],  # cap query length for embedding
            n_results=max_results + 10,
        )

        note_path = str(note.path)
        seen_paths: set[str] = set()
        matches: list[DuplicateMatch] = []

        for hit in raw_results:
            meta = hit.get("metadata", {})
            hit_path = meta.get("note_path", "")

            # Skip self-matches and already-seen note paths
            if hit_path == note_path or hit_path in seen_paths:
                continue
            seen_paths.add(hit_path)

            distance = hit.get("distance", 1.0)
            similarity = 1.0 - distance

            if distance <= _DUPLICATE_MAX_DISTANCE:
                match_type = MatchType.DUPLICATE
            elif distance <= _MERGE_MAX_DISTANCE:
                match_type = MatchType.MERGE
            else:
                continue  # below merge threshold

            matches.append(
                DuplicateMatch(
                    source_path=note_path,
                    source_title=note.title,
                    match_path=hit_path,
                    match_title=meta.get("note_title", "Untitled"),
                    similarity=round(similarity, 4),
                    match_type=match_type,
                    shared_heading=meta.get("heading", ""),
                )
            )

            if len(matches) >= max_results:
                break

        # Store results for later retrieval
        if matches:
            self._results[note_path] = matches
            logger.info(
                "Duplicates for %s: %d duplicate, %d merge",
                note.title,
                sum(1 for m in matches if m.match_type == MatchType.DUPLICATE),
                sum(1 for m in matches if m.match_type == MatchType.MERGE),
            )

        return matches

    # ------------------------------------------------------------------
    # Event bus integration
    # ------------------------------------------------------------------

    async def on_note_changed(
        self,
        event: NoteCreatedEvent | NoteModifiedEvent,
    ) -> None:
        """Event bus callback — fire-and-forget duplicate check."""
        if not self._config.enabled:
            return

        note = event.note
        if note is None:
            return

        self.find_duplicates(note)

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_results(self, note_path: str) -> list[DuplicateMatch]:
        """Retrieve cached detection results for a note path."""
        return self._results.get(note_path, [])

    def get_all_results(self) -> dict[str, list[DuplicateMatch]]:
        """All cached results, keyed by source note path."""
        return dict(self._results)

    def clear_results(self, note_path: str | None = None) -> None:
        """Clear cached results. If *note_path* is None, clear all."""
        if note_path is None:
            self._results.clear()
        else:
            self._results.pop(note_path, None)

    @property
    def result_count(self) -> int:
        """Number of notes with active duplicate/merge results."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Batch scan
    # ------------------------------------------------------------------

    def scan_vault(self, notes: list[Note]) -> dict[str, list[DuplicateMatch]]:
        """Run duplicate detection across all provided notes.

        Returns a dict of note_path → matches for notes that have
        at least one duplicate or merge candidate.
        """
        all_matches: dict[str, list[DuplicateMatch]] = {}

        for note in notes:
            matches = self.find_duplicates(note)
            if matches:
                all_matches[str(note.path)] = matches

        logger.info(
            "Vault scan: %d/%d notes have duplicates or merge candidates",
            len(all_matches),
            len(notes),
        )
        return all_matches
