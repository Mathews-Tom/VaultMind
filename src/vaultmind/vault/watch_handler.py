"""Incremental watch handler — debounced, hash-stable change processing.

Bridges the raw watchdog events from ``VaultWatcher`` to the indexing and
graph pipelines.  Key properties:

* **Debounce** — Coalesces rapid saves (Obsidian fires multiple events per
  save) into a single processing pass after ``debounce_ms`` of silence.
* **Hash stability** — After the debounce fires, the handler hashes the
  file content.  If the hash differs from the *last indexed* hash, it
  re-hashes after a second ``debounce_ms`` window.  Only when two
  consecutive hashes match (the file is no longer being written) does the
  handler proceed.  This eliminates the partial-write class of bugs.
* **Structured events** — Processing results are published to a
  ``VaultEventBus`` so downstream consumers (duplicate detection, note
  suggestions) can subscribe independently.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from vaultmind.vault.events import (
    NoteCreatedEvent,
    NoteDeletedEvent,
    NoteModifiedEvent,
    VaultEventBus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.config import WatchConfig
    from vaultmind.graph import EntityExtractor, KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)


def _file_content_hash(path: Path) -> str:
    """SHA-256 of file content, first 16 hex chars."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class IncrementalWatchHandler:
    """Processes vault file changes with debounce and hash-stability checks.

    Instantiate once per watch session.  Call :meth:`handle_change` from the
    ``VaultWatcher`` callback (it is sync-safe — schedules async work on the
    running loop).

    Parameters
    ----------
    config:
        Watch-specific settings (debounce, graph re-extraction toggle).
    parser:
        Vault parser for reading individual notes.
    store:
        ChromaDB vector store for incremental indexing.
    event_bus:
        Async event bus for publishing structured change events.
    graph:
        Knowledge graph instance (optional — required only when
        ``config.reextract_graph`` is True).
    extractor:
        Entity extractor (optional — same condition as *graph*).
    """

    def __init__(
        self,
        config: WatchConfig,
        parser: VaultParser,
        store: VaultStore,
        event_bus: VaultEventBus,
        graph: KnowledgeGraph | None = None,
        extractor: EntityExtractor | None = None,
    ) -> None:
        self._config = config
        self._parser = parser
        self._store = store
        self._event_bus = event_bus
        self._graph = graph
        self._extractor = extractor

        # Debounce state: path → scheduled asyncio.TimerHandle
        self._pending: dict[Path, asyncio.TimerHandle] = {}

        # Last-indexed content hash per note path (for stability check)
        self._hash_cache: dict[Path, str] = {}

        # Pending graph re-extraction paths (batched)
        self._graph_pending: set[Path] = set()
        self._graph_timer: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_change(self, path: Path, event_type: str) -> None:
        """Entry point called by ``VaultWatcher`` (sync context).

        Schedules debounced async processing on the running event loop.
        """
        loop = asyncio.get_event_loop()

        # Cancel any pending debounce for this path
        if path in self._pending:
            self._pending[path].cancel()

        debounce_s = self._config.debounce_ms / 1000.0

        if event_type == "deleted":

            def _delete_cb(p: Path = path) -> None:
                loop.create_task(self._process_delete(p))

            handle = loop.call_later(debounce_s, _delete_cb)
        else:

            def _change_cb(p: Path = path, et: str = event_type) -> None:
                loop.create_task(self._process_change(p, et))

            handle = loop.call_later(debounce_s, _change_cb)

        self._pending[path] = handle

    @property
    def pending_count(self) -> int:
        """Number of paths awaiting debounce resolution."""
        return len(self._pending)

    @property
    def indexed_hashes(self) -> dict[Path, str]:
        """Read-only view of tracked content hashes."""
        return dict(self._hash_cache)

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    async def _process_change(self, path: Path, event_type: str) -> None:
        """Debounce-triggered handler for created/modified events."""
        self._pending.pop(path, None)

        if not path.exists():
            logger.debug("File vanished before processing: %s", path)
            return

        # Stage 1: compute hash
        try:
            current_hash = _file_content_hash(path)
        except OSError:
            logger.warning("Cannot read %s — skipping", path)
            return

        # Hash stability check: if enabled, verify the hash is stable
        # across two consecutive reads separated by debounce_ms
        if self._config.hash_stability_check:
            cached = self._hash_cache.get(path)
            if cached != current_hash:
                # First observation of this hash — schedule re-check
                self._hash_cache[path] = current_hash
                await asyncio.sleep(self._config.debounce_ms / 1000.0)

                if not path.exists():
                    return

                try:
                    recheck_hash = _file_content_hash(path)
                except OSError:
                    logger.warning("Cannot re-read %s — skipping", path)
                    return

                if recheck_hash != current_hash:
                    # File still changing — re-debounce
                    logger.debug("Hash unstable for %s — re-debouncing", path)
                    self.handle_change(path, event_type)
                    return
        else:
            # Without stability check, still skip if content unchanged
            cached = self._hash_cache.get(path)
            if cached == current_hash:
                logger.debug("Content unchanged for %s — skipping", path)
                return

        # Hash is stable and differs from last indexed — proceed
        self._hash_cache[path] = current_hash

        try:
            note = await asyncio.to_thread(self._parser.parse_file, path)
        except Exception:
            logger.exception("Failed to parse %s", path)
            return

        try:
            chunks = await asyncio.to_thread(self._store.index_single_note, note, self._parser)
        except Exception:
            logger.exception("Failed to index %s", path)
            return

        logger.info(
            "Watch: %s %s (%d chunks)",
            event_type,
            path.name,
            chunks,
        )

        # Publish typed event
        event_cls = NoteCreatedEvent if event_type == "created" else NoteModifiedEvent
        await self._event_bus.publish(event_cls(path=path, note=note, chunks_indexed=chunks))

        # Queue graph re-extraction (batched)
        if self._config.reextract_graph and self._extractor and self._graph:
            self._graph_pending.add(path)
            self._schedule_graph_batch()

    async def _process_delete(self, path: Path) -> None:
        """Debounce-triggered handler for deleted events."""
        self._pending.pop(path, None)
        self._hash_cache.pop(path, None)

        # Derive the note_path string the store uses
        # VaultParser stores paths relative to vault root
        try:
            rel_path = str(path.relative_to(self._parser.vault_root))
        except ValueError:
            rel_path = str(path)

        try:
            await asyncio.to_thread(self._store.delete_note, rel_path)
        except Exception:
            logger.exception("Failed to delete index for %s", path)
            return

        logger.info("Watch: deleted %s", path.name)
        await self._event_bus.publish(NoteDeletedEvent(path=path))

    # ------------------------------------------------------------------
    # Batched graph re-extraction
    # ------------------------------------------------------------------

    def _schedule_graph_batch(self) -> None:
        """Schedule a batched graph re-extraction after the configured interval."""
        if self._graph_timer is not None:
            return  # already scheduled

        loop = asyncio.get_event_loop()
        self._graph_timer = loop.call_later(
            self._config.batch_graph_interval_seconds,
            lambda: loop.create_task(self._run_graph_batch()),
        )

    async def _run_graph_batch(self) -> None:
        """Re-extract entities for all queued notes in a single batch."""
        self._graph_timer = None
        paths = self._graph_pending.copy()
        self._graph_pending.clear()

        if not paths or not self._extractor or not self._graph:
            return

        notes = []
        for p in paths:
            if p.exists():
                try:
                    notes.append(self._parser.parse_file(p))
                except Exception:
                    logger.exception("Graph batch: failed to parse %s", p)

        if not notes:
            return

        logger.info("Graph batch: re-extracting %d notes", len(notes))
        try:
            stats = await asyncio.to_thread(
                self._extractor.extract_and_update_graph, notes, self._graph
            )
            logger.info(
                "Graph batch: +%d entities, +%d relationships",
                stats["entities_added"],
                stats["relationships_added"],
            )
        except Exception:
            logger.exception("Graph batch extraction failed")
