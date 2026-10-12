"""Incremental watch handler — debounced, hash-stable change processing.

Bridges the raw watchdog events from ``VaultWatcher`` to the indexing and
graph pipelines.  Key properties:

* **Debounce** — Coalesces rapid saves (Obsidian fires multiple events per
  save) into a single processing pass after ``debounce_ms`` of silence.
* **Hash stability** — After the debounce fires, the handler hashes the
  file content and skips it when the hash equals the last *successfully
  indexed* hash.  Otherwise, when ``hash_stability_check`` is on, it
  re-hashes after a second ``debounce_ms`` window and only proceeds when
  two consecutive hashes match (the file is no longer being written).  This
  eliminates the partial-write class of bugs.
* **Truthful state** — The tracked hash advances, and a created/modified
  event is published, only after the note was parsed *and* indexed.  A
  failure is logged, leaves the last-good hash untouched and publishes
  nothing, so the next event for the same bytes is retried.
* **Retirement** — A deleted event, or a change event whose file has
  vanished, retires the note from the vector and keyword indexes; the
  deleted event is published only once retirement succeeded.
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

        # Last successfully indexed content hash per note path
        self._hash_cache: dict[Path, str] = {}

        # Paths whose last index attempt failed: the index may no longer match the
        # cached hash, so the next event must re-index even when the bytes equal it.
        self._unreconciled: set[Path] = set()

        # One lock per note path: overlapping events for the same note (watchdog
        # often reports one save several times) are processed one after another,
        # so the second sees the first one's tracked hash instead of re-indexing.
        self._path_locks: dict[Path, asyncio.Lock] = {}

        # Pending graph re-extraction paths (batched)
        self._graph_pending: set[Path] = set()
        self._graph_timer: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_change(self, path: Path, event_type: str) -> None:
        """Entry point called by ``VaultWatcher`` (sync context).

        Schedules debounced async processing on the running event loop; it must
        be invoked on that loop's thread (``VaultWatcher.start`` marshals the
        watchdog observer thread onto the loop it was started from).
        """
        loop = asyncio.get_running_loop()

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
        async with self._path_lock(path):
            await self._index_current(path, event_type)

    def _path_lock(self, path: Path) -> asyncio.Lock:
        lock = self._path_locks.get(path)
        if lock is None:
            lock = self._path_locks[path] = asyncio.Lock()
        return lock

    async def _index_current(self, path: Path, event_type: str) -> None:
        """Index the file's current content, advancing tracked state only on success."""
        try:
            current_hash = _file_content_hash(path)
        except FileNotFoundError:
            logger.info("File vanished before processing: %s", path)
            await self._retire(path)
            return
        except OSError:
            logger.exception("Cannot read %s; index and tracked hash left unchanged", path)
            return

        # Only content already indexed successfully is skipped; a path whose last
        # attempt failed is re-indexed even if its bytes match the last-good hash.
        if current_hash == self._hash_cache.get(path) and path not in self._unreconciled:
            logger.debug("Content unchanged for %s — skipping", path)
            return

        # Hash stability check: if enabled, verify the hash is stable
        # across two consecutive reads separated by debounce_ms
        if self._config.hash_stability_check:
            await asyncio.sleep(self._config.debounce_ms / 1000.0)

            try:
                recheck_hash = _file_content_hash(path)
            except FileNotFoundError:
                logger.info("File vanished before processing: %s", path)
                await self._retire(path)
                return
            except OSError:
                logger.exception("Cannot re-read %s; index and tracked hash left unchanged", path)
                return

            if recheck_hash != current_hash:
                # File still changing — re-debounce
                logger.debug("Hash unstable for %s — re-debouncing", path)
                self.handle_change(path, event_type)
                return

        try:
            note = await asyncio.to_thread(self._parser.parse_file, path)
        except FileNotFoundError:
            logger.info("File vanished before processing: %s", path)
            await self._retire(path)
            return
        except Exception:
            logger.exception(
                "Failed to parse %s; last-good index content kept, next event retries", path
            )
            return

        try:
            chunks = await asyncio.to_thread(self._store.index_single_note, note, self._parser)
        except Exception:
            self._unreconciled.add(path)
            logger.exception(
                "Failed to index %s; last-good hash kept, next event retries this content", path
            )
            return

        # Parse and index both succeeded: only now does the tracked state advance.
        self._hash_cache[path] = current_hash
        self._unreconciled.discard(path)

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
        async with self._path_lock(path):
            await self._retire(path)

    async def _retire(self, path: Path) -> None:
        """Retire a deleted or vanished note from the indexes.

        The deleted event is published only after the store confirmed retirement.
        The tracked hash is dropped either way: after a failed retirement the
        index state is uncertain, so a re-created file must be indexed afresh.
        """
        self._hash_cache.pop(path, None)
        self._unreconciled.discard(path)

        # Derive the note_path string the store uses
        # VaultParser stores paths relative to vault root
        try:
            rel_path = str(path.relative_to(self._parser.vault_root))
        except ValueError:
            logger.warning(
                "%s is outside vault root %s; retiring it by absolute path",
                path,
                self._parser.vault_root,
            )
            rel_path = str(path)

        try:
            await asyncio.to_thread(self._store.delete_note, rel_path)
        except Exception:
            logger.exception(
                "Failed to retire %s from the index; stale rows may remain until reindex", path
            )
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
