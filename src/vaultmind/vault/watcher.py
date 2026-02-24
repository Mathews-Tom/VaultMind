"""Vault file watcher — monitors for changes and triggers re-indexing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from collections.abc import Callable

from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from vaultmind.config import VaultConfig

logger = logging.getLogger(__name__)


class _VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for .md files in the vault."""

    def __init__(
        self,
        vault_root: Path,
        excluded_folders: list[str],
        on_change: Callable[[Path, str], None],
    ) -> None:
        self.vault_root = vault_root
        self.excluded = set(excluded_folders)
        self.on_change = on_change

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        if p.suffix != ".md":
            return False
        try:
            rel = p.relative_to(self.vault_root)
        except ValueError:
            return False
        return not any(part in self.excluded for part in rel.parts)

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_process(event.src_path):
            logger.info("Note created: %s", event.src_path)
            self.on_change(Path(event.src_path), "created")

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_process(event.src_path):
            logger.debug("Note modified: %s", event.src_path)
            self.on_change(Path(event.src_path), "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_process(event.src_path):
            logger.info("Note deleted: %s", event.src_path)
            self.on_change(Path(event.src_path), "deleted")


class VaultWatcher:
    """Watches the Obsidian vault for file changes.

    Usage:
        watcher = VaultWatcher(config, on_change=my_callback)
        watcher.start()  # non-blocking
        ...
        watcher.stop()
    """

    def __init__(
        self,
        config: VaultConfig,
        on_change: Callable[[Path, str], None],
    ) -> None:
        self.config = config
        self.handler = _VaultEventHandler(
            vault_root=config.path,
            excluded_folders=config.excluded_folders,
            on_change=on_change,
        )
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching the vault directory (non-blocking)."""
        self._observer = Observer()
        self._observer.schedule(self.handler, str(self.config.path), recursive=True)
        self._observer.start()
        logger.info("Watching vault at %s", self.config.path)

    def stop(self) -> None:
        """Stop the watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Vault watcher stopped")

    async def run_async(self) -> None:
        """Run watcher in async context — blocks until cancelled."""
        self.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.stop()
