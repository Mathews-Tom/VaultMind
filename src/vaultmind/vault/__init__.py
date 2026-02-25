"""Vault operations — parsing, watching, and managing Obsidian vault files."""

from vaultmind.vault.events import (
    NoteCreatedEvent,
    NoteDeletedEvent,
    NoteModifiedEvent,
    VaultEventBus,
)
from vaultmind.vault.models import Note, NoteChunk, NoteType
from vaultmind.vault.parser import VaultParser
from vaultmind.vault.security import PathTraversalError, validate_vault_path
from vaultmind.vault.watch_handler import IncrementalWatchHandler
from vaultmind.vault.watcher import VaultWatcher

__all__ = [
    "IncrementalWatchHandler",
    "Note",
    "NoteChunk",
    "NoteCreatedEvent",
    "NoteDeletedEvent",
    "NoteModifiedEvent",
    "NoteType",
    "PathTraversalError",
    "VaultEventBus",
    "VaultParser",
    "VaultWatcher",
    "validate_vault_path",
]
