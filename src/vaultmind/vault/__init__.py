"""Vault operations — parsing, watching, and managing Obsidian vault files."""

from vaultmind.vault.models import Note, NoteChunk, NoteType
from vaultmind.vault.parser import VaultParser
from vaultmind.vault.security import PathTraversalError, validate_vault_path
from vaultmind.vault.watcher import VaultWatcher

__all__ = [
    "Note",
    "NoteChunk",
    "NoteType",
    "PathTraversalError",
    "VaultParser",
    "VaultWatcher",
    "validate_vault_path",
]
