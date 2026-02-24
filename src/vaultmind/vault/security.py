"""Path traversal protection for vault operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a user-supplied path escapes the vault root."""

    def __init__(self, user_path: str, vault_root: Path) -> None:
        self.user_path = user_path
        self.vault_root = vault_root
        super().__init__(
            f"Path traversal blocked: '{user_path}' escapes vault root '{vault_root}'"
        )


def validate_vault_path(user_path: str, vault_root: Path) -> Path:
    """Resolve a user-supplied path and verify it stays within the vault root.

    Returns the resolved absolute path if valid.
    Raises PathTraversalError if the resolved path escapes vault_root.
    """
    resolved_root = vault_root.resolve()
    candidate = (resolved_root / user_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise PathTraversalError(user_path, vault_root) from None
    return candidate
