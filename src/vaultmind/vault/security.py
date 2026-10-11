"""Path traversal protection for vault operations."""

from __future__ import annotations

import errno
from typing import TYPE_CHECKING

from vaultmind.errors import VaultMindError

if TYPE_CHECKING:
    from pathlib import Path


class PathTraversalError(VaultMindError, ValueError):
    """Raised when a user-supplied path escapes the vault root."""

    def __init__(self, user_path: str, vault_root: Path) -> None:
        self.user_path = user_path
        self.vault_root = vault_root
        super().__init__(f"Path traversal blocked: '{user_path}' escapes vault root '{vault_root}'")


class PathResolutionError(VaultMindError, ValueError):
    """Raised when a caller path cannot resolve, without exposing physical targets."""

    def __init__(self, user_path: str) -> None:
        self.user_path = user_path
        super().__init__(f"Path cannot be resolved: '{user_path}'")


def validate_vault_path(user_path: str, vault_root: Path) -> Path:
    """Resolve a user-supplied path and verify it stays within the vault root.

    Returns the resolved absolute path if valid.
    Raises PathTraversalError for containment violations and PathResolutionError
    for unresolvable paths. Ordinary strict-resolution failures retain non-strict
    lookup semantics; symlink loops always fail closed.
    """
    try:
        resolved_root = vault_root.resolve()
        try:
            # Strict resolution detects loops on both Python 3.12 and 3.13.
            # Python 3.13's non-strict resolver can leave a loop unresolved.
            candidate = (resolved_root / user_path).resolve(strict=True)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise
            # Preserve non-strict lookup for missing paths, file children,
            # inaccessible components and overlong names before policy checks.
            candidate = (resolved_root / user_path).resolve()
    except RuntimeError:
        # Python 3.12 reports symlink loops as RuntimeError.
        raise PathResolutionError(user_path) from None
    except OSError:
        # A genuine non-strict failure must still fail loudly, but never carry
        # a symlink's unknown physical filename into a caller response or log.
        raise PathResolutionError(user_path) from None
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise PathTraversalError(user_path, vault_root) from None
    return candidate
