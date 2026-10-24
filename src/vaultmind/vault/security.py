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


class ReservedNamespaceError(VaultMindError, ValueError):
    """An ordinary writer attempted to enter a wiki-owned namespace."""

    def __init__(self) -> None:
        # Never include a requested filename, resolved alias target, or note body.
        super().__init__(
            "Ordinary writes cannot target wiki or _wiki; use the wiki intake/approval workflow"
        )


def _case_relative_parts(path: Path, root: Path) -> tuple[str, ...] | None:
    """Compare conservatively on case-sensitive and macOS case-insensitive volumes."""
    parts = tuple(part.casefold() for part in path.parts)
    root_parts = tuple(part.casefold() for part in root.parts)
    if parts[: len(root_parts)] == root_parts:
        return parts[len(root_parts) :]
    return None


def _reject_reserved_path(requested: Path, resolved: Path, vault_root: Path) -> None:
    root = _resolve_output_path(vault_root)
    # Check spelling before resolution: wiki/../ordinary and wiki symlinks that
    # point outside the vault must not launder a reserved request into an ordinary one.
    for lexical_root in (vault_root.absolute(), root):
        parts = _case_relative_parts(requested, lexical_root)
        if parts and parts[0] in {"wiki", "_wiki"}:
            raise ReservedNamespaceError()
    # The selected vault may itself have been configured through another alias.
    # Preserve the root-level lexical rule even for alias/wiki/../ordinary.
    for parent in requested.parents:
        child = requested.relative_to(parent).parts[0]
        if (
            child.casefold() in {"wiki", "_wiki"}
            and _case_relative_parts(_resolve_output_path(parent), root) == ()
        ):
            raise ReservedNamespaceError()
    parts = _case_relative_parts(resolved, root)
    if parts and parts[0] in {"wiki", "_wiki"}:
        raise ReservedNamespaceError()
    for name in ("wiki", "_wiki"):
        reserved = _resolve_output_path(root / name)
        if _case_relative_parts(resolved, reserved) is not None:
            raise ReservedNamespaceError()


def _resolve_output_path(path: Path) -> Path:
    """Retain validate_vault_path's missing-path semantics and fail closed on loops."""
    try:
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise
            return path.resolve()
    except (RuntimeError, OSError):
        raise PathResolutionError("output destination") from None


def validate_ordinary_write_path(user_path: str, vault_root: Path) -> Path:
    """Return the checked absolute path for an ordinary vault mutation.

    This covers every file type and directories, not only Markdown. Genuine
    wiki/intake writers keep using validate_vault_path under their own policy.
    Callers must use the returned path before reading for mutation or side effects.
    """
    requested = vault_root.absolute() / user_path
    # Lexical policy precedes resolution/containment, including escaping aliases.
    _reject_reserved_path(requested, requested, vault_root)
    candidate = validate_vault_path(user_path, vault_root)
    _reject_reserved_path(requested, candidate, vault_root)
    return candidate


def validate_ordinary_output_path(output_path: Path, vault_root: Path) -> Path:
    """Check an external/configured output without requiring vault containment.

    Relative state/export paths retain their cwd-relative meaning. External
    destinations are allowed unless their spelling or symlink target overlaps
    wiki-owned storage in the selected vault. This is a preflight, not a sandbox.
    """
    requested = output_path.expanduser().absolute()
    _reject_reserved_path(requested, requested, vault_root)
    candidate = _resolve_output_path(requested)
    _reject_reserved_path(requested, candidate, vault_root)
    return candidate
