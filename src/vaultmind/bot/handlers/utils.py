"""Shared utilities for handler modules."""

from __future__ import annotations

import errno
from typing import TYPE_CHECKING

from vaultmind.vault.security import PathResolutionError, validate_vault_path

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


def _is_authorized(ctx: HandlerContext, message: Message) -> bool:
    """Check if the user is allowed to use the bot."""
    allowed = ctx.settings.telegram.allowed_user_ids
    if not allowed:
        return True
    return message.from_user is not None and message.from_user.id in allowed


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split a long message into Telegram-compatible chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find a good split point
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    return chunks


def _resolve_candidate(path: str, vault_root: Path) -> Path | None:
    """Return an existing in-vault file for ``path``, or None to try the next strategy.

    Unresolvable candidates (symlink loops, uninspectable components) fall
    through; containment violations still propagate as PathTraversalError.
    """
    try:
        candidate = validate_vault_path(path, vault_root)
    except PathResolutionError:
        return None
    try:
        return candidate if candidate.is_file() else None
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.ENAMETOOLONG):
            return None
        raise


def _resolve_note_path(ctx: HandlerContext, query: str) -> Path | None:
    """Resolve a note query to an absolute filepath.

    Tries in order:
    1. Exact relative path match
    2. Relative path with .md extension appended
    3. Filename search across vault
    4. Semantic search (first result)
    """
    # Exact path
    candidate = _resolve_candidate(query, ctx.vault_root)
    if candidate is not None:
        return candidate

    # With .md
    if not query.endswith(".md"):
        candidate = _resolve_candidate(query + ".md", ctx.vault_root)
        if candidate is not None:
            return candidate

    # Filename search
    query_lower = query.lower().strip()
    for md_file in ctx.vault_root.rglob("*.md"):
        if query_lower not in md_file.stem.lower():
            continue
        candidate = _resolve_candidate(
            md_file.relative_to(ctx.vault_root).as_posix(), ctx.vault_root
        )
        if candidate is not None:
            return candidate

    # Semantic search fallback
    results = ctx.store.search(query, n_results=1)
    if results:
        note_path: str = results[0]["metadata"].get("note_path", "")
        if note_path:
            candidate = _resolve_candidate(note_path, ctx.vault_root)
            if candidate is not None:
                return candidate

    return None
