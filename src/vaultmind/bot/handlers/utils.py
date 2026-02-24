"""Shared utilities for handler modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.vault.security import validate_vault_path

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


def _resolve_note_path(ctx: HandlerContext, query: str) -> Path | None:
    """Resolve a note query to an absolute filepath.

    Tries in order:
    1. Exact relative path match
    2. Relative path with .md extension appended
    3. Filename search across vault
    4. Semantic search (first result)
    """
    # Exact path
    candidate = validate_vault_path(query, ctx.vault_root)
    if candidate.is_file():
        return candidate

    # With .md
    if not query.endswith(".md"):
        candidate = validate_vault_path(query + ".md", ctx.vault_root)
        if candidate.is_file():
            return candidate

    # Filename search
    query_lower = query.lower().strip()
    for md_file in ctx.vault_root.rglob("*.md"):
        if md_file.stem.lower() == query_lower:
            return md_file
        if query_lower in md_file.stem.lower():
            return md_file

    # Semantic search fallback
    results = ctx.store.search(query, n_results=1)
    if results:
        note_path: str = results[0]["metadata"].get("note_path", "")
        if note_path:
            candidate = validate_vault_path(note_path, ctx.vault_root)
            if candidate.is_file():
                return candidate

    return None
