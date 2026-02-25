"""Suggestions handler — find and display link suggestions for a note."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import (
    _is_authorized,
    _resolve_note_path,
    _split_message,
)
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.indexer.note_suggester import NoteSuggester

logger = logging.getLogger(__name__)


async def handle_suggestions(
    ctx: HandlerContext,
    message: Message,
    query: str,
    suggester: NoteSuggester,
) -> None:
    """Find and display link suggestions for a note."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(
        query, max_length=MAX_QUERY_LENGTH, operation="suggestions"
    )
    query = san.text
    if not query:
        await message.answer(
            "Usage: `/suggest <note path or search term>`",
            parse_mode="Markdown",
        )
        return

    resolved = _resolve_note_path(ctx, query)
    if resolved is None:
        await message.answer(
            f"Note not found: `{query}`", parse_mode="Markdown"
        )
        return

    try:
        note = ctx.parser.parse_file(resolved)
    except Exception:
        logger.exception("Failed to parse %s", resolved)
        await message.answer("Failed to read note.")
        return

    suggestions = suggester.suggest_links(note)

    if not suggestions:
        await message.answer(
            f"No link suggestions for `{note.title}`.",
            parse_mode="Markdown",
        )
        return

    lines = [f"\U0001f517 **Link suggestions:** _{note.title}_\n"]

    for s in suggestions:
        parts = [f"\u2022 `{s.target_path}` — score {s.composite_score:.2f}"]
        details: list[str] = []
        details.append(f"similarity {s.similarity:.0%}")
        if s.shared_entities:
            details.append(
                f"shared: {', '.join(s.shared_entities[:3])}"
            )
        if s.graph_distance is not None:
            details.append(f"graph dist: {s.graph_distance}")
        parts.append(f"  _({', '.join(details)})_")
        lines.append("\n".join(parts))

    text = "\n".join(lines)
    for chunk in _split_message(text):
        await message.answer(chunk, parse_mode="Markdown")
