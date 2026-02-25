"""Duplicates handler — find and display semantic duplicates for a note."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path, _split_message
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text
from vaultmind.indexer.duplicate_detector import MatchType

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.indexer.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)


async def handle_duplicates(
    ctx: HandlerContext,
    message: Message,
    query: str,
    detector: DuplicateDetector,
) -> None:
    """Find and display duplicate/merge candidates for a note."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(query, max_length=MAX_QUERY_LENGTH, operation="duplicates")
    query = san.text
    if not query:
        await message.answer(
            "Usage: `/duplicates <note path or search term>`",
            parse_mode="Markdown",
        )
        return

    resolved = _resolve_note_path(ctx, query)
    if resolved is None:
        await message.answer(f"Note not found: `{query}`", parse_mode="Markdown")
        return

    try:
        note = ctx.parser.parse_file(resolved)
    except Exception:
        logger.exception("Failed to parse %s", resolved)
        await message.answer("Failed to read note.")
        return

    matches = detector.find_duplicates(note)

    if not matches:
        await message.answer(
            f"No duplicates or merge candidates for `{note.title}`.",
            parse_mode="Markdown",
        )
        return

    lines = [f"\U0001f50d **Duplicate scan:** _{note.title}_\n"]

    duplicates = [m for m in matches if m.match_type == MatchType.DUPLICATE]
    merges = [m for m in matches if m.match_type == MatchType.MERGE]

    if duplicates:
        lines.append("**\u26a0\ufe0f Duplicates** (similarity \u2265 92%):")
        for m in duplicates:
            lines.append(f"  \u2022 `{m.match_path}` — {m.similarity:.0%}")

    if merges:
        lines.append("\n**\U0001f504 Merge candidates** (80–92% similar):")
        for m in merges:
            lines.append(f"  \u2022 `{m.match_path}` — {m.similarity:.0%}")

    text = "\n".join(lines)
    for chunk in _split_message(text):
        await message.answer(chunk, parse_mode="Markdown")
