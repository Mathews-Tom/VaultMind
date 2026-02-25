"""Read handler — read a note by path or search query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.formatter import TelegramFormatter
from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path
from vaultmind.bot.sanitize import sanitize_path

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_read(ctx: HandlerContext, message: Message, query: str) -> None:
    """Read a note by path or search query."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_path(query)
    query = san.text
    if not query:
        await message.answer("Empty path after sanitization.")
        return

    filepath = _resolve_note_path(ctx, query)
    if filepath is None:
        await message.answer(
            f"Note not found: <code>{query}</code>\n"
            "Provide a path relative to vault root, or a search term.",
            parse_mode="HTML",
        )
        return

    note = ctx.parser.parse_file(filepath)
    formatted = TelegramFormatter.format_note(note)
    await message.answer(formatted, parse_mode="HTML")
