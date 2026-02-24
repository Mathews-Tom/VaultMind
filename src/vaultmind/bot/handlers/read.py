"""Read handler — read a note by path or search query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path, _split_message
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
            f"Note not found: `{query}`\nProvide a path relative to vault root, or a search term.",
            parse_mode="Markdown",
        )
        return

    content = filepath.read_text(encoding="utf-8")
    rel_path = filepath.relative_to(ctx.vault_root)
    header = f"\U0001f4d6 **{rel_path}**\n\n"

    for chunk in _split_message(header + content, max_len=4000):
        await message.answer(chunk, parse_mode="Markdown")
