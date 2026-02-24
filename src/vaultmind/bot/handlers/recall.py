"""Recall handler — semantic search over the vault."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_recall(ctx: HandlerContext, message: Message, query: str) -> None:
    """Semantic search over the vault."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(query, max_length=MAX_QUERY_LENGTH, operation="recall")
    query = san.text
    if not query:
        await message.answer("Empty query after sanitization.")
        return

    await message.answer("\U0001f50d Searching vault...")

    results = ctx.store.search(query, n_results=5)

    if not results:
        await message.answer("No matching notes found.")
        return

    response_parts = [f"\U0001f50d **Results for:** _{query}_\n"]
    for i, hit in enumerate(results, 1):
        meta = hit["metadata"]
        title = meta.get("note_title", "Untitled")
        note_path = meta.get("note_path", "")
        heading = meta.get("heading", "")
        distance = hit.get("distance", 0)
        relevance = max(0, round((1 - distance) * 100))

        # Truncate content preview
        content = hit["content"][:200].replace("\n", " ").strip()
        if len(hit["content"]) > 200:
            content += "..."

        location = f"`{note_path}`"
        if heading:
            location += f" \u2192 {heading}"

        response_parts.append(f"**{i}. {title}** ({relevance}% match)\n{location}\n_{content}_\n")

    await message.answer("\n".join(response_parts), parse_mode="Markdown")
