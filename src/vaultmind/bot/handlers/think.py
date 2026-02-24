"""Think handler — thinking partner sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message
from vaultmind.bot.sanitize import MAX_LLM_INPUT_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_think(ctx: HandlerContext, message: Message, topic: str) -> None:
    """Start or continue a thinking partner session."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(topic, max_length=MAX_LLM_INPUT_LENGTH, operation="think")
    topic = san.text
    if not topic:
        await message.answer("Empty topic after sanitization.")
        return

    user_id = message.from_user.id if message.from_user else 0
    await message.answer("\U0001f9e0 Thinking...")

    response = await ctx.thinking.think(
        user_id=user_id,
        topic=topic,
        store=ctx.store,
        graph=ctx.graph,
    )

    # Split long responses for Telegram's 4096 char limit
    for chunk in _split_message(response, max_len=4000):
        await message.answer(chunk, parse_mode="Markdown")
