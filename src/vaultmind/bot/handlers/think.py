"""Think handler — thinking partner sessions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.knowledge.render import render_knowledge_result, split_knowledge_text
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext


async def handle_think(
    ctx: HandlerContext,
    message: Message,
    topic: str,
    last_exchanges: dict[int, LastExchange] | None = None,
) -> None:
    """Start or continue a thinking partner session."""
    if not _is_authorized(ctx, message):
        return

    user_id = message.from_user.id if message.from_user else 0
    await message.answer("\U0001f9e0 Thinking...")

    try:
        result = await ctx.thinking.think(user_id, topic)
    except WikiError as exc:
        await message.answer(f"Knowledge query refused: {exc}", parse_mode=None)
        return
    response = render_knowledge_result(result)

    if last_exchanges is not None:
        from vaultmind.bot.handlers.bookmark import LastExchange as _LastExchange

        last_exchanges[user_id] = _LastExchange(
            query=topic,
            response=response,
            timestamp=time.monotonic(),
        )

    for chunk in split_knowledge_text(response):
        await message.answer(chunk, parse_mode=None)
