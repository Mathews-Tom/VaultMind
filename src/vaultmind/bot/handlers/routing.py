"""Heuristic capture/greeting dispatch and approved-current knowledge answers."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.router import Intent, MessageRouter
from vaultmind.knowledge.render import render_knowledge_result, split_knowledge_text
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext

GREETING_RESPONSES = [
    "Hey! What's on your mind?",
    "Hi there. Ask me anything or prefix with note: to capture something.",
    "Hey. Need an approved-wiki answer or to capture a source?",
    "Hi! Ready when you are.",
]

_router = MessageRouter()


async def handle_message(
    ctx: HandlerContext,
    message: Message,
    text: str,
    *,
    capture_fn: object,
    think_fn: object,
    last_exchanges: dict[int, LastExchange] | None = None,
) -> None:
    """Preserve exact capture boundaries; never truncate a knowledge question."""
    if not _is_authorized(ctx, message):
        await message.answer("\u26d4 Unauthorized")
        return
    original = text
    result = _router.classify(original)
    if ctx.settings.routing.capture_all:
        await capture_fn(ctx, message, original, raw_original=original)  # type: ignore[operator]
        return
    user_id = message.from_user.id if message.from_user else 0
    if ctx.thinking.has_active_session(user_id):
        await think_fn(ctx, message, original, last_exchanges)  # type: ignore[operator]
        return
    if result.intent is Intent.capture:
        await capture_fn(ctx, message, result.content, raw_original=original)  # type: ignore[operator]
    elif result.intent is Intent.greeting:
        await handle_greeting(message)
    else:
        await handle_smart_response(
            ctx,
            message,
            original,
            is_question=result.intent is Intent.question,
            last_exchanges=last_exchanges,
        )


async def handle_greeting(message: Message) -> None:
    """Respond to casual greetings without a provider."""
    await message.answer(random.choice(GREETING_RESPONSES))


async def handle_smart_response(
    ctx: HandlerContext,
    message: Message,
    text: str,
    *,
    is_question: bool,
    last_exchanges: dict[int, LastExchange] | None = None,
) -> None:
    """Question/chat classification changes neither evidence nor answer policy."""
    if not _is_authorized(ctx, message):
        return
    user_id = message.from_user.id if message.from_user else 0
    try:
        result = await ctx.thinking.think(user_id, text, use_session=False)
    except WikiError as exc:
        await message.answer(f"Knowledge query refused: {exc}", parse_mode=None)
        return
    response = render_knowledge_result(result)
    if last_exchanges is not None:
        from vaultmind.bot.handlers.bookmark import LastExchange as _LastExchange

        last_exchanges[user_id] = _LastExchange(
            query=text, response=response, timestamp=time.monotonic()
        )
    for chunk in split_knowledge_text(response):
        await message.answer(chunk, parse_mode=None)
