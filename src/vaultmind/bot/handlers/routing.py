"""Routing handler — message classification and dispatch."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message
from vaultmind.bot.router import Intent, MessageRouter
from vaultmind.bot.sanitize import MAX_LLM_INPUT_LENGTH, sanitize_text
from vaultmind.llm.client import LLMError
from vaultmind.llm.client import Message as LLMMessage

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)

GREETING_RESPONSES = [
    "Hey! What's on your mind?",
    "Hi there. Ask me anything or prefix with `note:` to capture something.",
    "What's up? I've got your vault loaded.",
    "Hey. Need to look something up, think through an idea, or capture a note?",
    "Hi! Ready when you are.",
]

QUESTION_SYSTEM_PROMPT = """\
You are a vault-aware assistant with access to the user's personal knowledge base (Obsidian vault).
Answer the question using the vault context provided below. Be concise and direct.
Reference notes with [[Note Title]] format when citing vault content.
If the vault context is insufficient, say so and suggest `/think <topic>` for deeper exploration.
"""

CHAT_SYSTEM_PROMPT = """\
You are a friendly vault-aware assistant. The user's knowledge base context is provided below.
Respond naturally and conversationally. Weave in relevant vault context when it adds value.
Keep responses brief — a few sentences unless the topic demands more.
Reference notes with [[Note Title]] format when citing vault content.
"""

# Module-level router instance
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
    """Route a plain text message based on heuristic classification."""
    if not _is_authorized(ctx, message):
        await message.answer("\u26d4 Unauthorized")
        return

    san = sanitize_text(text, max_length=MAX_LLM_INPUT_LENGTH, operation="message_routing")
    text = san.text
    if not text:
        await message.answer("Empty input after sanitization.")
        return

    # Escape hatch: old behavior (all text -> capture)
    if ctx.settings.routing.capture_all:
        await capture_fn(ctx, message, text)  # type: ignore[operator]
        return

    # Sticky thinking sessions -- continue if active
    user_id = message.from_user.id if message.from_user else 0
    if ctx.thinking.has_active_session(user_id):
        await think_fn(ctx, message, text, last_exchanges)  # type: ignore[operator]
        return

    # Classify and dispatch
    result = _router.classify(text)

    if result.intent is Intent.capture:
        await capture_fn(ctx, message, result.content)  # type: ignore[operator]
    elif result.intent is Intent.greeting:
        await handle_greeting(message)
    elif result.intent is Intent.question:
        await handle_smart_response(
            ctx, message, result.content, is_question=True, last_exchanges=last_exchanges
        )
    else:
        await handle_smart_response(
            ctx, message, result.content, is_question=False, last_exchanges=last_exchanges
        )


async def handle_greeting(message: Message) -> None:
    """Respond to casual greetings with a static response."""
    await message.answer(random.choice(GREETING_RESPONSES))


async def handle_smart_response(
    ctx: HandlerContext,
    message: Message,
    text: str,
    *,
    is_question: bool,
    last_exchanges: dict[int, LastExchange] | None = None,
) -> None:
    """Generate a vault-context-aware response using the LLM."""
    routing_cfg = ctx.settings.routing

    # Build vault context (reuse ThinkingPartner's method, offload to thread)
    vault_context = ""
    if routing_cfg.vault_context_enabled:
        vault_context = await asyncio.to_thread(
            ctx.thinking._build_vault_context, text, ctx.store, ctx.graph
        )

    # Select model and system prompt
    model = routing_cfg.chat_model or ctx.settings.llm.fast_model
    system = QUESTION_SYSTEM_PROMPT if is_question else CHAT_SYSTEM_PROMPT

    # Build user message with vault context
    if vault_context and vault_context != "No specific vault context found for this topic.":
        user_content = f"**Context from vault:**\n{vault_context}\n\n**Message:** {text}"
    else:
        user_content = text

    messages = [LLMMessage(role="user", content=user_content)]

    try:
        response = await asyncio.to_thread(
            ctx.llm_client.complete,
            messages=messages,
            model=model,
            max_tokens=routing_cfg.chat_max_tokens,
            system=system,
        )
        if last_exchanges is not None:
            user_id = message.from_user.id if message.from_user else 0
            from vaultmind.bot.handlers.bookmark import LastExchange as _LastExchange

            last_exchanges[user_id] = _LastExchange(
                query=text,
                response=response.text,
                timestamp=time.monotonic(),
            )
        for chunk in _split_message(response.text, max_len=4000):
            await message.answer(chunk, parse_mode="Markdown")
    except LLMError as e:
        logger.error("LLM error in smart response: %s", e)
        await message.answer(f"API error ({e.provider}): {e}")
