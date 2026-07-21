"""Distill handler — manual conversation distillation into qa-artifact notes."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.pipeline.distill import (
    distill_conversation,
    extract_and_store_episodes,
    mint_gap_for_unresolved,
)

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.config import DistillConfig

logger = logging.getLogger(__name__)


async def handle_distill(
    ctx: HandlerContext,
    message: Message,
    distill_config: DistillConfig,
    last_exchanges: dict[int, LastExchange],
) -> None:
    """Manually distill the current thinking session or last Q&A exchange into a qa-artifact note.

    Prefers the active thinking session (like /bookmark); falls back to the
    last Q&A exchange when no session is active.
    """
    if not _is_authorized(ctx, message):
        return

    user_id = message.from_user.id if message.from_user else 0

    if ctx.thinking.has_active_session(user_id):
        session = ctx.thinking._get_session(user_id)
        if not session.history:
            await message.answer(
                "Nothing to distill. Start a /think session or ask a question first."
            )
            return
        turns = session.history
        source_ref = f"telegram-thinking:{user_id}:{int(session.last_active)}"
        occurred_at = datetime.fromtimestamp(session.last_active, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    elif user_id in last_exchanges:
        exchange = last_exchanges[user_id]
        turns = [{"user": exchange.query, "assistant": exchange.response}]
        source_ref = f"telegram-chat:{user_id}:{int(exchange.timestamp)}"
        occurred_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        await message.answer("Nothing to distill. Start a /think session or ask a question first.")
        return

    await message.answer("\U0001f4dd Distilling conversation...")

    model = distill_config.model or ctx.settings.llm.thinking_model
    result = await asyncio.to_thread(
        distill_conversation,
        turns,
        ctx.llm_client,
        model,
        ctx.vault_root,
        distill_config.output_folder,
        source_ref,
        occurred_at,
        distill_config.max_tokens,
    )

    if not result.success:
        await message.answer(f"Distillation failed: {result.error}")
        return

    note = await asyncio.to_thread(ctx.parser.parse_file, ctx.vault_root / result.output_path)
    await asyncio.to_thread(ctx.store.index_single_note, note, ctx.parser)

    extracted = 0
    if ctx.episode_store is not None:
        try:
            extracted = await asyncio.to_thread(
                extract_and_store_episodes, note, ctx.llm_client, model, ctx.episode_store
            )
        except Exception:
            logger.exception("Episodic extraction failed for distilled note %s", result.output_path)

    await asyncio.to_thread(mint_gap_for_unresolved, result, ctx.gap_store, source_ref)

    suffix = f"\nExtracted {extracted} episode(s)." if extracted else ""
    await message.answer(
        f"Created qa-artifact: `{result.output_path}`{suffix}",
        parse_mode="Markdown",
    )
