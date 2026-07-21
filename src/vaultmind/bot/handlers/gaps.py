"""Knowledge gap ledger bot command — /gaps."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.memory.gaps import GapStore

logger = logging.getLogger(__name__)

_KIND_LABELS = {
    "unanswered_question": "❓",
    "weak_retrieval": "🔍",
    "contradiction_escalated": "⚠️",
    "stale_claim": "🕰️",
}


async def handle_gaps(
    ctx: HandlerContext,
    message: Message,
    gap_store: GapStore,
) -> None:
    """List open knowledge gaps, oldest first. /gaps"""
    if not _is_authorized(ctx, message):
        return

    limit = ctx.settings.gaps.max_shown
    gaps = await asyncio.to_thread(gap_store.list_open, limit)

    if not gaps:
        await message.answer("No open gaps.")
        return

    now = datetime.now()
    lines = [f"<b>Open gaps ({len(gaps)}, oldest first)</b>", ""]
    for gap in gaps:
        icon = _KIND_LABELS.get(gap.kind.value, "•")
        age_days = (now - gap.created).days
        lines.append(f"{icon} <code>{gap.gap_id[:8]}</code> ({age_days}d) — {gap.question}")

    text = "\n".join(lines)
    for chunk in _split_message(text, max_len=4000):
        await message.answer(chunk, parse_mode="HTML")
