"""Review handler — weekly review prompts + pending SKIM-lane autonomy items."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_review(ctx: HandlerContext, message: Message) -> None:
    """Generate weekly review prompts with context from vault, plus any
    pending SKIM-lane review-queue items awaiting a batched human pass."""
    if not _is_authorized(ctx, message):
        return

    stats = ctx.graph.stats
    bridges = ctx.graph.get_bridge_entities(3)
    orphans = ctx.graph.get_orphan_entities()[:5]

    review = [
        "\U0001f4cb **Weekly Review**\n",
        f"**Vault:** {ctx.store.count} chunks indexed",
        f"**Graph:** {stats['nodes']} entities, {stats['edges']} relationships\n",
        "**Reflection Questions:**",
        "1. What was the most important thing I learned this week?",
        "2. What project made the most progress?",
        "3. What's blocking me right now?",
        "4. What should I focus on next week?\n",
    ]

    if bridges:
        review.append("**\U0001f309 Bridge Entities** (connecting different knowledge areas):")
        for b in bridges:
            review.append(f"  \u2022 {b['entity']} ({b['type']})")
        review.append("")

    if orphans:
        review.append("**\U0001f3dd Orphan Entities** (consider connecting these):")
        for o in orphans:
            review.append(f"  \u2022 {o.get('label', o.get('id', 'unknown'))}")
        review.append("")

    keyboard = None
    from vaultmind.services.review_queue import Lane, ReviewQueue

    if isinstance(ctx.review_queue, ReviewQueue):
        pending = ctx.review_queue.list_pending(lane=Lane.SKIM)
        if pending:
            review.append(f"**\U0001f4e5 Pending Review** ({len(pending)} SKIM item(s)):")
            for p in pending[:10]:
                review.append(f"  \u2022 {p.summary}")
            if len(pending) > 10:
                review.append(f"  \u2026 and {len(pending) - 10} more")
            review.append("")

            from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=f"Approve all {len(pending)} SKIM item(s)",
                            callback_data="autonomy_approve_all_skim",
                        )
                    ]
                ]
            )

    await message.answer("\n".join(review), parse_mode="Markdown", reply_markup=keyboard)
