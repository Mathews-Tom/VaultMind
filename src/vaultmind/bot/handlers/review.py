"""Review handler — weekly review prompts with graph insights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_review(ctx: HandlerContext, message: Message) -> None:
    """Generate weekly review prompts with context from vault."""
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

    await message.answer("\n".join(review), parse_mode="Markdown")
