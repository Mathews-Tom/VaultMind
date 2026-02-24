"""Stats handler — vault and graph statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_stats(ctx: HandlerContext, message: Message) -> None:
    """Show vault and graph statistics."""
    if not _is_authorized(ctx, message):
        return

    graph_stats = ctx.graph.stats
    chunks = ctx.store.count

    await message.answer(
        "\U0001f4ca **VaultMind Stats**\n\n"
        f"**Vector Store:** {chunks} chunks indexed\n"
        f"**Knowledge Graph:**\n"
        f"  \u2022 Nodes: {graph_stats['nodes']}\n"
        f"  \u2022 Edges: {graph_stats['edges']}\n"
        f"  \u2022 Density: {graph_stats['density']:.3f}\n"
        f"  \u2022 Components: {graph_stats['components']}\n"
        f"  \u2022 Orphans: {graph_stats['orphans']}",
        parse_mode="Markdown",
    )
