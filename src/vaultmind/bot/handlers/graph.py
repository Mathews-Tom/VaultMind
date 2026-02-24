"""Graph handler — knowledge graph entity queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_graph(ctx: HandlerContext, message: Message, entity: str) -> None:
    """Query the knowledge graph for an entity's connections."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(entity, max_length=MAX_QUERY_LENGTH, operation="graph")
    entity = san.text
    if not entity:
        await message.answer("Empty entity after sanitization.")
        return

    result = ctx.graph.get_neighbors(entity, depth=2)

    if result["entity"] is None:
        await message.answer(
            f"Entity `{entity}` not found in knowledge graph.",
            parse_mode="Markdown",
        )
        return

    ent = result["entity"]
    lines = [
        f"\U0001f578 **{ent.get('label', entity)}** ({ent.get('type', 'unknown')})",
        f"Confidence: {ent.get('confidence', 0):.0%}",
        f"Source notes: {len(ent.get('source_notes', []))}",
        "",
    ]

    if result["outgoing"]:
        lines.append("**\u2192 Outgoing:**")
        for rel in result["outgoing"][:10]:
            lines.append(f"  \u2022 {rel['relation']} \u2192 {rel['target']}")
        lines.append("")

    if result["incoming"]:
        lines.append("**\u2190 Incoming:**")
        for rel in result["incoming"][:10]:
            lines.append(f"  \u2022 {rel['source']} \u2192 {rel['relation']}")
        lines.append("")

    if result["neighbors"]:
        neighbor_labels = [n.get("label", n["id"]) for n in result["neighbors"][:15]]
        lines.append(f"**Neighborhood:** {', '.join(neighbor_labels)}")

    await message.answer("\n".join(lines), parse_mode="Markdown")
