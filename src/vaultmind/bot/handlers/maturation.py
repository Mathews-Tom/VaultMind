"""Maturation handler -- Zettelkasten maturation pipeline commands."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.pipeline.maturation import MaturationPipeline

logger = logging.getLogger(__name__)


async def handle_mature(
    ctx: HandlerContext,
    message: Message,
    args: str,
    pipeline: MaturationPipeline,
) -> None:
    """Handle /mature command -- discover clusters and synthesize notes."""
    if not _is_authorized(ctx, message):
        return

    parts = args.strip().split(maxsplit=1)
    subcommand = parts[0] if parts else ""

    if subcommand == "synthesize" and len(parts) > 1:
        await _handle_synthesize(message, parts[1].strip(), pipeline)
        return

    if subcommand == "dismiss" and len(parts) > 1:
        await _handle_dismiss(message, parts[1].strip(), pipeline)
        return

    # Default: discover clusters
    await _handle_discover(message, pipeline)


async def _handle_discover(
    message: Message,
    pipeline: MaturationPipeline,
) -> None:
    """Discover and display maturation clusters."""
    clusters = await asyncio.to_thread(pipeline.discover)

    if not clusters:
        await message.answer("No maturation candidates found. Your notes are well-organized.")
        return

    lines = [f"Found {len(clusters)} maturation cluster(s):\n"]
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"**{i}. {cluster.top_entity}** ({len(cluster.note_paths)} notes)")
        sample = cluster.note_titles[:3]
        for title in sample:
            lines.append(f"  - {title}")
        if len(cluster.note_titles) > 3:
            lines.append(f"  - ...and {len(cluster.note_titles) - 3} more")
        lines.append(f"  ID: `{cluster.fingerprint[:8]}`")
        lines.append("")

    lines.append("Use `/mature synthesize <id>` to create a permanent note")
    lines.append("Use `/mature dismiss <id>` to skip a cluster")

    text = "\n".join(lines)
    for chunk in _split_message(text, max_len=4000):
        await message.answer(chunk, parse_mode="Markdown")


async def _handle_synthesize(
    message: Message,
    id_prefix: str,
    pipeline: MaturationPipeline,
) -> None:
    """Synthesize a permanent note from a cluster."""
    clusters = await asyncio.to_thread(pipeline.discover)
    match = None
    for cluster in clusters:
        if cluster.fingerprint.startswith(id_prefix):
            match = cluster
            break

    if match is None:
        await message.answer(f"No cluster found matching '{id_prefix}'.")
        return

    await message.answer(f"Synthesizing from {len(match.note_paths)} notes...")
    result = await asyncio.to_thread(pipeline.synthesize, match)

    if result.startswith("Synthesis failed"):
        msg = f"Synthesis failed for cluster '{match.top_entity}' — review manually."
        await message.answer(msg)
    else:
        await message.answer(f"Created permanent note: `{result}`", parse_mode="Markdown")


async def _handle_dismiss(
    message: Message,
    id_prefix: str,
    pipeline: MaturationPipeline,
) -> None:
    """Dismiss a maturation cluster."""
    clusters = await asyncio.to_thread(pipeline.discover)
    match = None
    for cluster in clusters:
        if cluster.fingerprint.startswith(id_prefix):
            match = cluster
            break

    if match is None:
        await message.answer(f"No cluster found matching '{id_prefix}'.")
        return

    pipeline.dismiss(match.fingerprint)
    await message.answer(f"Dismissed cluster '{match.top_entity}'.")
