"""Episodic memory bot commands — /decide, /outcome, /episodes."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message
from vaultmind.memory.models import OutcomeStatus

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.memory.store import EpisodeStore

logger = logging.getLogger(__name__)

_VALID_STATUSES = {s.value for s in OutcomeStatus}


async def handle_decide(
    ctx: HandlerContext,
    message: Message,
    decision: str,
    episode_store: EpisodeStore,
) -> None:
    """Create a pending episode from /decide <decision text>."""
    if not _is_authorized(ctx, message):
        return

    episode = await asyncio.to_thread(episode_store.create, decision)
    await message.answer(
        f"<b>Decision recorded.</b>\n"
        f"ID: <code>{episode.episode_id}</code>\n"
        f"Decision: {episode.decision}\n\n"
        f"Use <code>/outcome {episode.episode_id} &lt;status&gt; &lt;description&gt;</code> "
        f"to record the outcome.\n"
        f"Status options: success, failure, partial, unknown",
        parse_mode="HTML",
    )


async def handle_outcome(
    ctx: HandlerContext,
    message: Message,
    args: str,
    episode_store: EpisodeStore,
) -> None:
    """Resolve an episode: /outcome <id> <status> <description>."""
    if not _is_authorized(ctx, message):
        return

    parts = args.strip().split(maxsplit=2)
    if len(parts) < 2:
        await message.answer(
            "Usage: <code>/outcome &lt;id&gt; &lt;status&gt; [description]</code>\n"
            "Status: success, failure, partial, unknown",
            parse_mode="HTML",
        )
        return

    episode_id = parts[0]
    raw_status = parts[1].lower()
    description = parts[2] if len(parts) > 2 else ""

    if raw_status not in _VALID_STATUSES or raw_status == OutcomeStatus.PENDING:
        await message.answer(
            f"Invalid status <code>{raw_status}</code>. Choose: success, failure, partial, unknown",
            parse_mode="HTML",
        )
        return

    episode = await asyncio.to_thread(episode_store.get, episode_id)
    if episode is None:
        await message.answer(
            f"No episode found with ID <code>{episode_id}</code>.", parse_mode="HTML"
        )
        return

    status = OutcomeStatus(raw_status)
    await asyncio.to_thread(
        episode_store.resolve,
        episode_id,
        description,
        status,
        [],
    )
    await message.answer(
        f"<b>Episode resolved.</b>\n"
        f"ID: <code>{episode_id}</code>\n"
        f"Status: {status.value}\n"
        f"Outcome: {description or '(none recorded)'}",
        parse_mode="HTML",
    )


async def handle_episodes(
    ctx: HandlerContext,
    message: Message,
    entity: str,
    episode_store: EpisodeStore,
) -> None:
    """List episodes, optionally filtered by entity."""
    if not _is_authorized(ctx, message):
        return

    if entity:
        episodes = await asyncio.to_thread(episode_store.search_by_entity, entity)
        header = f"Episodes mentioning <b>{entity}</b>:"
    else:
        episodes = await asyncio.to_thread(episode_store.query_pending)
        header = "Pending episodes:"

    if not episodes:
        await message.answer("No episodes found.")
        return

    lines = [header, ""]
    for ep in episodes:
        status_icon = {
            "pending": "⏳",
            "success": "✅",
            "failure": "❌",
            "partial": "🔶",
            "unknown": "❓",
        }.get(ep.outcome_status, "•")
        lines.append(f"{status_icon} <code>{ep.episode_id}</code> — {ep.decision}")
        if ep.outcome:
            lines.append(f"   → {ep.outcome}")
        lines.append(f"   {ep.created.strftime('%Y-%m-%d')}")
        lines.append("")

    text = "\n".join(lines)
    for chunk in _split_message(text, max_len=4000):
        await message.answer(chunk, parse_mode="HTML")
