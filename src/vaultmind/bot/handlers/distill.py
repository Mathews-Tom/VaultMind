"""Explicit conversation selection, source preservation and unapproved distillation."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vaultmind.bot.handlers.capture import telegram_identity
from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.knowledge.intake import SourceIntake, StageDisposition
from vaultmind.pipeline.distill import (
    PreservedDistillation,
    distill_retained_conversation,
    extract_and_store_episodes,
    mint_gap_for_unresolved,
    preserve_conversation_snapshot,
)
from vaultmind.vault.security import validate_ordinary_write_path

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.config import DistillConfig
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)


def _distill_snapshot(
    ctx: HandlerContext,
    snapshot: dict[str, Any],
    config: DistillConfig,
    identity: dict[str, Any],
    source_ref: str,
    occurred_at: str,
    model: str,
) -> tuple[PreservedDistillation, int, StageDisposition | None]:
    output_dir = validate_ordinary_write_path(config.output_folder, ctx.vault_root)
    validate_ordinary_write_path(str(output_dir / "distillation.md"), ctx.vault_root)
    with SourceIntake(ctx.settings) as intake:
        captured = preserve_conversation_snapshot(
            intake,
            snapshot,
            origin=identity["origin"],
            adapter="telegram-distill",
            event_id=identity["event_id"],
        )
        result = distill_retained_conversation(
            intake,
            captured,
            ctx.llm_client,
            model,
            source_ref=source_ref,
            occurred_at=occurred_at,
            output_folder=config.output_folder,
            max_tokens=config.max_tokens,
        )
        if not result.success or result.derivative is None or result.generation is None:
            return result, 0, None
        derivative = result.derivative
        note: Note | None = None

        def index() -> None:
            nonlocal note
            intake.read(derivative.source_id)
            note = ctx.parser.parse_file(ctx.vault_root / result.output_path)
            ctx.store.index_single_note(note, ctx.parser)

        indexed = derivative.run_stage("index", index)
        if indexed.status != "succeeded" or note is None:
            return result, 0, indexed
        extracted = 0
        if ctx.episode_store is not None:
            try:
                extracted = extract_and_store_episodes(
                    note, ctx.llm_client, model, ctx.episode_store
                )
            except Exception:
                logger.exception("Episodic extraction failed for distilled source")
        mint_gap_for_unresolved(result.generation, ctx.gap_store, source_ref)
        return result, extracted, indexed


async def handle_distill(
    ctx: HandlerContext,
    message: Message,
    distill_config: DistillConfig,
    last_exchanges: dict[int, LastExchange],
) -> None:
    """Snapshot the selected available conversation before yielding control."""
    if not _is_authorized(ctx, message):
        return
    user_id = message.from_user.id if message.from_user else 0
    snapshot = ctx.thinking.snapshot_for_distillation(user_id)
    if snapshot is not None:
        if not snapshot["history"]:
            await message.answer(
                "Nothing to distill. Start a /think session or ask a question first."
            )
            return
        source_ref = f"telegram-thinking:{user_id}:{snapshot['last_active']}"
        occurred_at = datetime.fromtimestamp(snapshot["last_active"], tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    elif user_id in last_exchanges:
        exchange = last_exchanges[user_id]
        snapshot = dict(
            version=1,
            kind="selected-last-exchange",
            user_id=user_id,
            history=[dict(user=exchange.query, assistant=exchange.response)],
            exchange_timestamp=exchange.timestamp,
            exchange_clock="monotonic",
            summaries=[],
            participants=[dict(role="user", user_id=user_id), dict(role="assistant")],
        )
        source_ref = f"telegram-chat:{user_id}:{exchange.timestamp}"
        occurred_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        await message.answer("Nothing to distill. Start a /think session or ask a question first.")
        return
    identity = telegram_identity(message)
    snapshot["selection"] = identity
    model = distill_config.model or ctx.settings.llm.thinking_model
    await message.answer("\U0001f4dd Distilling conversation...")
    try:
        result, extracted, indexed = await asyncio.to_thread(
            _distill_snapshot,
            ctx,
            snapshot,
            distill_config,
            identity,
            source_ref,
            occurred_at,
            model,
        )
    except (ValueError, OSError):
        logger.exception("Conversation preservation or projection was refused")
        await message.answer("Distillation refused: source preservation or destination failed.")
        return
    if not result.success:
        await message.answer(
            f"Distillation failed: {result.error}. "
            f"Input source `{result.input_capture.source_id}` remains preserved and unapproved."
        )
        return
    suffix = f"\nExtracted {extracted} episode(s)." if extracted else ""
    index_status = indexed.status if indexed is not None else "not-attempted"
    assert result.derivative is not None
    await message.answer(
        f"Created qa-artifact: `{result.output_path}` (unapproved source projection).\n"
        f"Source: `{result.derivative.source_id}`; index: {index_status}.{suffix}",
        parse_mode="Markdown",
    )
