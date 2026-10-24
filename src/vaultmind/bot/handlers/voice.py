"""Bounded, memory-only voice transcription and explicit transcript capture."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.capture import BoundedDownload, telegram_identity
from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.knowledge.intake import SourceIntake

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.transcribe import Transcriber

logger = logging.getLogger(__name__)


# Transient transport deduplication, not a durable question/source journal.
_voice_events: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()
_MAX_VOICE_BYTES = 20 * 1024 * 1024


def _captured_transcript(
    ctx: HandlerContext, event: str, adapter: str, revision: str
) -> str | None:
    with SourceIntake(ctx.settings) as intake:
        record = intake.find_event(
            event_id=event,
            adapter=adapter,
            role="voice-transcript",
            revision=revision,
        )
        return record.text if record is not None else None


async def handle_voice(
    ctx: HandlerContext,
    message: Message,
    transcriber: Transcriber | None = None,
) -> None:
    if not _is_authorized(ctx, message):
        return
    if transcriber is None:
        await message.answer(
            "Voice transcription is not configured.\nSet VAULTMIND_OPENAI_API_KEY to enable."
        )
        return
    if not message.voice or message.bot is None:
        return
    if message.voice.file_size is not None and message.voice.file_size > _MAX_VOICE_BYTES:
        await message.answer("Voice file exceeds its byte limit; nothing captured.")
        return
    identity = telegram_identity(message)
    adapter = f"transcriber:openai:{ctx.settings.voice.whisper_model}"
    event = identity["event_id"]
    try:
        file = await message.bot.get_file(message.voice.file_id)
        if not file.file_path:
            raise ValueError("Voice file unavailable")
        with BoundedDownload(_MAX_VOICE_BYTES) as stream:
            stream.name = "voice.ogg"
            await message.bot.download_file(file.file_path, destination=stream)
            revision = hashlib.sha256(stream.getvalue()).hexdigest()
            key = (str(ctx.vault_root), str(event), adapter, revision)
            if event is not None and key in _voice_events:
                await message.answer(
                    "Voice event already handled; no transcription or capture repeated."
                )
                return
            transcript = None
            if event is not None:
                # Reserve before any lookup/provider await so concurrent copies
                # cannot pass the check together, including on failure.
                _voice_events[key] = None
                if len(_voice_events) > 256:
                    _voice_events.popitem(last=False)
                transcript = await asyncio.to_thread(
                    _captured_transcript, ctx, event, adapter, revision
                )
            if transcript is None:
                stream.seek(0)
                transcript = await asyncio.to_thread(transcriber.transcribe, stream)
    except Exception as exc:
        logger.warning("Transcription failed or refused: %s", type(exc).__name__)
        await message.answer("Transcription failed or voice input refused; nothing captured.")
        return
    if not transcript.strip():
        await message.answer("No speech detected in voice message.")
        return
    if transcript.rstrip().endswith("?"):
        from vaultmind.bot.handlers.routing import handle_smart_response

        await handle_smart_response(ctx, message, transcript, is_question=True)
        return
    from vaultmind.bot.handlers.capture import handle_capture

    await handle_capture(
        ctx,
        message,
        transcript,
        raw_original=transcript,
        adapter=adapter,
        role="voice-transcript",
        original_retained=False,
        revision=revision,
    )
