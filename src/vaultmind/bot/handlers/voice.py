"""Voice handler — transcribe voice messages and route to capture or question."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.transcribe import Transcriber

logger = logging.getLogger(__name__)


async def handle_voice(
    ctx: HandlerContext,
    message: Message,
    transcriber: Transcriber | None = None,
) -> None:
    """Handle voice messages — transcribe and route to capture or question flow."""
    if not _is_authorized(ctx, message):
        return

    if transcriber is None:
        await message.answer(
            "Voice transcription is not configured.\nSet VAULTMIND_OPENAI_API_KEY to enable.",
        )
        return

    if not message.voice:
        return

    bot = message.bot
    if bot is None:
        return

    await message.answer("Transcribing voice message...")

    # Download the voice file from Telegram
    try:
        file = await bot.get_file(message.voice.file_id)
        if not file.file_path:
            await message.answer("Failed to retrieve voice file.")
            return

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            await bot.download_file(file.file_path, destination=tmp)
    except Exception:
        logger.exception("Failed to download voice file")
        await message.answer("Failed to download voice file.")
        return

    # Transcribe (sync call, offload to thread)
    try:
        transcript = await asyncio.to_thread(transcriber.transcribe, tmp_path)
    except RuntimeError as e:
        await message.answer(f"Transcription failed: {e}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    if not transcript:
        await message.answer("No speech detected in voice message.")
        return

    # Route: if ends with '?' treat as question, otherwise capture
    is_question = transcript.rstrip().endswith("?")

    if is_question:
        from vaultmind.bot.handlers.routing import handle_smart_response

        await message.answer(f"Transcript: _{transcript}_", parse_mode="Markdown")
        await handle_smart_response(ctx, message, transcript, is_question=True)
    else:
        from vaultmind.bot.handlers.capture import handle_capture

        await handle_capture(ctx, message, transcript)
        await message.answer(f"Transcript: _{transcript}_", parse_mode="Markdown")
