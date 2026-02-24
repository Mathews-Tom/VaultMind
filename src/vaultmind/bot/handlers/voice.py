"""Voice handler — handle voice messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_voice(ctx: HandlerContext, message: Message) -> None:
    """Handle voice messages \u2014 transcribe and capture."""
    if not _is_authorized(ctx, message):
        return
    # Voice transcription requires whisper integration
    # For now, provide a helpful message
    await message.answer(
        "\U0001f3a4 Voice transcription is not yet configured.\n"
        "Enable it by installing the `whisper` extra: `uv add vaultmind[whisper]`",
        parse_mode="Markdown",
    )
