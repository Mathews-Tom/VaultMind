"""Proactive Telegram notification delivery.

Sends messages from scheduled jobs to a configured chat.
Wraps aiogram Bot.send_message with chunking and error handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _split_message

if TYPE_CHECKING:
    from aiogram import Bot

logger = logging.getLogger(__name__)


class Notifier:
    """Delivers proactive notifications to a Telegram chat."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return self._chat_id != 0

    async def send(self, text: str, parse_mode: str | None = "Markdown") -> None:
        """Send a notification, splitting long messages."""
        if not self.enabled:
            return
        chunks = _split_message(text)
        for chunk in chunks:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            except Exception:
                logger.exception("Failed to send notification chunk")

    async def send_if_significant(self, text: str, min_length: int = 20) -> None:
        """Send only if text exceeds min_length (drop trivial messages)."""
        if len(text.strip()) >= min_length:
            await self.send(text)
