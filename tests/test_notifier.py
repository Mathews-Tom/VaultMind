"""Tests for proactive notification delivery."""

from __future__ import annotations

from unittest.mock import AsyncMock

from vaultmind.bot.notifier import Notifier


class TestNotifier:
    async def test_send_disabled_when_chat_id_zero(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=0)
        assert notifier.enabled is False
        await notifier.send("hello")
        bot.send_message.assert_not_called()

    async def test_send_delivers_message(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        assert notifier.enabled is True
        await notifier.send("hello")
        bot.send_message.assert_called_once_with(chat_id=12345, text="hello", parse_mode="Markdown")

    async def test_send_splits_long_messages(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        long_text = "x" * 5000
        await notifier.send(long_text)
        assert bot.send_message.call_count >= 2

    async def test_send_if_significant_drops_short(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        await notifier.send_if_significant("hi")
        bot.send_message.assert_not_called()

    async def test_send_if_significant_sends_long(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        await notifier.send_if_significant("This is a significant notification message.")
        bot.send_message.assert_called_once()

    async def test_send_handles_exception(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = RuntimeError("network error")
        notifier = Notifier(bot=bot, chat_id=12345)
        # Should not raise
        await notifier.send("hello")
