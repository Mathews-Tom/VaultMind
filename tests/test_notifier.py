"""Tests for proactive notification delivery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from vaultmind.bot.notifier import Notifier
from vaultmind.services.review_delivery import DeliveryStatus


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


class TestSendWithKeyboardStatus:
    """The keyboard path reports what actually happened; `send` stays fire-and-forget."""

    async def test_disabled_chat_reports_disabled_and_sends_nothing(self) -> None:
        bot = AsyncMock()
        status = await Notifier(bot=bot, chat_id=0).send_with_keyboard("hi", MagicMock())
        assert status is DeliveryStatus.DISABLED
        bot.send_message.assert_not_called()

    async def test_delivered_message_reports_sent_with_keyboard_attached(self) -> None:
        bot = AsyncMock()
        keyboard = MagicMock()
        status = await Notifier(bot=bot, chat_id=7).send_with_keyboard("hi", keyboard)
        assert status is DeliveryStatus.SENT
        assert bot.send_message.call_args.kwargs["reply_markup"] is keyboard

    async def test_transport_error_reports_failed(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = RuntimeError("network error")
        status = await Notifier(bot=bot, chat_id=7).send_with_keyboard("hi", MagicMock())
        assert status is DeliveryStatus.FAILED
