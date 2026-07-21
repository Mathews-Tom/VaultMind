"""Contradiction escalation — proactive Telegram notification + acknowledge callback.

Mirrors the existing delete/edit inline-keyboard confirmation pattern
(`bot/handlers/delete.py`), adapted for a background, non-user-triggered push:
the escalation fires from `ContradictionDetector` running inside the event
bus, not from a live chat command, so it is built as a standalone callback
bound to a `Notifier` rather than a `HandlerContext`-driven message handler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import CallbackQuery

    from vaultmind.bot.notifier import Notifier

logger = logging.getLogger(__name__)


def build_escalation_notifier(
    notifier: Notifier,
) -> Callable[[str, str, str, str], Awaitable[None]]:
    """Build a `ContradictionDetector.on_escalate` callback bound to `notifier`.

    Sends an inline-keyboard message with a single "Acknowledge" button
    (`contradiction_ack:<gap_id>`), matching the existing delete/edit
    confirmation flow's keyboard + callback structure.
    """

    async def _send(note_a_title: str, note_b_title: str, rationale: str, gap_id: str) -> None:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Acknowledge",
                        callback_data=f"contradiction_ack:{gap_id}",
                    )
                ]
            ]
        )
        text = (
            "\u26a0\ufe0f **Contradiction detected**\n\n"
            f"`{note_a_title}` vs `{note_b_title}`\n\n"
            f"_{rationale}_\n\n"
            "Review with /gaps."
        )
        await notifier.send_with_keyboard(text, keyboard)

    return _send


async def handle_contradiction_callback(callback: CallbackQuery) -> None:
    """Process the contradiction escalation acknowledge callback."""
    data = callback.data or ""
    if data.startswith("contradiction_ack:"):
        await callback.message.edit_text(  # type: ignore[union-attr]
            "\u2705 Acknowledged \u2014 see /gaps to review or close."
        )
    await callback.answer()


__all__ = ["build_escalation_notifier", "handle_contradiction_callback"]
