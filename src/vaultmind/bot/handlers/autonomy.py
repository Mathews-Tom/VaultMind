"""Autonomy BLOCK-lane review — proactive Telegram notification + approve/reject.

Generalizes M6's contradiction-escalation Acknowledge-only flow
(`build_escalation_notifier`/`handle_contradiction_callback`, retired by this
module) into a reusable BLOCK-lane flow for any `services.review_queue`
proposal: the notification fires from whatever async orchestrator called
`ReviewQueue.propose()` and observed `proposal.lane is Lane.BLOCK` — not
from a live chat command — so `build_block_notifier` is a standalone
callback bound to a `Notifier`, mirroring `bot/handlers/delete.py`'s
inline-keyboard confirmation structure.

`Approve` applies the proposal's registered applier (or, for the
applier-less kinds like `CONTRADICTION_ESCALATION`, records an
acknowledgment — never a false "applied" claim). `Reject` marks it
rejected. Both are terminal: a second tap on either button reports the
proposal as already resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import CallbackQuery

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.notifier import Notifier
    from vaultmind.vault.events import VaultEventBus

# (note_a_title, note_b_title, rationale, gap_id, proposal_id) -> None
type EscalationCallback = Callable[[str, str, str, str, str], Awaitable[None]]


def build_block_notifier(notifier: Notifier) -> EscalationCallback:
    """Build a `ContradictionDetector.on_escalate`-compatible callback bound
    to `notifier`, sending an Approve/Reject inline keyboard for the given
    review-queue proposal instead of the old single Acknowledge button.
    """

    async def _send(
        note_a_title: str,
        note_b_title: str,
        rationale: str,
        gap_id: str,
        proposal_id: str,
    ) -> None:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Approve",
                        callback_data=f"autonomy_approve:{proposal_id}",
                    ),
                    InlineKeyboardButton(
                        text="Reject",
                        callback_data=f"autonomy_reject:{proposal_id}",
                    ),
                ]
            ]
        )
        text = (
            "\U0001f512 **Review needed \u2014 contradiction escalated**\n\n"
            f"`{note_a_title}` vs `{note_b_title}`\n\n"
            f"_{rationale}_\n\n"
            "Review with /gaps."
        )
        await notifier.send_with_keyboard(text, keyboard)

    return _send


async def handle_autonomy_callback(
    ctx: HandlerContext,
    callback: CallbackQuery,
    event_bus: VaultEventBus | None = None,
) -> None:
    """Process an `autonomy_approve:`/`autonomy_reject:`/`autonomy_approve_all_skim`
    callback against `ctx.review_queue`.

    `event_bus`, if given, triggers `sources/pipeline.py::finalize_ingested_note`
    for every newly-`APPLIED` `SOURCE_INGESTION` proposal this approval
    produced — the approve-later half of M8's "evaluated by the existing
    dedup/contradiction event-bus path" requirement (the AUTO-immediate
    half runs from `cli.py::bot`'s scheduler job / `vaultmind source run`
    directly after `propose()`). `None` (the default) skips this follow-up
    — every other proposal kind is unaffected either way, since only
    `SOURCE_INGESTION` proposals create a new vault note to publish an
    event for.
    """
    from vaultmind.services.review_queue import Lane, ProposalKind, ReviewProposal, ReviewQueue

    data = callback.data or ""
    queue = ctx.review_queue
    if not isinstance(queue, ReviewQueue):
        await callback.answer()
        return

    async def _finalize_if_ingested(*proposals: ReviewProposal) -> None:
        if event_bus is None:
            return
        from vaultmind.sources.pipeline import finalize_ingested_note

        for proposal in proposals:
            if proposal.kind is ProposalKind.SOURCE_INGESTION:
                await finalize_ingested_note(
                    proposal,
                    parser=ctx.parser,
                    store=ctx.store,
                    vault_root=ctx.vault_root,
                    event_bus=event_bus,
                )

    if data == "autonomy_approve_all_skim":
        approved = queue.approve_all(lane=Lane.SKIM)
        await _finalize_if_ingested(*approved)
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"\u2705 Approved {len(approved)} SKIM item(s)."
        )
        await callback.answer()
        return

    if data.startswith("autonomy_approve:"):
        proposal_id = data[len("autonomy_approve:") :]
        result = queue.approve(proposal_id)
        if result is None:
            text = "Already resolved or not found."
        else:
            await _finalize_if_ingested(result)
            suffix = f" \u2014 {result.result}" if result.result else ""
            text = f"\u2705 Approved{suffix}"
        await callback.message.edit_text(text)  # type: ignore[union-attr]
        await callback.answer()
        return

    if data.startswith("autonomy_reject:"):
        proposal_id = data[len("autonomy_reject:") :]
        ok = queue.reject(proposal_id)
        text = "\u274c Rejected." if ok else "Already resolved or not found."
        await callback.message.edit_text(text)  # type: ignore[union-attr]
        await callback.answer()
        return

    await callback.answer()


__all__ = ["EscalationCallback", "build_block_notifier", "handle_autonomy_callback"]
