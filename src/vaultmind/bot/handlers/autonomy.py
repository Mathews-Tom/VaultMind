"""Autonomy BLOCK-lane review — proactive Telegram notification + approve/reject/retry.

`build_block_notifier` is a standalone `BlockNotifier` bound to a `Notifier`: it
accepts any newly-pending BLOCK `ReviewProposal` (whatever producer minted it) and
sends one Approve/Reject inline keyboard. It runs from `services.review_delivery.
deliver_block`, after `ReviewQueue.propose()` committed — never inside the queue.

`Approve` applies the proposal's registered applier. Informational kinds
(`CONTRADICTION_ESCALATION`, `DUPLICATE_MERGE`) are only acknowledged — the reply
says no vault change was made. A missing or raising applier records FAILED, which
is reported as failed (never "applied") and offers an explicit `Retry` button.
`Reject` marks a pending proposal rejected. Applied, acknowledged and rejected are
terminal: a second tap reports the proposal as already resolved.

Telegram limits `callback_data` to 64 bytes, but a proposal ID is a 64-char SHA-256
hex digest. Keyboards therefore carry the lossless base64url form of the digest
(43 chars) after the action prefix; handlers decode back to the exact proposal ID.
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING, Literal

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, InlineKeyboardMarkup

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.notifier import Notifier
    from vaultmind.services.review_delivery import BlockNotifier, DeliveryStatus
    from vaultmind.services.review_queue import ReviewProposal
    from vaultmind.vault.events import VaultEventBus

type ReviewAction = Literal["approve", "reject", "retry"]

_APPROVE_ALL_SKIM = "autonomy_approve_all_skim"
_ENCODED_ID_LEN = 43  # 32-byte SHA-256 digest, base64url without padding
_MAX_SUMMARY_CHARS = 500


def encode_proposal_id(proposal_id: str) -> str:
    """Encode a 64-hex proposal ID as 43 base64url chars (lossless)."""
    return base64.urlsafe_b64encode(bytes.fromhex(proposal_id)).rstrip(b"=").decode("ascii")


def decode_proposal_id(token: str) -> str | None:
    """Invert `encode_proposal_id`; None for anything that is not a canonical encoding."""
    if len(token) != _ENCODED_ID_LEN:
        return None
    try:
        raw = base64.urlsafe_b64decode(token + "=")
    except (binascii.Error, ValueError):
        return None
    if len(raw) != 32 or encode_proposal_id(raw.hex()) != token:
        return None
    return raw.hex()


def callback_data(action: ReviewAction, proposal_id: str) -> str:
    """Telegram-valid (<=64 byte) callback payload for an action on one proposal."""
    return f"autonomy_{action}:{encode_proposal_id(proposal_id)}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def retry_keyboard(
    proposal: ReviewProposal, *, item_number: int | None = None
) -> InlineKeyboardMarkup:
    """Explicit retry for one identified failed operation."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    label = f"Retry F{item_number}" if item_number is not None else "Retry"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=callback_data("retry", proposal.proposal_id),
                )
            ]
        ]
    )


def block_keyboard(
    proposal: ReviewProposal, *, item_number: int | None = None
) -> InlineKeyboardMarkup:
    """Approve/Reject one proposal; optional display ordinal matches its listed text."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    suffix = f" B{item_number}" if item_number is not None else ""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Approve{suffix}",
                    callback_data=callback_data("approve", proposal.proposal_id),
                ),
                InlineKeyboardButton(
                    text=f"Reject{suffix}",
                    callback_data=callback_data("reject", proposal.proposal_id),
                ),
            ]
        ]
    )


def build_block_notifier(notifier: Notifier) -> BlockNotifier:
    """Build a `BlockNotifier` that sends `proposal` as a plain-text Approve/Reject
    keyboard message through `notifier` and returns the real delivery status.

    The text is sent without a parse mode: proposal summaries come from vault
    content and must not be interpreted as Markdown.
    """

    async def _send(proposal: ReviewProposal) -> DeliveryStatus:
        text = (
            f"\U0001f512 Review needed ({proposal.kind.value})\n\n"
            f"{_clip(proposal.summary, _MAX_SUMMARY_CHARS)}\n\n"
            f"Confidence {proposal.confidence:.0%}, impact {proposal.impact.value}. "
            "Nothing has been applied yet. Approve or reject below, or use /review."
        )
        return await notifier.send_with_keyboard(text, block_keyboard(proposal), parse_mode=None)

    return _send


async def handle_autonomy_callback(
    ctx: HandlerContext,
    callback: CallbackQuery,
    event_bus: VaultEventBus | None = None,
) -> None:
    """Process an `autonomy_approve:`/`autonomy_reject:`/`autonomy_retry:`/
    `autonomy_approve_all_skim` callback against `ctx.review_queue`.

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
    from vaultmind.services.review_queue import Lane, ProposalKind, ProposalStatus, ReviewQueue

    if not _is_authorized(ctx, callback):
        await callback.answer("\u26d4 Unauthorized")
        return

    data = callback.data or ""
    queue = ctx.review_queue
    if not isinstance(queue, ReviewQueue):
        await callback.answer()
        return

    def _outcome_text(proposal: ReviewProposal) -> str:
        detail = f" \u2014 {_clip(proposal.result, 500)}" if proposal.result else ""
        if proposal.status is ProposalStatus.APPLIED:
            return f"\u2705 Applied{detail}"
        if proposal.status is ProposalStatus.ACKNOWLEDGED:
            return "\u2705 Acknowledged \u2014 informational only; no vault change was made."
        if proposal.status is ProposalStatus.FAILED:
            return (
                f"\u26a0\ufe0f Failed{detail}\n"
                f"{_clip(proposal.summary, _MAX_SUMMARY_CHARS)}\n"
                "Operation did not complete. Inspect its state before an explicit retry."
            )
        return f"Status: {proposal.status.value}{detail}"

    async def _finalize_if_ingested(*proposals: ReviewProposal) -> None:
        if event_bus is None:
            return
        from vaultmind.sources.pipeline import finalize_ingested_note

        for proposal in proposals:
            if proposal.kind is ProposalKind.SOURCE_INGESTION and (
                proposal.status is ProposalStatus.APPLIED
            ):
                await finalize_ingested_note(
                    proposal,
                    parser=ctx.parser,
                    store=ctx.store,
                    vault_root=ctx.vault_root,
                    event_bus=event_bus,
                )

    async def _edit(text: str, keyboard: InlineKeyboardMarkup | None = None) -> None:
        await callback.message.edit_text(text, reply_markup=keyboard)  # type: ignore[union-attr]
        await callback.answer()

    if data == _APPROVE_ALL_SKIM:
        processed = queue.approve_all(lane=Lane.SKIM)
        await _finalize_if_ingested(*processed)
        applied = sum(1 for p in processed if p.status is ProposalStatus.APPLIED)
        acknowledged = sum(1 for p in processed if p.status is ProposalStatus.ACKNOWLEDGED)
        failed = [p for p in processed if p.status is ProposalStatus.FAILED]
        text = (
            f"Processed {len(processed)} SKIM item(s): {applied} applied, "
            f"{acknowledged} acknowledged (informational, no vault change), "
            f"{len(failed)} failed."
        )
        if failed:
            text += (
                "\nFailed operations did not complete; inspect state before retrying via /review."
            )
        await _edit(text)
        return

    action, _, token = data.partition(":")
    if action not in ("autonomy_approve", "autonomy_reject", "autonomy_retry"):
        await callback.answer()
        return

    proposal_id = decode_proposal_id(token)
    if proposal_id is None:
        await _edit("Invalid review reference.")
        return

    if action == "autonomy_approve":
        approved = queue.approve(proposal_id)
        if approved is None:
            await _edit("Already resolved or not found.")
            return
        await _finalize_if_ingested(approved)
        await _edit(
            _outcome_text(approved),
            retry_keyboard(approved) if approved.status is ProposalStatus.FAILED else None,
        )
        return

    if action == "autonomy_retry":
        retried = queue.retry(proposal_id)
        if retried is None:
            await _edit("Not failed or not found.")
            return
        await _finalize_if_ingested(retried)
        await _edit(
            _outcome_text(retried),
            retry_keyboard(retried) if retried.status is ProposalStatus.FAILED else None,
        )
        return

    rejected = queue.reject(proposal_id)
    await _edit("\u274c Rejected." if rejected else "Already resolved or not found.")


__all__ = [
    "ReviewAction",
    "block_keyboard",
    "build_block_notifier",
    "callback_data",
    "decode_proposal_id",
    "encode_proposal_id",
    "handle_autonomy_callback",
    "retry_keyboard",
]
