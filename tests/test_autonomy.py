"""Tests for the generic BLOCK-lane Telegram flow (M7 PR-3).

Generalizes M6's contradiction-escalation Acknowledge-only flow into
Approve/Reject buttons that operate on any `services.review_queue`
proposal via `ctx.review_queue`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from vaultmind.bot.handlers.autonomy import build_block_notifier, handle_autonomy_callback
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.notifier import Notifier
from vaultmind.services.review_queue import Impact, Lane, ProposalKind, ProposalStatus, ReviewQueue

if TYPE_CHECKING:
    from pathlib import Path


def _make_ctx(review_queue: ReviewQueue | None) -> HandlerContext:
    return HandlerContext(
        settings=object(),
        store=object(),
        graph=object(),
        parser=object(),
        thinking=object(),
        llm_client=object(),
        vault_root=object(),
        review_queue=review_queue,
    )


# ---------------------------------------------------------------------------
# build_block_notifier
# ---------------------------------------------------------------------------


class TestBuildBlockNotifier:
    async def test_sends_approve_reject_keyboard(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        send = build_block_notifier(notifier)

        await send("Note A", "Note B", "different facts", "gap123", "proposal-abc")

        bot.send_message.assert_called_once()
        _, kwargs = bot.send_message.call_args
        assert "Note A" in kwargs["text"]
        assert "Note B" in kwargs["text"]
        assert "different facts" in kwargs["text"]
        keyboard = kwargs["reply_markup"]
        approve, reject = keyboard.inline_keyboard[0]
        assert approve.callback_data == "autonomy_approve:proposal-abc"
        assert reject.callback_data == "autonomy_reject:proposal-abc"

    async def test_disabled_notifier_sends_nothing(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=0)
        send = build_block_notifier(notifier)

        await send("Note A", "Note B", "reason", "gap123", "proposal-abc")

        bot.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# handle_autonomy_callback
# ---------------------------------------------------------------------------


class TestHandleAutonomyCallback:
    async def test_approve_with_applier_edits_message_with_result(self, tmp_path: Path) -> None:
        def _apply(payload: dict[str, object]) -> str:
            return "marked contradicted"

        queue = ReviewQueue(
            tmp_path / "queue.db", appliers={ProposalKind.CONTRADICTION_RESOLUTION: _apply}
        )
        proposal = queue.propose(
            ProposalKind.CONTRADICTION_RESOLUTION,
            confidence=0.5,
            impact=Impact.HIGH,
            summary="test",
            payload={},
            lane_override=Lane.BLOCK,
        )
        ctx = _make_ctx(queue)
        callback = AsyncMock()
        callback.data = f"autonomy_approve:{proposal.proposal_id}"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_called_once()
        text = callback.message.edit_text.call_args[0][0]
        assert "Approved" in text
        assert "marked contradicted" in text
        callback.answer.assert_called_once()
        assert queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED

    async def test_approve_applier_less_kind_acknowledges(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=0.0,
            impact=Impact.HIGH,
            summary="escalated",
            payload={},
            lane_override=Lane.BLOCK,
        )
        ctx = _make_ctx(queue)
        callback = AsyncMock()
        callback.data = f"autonomy_approve:{proposal.proposal_id}"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_called_once()
        text = callback.message.edit_text.call_args[0][0]
        assert text == "\u2705 Approved"
        assert queue.get(proposal.proposal_id).status is ProposalStatus.ACKNOWLEDGED

    async def test_reject_marks_rejected(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=0.0,
            impact=Impact.HIGH,
            summary="escalated",
            payload={},
            lane_override=Lane.BLOCK,
        )
        ctx = _make_ctx(queue)
        callback = AsyncMock()
        callback.data = f"autonomy_reject:{proposal.proposal_id}"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_called_once_with("\u274c Rejected.")
        callback.answer.assert_called_once()
        assert queue.get(proposal.proposal_id).status is ProposalStatus.REJECTED

    async def test_approve_already_resolved_reports_it(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=0.0,
            impact=Impact.HIGH,
            summary="escalated",
            payload={},
            lane_override=Lane.BLOCK,
        )
        queue.reject(proposal.proposal_id)
        ctx = _make_ctx(queue)
        callback = AsyncMock()
        callback.data = f"autonomy_approve:{proposal.proposal_id}"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_called_once_with("Already resolved or not found.")

    async def test_approve_all_skim(self, tmp_path: Path) -> None:
        def _apply(payload: dict[str, object]) -> str:
            return "applied"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _apply})
        for i in range(2):
            queue.propose(
                ProposalKind.TAG_APPLICATION,
                confidence=0.6,
                impact=Impact.LOW,
                summary=f"tag {i}",
                payload={"note_path": f"n{i}.md", "tags": ["x"]},
            )
        ctx = _make_ctx(queue)
        callback = AsyncMock()
        callback.data = "autonomy_approve_all_skim"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_called_once_with("\u2705 Approved 2 SKIM item(s).")
        assert queue.list_pending(lane=Lane.SKIM) == []

    async def test_no_review_queue_is_a_noop(self) -> None:
        ctx = _make_ctx(None)
        callback = AsyncMock()
        callback.data = "autonomy_approve:whatever"
        callback.message = AsyncMock()

        await handle_autonomy_callback(ctx, callback)

        callback.message.edit_text.assert_not_called()
        callback.answer.assert_called_once()
