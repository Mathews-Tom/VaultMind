"""Tests for `/review` — weekly review prompts + pending/failed review-queue items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from vaultmind.bot.handlers.autonomy import callback_data, handle_autonomy_callback
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.handlers.review import handle_review
from vaultmind.services.review_queue import (
    Impact,
    Lane,
    ProposalKind,
    ProposalStatus,
    ReviewQueue,
)

if TYPE_CHECKING:
    from pathlib import Path


class FakeGraph:
    stats = {"nodes": 10, "edges": 5}

    def get_bridge_entities(self, n: int) -> list[dict[str, str]]:
        return []

    def get_orphan_entities(self) -> list[dict[str, str]]:
        return []


class FakeStore:
    count = 42


class FakeSettings:
    class telegram:  # noqa: N801
        allowed_user_ids: list[int] = []


def _make_ctx(review_queue: ReviewQueue | None) -> HandlerContext:
    return HandlerContext(
        settings=FakeSettings(),  # type: ignore[arg-type]
        store=FakeStore(),  # type: ignore[arg-type]
        graph=FakeGraph(),  # type: ignore[arg-type]
        parser=object(),  # type: ignore[arg-type]
        thinking=object(),  # type: ignore[arg-type]
        llm_client=object(),  # type: ignore[arg-type]
        vault_root=object(),  # type: ignore[arg-type]
        review_queue=review_queue,
    )


def _message() -> AsyncMock:
    message = AsyncMock()
    message.from_user.id = 1
    return message


class TestReviewRecovery:
    """`/review` is how a human recovers BLOCK and FAILED items after the fact."""

    async def test_undelivered_block_proposal_is_decidable_from_review(
        self, tmp_path: Path
    ) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        block = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.1,
            impact=Impact.HIGH,
            summary="needs a human",
            payload={"n": 1},
            lane_override=Lane.BLOCK,
        )
        message = _message()

        await handle_review(_make_ctx(queue), message)

        assert "needs a human" in message.answer.call_args[0][0]
        buttons = _buttons(message.answer.call_args.kwargs["reply_markup"])
        assert [b.callback_data for b in buttons] == [
            callback_data("approve", block.proposal_id),
            callback_data("reject", block.proposal_id),
        ]
        assert all(len(b.callback_data.encode()) <= 64 for b in buttons)

        # The button actually resolves the proposal through the real callback handler.
        callback = AsyncMock()
        callback.data = buttons[0].callback_data
        callback.from_user.id = 1
        await handle_autonomy_callback(_make_ctx(queue), callback)
        assert queue.get(block.proposal_id).status is ProposalStatus.ACKNOWLEDGED

    async def test_failed_proposal_is_listed_after_reopen_with_working_retry(
        self, tmp_path: Path
    ) -> None:
        attempts: list[int] = []

        def _flaky(payload: dict[str, object]) -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("disk full")
            return "done"

        db = tmp_path / "queue.db"
        queue = ReviewQueue(db, appliers={ProposalKind.TAG_VOCABULARY: _flaky})
        proposal = queue.propose(
            ProposalKind.TAG_VOCABULARY,
            confidence=0.1,
            impact=Impact.HIGH,
            summary="add tag",
            payload={"tag": "t"},
            lane_override=Lane.BLOCK,
        )
        queue.approve(proposal.proposal_id)
        queue.close()

        reopened = ReviewQueue(db, appliers={ProposalKind.TAG_VOCABULARY: _flaky})
        message = _message()
        await handle_review(_make_ctx(reopened), message)

        text = message.answer.call_args[0][0]
        assert "disk full" in text
        assert attempts == [1]  # listing never retries
        (retry,) = _buttons(message.answer.call_args.kwargs["reply_markup"])
        assert retry.callback_data == callback_data("retry", proposal.proposal_id)

        callback = AsyncMock()
        callback.data = retry.callback_data
        callback.from_user.id = 1
        await handle_autonomy_callback(_make_ctx(reopened), callback)
        assert reopened.get(proposal.proposal_id).status is ProposalStatus.APPLIED
        assert attempts == [1, 1]

        recovered = _message()
        await handle_review(_make_ctx(reopened), recovered)
        assert recovered.answer.call_args.kwargs.get("reply_markup") is None


@pytest.mark.parametrize("hostile", [False, True])
async def test_large_review_preserves_proposals_and_delivers_working_actions(
    tmp_path: Path,
    hostile: bool,
) -> None:
    import json

    import frontmatter

    from vaultmind.config import load_settings
    from vaultmind.indexer.tags import tag_appliers

    vault = tmp_path / "vault"
    vault.mkdir()
    database = tmp_path / "review.db"
    config = tmp_path / "fixture.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        "[telegram]\nallowed_user_ids = [1]\n"
        f"[autonomy]\ndb_path = {json.dumps(str(database))}\n"
    )
    queue = ReviewQueue(database, appliers=tag_appliers(vault, tmp_path / "vocabulary.json"))
    ctx = _make_ctx(queue)
    ctx.settings = load_settings(config)
    listed = []
    for lane, count in ((Lane.SKIM, 10), (Lane.BLOCK, 5)):
        for i in range(count):
            summary = f"Common producer prefix awaiting a distinct decision {i}: "
            summary += "inspection rationale " * 15
            if hostile and lane is Lane.BLOCK and i == 0:
                summary += " _*`[" * 600
            listed.append(
                queue.propose(
                    ProposalKind.DUPLICATE_MERGE,
                    0.5,
                    Impact.MEDIUM,
                    summary,
                    {"lane": lane.value, "item": i},
                    lane_override=lane,
                )
            )
    for i in range(6):
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION,
            0.5,
            Impact.LOW,
            f"missing {i}: " + "reviewed tag application " * 15,
            {"note_path": f"missing-{i}.md", "tags": ["reviewed"]},
            lane_override=Lane.BLOCK,
        )
        listed.append(queue.approve(proposal.proposal_id))
    before = {p.proposal_id: queue.get(p.proposal_id) for p in listed}
    delivered: list[tuple[str, Any]] = []

    async def bounded_transport(text: str, **kwargs: Any) -> None:
        if len(text) > 4096:
            raise ValueError("Telegram text exceeds 4096 characters")
        delivered.append((text, kwargs.get("reply_markup")))

    message = _message()
    message.answer.side_effect = bounded_transport
    try:
        await handle_review(ctx, message)
        assert {p.proposal_id: queue.get(p.proposal_id) for p in listed} == before
        buttons = [
            button for _, markup in delivered if markup is not None for button in _buttons(markup)
        ]
        actions = [button.callback_data for button in buttons]
        assert len({button.text for button in buttons}) == len(buttons)
        blocked = [p for p in listed if p.status is ProposalStatus.PENDING and p.lane is Lane.BLOCK]
        failed = [p for p in listed if p.status is ProposalStatus.FAILED]
        assert all(callback_data("approve", p.proposal_id) in actions for p in blocked)
        assert all(callback_data("reject", p.proposal_id) in actions for p in blocked)
        assert all(callback_data("retry", p.proposal_id) in actions for p in failed)
        assert all(len(action.encode()) <= 64 for action in actions)
        for action, proposal in (("approve", blocked[0]), ("retry", failed[-1])):
            if action == "retry":
                (vault / proposal.payload["note_path"]).write_text("Preserved note body.\n")
            callback = AsyncMock()
            callback.data = next(
                data for data in actions if data == callback_data(action, proposal.proposal_id)
            )
            callback.from_user.id = 1
            await handle_autonomy_callback(ctx, callback)
        assert queue.get(blocked[0].proposal_id).status is ProposalStatus.ACKNOWLEDGED
        assert queue.get(failed[-1].proposal_id).status is ProposalStatus.APPLIED
        assert all(queue.get(p.proposal_id).status is ProposalStatus.FAILED for p in failed[:-1])
        tagged = frontmatter.load(vault / failed[-1].payload["note_path"])
        assert tagged.metadata["tags"] == ["reviewed"]
    finally:
        queue.close()


def _buttons(markup: Any) -> list[Any]:
    return [b for row in markup.inline_keyboard for b in row]
