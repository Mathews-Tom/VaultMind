"""Tests for `/review` — weekly review prompts + pending SKIM-lane items (M7 PR-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.handlers.review import handle_review
from vaultmind.services.review_queue import Impact, ProposalKind, ReviewQueue

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


class TestHandleReview:
    async def test_no_review_queue_sends_plain_review(self) -> None:
        ctx = _make_ctx(None)
        message = _message()

        await handle_review(ctx, message)

        message.answer.assert_called_once()
        _, kwargs = message.answer.call_args
        assert kwargs.get("reply_markup") is None
        text = message.answer.call_args[0][0]
        assert "Weekly Review" in text
        assert "Pending Review" not in text

    async def test_no_pending_skim_items_omits_section(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        ctx = _make_ctx(queue)
        message = _message()

        await handle_review(ctx, message)

        text = message.answer.call_args[0][0]
        assert "Pending Review" not in text
        assert message.answer.call_args.kwargs.get("reply_markup") is None

    async def test_pending_skim_items_shown_with_approve_all_button(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.MEDIUM,
            summary="Merge candidate: 'A' ~ 'B' (85% similar)",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        ctx = _make_ctx(queue)
        message = _message()

        await handle_review(ctx, message)

        text = message.answer.call_args[0][0]
        assert "Pending Review" in text
        assert "Merge candidate" in text
        keyboard = message.answer.call_args.kwargs.get("reply_markup")
        assert keyboard is not None
        button = keyboard.inline_keyboard[0][0]
        assert button.callback_data == "autonomy_approve_all_skim"
        assert "1" in button.text
