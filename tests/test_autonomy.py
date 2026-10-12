"""Tests for the generic BLOCK-lane Telegram flow: delivery, callbacks, outcomes.

Operations under test are the real shared tag appliers (frontmatter merge and the
approved-vocabulary file), run against files under ``tmp_path``. Only the Telegram
``Bot`` is a double.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import frontmatter
import pytest

from vaultmind.bot.handlers.autonomy import (
    build_block_notifier,
    callback_data,
    decode_proposal_id,
    encode_proposal_id,
    handle_autonomy_callback,
)
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.notifier import Notifier
from vaultmind.indexer.tags import tag_appliers
from vaultmind.services.review_delivery import DeliveryStatus, deliver_block
from vaultmind.services.review_queue import (
    Impact,
    Lane,
    ProposalKind,
    ProposalStatus,
    ReviewQueue,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.bot.handlers.autonomy import ReviewAction
    from vaultmind.services.review_queue import ReviewProposal


def _make_ctx(
    review_queue: ReviewQueue | None, allowed: list[int] | None = None, vault_root: Any = None
) -> HandlerContext:
    return HandlerContext(
        settings=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=allowed or [])),
        store=object(),
        graph=object(),
        parser=object(),
        thinking=object(),
        llm_client=object(),
        vault_root=vault_root if vault_root is not None else object(),
        review_queue=review_queue,
    )


def _callback(data: str, user_id: int = 1) -> AsyncMock:
    callback = AsyncMock()
    callback.data = data
    callback.from_user = SimpleNamespace(id=user_id)
    callback.message = AsyncMock()
    return callback


def _tap(action: ReviewAction, proposal: ReviewProposal, user_id: int = 1) -> AsyncMock:
    return _callback(callback_data(action, proposal.proposal_id), user_id)


def _edited(callback: AsyncMock) -> tuple[str, Any]:
    callback.message.edit_text.assert_called_once()
    call = callback.message.edit_text.call_args
    return call.args[0], call.kwargs.get("reply_markup")


def _buttons(markup: Any) -> list[Any]:
    return [b for row in markup.inline_keyboard for b in row]


def _propose(
    queue: ReviewQueue,
    kind: ProposalKind,
    payload: dict[str, Any],
    lane: Lane = Lane.BLOCK,
    summary: str = "summary",
) -> ReviewProposal:
    return queue.propose(
        kind,
        confidence=0.5,
        impact=Impact.HIGH,
        summary=summary,
        payload=payload,
        lane_override=lane,
    )


class _Vault:
    """A real vault directory plus the real queued tag operations bound to it."""

    def __init__(self, root: Path) -> None:
        self.root = root / "vault"
        self.root.mkdir()
        self.vocabulary = root / "tag_vocabulary.json"
        self.queue = ReviewQueue(
            root / "queue.db", appliers=tag_appliers(self.root, self.vocabulary)
        )

    def note(self, name: str, text: str = "---\ntitle: T\n---\n\nbody\n") -> Path:
        path = self.root / name
        path.write_text(text)
        return path

    def tags(self, name: str) -> list[str]:
        return list(frontmatter.load(str(self.root / name)).metadata.get("tags", []))


# ---------------------------------------------------------------------------
# Callback payload transport
# ---------------------------------------------------------------------------


class TestCallbackPayload:
    def test_proposal_id_roundtrips_and_fits_telegram_limit(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {"a": 1})
        assert len(proposal.proposal_id) == 64  # raw ID would overflow callback_data

        for action in ("approve", "reject", "retry"):
            data = callback_data(action, proposal.proposal_id)  # type: ignore[arg-type]
            assert len(data.encode()) <= 64
            assert decode_proposal_id(data.partition(":")[2]) == proposal.proposal_id

    @pytest.mark.parametrize(
        "token",
        ["", "short", "x" * 43, "A" * 44, "!" * 43, "A" * 42 + "B"],
    )
    def test_non_canonical_tokens_are_rejected(self, token: str) -> None:
        assert decode_proposal_id(token) is None

    def test_raw_hex_id_is_not_accepted_as_a_token(self) -> None:
        raw = "ab" * 32
        assert decode_proposal_id(raw) is None
        assert decode_proposal_id(encode_proposal_id(raw)) == raw


# ---------------------------------------------------------------------------
# BLOCK delivery (deliver_block + build_block_notifier + Notifier)
# ---------------------------------------------------------------------------


class TestBlockDelivery:
    async def test_new_block_sends_one_keyboard_whose_buttons_resolve_it(
        self, tmp_path: Path
    ) -> None:
        vault = _Vault(tmp_path)
        bot = AsyncMock()
        send = build_block_notifier(Notifier(bot=bot, chat_id=555))
        proposal = _propose(vault.queue, ProposalKind.TAG_VOCABULARY, {"tag": "alpha"})

        status = await deliver_block(proposal, send)

        assert status is DeliveryStatus.SENT
        bot.send_message.assert_called_once()
        kwargs = bot.send_message.call_args.kwargs
        assert kwargs["parse_mode"] is None  # vault-derived text is never Markdown
        approve, reject = _buttons(kwargs["reply_markup"])
        assert len(approve.callback_data.encode()) <= 64
        assert len(reject.callback_data.encode()) <= 64
        # Delivery alone never applies anything.
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.PENDING
        assert not vault.vocabulary.exists()

        # The delivered Approve button drives the real vocabulary operation.
        callback = _callback(approve.callback_data)
        await handle_autonomy_callback(_make_ctx(vault.queue), callback)
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED
        assert json.loads(vault.vocabulary.read_text()) == {"approved": ["alpha"]}

    async def test_summary_text_is_sent_without_markup_interpretation(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        bot = AsyncMock()
        hostile = "[click](tg://x) *bold* `code` _it_ " + "z" * 5000
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {}, summary=hostile)

        await deliver_block(proposal, build_block_notifier(Notifier(bot=bot, chat_id=1)))

        kwargs = bot.send_message.call_args.kwargs
        assert kwargs["parse_mode"] is None
        assert len(kwargs["text"]) < 4096

    async def test_replayed_and_terminal_proposals_are_not_resent(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        bot = AsyncMock()
        send = build_block_notifier(Notifier(bot=bot, chat_id=1))

        first = _propose(queue, ProposalKind.DUPLICATE_MERGE, {"n": 1})
        assert await deliver_block(first, send) is DeliveryStatus.SENT

        replay = _propose(queue, ProposalKind.DUPLICATE_MERGE, {"n": 1})
        assert replay.occurrence_count == 2
        assert await deliver_block(replay, send) is DeliveryStatus.NOT_REQUIRED

        resolved = queue.approve(first.proposal_id)
        assert resolved is not None
        assert await deliver_block(resolved, send) is DeliveryStatus.NOT_REQUIRED
        assert bot.send_message.call_count == 1

    async def test_only_block_lane_needs_delivery(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        bot = AsyncMock()
        send = build_block_notifier(Notifier(bot=bot, chat_id=1))
        skim = _propose(queue, ProposalKind.DUPLICATE_MERGE, {"n": 2}, lane=Lane.SKIM)

        assert await deliver_block(skim, send) is DeliveryStatus.NOT_REQUIRED
        bot.send_message.assert_not_called()

    async def test_disabled_chat_reports_disabled_and_proposal_stays_pending(
        self, tmp_path: Path
    ) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        bot = AsyncMock()
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {})

        status = await deliver_block(proposal, build_block_notifier(Notifier(bot=bot, chat_id=0)))

        assert status is DeliveryStatus.DISABLED
        bot.send_message.assert_not_called()
        assert [p.proposal_id for p in queue.list_pending(Lane.BLOCK)] == [proposal.proposal_id]

    async def test_telegram_failure_reports_failed_and_proposal_stays_pending(
        self, tmp_path: Path
    ) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        bot = AsyncMock()
        bot.send_message.side_effect = RuntimeError("network down")
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {})

        status = await deliver_block(proposal, build_block_notifier(Notifier(bot=bot, chat_id=1)))

        assert status is DeliveryStatus.FAILED
        assert queue.get(proposal.proposal_id).status is ProposalStatus.PENDING

    async def test_missing_notifier_is_unavailable_not_sent(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {})

        assert await deliver_block(proposal, None) is DeliveryStatus.UNAVAILABLE
        assert queue.get(proposal.proposal_id).status is ProposalStatus.PENDING

    async def test_raising_notifier_is_failed_not_propagated(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.DUPLICATE_MERGE, {})

        async def _boom(_: ReviewProposal) -> DeliveryStatus:
            raise RuntimeError("boom")

        assert await deliver_block(proposal, _boom) is DeliveryStatus.FAILED
        assert queue.get(proposal.proposal_id).status is ProposalStatus.PENDING


# ---------------------------------------------------------------------------
# handle_autonomy_callback — single-proposal outcomes
# ---------------------------------------------------------------------------


class TestHandleAutonomyCallback:
    async def test_approve_applies_real_operation_and_reports_result(self, tmp_path: Path) -> None:
        vault = _Vault(tmp_path)
        vault.note("a.md")
        proposal = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "a.md", "tags": ["x", "y"]}
        )
        callback = _callback(callback_data("approve", proposal.proposal_id))

        await handle_autonomy_callback(_make_ctx(vault.queue), callback)

        text, markup = _edited(callback)
        assert "a.md" in text  # the applier's own result
        assert markup is None
        assert vault.tags("a.md") == ["x", "y"]
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED

    async def test_informational_kind_is_acknowledged_without_claiming_a_change(
        self, tmp_path: Path
    ) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.CONTRADICTION_ESCALATION, {})
        callback = _callback(callback_data("approve", proposal.proposal_id))

        await handle_autonomy_callback(_make_ctx(queue), callback)

        text, markup = _edited(callback)
        assert "Applied" not in text
        assert "no vault change" in text
        assert markup is None
        assert queue.get(proposal.proposal_id).status is ProposalStatus.ACKNOWLEDGED

    async def test_missing_mutation_operation_is_failed_never_applied(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")  # no applier for maturation synthesis
        proposal = _propose(queue, ProposalKind.MATURATION_SYNTHESIS, {"cluster": 1})
        callback = _callback(callback_data("approve", proposal.proposal_id))

        await handle_autonomy_callback(_make_ctx(queue), callback)

        text, markup = _edited(callback)
        assert "Applied" not in text
        assert "Acknowledged" not in text
        stored = queue.get(proposal.proposal_id)
        assert stored.status is ProposalStatus.FAILED
        assert ProposalKind.MATURATION_SYNTHESIS.value in stored.result
        # A retry button for exactly this proposal is offered.
        (retry,) = _buttons(markup)
        assert retry.callback_data == callback_data("retry", proposal.proposal_id)

    async def test_reject_marks_rejected(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.CONTRADICTION_ESCALATION, {})
        callback = _callback(callback_data("reject", proposal.proposal_id))

        await handle_autonomy_callback(_make_ctx(queue), callback)

        callback.message.edit_text.assert_called_once()
        callback.answer.assert_called_once()
        assert queue.get(proposal.proposal_id).status is ProposalStatus.REJECTED

    async def test_second_tap_on_resolved_proposal_changes_nothing(self, tmp_path: Path) -> None:
        vault = _Vault(tmp_path)
        vault.note("a.md")
        proposal = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "a.md", "tags": ["x"]}
        )
        ctx = _make_ctx(vault.queue)
        await handle_autonomy_callback(ctx, _tap("approve", proposal))
        before = (vault.root / "a.md").read_text()

        for action in ("approve", "reject", "retry"):
            await handle_autonomy_callback(ctx, _tap(action, proposal))

        assert (vault.root / "a.md").read_text() == before
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED

    async def test_malformed_reference_touches_nothing(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = _propose(queue, ProposalKind.CONTRADICTION_ESCALATION, {})
        # The raw 64-hex ID cannot be a valid token; it must not resolve the proposal.
        callback = _callback(f"autonomy_approve:{proposal.proposal_id}")

        await handle_autonomy_callback(_make_ctx(queue), callback)

        callback.message.edit_text.assert_called_once()
        assert queue.get(proposal.proposal_id).status is ProposalStatus.PENDING

    async def test_denied_actor_cannot_approve_reject_retry_or_approve_all(
        self, tmp_path: Path
    ) -> None:
        vault = _Vault(tmp_path)
        vault.note("a.md")
        block = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "a.md", "tags": ["b"]}
        )
        skim = _propose(
            vault.queue,
            ProposalKind.TAG_APPLICATION,
            {"note_path": "a.md", "tags": ["s"]},
            lane=Lane.SKIM,
        )
        ctx = _make_ctx(vault.queue, allowed=[7])

        for data in (
            callback_data("approve", block.proposal_id),
            callback_data("reject", block.proposal_id),
            callback_data("retry", block.proposal_id),
            "autonomy_approve_all_skim",
        ):
            callback = _callback(data, user_id=99)
            await handle_autonomy_callback(ctx, callback)
            callback.message.edit_text.assert_not_called()

        assert vault.tags("a.md") == []
        assert vault.queue.get(block.proposal_id).status is ProposalStatus.PENDING
        assert vault.queue.get(skim.proposal_id).status is ProposalStatus.PENDING

    async def test_no_review_queue_is_a_noop(self) -> None:
        callback = _callback("autonomy_approve:whatever")

        await handle_autonomy_callback(_make_ctx(None), callback)

        callback.message.edit_text.assert_not_called()
        callback.answer.assert_called_once()


# ---------------------------------------------------------------------------
# Failure -> explicit retry
# ---------------------------------------------------------------------------


class TestFailureAndExplicitRetry:
    async def test_failed_approval_offers_retry_that_succeeds_once_cause_is_fixed(
        self, tmp_path: Path
    ) -> None:
        vault = _Vault(tmp_path)
        broken = "---\ntitle: T\ntags:\n  nested: map\n---\n\nbody\n"
        vault.note("a.md", broken)  # real apply_tags rejects non-list tags
        proposal = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "a.md", "tags": ["x"]}
        )
        ctx = _make_ctx(vault.queue)

        approve = _callback(callback_data("approve", proposal.proposal_id))
        await handle_autonomy_callback(ctx, approve)

        text, markup = _edited(approve)
        assert "Applied" not in text
        failed = vault.queue.get(proposal.proposal_id)
        assert failed.status is ProposalStatus.FAILED
        assert "TypeError" in failed.result
        assert (vault.root / "a.md").read_text() == broken  # nothing written
        (retry,) = _buttons(markup)

        # Not automatic: approve again, and a stale approve-all, never re-run it.
        await handle_autonomy_callback(ctx, _tap("approve", proposal))
        await handle_autonomy_callback(ctx, _callback("autonomy_approve_all_skim"))
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.FAILED

        # Explicit retry with the cause still present fails again and re-offers retry.
        again = _callback(retry.callback_data)
        await handle_autonomy_callback(ctx, again)
        assert _buttons(_edited(again)[1])[0].callback_data == retry.callback_data
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.FAILED

        vault.note("a.md")  # operator fixes the note
        fixed = _callback(retry.callback_data)
        await handle_autonomy_callback(ctx, fixed)

        assert _edited(fixed)[1] is None
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED
        assert vault.tags("a.md") == ["x"]

        # Terminal: a replayed retry does not re-run the operation.
        after = (vault.root / "a.md").read_text()
        await handle_autonomy_callback(ctx, _callback(retry.callback_data))
        assert (vault.root / "a.md").read_text() == after

    async def test_pending_proposal_is_not_retryable(self, tmp_path: Path) -> None:
        vault = _Vault(tmp_path)
        vault.note("a.md")
        proposal = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "a.md", "tags": ["x"]}
        )

        await handle_autonomy_callback(
            _make_ctx(vault.queue), _callback(callback_data("retry", proposal.proposal_id))
        )

        assert vault.tags("a.md") == []
        assert vault.queue.get(proposal.proposal_id).status is ProposalStatus.PENDING


# ---------------------------------------------------------------------------
# Approve-all: mixed batch outcomes
# ---------------------------------------------------------------------------


class TestApproveAllOutcomes:
    async def test_mixed_batch_reports_truthful_counts_and_leaves_failures_unapplied(
        self, tmp_path: Path
    ) -> None:
        vault = _Vault(tmp_path)
        vault.note("good.md")
        broken = "---\ntitle: B\ntags:\n  nested: map\n---\n\nbody\n"
        vault.note("bad.md", broken)

        applied_tag = _propose(
            vault.queue,
            ProposalKind.TAG_APPLICATION,
            {"note_path": "good.md", "tags": ["ok"]},
            Lane.SKIM,
        )
        applied_vocab = _propose(
            vault.queue, ProposalKind.TAG_VOCABULARY, {"tag": "vocab"}, Lane.SKIM
        )
        informational = _propose(vault.queue, ProposalKind.DUPLICATE_MERGE, {"m": 1}, Lane.SKIM)
        raised = _propose(
            vault.queue,
            ProposalKind.TAG_APPLICATION,
            {"note_path": "bad.md", "tags": ["no"]},
            Lane.SKIM,
        )
        missing = _propose(
            vault.queue, ProposalKind.MATURATION_SYNTHESIS, {"cluster": 3}, Lane.SKIM
        )
        ctx = _make_ctx(vault.queue)

        callback = _callback("autonomy_approve_all_skim")
        await handle_autonomy_callback(ctx, callback)

        status = {
            name: vault.queue.get(p.proposal_id).status
            for name, p in {
                "tag": applied_tag,
                "vocab": applied_vocab,
                "info": informational,
                "raised": raised,
                "missing": missing,
            }.items()
        }
        assert status == {
            "tag": ProposalStatus.APPLIED,
            "vocab": ProposalStatus.APPLIED,
            "info": ProposalStatus.ACKNOWLEDGED,
            "raised": ProposalStatus.FAILED,
            "missing": ProposalStatus.FAILED,
        }
        # Real mutations happened exactly for the applied ones.
        assert vault.tags("good.md") == ["ok"]
        assert json.loads(vault.vocabulary.read_text()) == {"approved": ["vocab"]}
        assert (vault.root / "bad.md").read_text() == broken

        text, _ = _edited(callback)
        counts = {
            word: int(n) for n, word in re.findall(r"(\d+) (applied|acknowledged|failed)", text)
        }
        assert counts == {"applied": 2, "acknowledged": 1, "failed": 2}

        # A second approve-all has nothing pending and never retries the failures.
        again = _callback("autonomy_approve_all_skim")
        await handle_autonomy_callback(ctx, again)
        assert (vault.root / "bad.md").read_text() == broken
        assert vault.queue.get(raised.proposal_id).status is ProposalStatus.FAILED
        assert vault.queue.get(missing.proposal_id).status is ProposalStatus.FAILED

    async def test_failures_survive_reopen_and_stay_unretried(self, tmp_path: Path) -> None:
        vault = _Vault(tmp_path)
        vault.note("bad.md", "---\ntitle: B\ntags:\n  nested: map\n---\n\nbody\n")
        failed = _propose(
            vault.queue, ProposalKind.TAG_APPLICATION, {"note_path": "bad.md", "tags": ["n"]}
        )
        vault.queue.approve(failed.proposal_id)
        vault.queue.close()

        reopened = ReviewQueue(
            tmp_path / "queue.db", appliers=tag_appliers(vault.root, vault.vocabulary)
        )
        assert [p.proposal_id for p in reopened.list_failed()] == [failed.proposal_id]
        assert reopened.approve_all(Lane.BLOCK) == []
        assert reopened.get(failed.proposal_id).status is ProposalStatus.FAILED


# ---------------------------------------------------------------------------
# handle_autonomy_callback — SOURCE_INGESTION event-bus follow-up (M8)
# ---------------------------------------------------------------------------


class _FakeParser:
    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root

    def parse_file(self, path: Path) -> object:
        from vaultmind.vault.models import Note

        return Note(path=path.relative_to(self.vault_root), title=path.stem, content="body")


class _FakeStore:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    def index_single_note(self, note: object, parser: object) -> int:
        self.indexed.append(str(note.path))  # type: ignore[attr-defined]
        return 1


class _FakeEventBus:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


def _make_ingestion_ctx(review_queue: ReviewQueue, vault_root: Path) -> HandlerContext:
    return HandlerContext(
        settings=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[])),
        store=_FakeStore(),
        graph=object(),
        parser=_FakeParser(vault_root),
        thinking=object(),
        llm_client=object(),
        vault_root=vault_root,
        review_queue=review_queue,
    )


class TestHandleAutonomyCallbackSourceIngestionFollowUp:
    async def test_approve_publishes_note_created_event(self, tmp_path: Path) -> None:
        (tmp_path / "note.md").write_text("---\ntitle: X\n---\n\nbody\n")

        def _apply(payload: dict[str, object]) -> str:
            return "note.md"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.SOURCE_INGESTION: _apply})
        proposal = _propose(queue, ProposalKind.SOURCE_INGESTION, {})
        ctx = _make_ingestion_ctx(queue, tmp_path)
        event_bus = _FakeEventBus()
        callback = _callback(callback_data("approve", proposal.proposal_id))

        await handle_autonomy_callback(ctx, callback, event_bus)  # type: ignore[arg-type]

        assert len(event_bus.published) == 1
        assert ctx.store.indexed == ["note.md"]  # type: ignore[attr-defined]

    async def test_approve_all_skim_publishes_event_per_applied_ingestion(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_text("---\ntitle: A\n---\n\nbody\n")
        (tmp_path / "b.md").write_text("---\ntitle: B\n---\n\nbody\n")

        paths = iter(["a.md", "b.md"])

        def _apply(payload: dict[str, object]) -> str:
            return next(paths)

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.SOURCE_INGESTION: _apply})
        _propose(queue, ProposalKind.SOURCE_INGESTION, {"id": "a"}, Lane.SKIM, "a")
        _propose(queue, ProposalKind.SOURCE_INGESTION, {"id": "b"}, Lane.SKIM, "b")
        ctx = _make_ingestion_ctx(queue, tmp_path)
        event_bus = _FakeEventBus()

        await handle_autonomy_callback(
            ctx,
            _callback("autonomy_approve_all_skim"),
            event_bus,  # type: ignore[arg-type]
        )

        assert len(event_bus.published) == 2

    async def test_no_event_bus_skips_follow_up_without_error(self, tmp_path: Path) -> None:
        (tmp_path / "note.md").write_text("---\ntitle: X\n---\n\nbody\n")

        def _apply(payload: dict[str, object]) -> str:
            return "note.md"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.SOURCE_INGESTION: _apply})
        proposal = _propose(queue, ProposalKind.SOURCE_INGESTION, {})
        ctx = _make_ingestion_ctx(queue, tmp_path)
        callback = _callback(callback_data("approve", proposal.proposal_id))

        await handle_autonomy_callback(ctx, callback)  # no event_bus arg — defaults to None

        callback.message.edit_text.assert_called_once()
        assert queue.get(proposal.proposal_id).status is ProposalStatus.APPLIED
