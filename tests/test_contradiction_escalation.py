"""Tests for the contradiction detector orchestrator + escalation flow (M6 PR-4)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from vaultmind.bot.handlers.contradiction import (
    build_escalation_notifier,
    handle_contradiction_callback,
)
from vaultmind.bot.notifier import Notifier
from vaultmind.contradiction.detector import ContradictionDetector
from vaultmind.indexer.duplicate_detector import DuplicateMatch, MatchType
from vaultmind.llm.client import LLMResponse
from vaultmind.memory.gaps import GapKind, GapStatus, GapStore
from vaultmind.vault.events import NoteCreatedEvent
from vaultmind.vault.models import Note

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeContradictionConfig:
    def __init__(
        self, enabled: bool = True, auto_resolve: bool = False, max_tokens: int = 300
    ) -> None:
        self.enabled = enabled
        self.auto_resolve = auto_resolve
        self.max_tokens = max_tokens


class FakeDuplicateDetector:
    def __init__(self, matches: list[DuplicateMatch]) -> None:
        self._matches = matches

    def find_duplicates(self, note: Note, *, max_results: int = 10) -> list[DuplicateMatch]:
        return self._matches


class FakeParser:
    def __init__(self, note: Note) -> None:
        self._note = note

    def parse_file(self, filepath: object) -> Note:
        return self._note


def _make_llm(conflicts: bool, reasoning: str = "reason") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text=json.dumps({"materially_conflicts": conflicts, "reasoning": reasoning}),
        model="test-model",
        usage={},
    )
    return client


def _note(
    path: str = "new.md",
    title: str = "New Note",
    modified: datetime = datetime(2026, 1, 1),
    authority: int = 0,
    frontmatter: dict[str, object] | None = None,
) -> Note:
    return Note(
        path=path,
        title=title,
        content="body content",
        modified=modified,
        authority=authority,
        frontmatter=frontmatter or {},
    )


def _match(match_path: str = "candidate.md", match_title: str = "Candidate Note") -> DuplicateMatch:
    return DuplicateMatch(
        source_path="new.md",
        source_title="New Note",
        match_path=match_path,
        match_title=match_title,
        similarity=0.85,
        match_type=MatchType.MERGE,
    )


def _make_detector(
    tmp_path: Path,
    llm_client: object,
    candidate: Note,
    match: DuplicateMatch,
    auto_resolve: bool = False,
    gap_store: GapStore | None = None,
    on_escalate: object = None,
    review_queue: object = None,
) -> ContradictionDetector:
    candidate_path = tmp_path / match.match_path
    candidate_path.write_text("---\ntitle: Candidate\n---\n\ncandidate body\n", encoding="utf-8")

    return ContradictionDetector(
        FakeContradictionConfig(auto_resolve=auto_resolve),
        FakeDuplicateDetector([match]),
        llm_client,
        "test-model",
        tmp_path,
        FakeParser(candidate),
        gap_store=gap_store,
        on_escalate=on_escalate,
        review_queue=review_queue,
    )


# ---------------------------------------------------------------------------
# on_note_changed guards
# ---------------------------------------------------------------------------


class TestOnNoteChangedGuards:
    async def test_disabled_config_is_a_noop(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True)
        detector = ContradictionDetector(
            FakeContradictionConfig(enabled=False),
            FakeDuplicateDetector([match]),
            llm,
            "test-model",
            tmp_path,
            FakeParser(candidate),
        )
        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))
        llm.complete.assert_not_called()

    async def test_none_note_is_a_noop(self, tmp_path: Path) -> None:
        llm = _make_llm(True)
        detector = ContradictionDetector(
            FakeContradictionConfig(),
            FakeDuplicateDetector([]),
            llm,
            "test-model",
            tmp_path,
            FakeParser(_note()),
        )
        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=None))
        llm.complete.assert_not_called()

    async def test_already_marked_loser_skips_llm_call(self, tmp_path: Path) -> None:
        note = _note(frontmatter={"contradicted_by": ["some-winner.md"]})
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True)
        detector = _make_detector(tmp_path, llm, candidate, match)

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        llm.complete.assert_not_called()

    async def test_duplicate_band_matches_are_ignored(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        dup_match = DuplicateMatch(
            source_path="new.md",
            source_title="New Note",
            match_path="candidate.md",
            match_title="Candidate Note",
            similarity=0.95,
            match_type=MatchType.DUPLICATE,
        )
        llm = _make_llm(True)
        (tmp_path / "candidate.md").write_text("---\n---\n\nbody\n", encoding="utf-8")
        detector = ContradictionDetector(
            FakeContradictionConfig(),
            FakeDuplicateDetector([dup_match]),
            llm,
            "test-model",
            tmp_path,
            FakeParser(candidate),
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        llm.complete.assert_not_called()

    async def test_already_marked_candidate_skips_llm_call(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note(
            "candidate.md", "Candidate Note", frontmatter={"contradicted_by": ["x.md"]}
        )
        match = _match()
        llm = _make_llm(True)
        detector = _make_detector(tmp_path, llm, candidate, match)

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Non-conflicting verdict
# ---------------------------------------------------------------------------


class TestNonConflictingVerdict:
    async def test_no_conflict_takes_no_action(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(False)
        gap_store = GapStore(tmp_path / "gaps.db")
        on_escalate = AsyncMock()
        detector = _make_detector(
            tmp_path, llm, candidate, match, gap_store=gap_store, on_escalate=on_escalate
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        on_escalate.assert_not_called()
        assert gap_store.list_open() == []
        gap_store.close()


# ---------------------------------------------------------------------------
# Escalation (auto_resolve=False, the default)
# ---------------------------------------------------------------------------


class TestEscalationDefault:
    async def test_conflict_mints_gap_and_calls_escalate_callback(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True, reasoning="different facts")
        gap_store = GapStore(tmp_path / "gaps.db")
        on_escalate = AsyncMock()
        detector = _make_detector(
            tmp_path,
            llm,
            candidate,
            match,
            auto_resolve=False,
            gap_store=gap_store,
            on_escalate=on_escalate,
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        gaps = gap_store.list_open()
        assert len(gaps) == 1
        assert gaps[0].kind == GapKind.CONTRADICTION_ESCALATED
        assert gaps[0].status == GapStatus.OPEN
        on_escalate.assert_called_once()
        gap_store.close()

    async def test_conflict_never_mutates_the_vault_when_not_auto_resolving(
        self, tmp_path: Path
    ) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True)
        detector = _make_detector(tmp_path, llm, candidate, match, auto_resolve=False)

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" not in content

    async def test_escalates_even_when_policy_would_resolve(self, tmp_path: Path) -> None:
        # new note is more recently modified than candidate -> policy would
        # resolve via TEMPORAL, but auto_resolve=False forces escalation anyway.
        note = _note(modified=datetime(2026, 2, 1))
        candidate = _note("candidate.md", "Candidate Note", modified=datetime(2026, 1, 1))
        match = _match()
        llm = _make_llm(True)
        gap_store = GapStore(tmp_path / "gaps.db")
        detector = _make_detector(
            tmp_path, llm, candidate, match, auto_resolve=False, gap_store=gap_store
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        assert len(gap_store.list_open()) == 1
        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" not in content
        gap_store.close()

    async def test_gap_dedup_is_order_invariant(self, tmp_path: Path) -> None:
        # Same conceptual pair, titles swapped -> same normalized question.
        note_a = _note("a.md", "Alpha Note")
        note_b_as_candidate = _note("b.md", "Beta Note")
        match_a = _match(match_path="b.md", match_title="Beta Note")
        llm = _make_llm(True)
        gap_store = GapStore(tmp_path / "gaps.db")
        detector_a = _make_detector(
            tmp_path, llm, note_b_as_candidate, match_a, gap_store=gap_store
        )
        await detector_a.on_note_changed(NoteCreatedEvent(path=tmp_path / "a.md", note=note_a))

        note_b = _note("b.md", "Beta Note")
        note_a_as_candidate = _note("a.md", "Alpha Note")
        match_b = _match(match_path="a.md", match_title="Alpha Note")
        detector_b = _make_detector(
            tmp_path, llm, note_a_as_candidate, match_b, gap_store=gap_store
        )
        await detector_b.on_note_changed(NoteCreatedEvent(path=tmp_path / "b.md", note=note_b))

        gaps = gap_store.list_open()
        assert len(gaps) == 1
        assert gaps[0].occurrence_count == 2
        gap_store.close()


# ---------------------------------------------------------------------------
# Auto-resolve enabled
# ---------------------------------------------------------------------------


class TestAutoResolveEnabled:
    async def test_temporal_resolution_marks_loser_without_gap(self, tmp_path: Path) -> None:
        note = _note(modified=datetime(2026, 2, 1))  # more recent -> new wins
        candidate = _note("candidate.md", "Candidate Note", modified=datetime(2026, 1, 1))
        match = _match()
        llm = _make_llm(True)
        gap_store = GapStore(tmp_path / "gaps.db")
        on_escalate = AsyncMock()
        detector = _make_detector(
            tmp_path,
            llm,
            candidate,
            match,
            auto_resolve=True,
            gap_store=gap_store,
            on_escalate=on_escalate,
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" in content
        assert gap_store.list_open() == []
        on_escalate.assert_not_called()
        gap_store.close()

    async def test_tied_policy_still_escalates_under_auto_resolve(self, tmp_path: Path) -> None:
        same_time = datetime(2026, 1, 1)
        note = _note(modified=same_time, authority=3)
        candidate = _note("candidate.md", "Candidate Note", modified=same_time, authority=3)
        match = _match()
        llm = _make_llm(True)
        gap_store = GapStore(tmp_path / "gaps.db")
        on_escalate = AsyncMock()
        detector = _make_detector(
            tmp_path,
            llm,
            candidate,
            match,
            auto_resolve=True,
            gap_store=gap_store,
            on_escalate=on_escalate,
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" not in content
        assert len(gap_store.list_open()) == 1
        on_escalate.assert_called_once()
        gap_store.close()


# ---------------------------------------------------------------------------
# bot/handlers/contradiction.py
# ---------------------------------------------------------------------------


class TestBuildEscalationNotifier:
    async def test_sends_keyboard_with_acknowledge_button(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        send = build_escalation_notifier(notifier)

        await send("Note A", "Note B", "different facts", "gap123")

        bot.send_message.assert_called_once()
        _, kwargs = bot.send_message.call_args
        assert "Note A" in kwargs["text"]
        assert "Note B" in kwargs["text"]
        assert "different facts" in kwargs["text"]
        keyboard = kwargs["reply_markup"]
        button = keyboard.inline_keyboard[0][0]
        assert button.callback_data == "contradiction_ack:gap123"

    async def test_disabled_notifier_sends_nothing(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=0)
        send = build_escalation_notifier(notifier)

        await send("Note A", "Note B", "reason", "gap123")

        bot.send_message.assert_not_called()


class TestHandleContradictionCallback:
    async def test_acknowledge_edits_message_and_answers(self) -> None:
        callback = AsyncMock()
        callback.data = "contradiction_ack:gap123"
        callback.message = AsyncMock()

        await handle_contradiction_callback(callback)

        callback.message.edit_text.assert_called_once()
        callback.answer.assert_called_once()


class TestNotifierSendWithKeyboard:
    async def test_disabled_when_chat_id_zero(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=0)
        keyboard = MagicMock()
        await notifier.send_with_keyboard("hello", keyboard)
        bot.send_message.assert_not_called()

    async def test_sends_with_reply_markup(self) -> None:
        bot = AsyncMock()
        notifier = Notifier(bot=bot, chat_id=12345)
        keyboard = MagicMock()
        await notifier.send_with_keyboard("hello", keyboard)
        bot.send_message.assert_called_once_with(
            chat_id=12345, text="hello", parse_mode="Markdown", reply_markup=keyboard
        )

    async def test_handles_exception(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = RuntimeError("network error")
        notifier = Notifier(bot=bot, chat_id=12345)
        keyboard = MagicMock()
        await notifier.send_with_keyboard("hello", keyboard)  # should not raise


class TestMintGapErrorHandling:
    async def test_gap_store_none_returns_empty_string(self, tmp_path: Path) -> None:
        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True)
        on_escalate = AsyncMock()
        detector = _make_detector(
            tmp_path, llm, candidate, match, gap_store=None, on_escalate=on_escalate
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        on_escalate.assert_called_once()
        _, _, _, gap_id = on_escalate.call_args[0]
        assert gap_id == ""


# ---------------------------------------------------------------------------
# Review queue integration (M7 PR-2) — routes resolution/escalation through
# services.review_queue.ReviewQueue instead of applying/notifying directly.
# ---------------------------------------------------------------------------


class TestReviewQueueIntegration:
    async def test_temporal_resolution_routes_through_queue_as_auto(self, tmp_path: Path) -> None:
        from vaultmind.services.review_queue import Lane, ProposalKind, ProposalStatus, ReviewQueue

        note = _note(modified=datetime(2026, 2, 1))  # more recent -> new wins
        candidate = _note("candidate.md", "Candidate Note", modified=datetime(2026, 1, 1))
        match = _match()
        llm = _make_llm(True)
        queue = ReviewQueue(tmp_path / "queue.db")
        detector = _make_detector(
            tmp_path, llm, candidate, match, auto_resolve=True, review_queue=queue
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        # AUTO-lane proposals apply immediately and leave nothing pending —
        # inspect via a fresh query keyed by kind instead.
        all_rows = queue._conn.execute("SELECT * FROM review_proposals").fetchall()
        assert len(all_rows) == 1
        assert all_rows[0]["kind"] == ProposalKind.CONTRADICTION_RESOLUTION.value
        assert all_rows[0]["lane"] == Lane.AUTO.name
        assert all_rows[0]["status"] == ProposalStatus.APPLIED.value

        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" in content

    async def test_escalation_routes_through_queue_as_block(self, tmp_path: Path) -> None:
        from vaultmind.services.review_queue import Lane, ProposalKind, ReviewQueue

        note = _note()
        candidate = _note("candidate.md", "Candidate Note")
        match = _match()
        llm = _make_llm(True, reasoning="different facts")
        gap_store = GapStore(tmp_path / "gaps.db")
        on_escalate = AsyncMock()
        queue = ReviewQueue(tmp_path / "queue.db")
        detector = _make_detector(
            tmp_path,
            llm,
            candidate,
            match,
            auto_resolve=False,
            gap_store=gap_store,
            on_escalate=on_escalate,
            review_queue=queue,
        )

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        pending = queue.list_pending(lane=Lane.BLOCK)
        assert len(pending) == 1
        assert pending[0].kind is ProposalKind.CONTRADICTION_ESCALATION
        # Escalation still fires the existing Telegram notify + gap mint —
        # queue routing is additive, not a replacement for the notify path.
        on_escalate.assert_called_once()
        assert len(gap_store.list_open()) == 1
        gap_store.close()

    async def test_without_review_queue_behavior_is_unchanged(self, tmp_path: Path) -> None:
        """No `review_queue` passed (the default) -> identical to pre-M7
        behavior: direct `mark_contradicted` / direct escalate notify."""
        note = _note(modified=datetime(2026, 2, 1))
        candidate = _note("candidate.md", "Candidate Note", modified=datetime(2026, 1, 1))
        match = _match()
        llm = _make_llm(True)
        detector = _make_detector(tmp_path, llm, candidate, match, auto_resolve=True)

        await detector.on_note_changed(NoteCreatedEvent(path=tmp_path / "new.md", note=note))

        content = (tmp_path / "candidate.md").read_text(encoding="utf-8")
        assert "contradicted_by" in content
