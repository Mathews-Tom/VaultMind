"""Tests for the knowledge gap ledger — GapStore model + storage (M5)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from vaultmind.memory.gaps import GapKind, GapStatus, GapStore, normalize_question

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, stale_after_days: int = 30) -> GapStore:
    return GapStore(tmp_path / "gaps.db", stale_after_days=stale_after_days)


# ---------------------------------------------------------------------------
# GapKind / GapStatus schema
# ---------------------------------------------------------------------------


class TestGapEnumsModel:
    def test_model_kind_values_match_source_schema(self) -> None:
        assert {k.value for k in GapKind} == {
            "unanswered_question",
            "weak_retrieval",
            "contradiction_escalated",
            "stale_claim",
        }

    def test_model_status_values_match_source_schema(self) -> None:
        assert {s.value for s in GapStatus} == {"open", "answered", "invalidated", "stale"}


# ---------------------------------------------------------------------------
# Normalization + dedup key stability
# ---------------------------------------------------------------------------


class TestNormalizeQuestionModel:
    def test_model_lowercases_and_strips(self) -> None:
        assert normalize_question("  What is Kosha?  ") == "what is kosha"

    def test_model_collapses_internal_whitespace(self) -> None:
        assert normalize_question("what   is\tkosha") == "what is kosha"

    def test_model_strips_trailing_punctuation(self) -> None:
        assert normalize_question("What is Kosha???") == "what is kosha"
        assert normalize_question("What is Kosha.") == "what is kosha"


class TestDedupKeyStabilityModel:
    def test_model_same_kind_and_question_variants_dedup_to_one_row(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        first = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        second = store.mint("  what is kosha  ", GapKind.UNANSWERED_QUESTION)
        assert first.gap_id == second.gap_id
        assert len(store.list_open()) == 1
        store.close()

    def test_model_different_kind_same_question_is_a_separate_gap(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        a = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        b = store.mint("What is Kosha?", GapKind.WEAK_RETRIEVAL)
        assert a.gap_id != b.gap_id
        assert len(store.list_open()) == 2
        store.close()

    def test_model_different_question_is_a_separate_gap(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        a = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        b = store.mint("What is VaultMind?", GapKind.UNANSWERED_QUESTION)
        assert a.gap_id != b.gap_id
        store.close()


# ---------------------------------------------------------------------------
# mint() — creation, re-ask dedup, reopening
# ---------------------------------------------------------------------------


class TestMintModel:
    def test_model_mints_exactly_one_gap_on_first_ask(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION, evidence_ref="ref-1")
        assert gap.status == GapStatus.OPEN
        assert gap.occurrence_count == 1
        assert gap.evidence_ref == "ref-1"
        assert gap.resolution_ref == ""
        store.close()

    def test_model_reask_increments_occurrence_and_touches_last_seen(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        first = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        second = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        assert second.occurrence_count == 2
        assert second.last_seen >= first.last_seen
        store.close()

    def test_model_reasking_a_stale_gap_reopens_it(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, stale_after_days=1)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        old = (datetime.now() - timedelta(days=5)).isoformat()
        store._conn.execute("UPDATE gaps SET last_seen = ? WHERE gap_id = ?", (old, gap.gap_id))
        store._conn.commit()
        assert store.list_open() == []  # lazily staled by list_open()
        reopened = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        assert reopened.status == GapStatus.OPEN
        assert len(store.list_open()) == 1
        store.close()


# ---------------------------------------------------------------------------
# answer()
# ---------------------------------------------------------------------------


class TestAnswerModel:
    def test_model_answer_transitions_status_and_sets_resolution_ref(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        closed = store.answer(gap.gap_id, "research/kosha/summary.md")
        assert closed is True
        updated = store.get(gap.gap_id)
        assert updated is not None
        assert updated.status == GapStatus.ANSWERED
        assert updated.resolution_ref == "research/kosha/summary.md"
        assert updated.resolved is not None
        store.close()

    def test_model_answer_returns_false_for_unknown_gap(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.answer("nonexistent", "ref") is False
        store.close()

    def test_model_answer_returns_false_when_already_answered(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        store.answer(gap.gap_id, "ref-1")
        assert store.answer(gap.gap_id, "ref-2") is False
        store.close()

    def test_model_answered_gap_excluded_from_list_open(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        store.answer(gap.gap_id, "ref")
        assert store.list_open() == []
        store.close()


# ---------------------------------------------------------------------------
# list_open() ordering + limit
# ---------------------------------------------------------------------------


class TestListOpenModel:
    def test_model_orders_oldest_first(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.mint("First question?", GapKind.UNANSWERED_QUESTION)
        store.mint("Second question?", GapKind.UNANSWERED_QUESTION)
        gaps = store.list_open()
        assert [g.question for g in gaps] == ["First question?", "Second question?"]
        store.close()

    def test_model_respects_limit(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for i in range(5):
            store.mint(f"Question {i}?", GapKind.UNANSWERED_QUESTION)
        assert len(store.list_open(limit=2)) == 2
        store.close()


# ---------------------------------------------------------------------------
# find_open_by_question() — used by the research-closes-gap flow
# ---------------------------------------------------------------------------


class TestFindOpenByQuestionModel:
    def test_model_finds_matching_open_gap_regardless_of_kind(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.mint("What is Kosha?", GapKind.WEAK_RETRIEVAL)
        found = store.find_open_by_question("what is kosha")
        assert found is not None
        assert found.question == "What is Kosha?"
        store.close()

    def test_model_returns_none_when_no_match(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.find_open_by_question("nothing here") is None
        store.close()

    def test_model_excludes_answered_gaps(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        gap = store.mint("What is Kosha?", GapKind.WEAK_RETRIEVAL)
        store.answer(gap.gap_id, "ref")
        assert store.find_open_by_question("What is Kosha?") is None
        store.close()

    def test_model_includes_stale_gaps(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, stale_after_days=1)
        gap = store.mint("What is Kosha?", GapKind.WEAK_RETRIEVAL)
        old = (datetime.now() - timedelta(days=5)).isoformat()
        store._conn.execute("UPDATE gaps SET last_seen = ? WHERE gap_id = ?", (old, gap.gap_id))
        store._conn.commit()
        store.list_open()  # trigger lazy staleness transition
        found = store.find_open_by_question("What is Kosha?")
        assert found is not None
        assert found.status == GapStatus.STALE
        store.close()


# ---------------------------------------------------------------------------
# Auto-stale aging (lazy, applied by list_open())
# ---------------------------------------------------------------------------


class TestStalenessModel:
    def test_model_untouched_gap_past_window_auto_stales(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, stale_after_days=7)
        gap = store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        old = (datetime.now() - timedelta(days=10)).isoformat()
        store._conn.execute("UPDATE gaps SET last_seen = ? WHERE gap_id = ?", (old, gap.gap_id))
        store._conn.commit()
        assert store.list_open() == []
        staled = store.get(gap.gap_id)
        assert staled is not None
        assert staled.status == GapStatus.STALE
        store.close()

    def test_model_gap_within_window_stays_open(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, stale_after_days=7)
        store.mint("What is Kosha?", GapKind.UNANSWERED_QUESTION)
        assert len(store.list_open()) == 1
        store.close()
