"""Tests for the unified review queue + graduated autonomy Lane model (M7)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from vaultmind.indexer.duplicate_detector import DuplicateMatch, MatchType
from vaultmind.services.review_queue import (
    DEFAULT_THRESHOLDS,
    AutonomyThresholds,
    FatigueStats,
    Impact,
    Lane,
    ProposalKind,
    ProposalStatus,
    ReviewQueue,
    migrate_quarantine,
    mint_duplicate_proposals,
    route,
)
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Lane routing (route())
# ---------------------------------------------------------------------------


class TestLaneRouting:
    def test_lane_high_confidence_low_impact_routes_auto(self) -> None:
        lane, reason = route(0.95, Impact.LOW)
        assert lane is Lane.AUTO
        assert "confidence" in reason or "high" in reason

    def test_lane_medium_impact_routes_skim_even_at_high_confidence(self) -> None:
        lane, _ = route(0.99, Impact.MEDIUM)
        assert lane is Lane.SKIM

    def test_lane_high_impact_routes_block_even_at_high_confidence(self) -> None:
        lane, _ = route(0.99, Impact.HIGH)
        assert lane is Lane.BLOCK

    def test_lane_low_confidence_routes_block(self) -> None:
        lane, _ = route(0.1, Impact.LOW)
        assert lane is Lane.BLOCK

    def test_lane_mid_confidence_low_impact_routes_skim(self) -> None:
        lane, _ = route(0.6, Impact.LOW)
        assert lane is Lane.SKIM

    def test_lane_force_block_overrides_everything(self) -> None:
        thresholds = AutonomyThresholds(force_block=True)
        lane, reason = route(1.0, Impact.LOW, thresholds)
        assert lane is Lane.BLOCK
        assert "force" in reason

    def test_lane_thresholds_are_tunable(self) -> None:
        strict = AutonomyThresholds(block_below=0.8, skim_below=0.95)
        lane, _ = route(0.85, Impact.LOW, strict)
        assert lane is Lane.SKIM  # would have been AUTO under defaults

    def test_lane_boundary_values_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="0 <= block_below <= skim_below <= 1"):
            AutonomyThresholds(block_below=0.9, skim_below=0.4)

    def test_default_thresholds_match_kosha_precedent(self) -> None:
        assert DEFAULT_THRESHOLDS.block_below == 0.4
        assert DEFAULT_THRESHOLDS.skim_below == 0.9
        assert DEFAULT_THRESHOLDS.force_block is False

    def test_lane_label_property(self) -> None:
        assert Lane.AUTO.label == "auto"
        assert Lane.SKIM.label == "skim"
        assert Lane.BLOCK.label == "block"

    def test_lane_ordering_matches_attention_demand(self) -> None:
        assert Lane.AUTO < Lane.SKIM < Lane.BLOCK


# ---------------------------------------------------------------------------
# ReviewQueue.propose() — routing + apply/acknowledge semantics
# ---------------------------------------------------------------------------


class TestReviewQueuePropose:
    def test_lane_auto_with_applier_applies_immediately(self, tmp_path: Path) -> None:
        applied: list[dict[str, object]] = []

        def _apply(payload: dict[str, object]) -> str:
            applied.append(payload)
            return "applied ok"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _apply})
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION,
            confidence=1.0,
            impact=Impact.LOW,
            summary="apply known tags",
            payload={"note_path": "a.md", "tags": ["known"]},
        )
        assert proposal.lane is Lane.AUTO
        assert proposal.status is ProposalStatus.APPLIED
        assert proposal.result == "applied ok"
        assert applied == [{"note_path": "a.md", "tags": ["known"]}]

    def test_lane_skim_with_applier_stays_pending(self, tmp_path: Path) -> None:
        def _apply(payload: dict[str, object]) -> str:
            return "applied"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _apply})
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION,
            confidence=0.6,
            impact=Impact.LOW,
            summary="apply mixed tags",
            payload={"note_path": "b.md", "tags": ["novel"]},
        )
        assert proposal.lane is Lane.SKIM
        assert proposal.status is ProposalStatus.PENDING

    def test_kind_with_no_applier_never_routes_auto(self, tmp_path: Path) -> None:
        """A kind with no registered applier is capped at SKIM even when
        confidence/impact would otherwise route AUTO."""
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.99,
            impact=Impact.LOW,
            summary="duplicate candidate",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        assert proposal.lane is Lane.SKIM

    def test_kind_with_no_applier_approve_acknowledges_not_applies(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.HIGH,
            summary="duplicate candidate",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        assert proposal.status is ProposalStatus.PENDING
        result = queue.approve(proposal.proposal_id)
        assert result is not None
        assert result.status is ProposalStatus.ACKNOWLEDGED
        assert result.result == ""

    def test_lane_override_bypasses_confidence_routing(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=1.0,
            impact=Impact.LOW,
            summary="escalated conflict",
            payload={"gap_id": "abc"},
            lane_override=Lane.BLOCK,
        )
        assert proposal.lane is Lane.BLOCK

    def test_applier_exception_marks_failed_with_typed_error(self, tmp_path: Path) -> None:
        def _boom(payload: dict[str, object]) -> str:
            raise RuntimeError("write failed")

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _boom})
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION,
            confidence=1.0,
            impact=Impact.LOW,
            summary="apply tags",
            payload={"note_path": "c.md", "tags": ["x"]},
        )
        assert proposal.status is ProposalStatus.FAILED
        assert proposal.result == "RuntimeError: write failed"

    def test_mutation_kind_without_applier_fails_never_acknowledges(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.TAG_VOCABULARY,
            confidence=0.6,
            impact=Impact.LOW,
            summary="vocab",
            payload={"tag": "x"},
        )
        result = queue.approve(proposal.proposal_id)
        assert result is not None
        assert result.status is ProposalStatus.FAILED
        assert "tag_vocabulary" in result.result

    def test_repeat_identical_proposal_dedupes_and_bumps_occurrence(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        first = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.HIGH,
            summary="dup",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        second = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.HIGH,
            summary="dup",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        assert first.proposal_id == second.proposal_id
        assert second.occurrence_count == 2


# ---------------------------------------------------------------------------
# Human decisions — approve / reject / approve_all
# ---------------------------------------------------------------------------


class TestReviewQueueDecisions:
    def test_reject_pending_proposal(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.HIGH,
            summary="dup",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        assert queue.reject(proposal.proposal_id, reason="not a duplicate") is True
        row = queue.get(proposal.proposal_id)
        assert row is not None
        assert row.status is ProposalStatus.REJECTED
        assert row.result == "not a duplicate"

    def test_reject_already_resolved_returns_false(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        proposal = queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.HIGH,
            summary="dup",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        queue.reject(proposal.proposal_id)
        assert queue.reject(proposal.proposal_id) is False

    def test_approve_nonexistent_returns_none(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        assert queue.approve("does-not-exist") is None

    def test_approve_all_applies_every_pending_in_lane(self, tmp_path: Path) -> None:
        applied_paths: list[str] = []

        def _apply(payload: dict[str, object]) -> str:
            applied_paths.append(str(payload["note_path"]))
            return "ok"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _apply})
        for i in range(3):
            queue.propose(
                ProposalKind.TAG_APPLICATION,
                confidence=0.6,
                impact=Impact.LOW,
                summary=f"apply tags {i}",
                payload={"note_path": f"note{i}.md", "tags": ["novel"]},
            )
        results = queue.approve_all(lane=Lane.SKIM)
        assert len(results) == 3
        assert all(r.status is ProposalStatus.APPLIED for r in results)
        assert len(applied_paths) == 3
        assert queue.list_pending(lane=Lane.SKIM) == []

    def test_list_pending_filters_by_lane(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.MEDIUM,
            summary="dup a",
            payload={"source_path": "a.md", "match_path": "b.md"},
            lane_override=Lane.SKIM,
        )
        queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=0.0,
            impact=Impact.HIGH,
            summary="conflict",
            payload={"gap_id": "g1"},
            lane_override=Lane.BLOCK,
        )
        assert len(queue.list_pending(lane=Lane.SKIM)) == 1
        assert len(queue.list_pending(lane=Lane.BLOCK)) == 1
        assert len(queue.list_pending()) == 2


# ---------------------------------------------------------------------------
# Approval-fatigue metric
# ---------------------------------------------------------------------------


class TestFatigueStats:
    def test_fatigue_rate_counts_skim_and_block_as_reaching_human(self, tmp_path: Path) -> None:
        def _apply(payload: dict[str, object]) -> str:
            return "ok"

        queue = ReviewQueue(tmp_path / "queue.db", appliers={ProposalKind.TAG_APPLICATION: _apply})
        queue.propose(
            ProposalKind.TAG_APPLICATION,
            confidence=1.0,
            impact=Impact.LOW,
            summary="auto",
            payload={"note_path": "a.md", "tags": ["x"]},
        )
        queue.propose(
            ProposalKind.TAG_APPLICATION,
            confidence=0.6,
            impact=Impact.LOW,
            summary="skim",
            payload={"note_path": "b.md", "tags": ["y"]},
        )
        queue.propose(
            ProposalKind.CONTRADICTION_ESCALATION,
            confidence=0.0,
            impact=Impact.HIGH,
            summary="block",
            payload={"gap_id": "g2"},
            lane_override=Lane.BLOCK,
        )
        stats = queue.fatigue_stats()
        assert isinstance(stats, FatigueStats)
        assert stats.total == 3
        assert stats.auto_count == 1
        assert stats.skim_count == 1
        assert stats.block_count == 1
        assert stats.fatigue_rate == pytest.approx(2 / 3)

    def test_fatigue_rate_zero_proposals_is_zero(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        stats = queue.fatigue_stats()
        assert stats.total == 0
        assert stats.fatigue_rate == 0.0


# ---------------------------------------------------------------------------
# migrate_quarantine() — one-time tag-quarantine import (PR-2)
# ---------------------------------------------------------------------------


class TestMigrateQuarantine:
    def test_migration_moves_pending_tags_into_skim_proposals(self, tmp_path: Path) -> None:
        quarantine_path = tmp_path / "tag_quarantine.json"
        quarantine_path.write_text(
            json.dumps({"approved": ["existing"], "quarantined": ["novel-a", "novel-b"]})
        )
        queue = ReviewQueue(tmp_path / "queue.db")

        migrated = migrate_quarantine(queue, quarantine_path)

        assert migrated == 2
        pending = queue.list_pending(lane=Lane.SKIM)
        assert len(pending) == 2
        assert {p.kind for p in pending} == {ProposalKind.TAG_VOCABULARY}
        assert {p.payload["tag"] for p in pending} == {"novel-a", "novel-b"}

    def test_migration_empties_quarantined_list_leaves_approved_untouched(
        self, tmp_path: Path
    ) -> None:
        quarantine_path = tmp_path / "tag_quarantine.json"
        quarantine_path.write_text(
            json.dumps({"approved": ["existing"], "quarantined": ["novel-a"]})
        )
        queue = ReviewQueue(tmp_path / "queue.db")

        migrate_quarantine(queue, quarantine_path)

        data = json.loads(quarantine_path.read_text())
        assert data["quarantined"] == []
        assert data["approved"] == ["existing"]

    def test_migration_of_missing_file_is_a_noop(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")
        migrated = migrate_quarantine(queue, tmp_path / "does-not-exist.json")
        assert migrated == 0

    def test_migration_of_already_emptied_file_is_a_noop(self, tmp_path: Path) -> None:
        quarantine_path = tmp_path / "tag_quarantine.json"
        quarantine_path.write_text(json.dumps({"approved": ["existing"], "quarantined": []}))
        queue = ReviewQueue(tmp_path / "queue.db")

        migrated = migrate_quarantine(queue, quarantine_path)
        assert migrated == 0

    def test_migration_is_idempotent_across_repeated_runs(self, tmp_path: Path) -> None:
        quarantine_path = tmp_path / "tag_quarantine.json"
        quarantine_path.write_text(json.dumps({"approved": [], "quarantined": ["novel-a"]}))
        queue = ReviewQueue(tmp_path / "queue.db")

        first = migrate_quarantine(queue, quarantine_path)
        second = migrate_quarantine(queue, quarantine_path)

        assert first == 1
        assert second == 0
        assert len(queue.list_pending()) == 1


# ---------------------------------------------------------------------------
# mint_duplicate_proposals() — duplicate-detector bridge (PR-2)
# ---------------------------------------------------------------------------


class TestMintDuplicateProposals:
    def test_merge_band_matches_become_skim_proposals(self, tmp_path: Path) -> None:
        note = Note(path=Path("a.md"), title="A", content="body")
        match = DuplicateMatch(
            source_path="a.md",
            source_title="A",
            match_path="b.md",
            match_title="B",
            similarity=0.85,
            match_type=MatchType.MERGE,
        )
        queue = ReviewQueue(tmp_path / "queue.db")

        proposals = mint_duplicate_proposals(queue, note, [match])

        assert len(proposals) == 1
        assert proposals[0].kind is ProposalKind.DUPLICATE_MERGE
        assert proposals[0].lane is Lane.SKIM
        assert proposals[0].status is ProposalStatus.PENDING

    def test_duplicate_band_matches_are_ignored(self, tmp_path: Path) -> None:
        note = Note(path=Path("a.md"), title="A", content="body")
        match = DuplicateMatch(
            source_path="a.md",
            source_title="A",
            match_path="b.md",
            match_title="B",
            similarity=0.98,
            match_type=MatchType.DUPLICATE,
        )
        queue = ReviewQueue(tmp_path / "queue.db")

        proposals = mint_duplicate_proposals(queue, note, [match])

        assert proposals == []
        assert queue.list_pending() == []

    def test_approving_a_duplicate_proposal_acknowledges_never_applies(
        self, tmp_path: Path
    ) -> None:
        note = Note(path=Path("a.md"), title="A", content="body")
        match = DuplicateMatch(
            source_path="a.md",
            source_title="A",
            match_path="b.md",
            match_title="B",
            similarity=0.85,
            match_type=MatchType.MERGE,
        )
        queue = ReviewQueue(tmp_path / "queue.db")
        proposals = mint_duplicate_proposals(queue, note, [match])

        result = queue.approve(proposals[0].proposal_id)
        assert result is not None
        assert result.status is ProposalStatus.ACKNOWLEDGED


# ---------------------------------------------------------------------------
# DigestGenerator SKIM-lane section (PR-3)
# ---------------------------------------------------------------------------


class _FakeDigestStore:
    def search(
        self, query: str, n_results: int = 5, where: dict[str, str] | None = None
    ) -> list[dict[str, str]]:
        return []


class _FakeDigestGraph:
    def __init__(self) -> None:
        import networkx as nx

        self._graph = nx.DiGraph()

    @property
    def stats(self) -> dict[str, int]:
        return {"nodes": 0, "edges": 0}


class _FakeDigestParser:
    def iter_notes(self) -> list[Note]:
        return []


class _FakeDigestConfig:
    period_days = 7
    max_trending = 10
    max_suggestions = 5
    connection_threshold_low = 0.70
    connection_threshold_high = 0.85
    inbox_folder = "00-inbox"
    inbox_age_warning_days = 7
    max_inbox_shown = 10


class TestDigestReviewQueueLifecycle:
    def test_report_carries_skim_pending_count_and_summaries(self, tmp_path: Path) -> None:
        from vaultmind.indexer.digest import DigestGenerator

        queue = ReviewQueue(tmp_path / "queue.db")
        queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.MEDIUM,
            summary="Merge candidate: 'A' ~ 'B' (85% similar)",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        generator = DigestGenerator(
            store=_FakeDigestStore(),
            graph=_FakeDigestGraph(),
            parser=_FakeDigestParser(),
            config=_FakeDigestConfig(),
            review_queue=queue,
        )
        report = generator.generate()
        assert report.skim_pending_count == 1
        assert "Merge candidate" in report.skim_pending[0]

    def test_telegram_format_includes_pending_review_section(self, tmp_path: Path) -> None:
        from vaultmind.indexer.digest import DigestGenerator

        queue = ReviewQueue(tmp_path / "queue.db")
        queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.MEDIUM,
            summary="Merge candidate: 'A' ~ 'B'",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        generator = DigestGenerator(
            store=_FakeDigestStore(),
            graph=_FakeDigestGraph(),
            parser=_FakeDigestParser(),
            config=_FakeDigestConfig(),
            review_queue=queue,
        )
        report = generator.generate()
        text = generator.format_telegram(report)
        assert "Pending Review" in text
        assert "1 SKIM item" in text

    def test_markdown_format_includes_pending_review_section(self, tmp_path: Path) -> None:
        from vaultmind.indexer.digest import DigestGenerator

        queue = ReviewQueue(tmp_path / "queue.db")
        queue.propose(
            ProposalKind.DUPLICATE_MERGE,
            confidence=0.85,
            impact=Impact.MEDIUM,
            summary="Merge candidate: 'A' ~ 'B'",
            payload={"source_path": "a.md", "match_path": "b.md"},
        )
        generator = DigestGenerator(
            store=_FakeDigestStore(),
            graph=_FakeDigestGraph(),
            parser=_FakeDigestParser(),
            config=_FakeDigestConfig(),
            review_queue=queue,
        )
        report = generator.generate()
        text = generator._format_markdown(report)
        assert "## Pending Review" in text
        assert "Merge candidate" in text

    def test_no_review_queue_leaves_report_zeroed(self) -> None:
        from vaultmind.indexer.digest import DigestGenerator

        generator = DigestGenerator(
            store=_FakeDigestStore(),
            graph=_FakeDigestGraph(),
            parser=_FakeDigestParser(),
            config=_FakeDigestConfig(),
        )
        report = generator.generate()
        assert report.skim_pending_count == 0
        assert report.skim_pending == []

    def test_no_pending_skim_items_omits_section_from_telegram(self, tmp_path: Path) -> None:
        from vaultmind.indexer.digest import DigestGenerator

        queue = ReviewQueue(tmp_path / "queue.db")
        generator = DigestGenerator(
            store=_FakeDigestStore(),
            graph=_FakeDigestGraph(),
            parser=_FakeDigestParser(),
            config=_FakeDigestConfig(),
            review_queue=queue,
        )
        report = generator.generate()
        text = generator.format_telegram(report)
        assert "Pending Review" not in text


# ---------------------------------------------------------------------------
# FAILED state, explicit retry, legacy migration
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA = """
CREATE TABLE review_proposals (
    proposal_id TEXT PRIMARY KEY, kind TEXT NOT NULL, lane TEXT NOT NULL,
    lane_reason TEXT NOT NULL, confidence REAL NOT NULL, impact TEXT NOT NULL,
    summary TEXT NOT NULL, payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', result TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL, last_seen TEXT NOT NULL, resolved TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1
);
"""

_LEGACY_ROWS: list[tuple[str, str, str, str, str | None, int]] = [
    # (id, kind, status, result, resolved, occurrence_count)
    ("p-pending", "tag_application", "pending", "", None, 1),
    ("p-applied", "tag_application", "applied", "done", "2026-01-02T00:00:00", 2),
    ("p-ack", "duplicate_merge", "acknowledged", "", "2026-01-03T00:00:00", 3),
    ("p-unexecuted", "tag_application", "acknowledged", "", "2026-01-03T01:00:00", 2),
    ("p-ack-result", "tag_application", "acknowledged", "reviewed", "2026-01-03T02:00:00", 1),
    ("p-human", "tag_application", "rejected", "not wanted", "2026-01-04T00:00:00", 1),
    ("p-known", "tag_application", "rejected", "applier raised — see logs", "2026-01-05", 4),
    (
        "p-ambig",
        "tag_application",
        "rejected",
        "applier raised — see logs (human)",
        "2026-01-06",
        1,
    ),
    ("p-empty", "tag_application", "rejected", "", "2026-01-07T00:00:00", 1),
]


def _make_legacy_db(path: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SCHEMA)
    for pid, kind, status, result, resolved, occ in _LEGACY_ROWS:
        conn.execute(
            "INSERT INTO review_proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                pid,
                kind,
                "SKIM",
                "r",
                0.7,
                "low",
                f"s-{pid}",
                json.dumps({"id": pid}),
                status,
                result,
                "2026-01-01T00:00:00",
                "2026-01-01T12:00:00",
                resolved,
                occ,
            ),
        )
    conn.commit()
    conn.close()


class TestFailedStateAndRetry:
    def _failing_queue(self, tmp_path: Path) -> tuple[ReviewQueue, list[dict[str, object]]]:
        calls: list[dict[str, object]] = []
        state = {"fail": True}

        def _applier(payload: dict[str, object]) -> str:
            calls.append(payload)
            if state["fail"]:
                raise OSError("disk full")
            return "ok"

        queue = ReviewQueue(tmp_path / "q.db", appliers={ProposalKind.TAG_APPLICATION: _applier})
        queue._test_state = state  # type: ignore[attr-defined]
        return queue, calls

    def test_failed_is_distinct_from_rejected_and_only_retry_reopens(self, tmp_path: Path) -> None:
        queue, calls = self._failing_queue(tmp_path)
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION, 0.6, Impact.LOW, "t", {"note_path": "a.md"}
        )
        assert proposal.status is ProposalStatus.PENDING
        failed = queue.approve(proposal.proposal_id)
        assert failed is not None and failed.status is ProposalStatus.FAILED
        assert failed.status is not ProposalStatus.REJECTED
        assert failed.result == "OSError: disk full"
        # ordinary approve/reject/approve_all/dedup never retry
        assert queue.approve(proposal.proposal_id) is None
        assert queue.reject(proposal.proposal_id) is False
        assert queue.approve_all(Lane.SKIM) == []
        queue.propose(ProposalKind.TAG_APPLICATION, 0.6, Impact.LOW, "t", {"note_path": "a.md"})
        assert len(calls) == 1
        assert queue.list_pending() == []

    def test_retry_applies_then_terminal_replay_is_inert(self, tmp_path: Path) -> None:
        queue, calls = self._failing_queue(tmp_path)
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION, 0.6, Impact.LOW, "t", {"note_path": "a.md"}
        )
        queue.approve(proposal.proposal_id)
        queue._test_state["fail"] = False  # type: ignore[attr-defined]
        retried = queue.retry(proposal.proposal_id)
        assert retried is not None and retried.status is ProposalStatus.APPLIED
        assert retried.result == "ok"
        assert queue.retry(proposal.proposal_id) is None
        assert queue.approve(proposal.proposal_id) is None
        assert len(calls) == 2

    def test_retry_only_operates_on_failed(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "q.db")
        pending = queue.propose(ProposalKind.DUPLICATE_MERGE, 0.8, Impact.LOW, "d", {"a": 1})
        assert queue.retry(pending.proposal_id) is None
        queue.approve(pending.proposal_id)
        assert queue.retry(pending.proposal_id) is None
        assert queue.retry("missing") is None

    def test_retry_missing_applier_stays_failed_until_registered(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "q.db")
        proposal = queue.propose(ProposalKind.MATURATION_SYNTHESIS, 0.6, Impact.LOW, "m", {"c": 1})
        failed = queue.approve(proposal.proposal_id)
        assert failed is not None and failed.status is ProposalStatus.FAILED
        again = queue.retry(proposal.proposal_id)
        assert again is not None and again.status is ProposalStatus.FAILED
        queue.register_applier(ProposalKind.MATURATION_SYNTHESIS, lambda _p: "synth")
        done = queue.retry(proposal.proposal_id)
        assert done is not None and done.status is ProposalStatus.APPLIED

    def test_informational_kinds_acknowledge_without_applier(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "q.db")
        for kind in (ProposalKind.CONTRADICTION_ESCALATION, ProposalKind.DUPLICATE_MERGE):
            p = queue.propose(kind, 0.5, Impact.HIGH, "i", {"k": kind.value})
            result = queue.approve(p.proposal_id)
            assert result is not None and result.status is ProposalStatus.ACKNOWLEDGED
            assert queue.approve(p.proposal_id) is None


class TestLegacyMigration:
    @pytest.mark.parametrize(
        ("kind", "result", "recover"),
        [
            ("maturation_synthesis", "Synthesis failed: No source notes found on disk", True),
            ("maturation_synthesis", "Synthesis failed: No YAML frontmatter found", True),
            (
                "maturation_synthesis",
                "Synthesis failed: YAML frontmatter parse error: invalid",
                True,
            ),
            ("maturation_synthesis", "Synthesis failed: Unresolved wikilinks: missing", True),
            (
                "maturation_synthesis",
                "Synthesis failed: Unresolved wikilinks: x/synthesis-a-2026-01-01.md",
                False,
            ),
            ("maturation_synthesis", "Synthesis failed: Unresolved wikilinks: missing.md", False),
            (
                "maturation_synthesis",
                "cluster no longer available (dismissed or already synthesized)",
                False,
            ),
            ("maturation_synthesis", "synthesis failed: no yaml frontmatter found", False),
            ("tag_application", "Synthesis failed: No YAML frontmatter found", False),
        ],
    )
    def test_legacy_synthesis_failure_recovery_is_conservative(
        self, tmp_path: Path, kind: str, result: str, recover: bool
    ) -> None:
        import sqlite3

        db = tmp_path / "legacy.db"
        _make_legacy_db(db)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE review_proposals SET kind = ?, result = ? WHERE proposal_id = ?",
                (kind, result, "p-applied"),
            )
        queue = ReviewQueue(db)
        row = queue.get("p-applied")
        assert row is not None
        assert row.status is (ProposalStatus.FAILED if recover else ProposalStatus.APPLIED)
        assert row.result == result
        assert row.legacy_status == ("applied" if recover else None)
        assert row.legacy_result == (result if recover else None)
        queue.close()

        output = tmp_path / "synthesis-recovered.md"

        def apply_synthesis(_payload: dict[str, object]) -> str:
            output.write_text("---\ntype: permanent\n---\nRecovered current synthesis.\n")
            return output.name

        queue = ReviewQueue(db, appliers={ProposalKind.MATURATION_SYNTHESIS: apply_synthesis})
        reopened = queue.get("p-applied")
        assert reopened == row
        assert not output.exists()
        retried = queue.retry("p-applied")
        if recover:
            assert retried is not None and retried.status is ProposalStatus.APPLIED
            assert output.read_text().endswith("Recovered current synthesis.\n")
            assert retried.legacy_status == "applied"
            assert retried.legacy_result == result
        else:
            assert retried is None
            assert not output.exists()
        assert queue.retry("p-applied") is None
        queue.close()

    def test_migration_preserves_rows_and_reclassifies_only_exact_failure(
        self, tmp_path: Path
    ) -> None:
        import shutil

        legacy = tmp_path / "legacy.db"
        _make_legacy_db(legacy)
        copy = tmp_path / "copy.db"
        shutil.copy(legacy, copy)

        queue = ReviewQueue(copy)
        expected_status = {
            "p-pending": ProposalStatus.PENDING,
            "p-applied": ProposalStatus.APPLIED,
            "p-ack": ProposalStatus.ACKNOWLEDGED,
            "p-unexecuted": ProposalStatus.FAILED,
            "p-ack-result": ProposalStatus.ACKNOWLEDGED,
            "p-human": ProposalStatus.REJECTED,
            "p-known": ProposalStatus.FAILED,
            "p-ambig": ProposalStatus.REJECTED,
            "p-empty": ProposalStatus.REJECTED,
        }
        for pid, kind, _status, result, resolved, occ in _LEGACY_ROWS:
            row = queue.get(pid)
            assert row is not None
            assert row.status is expected_status[pid]
            assert row.kind.value == kind
            assert row.payload == {"id": pid}
            assert row.summary == f"s-{pid}"
            assert row.created.isoformat() == "2026-01-01T00:00:00"
            assert row.last_seen.isoformat() == "2026-01-01T12:00:00"
            assert row.result == result
            assert (row.resolved.isoformat() if row.resolved else None) == (
                datetime.fromisoformat(resolved).isoformat() if resolved else None
            )
            assert row.occurrence_count == occ
            if pid in ("p-known", "p-unexecuted"):
                assert row.legacy_status == _status
                assert row.legacy_result == result
            else:
                assert row.legacy_status is None and row.legacy_result is None
        queue.close()

    def test_migration_is_idempotent_and_later_human_reject_not_reclassified(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "legacy.db"
        _make_legacy_db(db)
        ReviewQueue(db).close()
        queue = ReviewQueue(db)
        pending = queue.get("p-pending")
        assert pending is not None
        queue.reject("p-pending", reason="applier raised — see logs")
        queue.close()
        reopened = ReviewQueue(db)
        row = reopened.get("p-pending")
        assert row is not None and row.status is ProposalStatus.REJECTED
        known = reopened.get("p-known")
        assert known is not None and known.legacy_result == "applier raised — see logs"
        assert known.status is ProposalStatus.FAILED

    def test_explicit_retry_after_reopen_of_migrated_failure(self, tmp_path: Path) -> None:
        db = tmp_path / "legacy.db"
        _make_legacy_db(db)
        ReviewQueue(db).close()
        applied: list[dict[str, object]] = []

        def _ok(payload: dict[str, object]) -> str:
            applied.append(payload)
            return "applied now"

        queue = ReviewQueue(db, appliers={ProposalKind.TAG_APPLICATION: _ok})
        assert applied == []  # opening never retries
        assert queue.approve_all(Lane.SKIM)[0].proposal_id == "p-pending"
        assert applied == [{"id": "p-pending"}]
        assert queue.retry("p-human") is None
        retried = queue.retry("p-known")
        assert retried is not None and retried.status is ProposalStatus.APPLIED
        assert retried.result == "applied now"
        assert retried.legacy_status == "rejected"
        assert applied[-1] == {"id": "p-known"}
