"""Tests for the unified review queue + graduated autonomy Lane model (M7)."""

from __future__ import annotations

import json
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
        assert "no applier" in proposal.lane_reason

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
        assert "fixed lane" in proposal.lane_reason

    def test_applier_exception_marks_rejected(self, tmp_path: Path) -> None:
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
        assert proposal.status is ProposalStatus.REJECTED

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
