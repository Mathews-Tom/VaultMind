"""Tests for the deterministic contradiction-resolution policy (M6 PR-2)."""

from __future__ import annotations

from datetime import datetime

from vaultmind.contradiction.policy import Resolution, resolve_conflict
from vaultmind.vault.models import Note


class FakeRankingConfig:
    """Minimal RankingConfig stand-in for authority_default overrides."""

    def __init__(self, authority_default: int = 3) -> None:
        self.authority_default = authority_default


def _note(
    path: str = "note.md",
    modified: datetime = datetime(2026, 1, 1),
    authority: int = 0,
) -> Note:
    return Note(
        path=path,
        title=path,
        content="body",
        modified=modified,
        authority=authority,
    )


class TestTemporalPrecedence:
    def test_more_recently_modified_new_note_wins(self) -> None:
        new = _note("new.md", modified=datetime(2026, 2, 1))
        candidate = _note("old.md", modified=datetime(2026, 1, 1))

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.TEMPORAL
        assert outcome.winner == "new"
        assert not outcome.escalated

    def test_more_recently_modified_candidate_wins(self) -> None:
        new = _note("new.md", modified=datetime(2026, 1, 1))
        candidate = _note("old.md", modified=datetime(2026, 2, 1))

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.TEMPORAL
        assert outcome.winner == "candidate"

    def test_temporal_wins_even_over_higher_candidate_authority(self) -> None:
        new = _note("new.md", modified=datetime(2026, 2, 1), authority=1)
        candidate = _note("old.md", modified=datetime(2026, 1, 1), authority=5)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.TEMPORAL
        assert outcome.winner == "new"


class TestAuthorityPrecedence:
    def test_higher_authority_new_note_wins_when_temporal_tied(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=5)
        candidate = _note("old.md", modified=same_time, authority=2)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.AUTHORITY
        assert outcome.winner == "new"

    def test_higher_authority_candidate_wins_when_temporal_tied(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=2)
        candidate = _note("old.md", modified=same_time, authority=5)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.AUTHORITY
        assert outcome.winner == "candidate"

    def test_unstamped_authority_normalizes_to_neutral_default(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=0)  # unstamped
        candidate = _note("old.md", modified=same_time, authority=4)

        outcome = resolve_conflict(new, candidate)

        # Default level 3 < candidate's stamped 4 -> candidate wins on authority
        assert outcome.resolution == Resolution.AUTHORITY
        assert outcome.winner == "candidate"

    def test_respects_ranking_config_authority_default_override(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=0)  # unstamped
        candidate = _note("old.md", modified=same_time, authority=4)
        config = FakeRankingConfig(authority_default=5)

        outcome = resolve_conflict(new, candidate, ranking_config=config)

        # Default level 5 > candidate's stamped 4 -> new wins on authority
        assert outcome.resolution == Resolution.AUTHORITY
        assert outcome.winner == "new"

    def test_out_of_range_authority_normalizes_to_default(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=99)  # invalid, treated as unstamped
        candidate = _note("old.md", modified=same_time, authority=4)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.AUTHORITY
        assert outcome.winner == "candidate"


class TestEscalation:
    def test_equal_temporal_and_equal_authority_escalates(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=3)
        candidate = _note("old.md", modified=same_time, authority=3)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.ESCALATE
        assert outcome.winner is None
        assert outcome.escalated is True

    def test_equal_unstamped_authority_escalates(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time, authority=0)
        candidate = _note("old.md", modified=same_time, authority=0)

        outcome = resolve_conflict(new, candidate)

        assert outcome.resolution == Resolution.ESCALATE


class TestResolutionOutcomeDataclass:
    def test_rationale_is_populated_for_every_resolution(self) -> None:
        same_time = datetime(2026, 1, 1)
        new = _note("new.md", modified=same_time)
        candidate = _note("old.md", modified=same_time)

        outcome = resolve_conflict(new, candidate)

        assert outcome.rationale
