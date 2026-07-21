"""Deterministic contradiction-resolution policy.

Detection is not resolution (`contradiction.detection`). Once a material
conflict is found, this module decides the winner with a fixed precedence so
the common cases are automatic and only genuine ambiguity reaches a human —
ported from Kosha's `contradiction/policy.py` shape:

1. **Temporal first.** The more recently modified note wins — a later edit is
   treated as the newer assertion, regardless of which note is older overall.
2. **Authority next.** Applies only when both notes were modified at the same
   time (temporal doesn't distinguish them). The higher stamped `authority`
   level (M2, normalized to the configured neutral default when unstamped)
   wins.
3. **Escalate the rest.** Equal modification time and equal authority is
   genuine ambiguity — it goes to human review, not an automatic winner.

This module is the decision only; applying it (marking the loser, never
deleting) is `contradiction.marking`, invoked from the escalation/detector
orchestrator (`contradiction.detector`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from vaultmind.indexer.ranking import DEFAULT_AUTHORITY_LEVEL

if TYPE_CHECKING:
    from vaultmind.vault.models import Note

Winner = Literal["new", "candidate"]


class Resolution(StrEnum):
    """Which rule resolved a conflict (or that it could not be resolved)."""

    TEMPORAL = "temporal"
    AUTHORITY = "authority"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class ResolutionOutcome:
    """The policy's decision for one detected conflict between two notes."""

    resolution: Resolution
    winner: Winner | None
    rationale: str

    @property
    def escalated(self) -> bool:
        """Whether the conflict needs human judgment."""
        return self.resolution is Resolution.ESCALATE


def _normalized_authority(authority: int, ranking_config: object = None) -> int:
    """Resolve a note's authority to a comparable level, substituting the
    configured neutral default for missing/unstamped (0) or out-of-range
    values — never raises, mirrors `indexer.ranking.authority_multiplier`'s
    own back-compat substitution without depending on ranking weights.
    """
    default_level = DEFAULT_AUTHORITY_LEVEL
    if ranking_config is not None:
        default_level = getattr(ranking_config, "authority_default", DEFAULT_AUTHORITY_LEVEL)
    if 1 <= authority <= 5:
        return authority
    return int(default_level)


def resolve_conflict(
    new_note: Note,
    candidate_note: Note,
    *,
    ranking_config: object = None,
) -> ResolutionOutcome:
    """Resolve a detected conflict between `new_note` and `candidate_note`.

    `new_note` is the note whose creation/modification triggered detection;
    `candidate_note` is the pre-existing merge-band match it conflicts with.
    Temporal precedence (`Note.modified`) is checked first and is absolute —
    the more recently modified note wins even if the other carries higher
    authority, mirroring Kosha's own "a dated version change is a new
    assertion, not a competing one" precedent. Authority breaks ties only
    when neither note is more recently modified; equal authority with equal
    modification time escalates.
    """
    if new_note.modified != candidate_note.modified:
        winner: Winner = "new" if new_note.modified > candidate_note.modified else "candidate"
        return ResolutionOutcome(
            Resolution.TEMPORAL,
            winner,
            f"more recently modified note ({winner}) supersedes the older assertion",
        )

    new_level = _normalized_authority(new_note.authority, ranking_config)
    candidate_level = _normalized_authority(candidate_note.authority, ranking_config)
    if new_level != candidate_level:
        winner = "new" if new_level > candidate_level else "candidate"
        return ResolutionOutcome(
            Resolution.AUTHORITY,
            winner,
            f"source authority {new_level} vs {candidate_level}: {winner} note wins",
        )

    return ResolutionOutcome(
        Resolution.ESCALATE,
        None,
        f"equal modification time and equal authority ({new_level}); needs human judgment",
    )


__all__ = [
    "Resolution",
    "ResolutionOutcome",
    "Winner",
    "resolve_conflict",
]
