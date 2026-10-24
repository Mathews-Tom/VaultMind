"""Unified review queue — graduated autonomy lanes for automated mutation proposals.

Replaces the scattered ad-hoc approval mechanisms (`indexer/auto_tagger.py`'s
tag quarantine, `indexer/duplicate_detector.py`'s cached merge candidates,
`contradiction/detector.py`'s bespoke escalation push) with one SQLite-backed
queue. Every automated mutation proposal is routed into one of three lanes by
confidence x impact, ported directly from Kosha's `approve/autonomy.py` Lane
model (`~/WorkSpace/AetherForge/Kosha/src/kosha/approve/autonomy.py`):

* ``AUTO``  — high confidence, low impact: applies immediately and logs.
* ``SKIM``  — medium confidence or impact: applies (if it has an applier) and
  is surfaced for a batched, unhurried human pass (`vaultmind digest`, `/review`).
* ``BLOCK`` — low confidence or high impact: withheld until a human approves
  or rejects it (Telegram inline keyboard).

Some proposal kinds have no real "apply" action in this codebase today (e.g.
duplicate-merge suggestions — no note-merge execution exists anywhere in
VaultMind). Those kinds are registered with no applier; approving one records
an ``ACKNOWLEDGED`` status, never a false ``APPLIED`` claim of a vault mutation
that did not happen. A handful of kinds bypass confidence-based routing
entirely via a fixed ``lane_override`` when their risk profile is not
genuinely a confidence spectrum (see call sites for the specific reasoning,
recorded in `.docs/DEVELOPMENT_PLAN_HISTORY.md`'s M7 entry).

``ReviewQueue`` itself is a plain synchronous SQLite store (mirrors
`memory/gaps.py::GapStore`) — it never awaits anything. Network delivery of a
Telegram approval keyboard for a newly pending ``BLOCK``-lane proposal is done
by producers via `vaultmind.services.review_delivery.deliver_block`, called
after `propose()` has committed and outside any SQL transaction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.indexer.duplicate_detector import DuplicateMatch
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)


class Lane(IntEnum):
    """Review lanes, ordered by how much human attention they demand."""

    AUTO = 0
    SKIM = 1
    BLOCK = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class ProposalKind(StrEnum):
    """The mutation-proposal call sites this queue unifies."""

    TAG_APPLICATION = "tag_application"
    TAG_VOCABULARY = "tag_vocabulary"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    CONTRADICTION_ESCALATION = "contradiction_escalation"
    DUPLICATE_MERGE = "duplicate_merge"
    MATURATION_SYNTHESIS = "maturation_synthesis"
    SOURCE_INGESTION = "source_ingestion"
    WIKI_PAGE = "wiki_page"


class Impact(StrEnum):
    """How much a proposal, if wrong, would cost to undo or live with."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposalStatus(StrEnum):
    """Lifecycle state of a queued proposal."""

    PENDING = "pending"
    APPLIED = "applied"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    FAILED = "failed"


#: Kinds with no mutation to execute (suggestions only); the sole kinds that may
#: ACKNOWLEDGE without a registered applier.
INFORMATIONAL_KINDS = frozenset(
    {ProposalKind.CONTRADICTION_ESCALATION, ProposalKind.DUPLICATE_MERGE}
)

#: Exact result written by the pre-FAILED queue when an applier raised.
_LEGACY_APPLIER_RAISED = "applier raised — see logs"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AutonomyThresholds:
    """Confidence cutoffs for the routing lanes (ported from Kosha's
    `AutonomyThresholds`, `approve/autonomy.py`).

    ``force_block`` routes every confidence-routed proposal to ``BLOCK``.
    """

    block_below: float = 0.4
    skim_below: float = 0.9
    force_block: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.block_below <= self.skim_below <= 1.0:
            raise ValueError("require 0 <= block_below <= skim_below <= 1")


DEFAULT_THRESHOLDS = AutonomyThresholds()


def route(
    confidence: float,
    impact: Impact,
    thresholds: AutonomyThresholds = DEFAULT_THRESHOLDS,
) -> tuple[Lane, str]:
    """Assign a proposal to a review lane by confidence and impact.

    Direct port of Kosha's `route_change()` decision order, minus the
    secret-detector and `ContradictionState` branches (no analogue here —
    escalated contradictions use a fixed `lane_override` instead, since the
    policy has already decided there is no confidence axis left to arbitrate).
    """
    if thresholds.force_block:
        return Lane.BLOCK, "thresholds force every proposal to block"
    if impact is Impact.HIGH:
        return Lane.BLOCK, "high impact"
    if confidence < thresholds.block_below:
        return Lane.BLOCK, f"confidence {confidence:.2f} < {thresholds.block_below:.2f}"
    if impact is Impact.MEDIUM:
        return Lane.SKIM, "medium impact"
    if confidence < thresholds.skim_below:
        return Lane.SKIM, f"confidence {confidence:.2f} < {thresholds.skim_below:.2f}"
    return Lane.AUTO, "high confidence, low impact"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_proposals (
    proposal_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    lane TEXT NOT NULL,
    lane_reason TEXT NOT NULL,
    confidence REAL NOT NULL,
    impact TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    resolved TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    legacy_status TEXT,
    legacy_result TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_proposals_status ON review_proposals(status);
CREATE INDEX IF NOT EXISTS idx_review_proposals_lane ON review_proposals(lane);
CREATE INDEX IF NOT EXISTS idx_review_proposals_kind ON review_proposals(kind);
"""


def _dedup_key(kind: ProposalKind, payload: Mapping[str, Any]) -> str:
    """Stable idempotency key over `kind` + canonical JSON payload.

    Re-proposing the same mutation (e.g. the same tag suggested again before
    review) touches the existing row instead of growing the table — mirrors
    `memory/gaps.py`'s dedup-key convention.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{kind.value}:{canonical}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewProposal:
    """A single automated mutation proposal and its routing/lifecycle state."""

    proposal_id: str
    kind: ProposalKind
    lane: Lane
    lane_reason: str
    confidence: float
    impact: Impact
    summary: str
    payload: dict[str, Any]
    status: ProposalStatus
    result: str
    created: datetime
    last_seen: datetime
    resolved: datetime | None
    occurrence_count: int
    legacy_status: str | None = None
    legacy_result: str | None = None


def _row_to_proposal(row: sqlite3.Row) -> ReviewProposal:
    return ReviewProposal(
        proposal_id=row["proposal_id"],
        kind=ProposalKind(row["kind"]),
        lane=Lane[row["lane"]],
        lane_reason=row["lane_reason"],
        confidence=row["confidence"],
        impact=Impact(row["impact"]),
        summary=row["summary"],
        payload=json.loads(row["payload"]),
        status=ProposalStatus(row["status"]),
        result=row["result"],
        created=datetime.fromisoformat(row["created"]),
        last_seen=datetime.fromisoformat(row["last_seen"]),
        resolved=datetime.fromisoformat(row["resolved"]) if row["resolved"] else None,
        occurrence_count=row["occurrence_count"],
        legacy_status=row["legacy_status"],
        legacy_result=row["legacy_result"],
    )


@dataclass(frozen=True, slots=True)
class FatigueStats:
    """Approval-fatigue snapshot: how often a human is actually needed."""

    total: int
    auto_count: int
    skim_count: int
    block_count: int

    @property
    def fatigue_rate(self) -> float:
        """Fraction of proposals that reached a human (SKIM + BLOCK)."""
        if self.total == 0:
            return 0.0
        return (self.skim_count + self.block_count) / self.total


Applier = Callable[[dict[str, Any]], str]


class ReviewQueue:
    """SQLite-backed store routing automated mutation proposals into lanes."""

    def __init__(
        self,
        db_path: Path,
        thresholds: AutonomyThresholds = DEFAULT_THRESHOLDS,
        appliers: Mapping[ProposalKind, Applier] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        if read_only:
            self._conn = sqlite3.connect(
                db_path.absolute().as_uri() + "?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if not read_only:
            self._migrate()
        self._thresholds = thresholds
        self._appliers: dict[ProposalKind, Applier] = dict(appliers or {})

    def register_applier(self, kind: ProposalKind, applier: Applier) -> None:
        """Register (or replace) the applier for `kind` on this queue instance.

        Lets independently-constructed subsystems (e.g. `ContradictionDetector`)
        contribute their own applier to a shared `ReviewQueue` built elsewhere
        (e.g. `cli.py`'s `bot` command), without the constructing call site
        needing to know every downstream payload shape.
        """
        self._appliers[kind] = applier

    # ------------------------------------------------------------------
    # Proposal creation + routing
    # ------------------------------------------------------------------

    def propose(
        self,
        kind: ProposalKind,
        confidence: float,
        impact: Impact,
        summary: str,
        payload: dict[str, Any],
        *,
        lane_override: Lane | None = None,
    ) -> ReviewProposal:
        """Route a mutation proposal into a lane and store it.

        AUTO-lane proposals with a registered applier are applied immediately.
        Without an applier, AUTO is capped at SKIM for explicit human review.
        Only informational kinds may be acknowledged; missing mutation operations fail
        on approval rather than pretending the mutation happened.
        """
        if lane_override is not None:
            lane, reason = lane_override, f"fixed lane for {kind.value}"
        else:
            lane, reason = route(confidence, impact, self._thresholds)

        has_applier = kind in self._appliers
        if not has_applier and lane is Lane.AUTO:
            lane, reason = Lane.SKIM, f"{reason} (capped: no applier registered for {kind.value})"

        proposal_id = _dedup_key(kind, payload)
        now = datetime.now()
        existing = self.get(proposal_id)

        if existing is not None:
            self._conn.execute(
                """
                UPDATE review_proposals
                SET last_seen = ?, occurrence_count = occurrence_count + 1
                WHERE proposal_id = ?
                """,
                (now.isoformat(), proposal_id),
            )
            self._conn.commit()
            updated = self.get(proposal_id)
            assert updated is not None
            return updated

        row = ReviewProposal(
            proposal_id=proposal_id,
            kind=kind,
            lane=lane,
            lane_reason=reason,
            confidence=confidence,
            impact=impact,
            summary=summary,
            payload=payload,
            status=ProposalStatus.PENDING,
            result="",
            created=now,
            last_seen=now,
            resolved=None,
            occurrence_count=1,
        )
        self._insert(row)

        if lane is Lane.AUTO:
            row = self._apply(row)

        return row

    # ------------------------------------------------------------------
    # Human decisions
    # ------------------------------------------------------------------

    def approve(self, proposal_id: str) -> ReviewProposal | None:
        """Apply a pending proposal, acknowledge explicit information, or record failure.

        Returns None for missing or non-pending proposals; failed operations need retry().
        """
        proposal = self.get(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.PENDING:
            return None
        return self._apply(proposal)

    def reject(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a pending proposal. Returns False if not found or not
        pending."""
        proposal = self.get(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.PENDING:
            return False
        self._conn.execute(
            "UPDATE review_proposals SET status = ?, result = ?, resolved = ? "
            "WHERE proposal_id = ?",
            (ProposalStatus.REJECTED.value, reason, datetime.now().isoformat(), proposal_id),
        )
        self._conn.commit()
        return True

    def retry(self, proposal_id: str) -> ReviewProposal | None:
        """Explicitly re-run the applier of a FAILED proposal. Returns None if
        not found or not FAILED. Never invoked automatically."""
        proposal = self.get(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.FAILED:
            return None
        return self._apply(proposal)

    def approve_all(self, lane: Lane = Lane.SKIM) -> list[ReviewProposal]:
        """Approve every pending proposal in `lane` — the digest/`/review`
        "one-tap approve-all" action. FAILED rows are never included."""
        pending = self.list_pending(lane=lane)
        approved = []
        for proposal in pending:
            result = self.approve(proposal.proposal_id)
            if result is not None:
                approved.append(result)
        return approved

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, proposal_id: str) -> ReviewProposal | None:
        row = self._conn.execute(
            "SELECT * FROM review_proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        return _row_to_proposal(row) if row else None

    def list_pending(self, lane: Lane | None = None) -> list[ReviewProposal]:
        """List pending proposals, oldest first, optionally filtered to one lane."""
        if lane is None:
            rows = self._conn.execute(
                "SELECT * FROM review_proposals WHERE status = ? ORDER BY created ASC",
                (ProposalStatus.PENDING.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM review_proposals WHERE status = ? AND lane = ? ORDER BY created ASC",
                (ProposalStatus.PENDING.value, lane.name),
            ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def list_failed(self) -> list[ReviewProposal]:
        """List FAILED proposals, oldest first — the explicit-retry candidates."""
        rows = self._conn.execute(
            "SELECT * FROM review_proposals WHERE status = ? ORDER BY created ASC",
            (ProposalStatus.FAILED.value,),
        ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def list_applied(self, kind: ProposalKind) -> list[ReviewProposal]:
        """Read completed mutations of one kind, including rows with lost external evidence."""
        rows = self._conn.execute(
            "SELECT * FROM review_proposals WHERE status = ? AND kind = ? ORDER BY resolved ASC",
            (ProposalStatus.APPLIED.value, kind.value),
        ).fetchall()
        return [_row_to_proposal(row) for row in rows]

    def fatigue_stats(self) -> FatigueStats:
        """Approval-fatigue snapshot across every proposal ever routed."""
        rows = self._conn.execute(
            "SELECT lane, COUNT(*) as n FROM review_proposals GROUP BY lane"
        ).fetchall()
        counts = {Lane[r["lane"]]: r["n"] for r in rows}
        auto_count = counts.get(Lane.AUTO, 0)
        skim_count = counts.get(Lane.SKIM, 0)
        block_count = counts.get(Lane.BLOCK, 0)
        return FatigueStats(
            total=auto_count + skim_count + block_count,
            auto_count=auto_count,
            skim_count=skim_count,
            block_count=block_count,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _insert(self, row: ReviewProposal) -> None:
        self._conn.execute(
            """
            INSERT INTO review_proposals
                (proposal_id, kind, lane, lane_reason, confidence, impact,
                 summary, payload, status, result, created, last_seen,
                 resolved, occurrence_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.proposal_id,
                row.kind.value,
                row.lane.name,
                row.lane_reason,
                row.confidence,
                row.impact.value,
                row.summary,
                json.dumps(row.payload, sort_keys=True),
                row.status.value,
                row.result,
                row.created.isoformat(),
                row.last_seen.isoformat(),
                None,
                row.occurrence_count,
            ),
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive, deterministic schema migration gated by `user_version`.

        Adds evidence columns and recovers deterministic pre-FAILED execution
        failures: the exact applier-raised rejection marker, a known mutation
        acknowledged without an applier, or a known maturation failure result.
        Possible successful synthesis paths and ambiguous history stay unchanged.
        Original fields are preserved; opening never retries an operation.
        """
        fresh = (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_proposals'"
            ).fetchone()
            is None
        )
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if fresh:
            self._conn.executescript(_SCHEMA)
        elif version < _SCHEMA_VERSION:
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(review_proposals)")}
            with self._conn:
                if "legacy_status" not in columns:
                    self._conn.execute("ALTER TABLE review_proposals ADD COLUMN legacy_status TEXT")
                if "legacy_result" not in columns:
                    self._conn.execute("ALTER TABLE review_proposals ADD COLUMN legacy_result TEXT")
                self._conn.execute(
                    "UPDATE review_proposals SET legacy_status = status, legacy_result = result, "
                    "status = ? WHERE status = ? AND result = ?",
                    (
                        ProposalStatus.FAILED.value,
                        ProposalStatus.REJECTED.value,
                        _LEGACY_APPLIER_RAISED,
                    ),
                )
                mutation_kinds = tuple(
                    kind.value for kind in ProposalKind if kind not in INFORMATIONAL_KINDS
                )
                placeholders = ", ".join("?" for _ in mutation_kinds)
                self._conn.execute(
                    "UPDATE review_proposals SET legacy_status = status, legacy_result = result, "
                    "status = ? WHERE status = ? AND result = '' "
                    f"AND kind IN ({placeholders})",
                    (
                        ProposalStatus.FAILED.value,
                        ProposalStatus.ACKNOWLEDGED.value,
                        *mutation_kinds,
                    ),
                )
                synthesis_failures = (
                    "Synthesis failed: No source notes found on disk",
                    "Synthesis failed: No YAML frontmatter found",
                )
                synthesis_prefixes = (
                    "Synthesis failed: YAML frontmatter parse error: ",
                    "Synthesis failed: Unresolved wikilinks: ",
                )
                # Every old successful synthesis returns a path ending in .md,
                # even when its configurable inbox begins with a failure prefix.
                self._conn.execute(
                    "UPDATE review_proposals SET legacy_status = status, legacy_result = result, "
                    "status = ? WHERE kind = ? AND status = ? AND substr(result, -3) <> '.md' "
                    "AND (result IN (?, ?) OR substr(result, 1, ?) = ? "
                    "OR substr(result, 1, ?) = ?)",
                    (
                        ProposalStatus.FAILED.value,
                        ProposalKind.MATURATION_SYNTHESIS.value,
                        ProposalStatus.APPLIED.value,
                        *synthesis_failures,
                        len(synthesis_prefixes[0]),
                        synthesis_prefixes[0],
                        len(synthesis_prefixes[1]),
                        synthesis_prefixes[1],
                    ),
                )
            self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._conn.commit()

    def _apply(self, proposal: ReviewProposal) -> ReviewProposal:
        """Apply (or acknowledge) a PENDING or FAILED proposal and persist the outcome."""
        applier = self._appliers.get(proposal.kind)
        if applier is None:
            if proposal.kind in INFORMATIONAL_KINDS:
                status, result = ProposalStatus.ACKNOWLEDGED, ""
            else:
                status = ProposalStatus.FAILED
                result = f"no applier registered for operation {proposal.kind.value}"
        else:
            try:
                result = applier(proposal.payload)
                status = ProposalStatus.APPLIED
            except Exception as exc:
                logger.exception(
                    "Applier failed for proposal %s (%s)", proposal.proposal_id, proposal.kind
                )
                status = ProposalStatus.FAILED
                result = f"{type(exc).__name__}: {exc}"

        self._conn.execute(
            "UPDATE review_proposals SET status = ?, result = ?, resolved = ? "
            "WHERE proposal_id = ?",
            (status.value, result, datetime.now().isoformat(), proposal.proposal_id),
        )
        self._conn.commit()
        updated = self.get(proposal.proposal_id)
        assert updated is not None
        return updated

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def migrate_quarantine(queue: ReviewQueue, quarantine_path: Path) -> int:
    """One-time import of the retired `auto_tagger.py` tag-quarantine JSON.

    Reads the old `{"approved": [...], "quarantined": [...]}` file, mints a
    `TAG_VOCABULARY` SKIM proposal per pending (quarantined) tag so nothing
    in flight is lost, then empties the file's `quarantined` list — leaving
    it on disk, confirmed-empty, as a read-only historical artifact (the
    `queue` is the sole approval mechanism for new tags going forward).
    Returns the number of tags migrated. A no-op (returns 0) if the file
    does not exist or has already been emptied.
    """
    if not quarantine_path.exists():
        return 0

    data = json.loads(quarantine_path.read_text())
    pending = data.get("quarantined", [])
    if not pending:
        return 0

    for tag in pending:
        queue.propose(
            ProposalKind.TAG_VOCABULARY,
            confidence=0.6,
            impact=Impact.LOW,
            summary=f"New tag vocabulary: '{tag}' (migrated from tag quarantine)",
            payload={"tag": tag},
        )

    data["quarantined"] = []
    quarantine_path.write_text(json.dumps(data, indent=2))
    return len(pending)


def mint_duplicate_proposals(
    queue: ReviewQueue,
    note: Note,
    matches: list[DuplicateMatch],
) -> list[ReviewProposal]:
    """Route a note's merge-band duplicate candidates into the queue.

    `DUPLICATE_MERGE` has no registered applier — no note-merge execution
    exists anywhere in VaultMind (`/duplicates`, `scan-duplicates`, and the
    MCP `find_duplicates` tool are all read-only). Every candidate is fixed
    at `SKIM` by default: routing merge-band hits to an interrupting Telegram
    push on every ordinary note edit would page the user for a purely
    informational signal — directly counter to the approval-fatigue goal. An
    explicit `[autonomy].force_block` is honored and routes them to `BLOCK`.
    """
    from vaultmind.indexer.duplicate_detector import MatchType

    proposals = []
    for match in matches:
        if match.match_type is not MatchType.MERGE:
            continue
        proposals.append(
            queue.propose(
                ProposalKind.DUPLICATE_MERGE,
                confidence=match.similarity,
                impact=Impact.MEDIUM,
                summary=(
                    f"Merge candidate: '{note.title}' ~ '{match.match_title}' "
                    f"({match.similarity:.0%} similar)"
                ),
                payload={"source_path": str(note.path), "match_path": match.match_path},
                lane_override=Lane.BLOCK if queue._thresholds.force_block else Lane.SKIM,
            )
        )
    return proposals


__all__ = [
    "DEFAULT_THRESHOLDS",
    "INFORMATIONAL_KINDS",
    "Applier",
    "AutonomyThresholds",
    "FatigueStats",
    "Impact",
    "Lane",
    "ProposalKind",
    "ProposalStatus",
    "ReviewProposal",
    "ReviewQueue",
    "migrate_quarantine",
    "mint_duplicate_proposals",
    "route",
]
