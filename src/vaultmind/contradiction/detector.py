"""Contradiction-detection orchestrator — wires detection, policy, and
non-destructive marking into the vault event bus.

Subscribes alongside `DuplicateDetector` to `NoteCreatedEvent`/`NoteModifiedEvent`
and independently calls `DuplicateDetector.find_duplicates()` (idempotent,
reuses embeddings) to find the 80-92% merge-band candidates this milestone
checks for material conflict. Never mutates the vault unless `auto_resolve`
is enabled and the policy resolves via `TEMPORAL`/`AUTHORITY` — the default
path always escalates (mints an M5 gap + notifies) without writing anything.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol

from vaultmind.contradiction.detection import detect_conflict
from vaultmind.contradiction.marking import mark_contradicted
from vaultmind.contradiction.policy import resolve_conflict
from vaultmind.indexer.duplicate_detector import MatchType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vaultmind.config import ContradictionConfig
    from vaultmind.contradiction.policy import ResolutionOutcome
    from vaultmind.indexer.duplicate_detector import DuplicateDetector, DuplicateMatch
    from vaultmind.llm.client import LLMClient
    from vaultmind.services.review_queue import ReviewQueue
    from vaultmind.vault.events import NoteCreatedEvent, NoteModifiedEvent
    from vaultmind.vault.models import Note
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)

# (note_a_title, note_b_title, rationale, gap_id, proposal_id) -> None
type EscalationCallback = Callable[[str, str, str, str, str], Any]


class RankingConfigLike(Protocol):
    """Structural shape `contradiction.policy.resolve_conflict` needs."""

    authority_default: int


class ContradictionDetector:
    """Detects and resolves note-vs-note contradictions in the merge band."""

    def __init__(
        self,
        config: ContradictionConfig,
        duplicate_detector: DuplicateDetector,
        llm_client: LLMClient,
        model: str,
        vault_root: Path,
        parser: VaultParser,
        ranking_config: RankingConfigLike | None = None,
        gap_store: object = None,
        on_escalate: EscalationCallback | None = None,
        review_queue: ReviewQueue | None = None,
    ) -> None:
        self._config = config
        self._duplicate_detector = duplicate_detector
        self._llm_client = llm_client
        self._model = model
        self._vault_root = vault_root
        self._parser = parser
        self._ranking_config = ranking_config
        self._gap_store = gap_store
        self._on_escalate = on_escalate
        self._review_queue = review_queue
        if review_queue is not None:
            from vaultmind.services.review_queue import ProposalKind

            review_queue.register_applier(
                ProposalKind.CONTRADICTION_RESOLUTION, self._resolution_applier
            )

    def _resolution_applier(self, payload: dict[str, Any]) -> str:
        """Sync applier registered onto `review_queue` for
        `CONTRADICTION_RESOLUTION` proposals — invoked by the queue's own
        `_apply()`, on a worker thread via `asyncio.to_thread`."""
        from pathlib import Path as _Path

        mark_contradicted(
            _Path(str(payload["loser_path"])),
            str(payload["winner_path"]),
            str(payload["winner_title"]),
            str(payload["rationale"]),
        )
        return f"marked {payload['loser_path']} contradicted by {payload['winner_path']}"

    async def on_note_changed(self, event: NoteCreatedEvent | NoteModifiedEvent) -> None:
        """Event bus callback — checks the note's merge-band candidates for conflict."""
        if not self._config.enabled:
            return

        note = event.note
        if note is None:
            return
        if note.frontmatter.get("contradicted_by"):
            # Already a marked loser — skip re-check to avoid reprocessing the
            # loop the marking write itself triggers on a live-watched vault.
            return

        matches = await asyncio.to_thread(self._duplicate_detector.find_duplicates, note)
        merge_matches = [m for m in matches if m.match_type == MatchType.MERGE]
        for match in merge_matches:
            await self._check_pair(note, match)

    async def _check_pair(self, note: Note, match: DuplicateMatch) -> None:
        candidate_path = self._vault_root / match.match_path
        if not candidate_path.exists():
            return

        try:
            candidate = await asyncio.to_thread(self._parser.parse_file, candidate_path)
        except Exception:
            logger.exception("Failed to parse candidate note %s", candidate_path)
            return

        if candidate.frontmatter.get("contradicted_by"):
            return

        verdict = await asyncio.to_thread(
            detect_conflict,
            note.title,
            note.body_without_frontmatter(),
            candidate.title,
            candidate.body_without_frontmatter(),
            self._llm_client,
            self._model,
            self._config.max_tokens,
        )
        if verdict.error or not verdict.materially_conflicts:
            return

        outcome = resolve_conflict(note, candidate, ranking_config=self._ranking_config)

        if self._config.auto_resolve and not outcome.escalated:
            await self._apply_resolution(note, match, outcome)
            return

        await self._escalate(note, candidate, verdict.reasoning or outcome.rationale)

    async def _apply_resolution(
        self, note: Note, match: DuplicateMatch, outcome: ResolutionOutcome
    ) -> None:
        if outcome.winner == "new":
            winner_title, winner_rel = note.title, str(note.path)
            loser_path, loser_rel = self._vault_root / match.match_path, match.match_path
        else:
            winner_title, winner_rel = match.match_title, match.match_path
            loser_path, loser_rel = self._vault_root / note.path, str(note.path)

        if self._review_queue is None:
            try:
                await asyncio.to_thread(
                    mark_contradicted, loser_path, winner_rel, winner_title, outcome.rationale
                )
            except Exception:
                logger.exception("Failed to mark %s as contradicted by %s", loser_rel, winner_rel)
            return

        from vaultmind.services.review_queue import Impact, ProposalKind

        payload = {
            "loser_path": str(loser_path),
            "winner_path": winner_rel,
            "winner_title": winner_title,
            "rationale": outcome.rationale,
        }
        try:
            await asyncio.to_thread(
                self._review_queue.propose,
                ProposalKind.CONTRADICTION_RESOLUTION,
                1.0,
                Impact.LOW,
                f"Mark '{loser_rel}' contradicted by '{winner_rel}'",
                payload,
            )
        except Exception:
            logger.exception("Failed to mark %s as contradicted by %s", loser_rel, winner_rel)

    async def _escalate(self, note: Note, candidate: Note, rationale: str) -> None:
        proposal_id = ""
        if self._review_queue is not None:
            from vaultmind.services.review_queue import Impact, Lane, ProposalKind

            try:
                proposal = await asyncio.to_thread(
                    self._review_queue.propose,
                    ProposalKind.CONTRADICTION_ESCALATION,
                    0.0,
                    Impact.HIGH,
                    f"Contradiction escalated: '{note.title}' vs '{candidate.title}'",
                    {"note_path": str(note.path), "candidate_path": str(candidate.path)},
                    lane_override=Lane.BLOCK,
                )
                proposal_id = proposal.proposal_id
            except Exception:
                logger.exception(
                    "Escalation queue-proposal failed for %s vs %s", note.title, candidate.title
                )

        gap_id = self._mint_gap(note, candidate)
        if self._on_escalate is not None:
            try:
                await self._on_escalate(note.title, candidate.title, rationale, gap_id, proposal_id)
            except Exception:
                logger.exception(
                    "Escalation notification failed for %s vs %s", note.title, candidate.title
                )

    def _mint_gap(self, note: Note, candidate: Note) -> str:
        from vaultmind.memory.gaps import GapKind
        from vaultmind.memory.gaps import GapStore as _GapStore

        if not isinstance(self._gap_store, _GapStore):
            return ""

        titles = sorted([note.title, candidate.title])
        question = f"Contradiction: {titles[0]} vs {titles[1]}"
        evidence_ref = f"{note.path}|{candidate.path}"
        try:
            gap = self._gap_store.mint(
                question, GapKind.CONTRADICTION_ESCALATED, evidence_ref=evidence_ref
            )
        except Exception:
            logger.exception("Gap minting failed for contradiction %s", question)
            return ""
        return gap.gap_id


__all__ = ["ContradictionDetector", "EscalationCallback", "RankingConfigLike"]
