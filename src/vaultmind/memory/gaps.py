"""Knowledge gap ledger — SQLite-backed store for unresolved questions.

Unanswered questions, weak-retrieval recalls, contradiction escalations, and
stale claims are recorded as deduplicated, lifecycle-tracked gaps instead of
being silently discarded. `mint()` is the single entry point every minting
site calls; re-asking the same normalized question against the same gap
`kind` touches (and, if stale, reopens) the existing row rather than creating
a duplicate. Mirrors `memory/store.py`'s `EpisodeStore` SQLite convention.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — used at runtime for mkdir/connect

logger = logging.getLogger(__name__)


class GapKind(StrEnum):
    """Origin of a recorded knowledge gap."""

    UNANSWERED_QUESTION = "unanswered_question"
    WEAK_RETRIEVAL = "weak_retrieval"
    CONTRADICTION_ESCALATED = "contradiction_escalated"
    STALE_CLAIM = "stale_claim"


class GapStatus(StrEnum):
    """Lifecycle state of a knowledge gap."""

    OPEN = "open"
    ANSWERED = "answered"
    INVALIDATED = "invalidated"
    STALE = "stale"


_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?.!,;:]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gaps (
    gap_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    question TEXT NOT NULL,
    normalized_question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    evidence_ref TEXT NOT NULL DEFAULT '',
    resolution_ref TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    resolved TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_gaps_status ON gaps(status);
CREATE INDEX IF NOT EXISTS idx_gaps_created ON gaps(created);
CREATE INDEX IF NOT EXISTS idx_gaps_normalized_question ON gaps(normalized_question);
"""


def normalize_question(question: str) -> str:
    """Normalize a question for stable dedup-key derivation.

    Lowercases, strips leading/trailing whitespace and trailing punctuation,
    and collapses internal whitespace runs — so re-asks that differ only in
    casing, punctuation, or spacing dedup to the same gap.
    """
    text = _TRAILING_PUNCT_RE.sub("", question.strip().lower())
    return _WS_RE.sub(" ", text).strip()


def _dedup_key(kind: GapKind, normalized_question: str) -> str:
    """SHA-256 dedup key over `kind` + normalized question text."""
    payload = f"{kind.value}:{normalized_question}".encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Gap:
    """A recorded knowledge gap."""

    gap_id: str
    kind: GapKind
    question: str
    normalized_question: str
    status: GapStatus
    evidence_ref: str
    resolution_ref: str
    created: datetime
    last_seen: datetime
    resolved: datetime | None
    occurrence_count: int


def _row_to_gap(row: sqlite3.Row) -> Gap:
    return Gap(
        gap_id=row["gap_id"],
        kind=GapKind(row["kind"]),
        question=row["question"],
        normalized_question=row["normalized_question"],
        status=GapStatus(row["status"]),
        evidence_ref=row["evidence_ref"],
        resolution_ref=row["resolution_ref"],
        created=datetime.fromisoformat(row["created"]),
        last_seen=datetime.fromisoformat(row["last_seen"]),
        resolved=datetime.fromisoformat(row["resolved"]) if row["resolved"] else None,
        occurrence_count=row["occurrence_count"],
    )


class GapStore:
    """SQLite-backed store for the knowledge gap ledger."""

    def __init__(self, db_path: Path, stale_after_days: int = 30) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._stale_after_days = stale_after_days

    def mint(self, question: str, kind: GapKind, evidence_ref: str = "") -> Gap:
        """Mint a gap, or touch (and reopen if stale) an existing dedup match.

        Deduplicated against re-asks of the same normalized question for the
        same `kind` — the row count never grows past one per (kind,
        normalized question) pair; `occurrence_count` tracks re-asks.
        """
        normalized = normalize_question(question)
        gap_id = _dedup_key(kind, normalized)
        now = datetime.now()
        existing = self.get(gap_id)

        if existing is not None:
            status = GapStatus.OPEN if existing.status == GapStatus.STALE else existing.status
            self._conn.execute(
                """
                UPDATE gaps
                SET status = ?, last_seen = ?, occurrence_count = occurrence_count + 1
                WHERE gap_id = ?
                """,
                (status.value, now.isoformat(), gap_id),
            )
            self._conn.commit()
            updated = self.get(gap_id)
            assert updated is not None
            return updated

        gap = Gap(
            gap_id=gap_id,
            kind=kind,
            question=question,
            normalized_question=normalized,
            status=GapStatus.OPEN,
            evidence_ref=evidence_ref,
            resolution_ref="",
            created=now,
            last_seen=now,
            resolved=None,
            occurrence_count=1,
        )
        self._conn.execute(
            """
            INSERT INTO gaps
                (gap_id, kind, question, normalized_question, status,
                 evidence_ref, resolution_ref, created, last_seen, resolved,
                 occurrence_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gap.gap_id,
                gap.kind.value,
                gap.question,
                gap.normalized_question,
                gap.status.value,
                gap.evidence_ref,
                gap.resolution_ref,
                gap.created.isoformat(),
                gap.last_seen.isoformat(),
                None,
                gap.occurrence_count,
            ),
        )
        self._conn.commit()
        return gap

    def get(self, gap_id: str) -> Gap | None:
        """Retrieve a single gap by ID."""
        row = self._conn.execute("SELECT * FROM gaps WHERE gap_id = ?", (gap_id,)).fetchone()
        return _row_to_gap(row) if row else None

    def answer(self, gap_id: str, resolution_ref: str) -> bool:
        """Mark a gap answered with a link to its resolving note/research.

        Returns False if no such gap exists or it is already answered.
        """
        gap = self.get(gap_id)
        if gap is None or gap.status == GapStatus.ANSWERED:
            return False
        self._conn.execute(
            "UPDATE gaps SET status = ?, resolution_ref = ?, resolved = ? WHERE gap_id = ?",
            (GapStatus.ANSWERED.value, resolution_ref, datetime.now().isoformat(), gap_id),
        )
        self._conn.commit()
        return True

    def list_open(self, limit: int = 50) -> list[Gap]:
        """List open gaps ordered oldest-first (by age).

        Lazily transitions gaps untouched past the configured staleness
        window to `stale` before listing (mirrors `SessionStore`'s
        lazy-reap-on-read convention — no separate scheduler job).
        """
        self._apply_staleness()
        rows = self._conn.execute(
            "SELECT * FROM gaps WHERE status = ? ORDER BY created ASC LIMIT ?",
            (GapStatus.OPEN.value, limit),
        ).fetchall()
        return [_row_to_gap(r) for r in rows]

    def find_open_by_question(self, question: str) -> Gap | None:
        """Find the most recent open/stale gap matching a normalized question.

        Used by the `research`-closes-gap flow: the caller supplies the
        gap's question text (e.g. copied from `/gaps` output) without
        needing to know its `kind`.
        """
        normalized = normalize_question(question)
        row = self._conn.execute(
            """
            SELECT * FROM gaps
            WHERE normalized_question = ? AND status IN (?, ?)
            ORDER BY created DESC LIMIT 1
            """,
            (normalized, GapStatus.OPEN.value, GapStatus.STALE.value),
        ).fetchone()
        return _row_to_gap(row) if row else None

    def close_from_research(self, question: str, resolution_ref: str) -> Gap | None:
        """Close the open/stale gap matching `question`, if any.

        Combines `find_open_by_question()` + `answer()` for the
        `research`-closes-gap flow: `vaultmind research "<gap question>"`
        closes the gap it answers, linking to the resulting note.
        Returns the closed `Gap`, or `None` if no matching gap exists.
        """
        gap = self.find_open_by_question(question)
        if gap is None or not self.answer(gap.gap_id, resolution_ref):
            return None
        return self.get(gap.gap_id)

    def _apply_staleness(self) -> int:
        """Transition open gaps untouched past the staleness window to stale."""
        cutoff = (datetime.now() - timedelta(days=self._stale_after_days)).isoformat()
        cursor = self._conn.execute(
            "UPDATE gaps SET status = ? WHERE status = ? AND last_seen < ?",
            (GapStatus.STALE.value, GapStatus.OPEN.value, cutoff),
        )
        self._conn.commit()
        staled = cursor.rowcount
        if staled > 0:
            logger.info("Auto-staled %d gap(s) past the %dd window", staled, self._stale_after_days)
        return staled

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


__all__ = [
    "Gap",
    "GapKind",
    "GapStatus",
    "GapStore",
    "normalize_question",
]
