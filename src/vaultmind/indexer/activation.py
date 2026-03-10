"""Activation-based note decay tracking.

Records per-note access events in SQLite and computes a sigmoid-normalized
activation score that boosts frequently-accessed notes in search ranking.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS activations (
    note_path TEXT NOT NULL,
    timestamp REAL NOT NULL,
    activation_type TEXT NOT NULL DEFAULT 'access'
);
CREATE INDEX IF NOT EXISTS idx_activations_path ON activations(note_path);
CREATE INDEX IF NOT EXISTS idx_activations_ts ON activations(timestamp);
"""

_RECENT_DAYS = 30


@dataclass(frozen=True, slots=True)
class ActivationScore:
    """Computed activation score for a single note."""

    note_path: str
    total_activations: int
    recent_activations: int  # last 30 days
    score: float  # 0.0–1.0 sigmoid-normalized


class ActivationTracker:
    """SQLite-backed activation tracking for notes."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, note_path: str, activation_type: str = "access") -> None:
        """Record a single activation event for *note_path*."""
        self._conn.execute(
            "INSERT INTO activations (note_path, timestamp, activation_type) VALUES (?, ?, ?)",
            (note_path, time.time(), activation_type),
        )
        self._conn.commit()

    def get_score(self, note_path: str, half_life_days: float = 14.0) -> ActivationScore:
        """Compute activation score for *note_path*.

        Each activation contributes exp(-0.693 * age_days / half_life_days).
        The raw sum is normalized through sigmoid centered at 3 activations:
            score = 1 / (1 + exp(-(raw - 3)))
        """
        now = time.time()
        cutoff_recent = now - _RECENT_DAYS * 86400.0

        cur = self._conn.execute(
            "SELECT timestamp FROM activations WHERE note_path = ?",
            (note_path,),
        )
        rows = cur.fetchall()

        if not rows:
            raw = 0.0
            total = 0
            recent = 0
        else:
            total = len(rows)
            recent = 0
            raw = 0.0
            for (ts,) in rows:
                age_days = (now - ts) / 86400.0
                raw += math.exp(-0.693 * age_days / half_life_days)
                if ts >= cutoff_recent:
                    recent += 1

        sigmoid = 1.0 / (1.0 + math.exp(-(raw - 3.0)))
        return ActivationScore(
            note_path=note_path,
            total_activations=total,
            recent_activations=recent,
            score=sigmoid,
        )

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
