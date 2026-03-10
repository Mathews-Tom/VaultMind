"""Note-type-aware search result ranking.

Applies type multipliers, temporal decay, mode multipliers, activation boost,
and status suppression post-retrieval to produce Zettelkasten-aware search ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

NOTE_TYPE_CONFIG: dict[str, dict[str, float | None]] = {
    "permanent": {"multiplier": 1.3, "half_life_days": None},
    "literature": {"multiplier": 1.1, "half_life_days": 90},
    "concept": {"multiplier": 1.2, "half_life_days": None},
    "project": {"multiplier": 1.0, "half_life_days": None},
    "person": {"multiplier": 1.0, "half_life_days": None},
    "daily": {"multiplier": 0.9, "half_life_days": 30},
    "fleeting": {"multiplier": 0.8, "half_life_days": 7},
}

DEFAULT_CONFIG: dict[str, float | None] = {"multiplier": 1.0, "half_life_days": 30}

ARCHIVED_MULTIPLIER = 0.4

MODE_MULTIPLIERS: dict[str, float] = {
    "operational": 1.2,
    "learning": 1.0,
}


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A search result with computed ranking score."""

    chunk_id: str
    raw_score: float
    final_score: float
    note_type: str
    metadata: dict[str, str | int]
    content: str


def score(
    raw_score: float,
    note_type: str,
    created_at: str,
    status: str,
    mode: str = "",
    activation_score: float = 0.0,
) -> float:
    """Compute ranked score from raw similarity score.

    Pipeline: raw_score -> type_multiplier -> decay_multiplier ->
              mode_multiplier -> activation_boost -> status_multiplier
    """
    cfg = NOTE_TYPE_CONFIG.get(note_type, DEFAULT_CONFIG)
    multiplier = cfg["multiplier"]
    assert multiplier is not None
    s = raw_score * multiplier

    half_life = cfg["half_life_days"]
    if half_life is not None and created_at:
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - created).days
            if age_days > 0:
                s *= math.exp(-0.693 * age_days / half_life)
        except (ValueError, TypeError):
            pass  # unparseable created_at — skip decay

    mode_mult = MODE_MULTIPLIERS.get(mode, 1.0)
    s *= mode_mult

    if activation_score > 0:
        s *= 1.0 + 0.3 * activation_score  # up to 30% boost

    if status in ("archived", "completed"):
        s *= ARCHIVED_MULTIPLIER

    return s


def rank_results(
    hits: list[dict[str, Any]],
    enabled: bool = True,
) -> list[RankedResult]:
    """Rank a list of search hits using the scoring pipeline.

    Args:
        hits: Raw search results from VaultStore.search().
        enabled: If False, return results with final_score = raw_score (passthrough).

    Returns:
        List of RankedResult sorted by final_score descending.
    """
    ranked: list[RankedResult] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        # ChromaDB distances are lower = better, convert to similarity score
        # cosine distance ranges [0, 2], similarity = 1 - (distance / 2)
        raw = 1.0 - (hit.get("distance", 0.0) / 2.0)

        if enabled:
            final = score(
                raw_score=raw,
                note_type=meta.get("note_type", ""),
                created_at=meta.get("created", ""),
                status=meta.get("status", ""),
                mode=meta.get("mode", ""),
                activation_score=hit.get("activation_score", 0.0),
            )
        else:
            final = raw

        ranked.append(
            RankedResult(
                chunk_id=hit.get("chunk_id", ""),
                raw_score=raw,
                final_score=final,
                note_type=meta.get("note_type", ""),
                metadata=meta,
                content=hit.get("content", ""),
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
