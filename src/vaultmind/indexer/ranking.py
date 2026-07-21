"""Note-type-aware search result ranking.

Applies composite weighted scoring with semantic similarity, temporal recency,
connection density, activation boost, note-type normalization, mode multipliers,
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

# Authority (source-provenance) multiplier — mirrors MODE_MULTIPLIERS' role as a
# post-sum multiplicative factor. Fallback defaults when no RankingConfig is
# supplied; RankingConfig.authority_weight/authority_default override these.
DEFAULT_AUTHORITY_WEIGHT: dict[int, float] = {1: 0.85, 2: 0.92, 3: 1.0, 4: 1.08, 5: 1.15}
DEFAULT_AUTHORITY_LEVEL = 3

# Normalization range for note-type multipliers.
_NOTE_TYPE_MIN = 0.8
_NOTE_TYPE_MAX = 1.3

# Default composite weights (must sum to 1.0).
_DEFAULT_SEMANTIC_WEIGHT = 0.40
_DEFAULT_RECENCY_WEIGHT = 0.20
_DEFAULT_DENSITY_WEIGHT = 0.25
_DEFAULT_ACTIVATION_WEIGHT = 0.05
_DEFAULT_NOTE_TYPE_WEIGHT = 0.10


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A search result with computed ranking score."""

    chunk_id: str
    raw_score: float
    final_score: float
    note_type: str
    metadata: dict[str, str | int]
    content: str
    semantic_score: float = 0.0
    recency_score: float = 0.0
    connection_density_score: float = 0.0
    activation_score_value: float = 0.0
    note_type_score: float = 0.0
    reranker_score: float = 0.0
    importance_score: float = 0.0


def compute_semantic_score(raw_score: float) -> float:
    """Clamp raw similarity score to [0.0, 1.0]."""
    return max(0.0, min(1.0, raw_score))


def compute_recency_score(created_at: str, half_life_days: float) -> float:
    """Compute temporal recency as exponential decay.

    Returns 1.0 (no decay) when created_at is empty or unparseable.
    """
    if not created_at:
        return 1.0
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - created).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0
        value = math.exp(-0.693 * age_days / half_life_days)
        return max(0.0, min(1.0, value))
    except (ValueError, TypeError):
        return 1.0


def compute_note_type_score(note_type: str) -> float:
    """Normalize note-type multiplier to [0.0, 1.0].

    Maps: fleeting(0.8)->0.0, daily(0.9)->0.2, project/person(1.0)->0.4,
    literature(1.1)->0.6, concept(1.2)->0.8, permanent(1.3)->1.0.
    """
    cfg = NOTE_TYPE_CONFIG.get(note_type, DEFAULT_CONFIG)
    multiplier = cfg["multiplier"]
    assert multiplier is not None
    normalized = (multiplier - _NOTE_TYPE_MIN) / (_NOTE_TYPE_MAX - _NOTE_TYPE_MIN)
    return max(0.0, min(1.0, normalized))


def authority_multiplier(authority: Any, ranking_config: Any = None) -> float:
    """Multiplier for a note's stamped `authority` level (1-5, source-provenance signal).

    Missing, zero, or out-of-range authority values fall back to the configured
    neutral default level — this never raises, satisfying the back-compat
    requirement that pre-existing vault content (or content indexed before the
    `authority` field existed) ranks as if it had the neutral default, not as
    an error or an implicit zero-weight penalty.
    """
    if ranking_config is not None:
        weights: dict[int, float] = ranking_config.authority_weight
        default_level: int = ranking_config.authority_default
    else:
        weights = DEFAULT_AUTHORITY_WEIGHT
        default_level = DEFAULT_AUTHORITY_LEVEL

    try:
        level = int(authority)
    except (TypeError, ValueError):
        level = default_level
    if level not in weights:
        level = default_level
    return weights.get(level, 1.0)


def composite_score(
    semantic: float,
    recency: float,
    connection_density: float,
    activation: float,
    note_type_normalized: float,
    status: str,
    mode: str = "",
    authority: Any = None,
    config: Any = None,
) -> float:
    """Compute weighted composite score from individual factors.

    All input factors should be in [0.0, 1.0].
    Weights come from config (RankingConfig). If config is None, use defaults.
    Status suppression (archived/completed: 0.4x) applied last.
    Mode and authority multipliers applied before status suppression.
    """
    if config is not None:
        w_s: float = config.semantic_weight
        w_r: float = config.recency_weight
        w_d: float = config.connection_density_weight
        w_a: float = config.activation_weight
        w_t: float = config.note_type_weight
    else:
        w_s = _DEFAULT_SEMANTIC_WEIGHT
        w_r = _DEFAULT_RECENCY_WEIGHT
        w_d = _DEFAULT_DENSITY_WEIGHT
        w_a = _DEFAULT_ACTIVATION_WEIGHT
        w_t = _DEFAULT_NOTE_TYPE_WEIGHT

    s: float = (
        semantic * w_s
        + recency * w_r
        + connection_density * w_d
        + activation * w_a
        + note_type_normalized * w_t
    )

    s *= MODE_MULTIPLIERS.get(mode, 1.0)
    s *= authority_multiplier(authority, config)

    if status in ("archived", "completed"):
        s *= ARCHIVED_MULTIPLIER

    return s


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

    Kept for backward compatibility. New callers should use composite_score().
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
    knowledge_graph: Any = None,
    ranking_config: Any = None,
) -> list[RankedResult]:
    """Rank a list of search hits using the scoring pipeline.

    Args:
        hits: Raw search results from VaultStore.search().
        enabled: If False, return results with final_score = raw_score (passthrough).
        knowledge_graph: Optional NetworkX knowledge graph for connection density.
        ranking_config: Optional RankingConfig for composite weights.

    Returns:
        List of RankedResult sorted by final_score descending.
    """
    half_life = 30.0
    if ranking_config is not None:
        half_life = ranking_config.recency_half_life_days

    ranked: list[RankedResult] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        # ChromaDB distances are lower = better, convert to similarity score
        # cosine distance ranges [0, 2], similarity = 1 - (distance / 2)
        raw = 1.0 - (hit.get("distance", 0.0) / 2.0)

        if not enabled:
            ranked.append(
                RankedResult(
                    chunk_id=hit.get("chunk_id", ""),
                    raw_score=raw,
                    final_score=raw,
                    note_type=meta.get("note_type", ""),
                    metadata=meta,
                    content=hit.get("content", ""),
                )
            )
            continue

        note_type = meta.get("note_type", "")
        created_at = meta.get("created", "")
        status = meta.get("status", "")
        mode = meta.get("mode", "")
        authority = meta.get("authority", 0)
        activation = float(hit.get("activation_score", 0.0))
        importance = float(meta.get("importance_score", 0.0))

        sem = compute_semantic_score(raw)
        rec = compute_recency_score(created_at, half_life)
        nt = compute_note_type_score(note_type)

        connection_density = 0.0
        if knowledge_graph is not None:
            entities_raw = meta.get("entities", "")
            entities = [e.strip() for e in str(entities_raw).split(",") if e.strip()]
            note_path = meta.get("note_path", "")
            try:
                from vaultmind.indexer.connection_density import ConnectionDensityCalculator

                calculator = ConnectionDensityCalculator(knowledge_graph, ranking_config)
                density_result = calculator.score_note(str(note_path), entities)
                connection_density = density_result.density_score
            except ImportError:
                pass

        final = composite_score(
            semantic=sem,
            recency=rec,
            connection_density=connection_density,
            activation=activation,
            note_type_normalized=nt,
            status=status,
            mode=mode,
            authority=authority,
            config=ranking_config,
        )

        # Importance boost: up to 15% additional for high-importance notes
        if importance > 0:
            final *= 1.0 + 0.15 * importance

        ranked.append(
            RankedResult(
                chunk_id=hit.get("chunk_id", ""),
                raw_score=raw,
                final_score=final,
                note_type=note_type,
                metadata=meta,
                content=hit.get("content", ""),
                semantic_score=sem,
                recency_score=rec,
                connection_density_score=connection_density,
                activation_score_value=activation,
                note_type_score=nt,
                reranker_score=float(hit.get("reranker_score", 0.0)),
                importance_score=importance,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked


def apply_authority(hits: list[dict[str, Any]], ranking_config: Any = None) -> list[dict[str, Any]]:
    """Re-rank retrieval hits by their stamped `authority` level.

    `/recall` (`bot/handlers/recall.py::handle_recall`) and `vaultmind bench`
    (`bench/runner.py::run_query`, which mirrors it exactly) call
    `VaultStore.hybrid_search()`/`search()` directly — the composite ranking
    pipeline above (`rank_results()`) is not wired into that live path. This
    applies the same `authority_multiplier()` directly to the score those
    functions already produced, so the authority signal is observable
    wherever `/recall` actually is, not only in the not-yet-live composite
    pipeline.

    Returns a new list of hit dicts (inputs are not mutated), each carrying
    an added `authority_score` key, sorted by that score descending.
    """
    if ranking_config is not None and not ranking_config.enabled:
        return hits

    def _base_score(hit: dict[str, Any]) -> float:
        rrf = hit.get("rrf_score")
        if rrf is not None:
            return float(rrf)
        return 1.0 - float(hit.get("distance", 0.0))

    scored: list[dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        mult = authority_multiplier(meta.get("authority"), ranking_config)
        adjusted = dict(hit)
        adjusted["authority_score"] = _base_score(hit) * mult
        scored.append(adjusted)

    scored.sort(key=lambda h: h["authority_score"], reverse=True)
    return scored


def best_distance(hits: list[dict[str, Any]]) -> float | None:
    """Lowest (best) `distance` across retrieval hits, or `None` if there are none.

    Shared by `bench/runner.py`'s deterministic honest-decline scoring and
    the live weak-retrieval gap-minting checks in `bot/handlers/recall.py`
    and `bot/thinking.py` — one implementation of "nothing confidently
    relevant was retrieved" for every caller that needs it.
    """
    if not hits:
        return None
    return min(float(h.get("distance", 1.0)) for h in hits)


def is_weak_retrieval(hits: list[dict[str, Any]], score_floor: float) -> bool:
    """True when nothing confidently relevant was retrieved for a query.

    Mirrors `bench/runner.py::score_question`'s honest-decline check: no
    hits, or the best hit's distance is at or above `score_floor`.
    """
    best = best_distance(hits)
    return best is None or best >= score_floor
