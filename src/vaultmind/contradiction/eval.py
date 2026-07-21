"""Labeled eval set + scoring for the contradiction conflict-detector.

Heeds Kosha's Gate-0 NO-GO lesson: ship a small labeled eval set and require
the detector to beat a trivial always-escalate baseline before auto-resolution
is ever enabled. `vaultmind eval contradict` reports both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from vaultmind.contradiction.detection import detect_conflict
from vaultmind.errors import VaultMindError

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.llm.client import LLMClient


class ContradictEvalError(VaultMindError):
    """Raised when the contradiction eval set is missing or malformed."""


@dataclass(frozen=True, slots=True)
class ContradictEvalPair:
    """One labeled note-pair entry in the contradiction eval set."""

    id: str
    note_a_title: str
    note_a_body: str
    note_b_title: str
    note_b_body: str
    materially_conflicts: bool


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Precision/recall/F1 over a set of boolean predictions vs labels."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ContradictEvalReport:
    """Aggregate result of scoring the detector against the labeled eval set."""

    n_pairs: int
    detector: ScoreSummary
    baseline: ScoreSummary
    beats_baseline: bool


def _require_str(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"pair[{index}]: '{key}' must be a non-empty string, got {value!r}"
        raise ContradictEvalError(msg)
    return value


def _parse_pair(entry: dict[str, Any], index: int, seen_ids: set[str]) -> ContradictEvalPair:
    if not isinstance(entry, dict):
        msg = f"pair[{index}]: expected a mapping, got {type(entry).__name__}"
        raise ContradictEvalError(msg)

    pair_id = _require_str(entry, "id", index)
    if pair_id in seen_ids:
        msg = f"pair[{index}]: duplicate id {pair_id!r}"
        raise ContradictEvalError(msg)
    seen_ids.add(pair_id)

    conflicts = entry.get("materially_conflicts")
    if not isinstance(conflicts, bool):
        msg = f"pair[{index}]: 'materially_conflicts' must be a boolean, got {conflicts!r}"
        raise ContradictEvalError(msg)

    return ContradictEvalPair(
        id=pair_id,
        note_a_title=_require_str(entry, "note_a_title", index),
        note_a_body=_require_str(entry, "note_a_body", index),
        note_b_title=_require_str(entry, "note_b_title", index),
        note_b_body=_require_str(entry, "note_b_body", index),
        materially_conflicts=conflicts,
    )


def load_eval_set(path: Path) -> list[ContradictEvalPair]:
    """Load and validate the labeled contradiction eval set from a YAML file."""
    if not path.exists():
        msg = f"Contradiction eval set not found: {path}"
        raise ContradictEvalError(msg)

    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {path}: {exc}"
            raise ContradictEvalError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{path}: expected a mapping at the document root"
        raise ContradictEvalError(msg)

    raw_pairs = data.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        msg = f"{path}: 'pairs' must be a non-empty list"
        raise ContradictEvalError(msg)

    seen_ids: set[str] = set()
    return [_parse_pair(entry, i, seen_ids) for i, entry in enumerate(raw_pairs)]


def _score(predictions: list[bool], labels: list[bool]) -> ScoreSummary:
    tp = sum(1 for p, y in zip(predictions, labels, strict=True) if p and y)
    fp = sum(1 for p, y in zip(predictions, labels, strict=True) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels, strict=True) if not p and y)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ScoreSummary(precision=precision, recall=recall, f1=f1)


def run_eval(
    pairs: list[ContradictEvalPair],
    llm_client: LLMClient,
    model: str,
    max_tokens: int = 300,
) -> ContradictEvalReport:
    """Score the LLM conflict-detector against the labeled eval set.

    Also scores a trivial always-escalate baseline (predicts `True` for every
    pair) over the same labels, so callers can determine whether the detector
    is worth gating auto-resolution behind — the Kosha Gate-0 precedent this
    milestone exists to avoid repeating.
    """
    labels = [p.materially_conflicts for p in pairs]
    predictions = [
        detect_conflict(
            p.note_a_title,
            p.note_a_body,
            p.note_b_title,
            p.note_b_body,
            llm_client,
            model,
            max_tokens=max_tokens,
        ).materially_conflicts
        for p in pairs
    ]
    baseline_predictions = [True] * len(pairs)

    detector_score = _score(predictions, labels)
    baseline_score = _score(baseline_predictions, labels)

    return ContradictEvalReport(
        n_pairs=len(pairs),
        detector=detector_score,
        baseline=baseline_score,
        beats_baseline=detector_score.f1 > baseline_score.f1,
    )


__all__ = [
    "ContradictEvalError",
    "ContradictEvalPair",
    "ContradictEvalReport",
    "ScoreSummary",
    "load_eval_set",
    "run_eval",
]
