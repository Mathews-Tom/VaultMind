"""Golden question set schema and loader for the retrieval self-benchmark.

The golden set is a user-authored YAML file (see `benchmarks/golden.yaml` for
the schema and a starter template) — each question names the note(s) that
correctly answer it, or is flagged unanswerable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from vaultmind.errors import VaultMindError

if TYPE_CHECKING:
    from pathlib import Path


class GoldenSetError(VaultMindError):
    """Raised when a golden question set file is missing, malformed, or fails
    schema validation. Never swallowed — an invalid golden set makes every
    downstream bench score meaningless.
    """


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """A single golden-set entry: a question plus its ground-truth answer."""

    id: str
    question: str
    answerable: bool
    expected_notes: tuple[str, ...]


def _require_str(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"question[{index}]: '{key}' must be a non-empty string, got {value!r}"
        raise GoldenSetError(msg)
    return value


def _parse_question(entry: dict[str, Any], index: int, seen_ids: set[str]) -> GoldenQuestion:
    if not isinstance(entry, dict):
        msg = f"question[{index}]: expected a mapping, got {type(entry).__name__}"
        raise GoldenSetError(msg)

    question_id = _require_str(entry, "id", index)
    if question_id in seen_ids:
        msg = f"question[{index}]: duplicate id '{question_id}'"
        raise GoldenSetError(msg)
    seen_ids.add(question_id)

    question_text = _require_str(entry, "question", index)

    answerable = entry.get("answerable")
    if not isinstance(answerable, bool):
        msg = f"question[{index}] ({question_id}): 'answerable' must be a bool, got {answerable!r}"
        raise GoldenSetError(msg)

    raw_notes = entry.get("expected_notes", [])
    if not isinstance(raw_notes, list) or not all(isinstance(p, str) for p in raw_notes):
        msg = f"question[{index}] ({question_id}): 'expected_notes' must be a list of strings"
        raise GoldenSetError(msg)

    if answerable and not raw_notes:
        msg = (
            f"question[{index}] ({question_id}): answerable questions require "
            "non-empty 'expected_notes'"
        )
        raise GoldenSetError(msg)
    if not answerable and raw_notes:
        msg = (
            f"question[{index}] ({question_id}): unanswerable questions must have "
            "empty 'expected_notes'"
        )
        raise GoldenSetError(msg)

    return GoldenQuestion(
        id=question_id,
        question=question_text,
        answerable=answerable,
        expected_notes=tuple(raw_notes),
    )


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Load and validate a golden question set from a YAML file.

    Raises:
        GoldenSetError: file missing, not valid YAML, or fails schema
            validation (loud failure — no silent fallback to an empty set).
    """
    if not path.exists():
        msg = f"Golden question set not found: {path}"
        raise GoldenSetError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Golden question set is not valid YAML: {path}"
        raise GoldenSetError(msg) from exc

    if not isinstance(raw, dict) or "questions" not in raw:
        msg = f"Golden question set must be a mapping with a 'questions' key: {path}"
        raise GoldenSetError(msg)

    raw_questions = raw["questions"]
    if not isinstance(raw_questions, list) or not raw_questions:
        msg = f"Golden question set 'questions' must be a non-empty list: {path}"
        raise GoldenSetError(msg)

    seen_ids: set[str] = set()
    return [_parse_question(entry, i, seen_ids) for i, entry in enumerate(raw_questions)]
