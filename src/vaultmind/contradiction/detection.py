"""LLM conflict detection for candidate pairs in the duplicate-detector merge band.

Adopts Kosha's split: **LLM detects, code decides.** This module only judges
whether two notes materially conflict — resolution precedence (temporal ->
authority -> escalate) lives in ``contradiction.policy``, kept separate so the
detection surface can be evaluated (``vaultmind eval contradict``) and improved
independently of the deterministic resolver.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vaultmind.llm.client import Message

if TYPE_CHECKING:
    from vaultmind.llm.client import LLMClient

logger = logging.getLogger(__name__)

CONFLICT_DETECTION_SYSTEM_PROMPT = """\
You compare two notes from a personal knowledge vault and judge whether they \
materially conflict — i.e. they assert different, incompatible facts about the \
same subject (not merely overlapping or complementary content).

Examples of a material conflict: one note says a project uses PostgreSQL, \
another says the same project uses MongoDB. One note says a decision was made \
on option A, another says option B was chosen instead.

Examples that are NOT a material conflict: two notes covering the same topic \
from different angles, a note that extends or elaborates another without \
contradicting it, near-duplicate notes with the same claim repeated, or notes \
whose overlap is topical only.

Respond with ONLY valid JSON:
{"materially_conflicts": true or false, "reasoning": "one sentence explaining why"}"""

_MAX_BODY_CHARS = 2000
_DEFAULT_MAX_TOKENS = 300


@dataclass(frozen=True, slots=True)
class ConflictVerdict:
    """Result of one LLM conflict-detection call over a note pair."""

    materially_conflicts: bool
    reasoning: str = ""
    error: str = ""


def build_prompt(note_a_title: str, note_a_body: str, note_b_title: str, note_b_body: str) -> str:
    """Build the single-shot conflict-detection prompt for one note pair."""
    return (
        f"Note A ({note_a_title}):\n{note_a_body[:_MAX_BODY_CHARS]}\n\n"
        f"Note B ({note_b_title}):\n{note_b_body[:_MAX_BODY_CHARS]}"
    )


def _parse_json_response(raw: str) -> dict[str, object] | None:
    """Parse an LLM JSON response, tolerating a markdown code fence wrapper."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def detect_conflict(
    note_a_title: str,
    note_a_body: str,
    note_b_title: str,
    note_b_body: str,
    llm_client: LLMClient,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> ConflictVerdict:
    """Judge whether two notes materially conflict via a single LLM call.

    Never raises — LLM failures and malformed responses are logged and
    returned as a non-conflicting verdict with `error` set, so a fire-and-forget
    background caller (`contradiction.detector.ContradictionDetector`) never
    escalates or marks a note it could not actually evaluate.
    """
    prompt = build_prompt(note_a_title, note_a_body, note_b_title, note_b_body)
    try:
        response = llm_client.complete(
            messages=[Message(role="user", content=prompt)],
            model=model,
            max_tokens=max_tokens,
            system=CONFLICT_DETECTION_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception(
            "LLM call failed during conflict detection for %r vs %r", note_a_title, note_b_title
        )
        return ConflictVerdict(materially_conflicts=False, error="LLM call failed")

    parsed = _parse_json_response(response.text)
    if parsed is None:
        logger.warning(
            "Invalid JSON from LLM during conflict detection for %r vs %r: %r",
            note_a_title,
            note_b_title,
            response.text[:200],
        )
        return ConflictVerdict(materially_conflicts=False, error="Invalid JSON from LLM")

    conflicts = bool(parsed.get("materially_conflicts", False))
    reasoning = str(parsed.get("reasoning", "")).strip()
    return ConflictVerdict(materially_conflicts=conflicts, reasoning=reasoning)


__all__ = [
    "CONFLICT_DETECTION_SYSTEM_PROMPT",
    "ConflictVerdict",
    "build_prompt",
    "detect_conflict",
]
