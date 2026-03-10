"""LLM-based episode extraction from vault notes."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from vaultmind.llm.client import Message

if TYPE_CHECKING:
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Analyze the following note and extract any decision-outcome pairs.
Return a JSON array of objects with these fields:
- decision: what was decided
- context: why/when the decision was made
- outcome: what happened (empty string if not yet resolved)
- outcome_status: one of "pending", "success", "failure", "partial", "unknown"
- lessons: array of lessons learned (empty array if none)
- entities: array of key entities/concepts mentioned

Return [] if no decisions found. Return ONLY valid JSON, no other text.\
"""

_MIN_BODY_LENGTH = 100
_MAX_CONTENT_LENGTH = 4000


def extract_episodes(
    note: Note,
    llm_client: LLMClient,
    model: str,
) -> list[dict[str, object]]:
    """Extract decision-outcome pairs from a note using an LLM.

    Returns a list of dicts with keys: decision, context, outcome,
    outcome_status, lessons, entities.  Returns [] for short notes or
    on any parse error.
    """
    body = note.content.strip()
    if len(body) < _MIN_BODY_LENGTH:
        return []

    content = body[:_MAX_CONTENT_LENGTH]

    try:
        response = llm_client.complete(
            messages=[Message(role="user", content=content)],
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("LLM call failed during episode extraction for %s", note.path)
        return []

    try:
        parsed = json.loads(response.text.strip())
    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON from LLM during episode extraction for %s: %r",
            note.path,
            response.text[:200],
        )
        return []

    if not isinstance(parsed, list):
        logger.warning(
            "Expected JSON array from LLM for %s, got %s", note.path, type(parsed).__name__
        )
        return []

    return parsed
