"""Non-destructive contradiction marking — writes `contradicted_by` frontmatter
and an Obsidian callout to a losing note's file.

Never edits or deletes existing body text. The callout is prepended above the
existing content; `contradicted_by` accumulates winner note paths (a note can
be marked by more than one later contradiction). Idempotent: marking the same
winner twice on the same note is a no-op, both to satisfy "exactly one gap
per escalation" upstream and to terminate the reprocessing loop that a file
write triggers when the vault is under live watch (a fresh `NoteModifiedEvent`
fires for the just-marked note).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import frontmatter

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CALLOUT_TYPE = "warning"


def mark_contradicted(
    loser_path: Path,
    winner_path: str,
    winner_title: str,
    rationale: str,
) -> bool:
    """Mark `loser_path` as contradicted by `winner_path`, non-destructively.

    Adds `contradicted_by` frontmatter (list of winner note paths) and
    prepends an Obsidian callout (`bot/formatter.py`'s recognized
    `> [!warning] ...` syntax) above the existing body. Never removes or
    rewrites any existing line. Returns `True` if the file was written,
    `False` if it was already marked with this exact winner (no-op).
    """
    with open(loser_path, encoding="utf-8") as f:
        post = frontmatter.load(f)

    existing = post.metadata.get("contradicted_by", [])
    if isinstance(existing, str):
        existing = [existing]
    elif not isinstance(existing, list):
        existing = []

    if winner_path in existing:
        return False

    post.metadata["contradicted_by"] = [*existing, winner_path]

    callout = f"> [!{CALLOUT_TYPE}] Contradicted by [[{winner_title}]]\n> {rationale}\n\n"
    post.content = callout + post.content

    with open(loser_path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    logger.info("Marked %s as contradicted by %s", loser_path, winner_path)
    return True


__all__ = ["CALLOUT_TYPE", "mark_contradicted"]
