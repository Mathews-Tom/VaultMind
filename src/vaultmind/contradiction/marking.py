"""Non-destructive supersede marking — writes a dated (or undated) Obsidian
callout plus a frontmatter marker to a note's file, instead of deleting or
rewriting its content.

`write_superseded_block()` is the shared primitive. Never edits or deletes
existing body text: the callout (and, optionally, new primary content) is
prepended above the existing content, which is always preserved verbatim
below it. Idempotent per `(frontmatter_key, frontmatter_value)` pair —
writing the same marker twice on the same note is a no-op, both to satisfy
"exactly one marking per event" upstream and to terminate any reprocessing
loop a file write triggers when the vault is under live watch (a fresh
`NoteModifiedEvent` fires for the just-marked note).

`mark_contradicted()` (M6) is a thin wrapper: an undated `warning` callout
recording `contradicted_by` frontmatter for a losing note in a contradiction
resolution. M9's bot-initiated destructive-edit marking (`bot/handlers/
delete.py`, `bot/handlers/edit.py`) calls `write_superseded_block()`
directly with a dated `superseded` callout and, for edits, `new_content` —
the same primitive, reused rather than reimplemented.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import frontmatter

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CALLOUT_TYPE = "warning"


def write_superseded_block(
    path: Path,
    *,
    callout_type: str,
    title: str,
    rationale: str,
    frontmatter_key: str,
    frontmatter_value: str,
    dated: bool = False,
    timestamp: datetime | None = None,
    new_content: str | None = None,
) -> bool:
    """Write a non-destructive supersede marker to `path`.

    Adds `frontmatter_value` to the list at `frontmatter_key` (idempotent —
    a repeat identical value is a no-op, returns `False`) and prepends a
    `> [!callout_type] title` Obsidian callout above the existing body.
    Existing content is never removed or rewritten. When `new_content` is
    given, it is placed above the callout (and, beneath it, the fully
    preserved prior body) so a bot-applied edit's new text becomes what
    search/recall see as primary while the prior text stays intact
    underneath. Returns `True` if the file was written.
    """
    with open(path, encoding="utf-8") as f:
        post = frontmatter.load(f)

    existing = post.metadata.get(frontmatter_key, [])
    if isinstance(existing, str):
        existing = [existing]
    elif not isinstance(existing, list):
        existing = []

    if frontmatter_value in existing:
        return False

    post.metadata[frontmatter_key] = [*existing, frontmatter_value]

    date_suffix = ""
    if dated:
        ts = timestamp or datetime.now(UTC)
        date_suffix = f" — {ts.date().isoformat()}"

    callout = f"> [!{callout_type}] {title}{date_suffix}\n> {rationale}\n\n"
    prefix = f"{new_content.rstrip()}\n\n" if new_content is not None else ""
    post.content = prefix + callout + post.content

    with open(path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    logger.info("Marked %s: %s+=%s", path, frontmatter_key, frontmatter_value)
    return True


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
    return write_superseded_block(
        loser_path,
        callout_type=CALLOUT_TYPE,
        title=f"Contradicted by [[{winner_title}]]",
        rationale=rationale,
        frontmatter_key="contradicted_by",
        frontmatter_value=winner_path,
    )


__all__ = ["CALLOUT_TYPE", "mark_contradicted", "write_superseded_block"]
