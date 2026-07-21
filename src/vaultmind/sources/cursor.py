"""Shared ID-position cursor filtering for connectors whose source API
returns items newest-first with no reliable per-item timestamp
(`youtube-channel`'s `yt-dlp` channel listing, `github-activity`'s commits
API) — distinct from `connectors/rss.py`'s own timestamp-based filtering,
which RSS/Atom feeds need because they have no guaranteed item order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaultmind.sources.models import SourceItem


def filter_new_by_id(items: list[SourceItem], last_seen_id: str) -> list[SourceItem]:
    """Return items ordered before `last_seen_id` in `items` (assumed
    newest-first). Returns every item if `last_seen_id` is empty (first
    run) or not found (source rotated stale entries out, or the cursor
    predates the source's retained history) — never raises, never silently
    drops the whole batch."""
    if not last_seen_id:
        return list(items)
    ids = [item.item_id for item in items]
    if last_seen_id not in ids:
        return list(items)
    idx = ids.index(last_seen_id)
    return items[:idx]


__all__ = ["filter_new_by_id"]
