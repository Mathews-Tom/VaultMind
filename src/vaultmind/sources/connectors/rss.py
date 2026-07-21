"""RSS/Atom connector — blogs and newsletters (DEVELOPMENT_PLAN.md M8).

Parses RSS 2.0 `<item>` and Atom `<entry>` elements via stdlib
`xml.etree.ElementTree` (no `feedparser` dependency, per the milestone's
"minimal dependencies" constraint). `target` is either an `http(s)://` feed
URL (fetched via `urllib.request`, matching `vault/ingest.py::fetch_article`'s
existing "urllib, no extra dependencies" convention) or a local filesystem
path (read directly — used by the `rss-fixture` instance in
`config/sources.toml` for deterministic tests and manual CLI verification).

Feeds have no guaranteed item ordering, so "newer than the stored cursor" is
determined by parsed timestamp (`pubDate`/`updated`/`published`), not by
document position or `last_seen_id` alone.
"""

from __future__ import annotations

import asyncio
import email.utils
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vaultmind.sources.models import ConnectorDefinition, FetchResult, SourceItem

if TYPE_CHECKING:
    from vaultmind.sources.models import ConnectorState, SourceInstance

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_TIMEOUT_SECONDS = 15


def _fetch_raw_feed(target: str) -> str:
    """Fetch feed XML from an `http(s)://` URL or a local filesystem path."""
    if target.startswith(("http://", "https://")):
        req = urllib.request.Request(target, headers={"User-Agent": "VaultMind-Sources/0.1"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            raw_bytes: bytes = resp.read()
            return raw_bytes.decode("utf-8", errors="replace")
    return Path(target).read_text(encoding="utf-8")


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_pub_date(raw: str) -> datetime | None:
    """Parse an RSS `pubDate` (RFC 822) or Atom `updated`/`published`
    (ISO 8601) timestamp. Returns `None` for empty or unparseable input —
    the caller treats an unparseable timestamp as "always new" rather than
    silently dropping the item."""
    if not raw:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _parse_rss_item(item: ET.Element) -> SourceItem | None:
    guid = _text(item.find("guid")) or _text(item.find("link"))
    if not guid:
        return None
    title = _text(item.find("title")) or guid
    link = _text(item.find("link"))
    description = _text(item.find("description"))
    pub_date = _text(item.find("pubDate"))
    return SourceItem(
        item_id=guid, title=title, content=description, url=link or guid, published_at=pub_date
    )


def _parse_atom_entry(entry: ET.Element) -> SourceItem | None:
    entry_id = _text(entry.find(f"{_ATOM_NS}id"))
    if not entry_id:
        return None
    title = _text(entry.find(f"{_ATOM_NS}title")) or entry_id
    link_el = entry.find(f"{_ATOM_NS}link")
    link = link_el.get("href", "") if link_el is not None else ""
    summary = _text(entry.find(f"{_ATOM_NS}summary")) or _text(entry.find(f"{_ATOM_NS}content"))
    updated = _text(entry.find(f"{_ATOM_NS}updated")) or _text(entry.find(f"{_ATOM_NS}published"))
    return SourceItem(
        item_id=entry_id, title=title, content=summary, url=link or entry_id, published_at=updated
    )


def parse_feed(raw_xml: str) -> list[SourceItem]:
    """Parse every RSS `<item>` and Atom `<entry>` element from raw feed XML.

    Items without a usable id/guid (RSS) or `<id>` (Atom) are skipped —
    there is nothing stable to key cursor advancement on. Order is
    whatever the document provides; callers must not assume it is
    chronological (see `filter_new_items`).
    """
    root = ET.fromstring(raw_xml)  # noqa: S314 - configured/fixture feed, not arbitrary user input
    items: list[SourceItem] = []
    for item_el in root.iter("item"):
        parsed = _parse_rss_item(item_el)
        if parsed is not None:
            items.append(parsed)
    for entry_el in root.iter(f"{_ATOM_NS}entry"):
        parsed = _parse_atom_entry(entry_el)
        if parsed is not None:
            items.append(parsed)
    return items


def filter_new_items(items: list[SourceItem], last_seen_at: datetime | None) -> list[SourceItem]:
    """Return only items published strictly after `last_seen_at`.

    An item with an unparseable or missing timestamp is always included
    (never silently dropped) — a feed using a nonstandard date format
    should not lose content, just fall back to "always new" for that item.
    `last_seen_at=None` (first run, or a cursor with no advanced timestamp
    yet) includes every item.
    """
    if last_seen_at is None:
        return list(items)
    new_items = []
    for item in items:
        published = parse_pub_date(item.published_at)
        if published is None or published > last_seen_at:
            new_items.append(item)
    return new_items


def _latest_item(items: list[SourceItem]) -> SourceItem | None:
    """The item with the latest parseable timestamp, or `None` if `items`
    is empty or none of them have a parseable timestamp."""
    latest: SourceItem | None = None
    latest_at: datetime | None = None
    for item in items:
        published = parse_pub_date(item.published_at)
        if published is not None and (latest_at is None or published > latest_at):
            latest, latest_at = item, published
    return latest


async def fetch(instance: SourceInstance, state: ConnectorState) -> FetchResult:
    """Fetch every item from `instance.target` published after
    `state.last_seen_at`."""
    raw = await asyncio.to_thread(_fetch_raw_feed, instance.target)
    all_items = parse_feed(raw)
    new_items = filter_new_items(all_items, state.last_seen_at)
    latest = _latest_item(all_items)
    next_cursor_at = parse_pub_date(latest.published_at) if latest else None
    if next_cursor_at is not None and state.last_seen_at is not None:
        next_cursor_at = max(next_cursor_at, state.last_seen_at)
    next_cursor_id = latest.item_id if latest else None
    return FetchResult(
        items=new_items, next_cursor_id=next_cursor_id, next_cursor_at=next_cursor_at
    )


RSS_CONNECTOR = ConnectorDefinition(
    kind="rss", fetch=fetch, description="RSS/Atom blogs and newsletters"
)


__all__ = [
    "RSS_CONNECTOR",
    "fetch",
    "filter_new_items",
    "parse_feed",
    "parse_pub_date",
]
