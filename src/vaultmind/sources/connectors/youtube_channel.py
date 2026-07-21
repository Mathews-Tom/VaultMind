"""YouTube-channel connector — new uploads reuse the existing transcript
pipeline (DEVELOPMENT_PLAN.md M8).

Channel-upload listing uses the same `yt-dlp` *subprocess* convention as
`research/searcher.py::_run_yt_search` (an external binary, not a Python
package dependency — `yt-dlp` is absent from `pyproject.toml` deps). Per-video
transcript fetch reuses `vault/ingest.py::fetch_youtube()` unmodified.

Cursor is ID-position based (`sources/cursor.py::filter_new_by_id`): a
channel's `/videos` listing is newest-first by YouTube's own default sort, so
"newer than cursor" means "listed before `last_seen_id`" — the same shape
`github_activity.py`'s commit listing uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import TYPE_CHECKING

from vaultmind.sources.cursor import filter_new_by_id
from vaultmind.sources.models import ConnectorDefinition, FetchResult, SourceItem
from vaultmind.vault.ingest import fetch_youtube

if TYPE_CHECKING:
    from vaultmind.sources.models import ConnectorState, SourceInstance

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LISTING = 25
_LISTING_TIMEOUT_SECONDS = 60


def _run_channel_listing(channel_url: str, limit: int) -> str:
    """Run yt-dlp's `--flat-playlist --dump-json` channel listing
    synchronously (called via `to_thread`). Returns raw NDJSON stdout."""
    cmd = [
        "yt-dlp",
        f"{channel_url.rstrip('/')}/videos",
        "--dump-json",
        "--flat-playlist",
        "--playlist-end",
        str(limit),
        "--no-warnings",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_LISTING_TIMEOUT_SECONDS)  # noqa: S603
    if proc.returncode != 0:
        logger.error("yt-dlp channel listing failed: %s", proc.stderr[:500])
        msg = f"yt-dlp channel listing failed: {proc.stderr[:200]}"
        raise RuntimeError(msg)
    return proc.stdout


def parse_channel_listing(raw_ndjson: str) -> list[SourceItem]:
    """Parse yt-dlp `--dump-json --flat-playlist` NDJSON into listing stubs.

    `content` is left empty here — `fetch()` fills it in per new item via
    `fetch_youtube()`'s transcript fetch, keeping listing (cheap, one call)
    and transcript fetch (one call per new video) separate.
    """
    items: list[SourceItem] = []
    for line in raw_ndjson.strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        video_id = data.get("id", "")
        if not video_id:
            continue
        upload_date = str(data.get("upload_date", "") or "")
        published_at = (
            f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
            if len(upload_date) == 8
            else ""
        )
        url = (
            data.get("url")
            or data.get("webpage_url")
            or f"https://www.youtube.com/watch?v={video_id}"
        )
        items.append(
            SourceItem(
                item_id=video_id,
                title=str(data.get("title", "Unknown")),
                content="",
                url=url,
                published_at=published_at,
            )
        )
    return items


async def fetch(instance: SourceInstance, state: ConnectorState) -> FetchResult:
    """Fetch new-upload transcripts from `instance.target` (a channel URL)
    since `state.last_seen_id`."""
    limit = int(instance.options.get("max_listing", str(_DEFAULT_MAX_LISTING)))
    language = instance.options.get("language", "en")

    raw = await asyncio.to_thread(_run_channel_listing, instance.target, limit)
    listing = parse_channel_listing(raw)
    new_stubs = filter_new_by_id(listing, state.last_seen_id)

    items: list[SourceItem] = []
    for stub in new_stubs:
        try:
            transcript = await fetch_youtube(stub.url, language=language)
            items.append(
                SourceItem(
                    item_id=stub.item_id,
                    title=transcript.title,
                    content=transcript.content,
                    url=stub.url,
                    published_at=stub.published_at,
                )
            )
        except Exception:
            logger.exception("Transcript fetch failed for %s — ingesting title only", stub.url)
            items.append(stub)

    next_cursor_id = listing[0].item_id if listing else None
    return FetchResult(items=items, next_cursor_id=next_cursor_id)


YOUTUBE_CHANNEL_CONNECTOR = ConnectorDefinition(
    kind="youtube-channel",
    fetch=fetch,
    description="YouTube channel new uploads, via the existing transcript pipeline",
)


__all__ = ["YOUTUBE_CHANNEL_CONNECTOR", "fetch", "parse_channel_listing"]
