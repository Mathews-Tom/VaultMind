"""YouTube search via yt-dlp subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    video_id: str
    title: str
    url: str
    channel: str
    duration_seconds: int
    view_count: int
    description: str


def _run_yt_search(query: str, max_results: int) -> list[SearchResult]:
    """Run yt-dlp search synchronously (called via to_thread)."""
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
        "--no-warnings",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # noqa: S603

    if proc.returncode != 0:
        logger.error("yt-dlp search failed: %s", proc.stderr[:500])
        msg = f"yt-dlp search failed: {proc.stderr[:200]}"
        raise RuntimeError(msg)

    results: list[SearchResult] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        video_id = data.get("id", "")
        results.append(
            SearchResult(
                video_id=video_id,
                title=data.get("title", "Unknown"),
                url=data.get("url", "")
                or data.get("webpage_url", "")
                or f"https://www.youtube.com/watch?v={video_id}",
                channel=data.get("channel", "") or data.get("uploader", "Unknown"),
                duration_seconds=int(data.get("duration", 0) or 0),
                view_count=int(data.get("view_count", 0) or 0),
                description=data.get("description", "") or "",
            )
        )

    return results


async def search_youtube(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search YouTube using yt-dlp. Non-blocking."""
    return await asyncio.to_thread(_run_yt_search, query, max_results)
