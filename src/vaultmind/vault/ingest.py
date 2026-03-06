"""URL ingestion — detect, fetch, and create vault notes from URLs."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from vaultmind.vault.security import validate_vault_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class SourceType(StrEnum):
    YOUTUBE = "youtube"
    ARTICLE = "article"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IngestResult:
    source_type: SourceType
    title: str
    content: str
    url: str
    metadata: dict[str, str] = field(default_factory=dict)


_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")

_YT_PATTERNS = [
    re.compile(
        r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/(?:embed|shorts|live)/)"
        r"([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:m\.youtube\.com|music\.youtube\.com)/watch\?.*v=([a-zA-Z0-9_-]{11})"
    ),
]


def detect_url(text: str) -> str | None:
    """Extract first URL from text."""
    m = _URL_RE.search(text)
    return m.group(0) if m else None


def classify_url(url: str) -> SourceType:
    """Classify URL as YouTube, article, or unknown."""
    for pat in _YT_PATTERNS:
        if pat.search(url):
            return SourceType.YOUTUBE
    if url.startswith(("http://", "https://")):
        return SourceType.ARTICLE
    return SourceType.UNKNOWN


def _extract_youtube_video_id(url: str) -> str:
    """Extract the 11-char video ID from a YouTube URL."""
    for pat in _YT_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    msg = f"Cannot extract video ID from: {url}"
    raise ValueError(msg)


async def fetch_youtube(url: str, language: str = "en") -> IngestResult:
    """Fetch YouTube transcript via youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _extract_youtube_video_id(url)
    segments = await asyncio.to_thread(
        YouTubeTranscriptApi.get_transcript, video_id, languages=[language]
    )
    transcript = " ".join(s["text"] for s in segments)

    # Attempt to get video title from page
    title = f"YouTube: {video_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=10)  # noqa: S310
        page = resp.read().decode("utf-8", errors="replace")[:50_000]
        m = re.search(r"<title>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
        if m:
            title = html.unescape(m.group(1).replace(" - YouTube", "").strip())
    except Exception:
        logger.debug("Could not fetch YouTube page title for %s", video_id)

    return IngestResult(
        source_type=SourceType.YOUTUBE,
        title=title,
        content=transcript,
        url=url,
        metadata={"video_id": video_id, "language": language},
    )


async def fetch_article(url: str) -> IngestResult:
    """Fetch article content via urllib (no extra dependencies)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=15)  # noqa: S310
    raw = resp.read().decode("utf-8", errors="replace")

    # Extract title
    title_match = re.search(r"<title>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    title = html.unescape(title_match.group(1).strip()) if title_match else url

    # Strip scripts, styles, then HTML tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return IngestResult(
        source_type=SourceType.ARTICLE,
        title=title,
        content=text,
        url=url,
    )


async def ingest_url(url: str, language: str = "en") -> IngestResult:
    """Route to appropriate fetcher based on URL classification."""
    source_type = classify_url(url)
    if source_type == SourceType.YOUTUBE:
        return await fetch_youtube(url, language=language)
    return await fetch_article(url)


def _sanitize_filename(name: str) -> str:
    """Create a filesystem-safe filename from a title."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100].strip("_")


def create_vault_note(
    result: IngestResult,
    vault_root: Path,
    inbox_folder: str = "00-inbox",
) -> Path:
    """Create a markdown note in the vault from an IngestResult."""
    folder = vault_root / inbox_folder
    folder.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(result.title) + ".md"
    note_path = folder / filename
    validate_vault_path(str(note_path), vault_root)

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    meta_lines = ""
    for k, v in result.metadata.items():
        meta_lines += f"{k}: {v}\n"

    content = f"""\
---
title: "{result.title}"
source: "{result.url}"
source_type: {result.source_type}
tags: [ingested]
created: {now}
{meta_lines}---

# {result.title}

**Source:** {result.url}
**Type:** {result.source_type}

---

{result.content}
"""
    note_path.write_text(content, encoding="utf-8")
    logger.info("Created vault note: %s", note_path)
    return note_path
