"""URL ingestion — detect, fetch, and create vault notes from URLs."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum

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
    re.compile(r"(?:m\.youtube\.com|music\.youtube\.com)/watch\?.*v=([a-zA-Z0-9_-]{11})"),
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

        def read_title() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                raw: bytes = resp.read(50_000)
            return raw.decode("utf-8", errors="replace")

        page = await asyncio.to_thread(read_title)
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

    def read_article() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw: bytes = resp.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Fetched article exceeds its byte limit")
        return raw.decode("utf-8", errors="replace")

    raw = await asyncio.to_thread(read_article)

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
