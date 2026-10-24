"""URL classification and bounded extraction through the existing fetch seam."""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock

import pytest

from vaultmind.vault.ingest import (
    SourceType,
    _extract_youtube_video_id,
    classify_url,
    detect_url,
)


class TestDetectUrl:
    def test_plain_text_no_url(self) -> None:
        assert detect_url("hello world no urls here") is None

    def test_extracts_http_url(self) -> None:
        assert detect_url("check out https://example.com/page") == "https://example.com/page"

    def test_extracts_first_url(self) -> None:
        result = detect_url("visit https://a.com and https://b.com")
        assert result == "https://a.com"

    def test_youtube_url(self) -> None:
        text = "watch this https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert detect_url(text) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class TestClassifyUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/live/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
    )
    def test_youtube_urls(self, url: str) -> None:
        assert classify_url(url) == SourceType.YOUTUBE

    def test_article_url(self) -> None:
        assert classify_url("https://example.com/article") == SourceType.ARTICLE

    def test_unknown_url(self) -> None:
        assert classify_url("ftp://files.example.com") == SourceType.UNKNOWN


class TestExtractYoutubeVideoId:
    def test_standard_watch(self) -> None:
        assert _extract_youtube_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        assert _extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot extract video ID"):
            _extract_youtube_video_id("https://example.com")


def test_article_extraction_reads_and_closes_response_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vaultmind.vault.ingest import fetch_article

    response = io.BytesIO(b"<title>Example</title><script>hidden</script><p>Exact article.</p>")
    opener = MagicMock(return_value=response)
    monkeypatch.setattr("vaultmind.vault.ingest.urllib.request.urlopen", opener)
    result = asyncio.run(fetch_article("https://example.com"))
    assert result.title == "Example"
    assert result.content == "Example Exact article."
    assert result.source_type is SourceType.ARTICLE
    assert "hidden" not in result.content
    assert response.closed


def test_article_byte_limit_refuses_partial_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from vaultmind.vault.ingest import fetch_article

    response = io.BytesIO(b"x" * (2 * 1024 * 1024 + 1))
    monkeypatch.setattr(
        "vaultmind.vault.ingest.urllib.request.urlopen", MagicMock(return_value=response)
    )
    with pytest.raises(ValueError, match="byte limit"):
        asyncio.run(fetch_article("https://example.com"))
    assert response.closed
