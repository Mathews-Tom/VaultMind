"""Tests for vault/ingest.py — URL detection, classification, and note creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.vault.ingest import (
    IngestResult,
    SourceType,
    _extract_youtube_video_id,
    _sanitize_filename,
    classify_url,
    create_vault_note,
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


class TestSanitizeFilename:
    def test_removes_special_chars(self) -> None:
        result = _sanitize_filename('hello: "world"')
        assert ":" not in result
        assert '"' not in result

    def test_truncates_long_names(self) -> None:
        assert len(_sanitize_filename("a" * 200)) <= 100


class TestCreateVaultNote:
    def test_creates_note_file(self, tmp_path: Path) -> None:
        result = IngestResult(
            source_type=SourceType.YOUTUBE,
            title="Test Video",
            content="This is the transcript.",
            url="https://youtube.com/watch?v=abc12345678",
            metadata={"video_id": "abc12345678"},
        )
        note_path = create_vault_note(result, tmp_path, inbox_folder="inbox")
        assert note_path.exists()
        content = note_path.read_text()
        assert "Test Video" in content
        assert "youtube" in content
        assert "This is the transcript." in content
        assert "abc12345678" in content

    def test_creates_inbox_directory(self, tmp_path: Path) -> None:
        result = IngestResult(
            source_type=SourceType.ARTICLE,
            title="Article",
            content="Content.",
            url="https://example.com",
        )
        note_path = create_vault_note(result, tmp_path, inbox_folder="new-inbox")
        assert (tmp_path / "new-inbox").is_dir()
        assert note_path.parent == tmp_path / "new-inbox"
