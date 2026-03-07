"""Tests for research/ package — searcher, analyzer, pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from vaultmind.research.analyzer import _parse_analysis
from vaultmind.research.searcher import SearchResult, _run_yt_search


class TestParseAnalysis:
    def test_parse_valid_json(self) -> None:
        raw = json.dumps(
            {
                "summary": "Test summary",
                "key_themes": ["theme1", "theme2"],
                "comparative_insights": "Comparison here",
                "gaps": "Gap analysis",
                "recommendations": "Do this",
            }
        )
        result = _parse_analysis(raw)
        assert result.summary == "Test summary"
        assert result.key_themes == ["theme1", "theme2"]
        assert result.comparative_insights == "Comparison here"

    def test_parse_json_in_code_block(self) -> None:
        raw = '```json\n{"summary": "Block summary", "key_themes": []}\n```'
        result = _parse_analysis(raw)
        assert result.summary == "Block summary"

    def test_parse_invalid_json_fallback(self) -> None:
        result = _parse_analysis("This is not JSON at all")
        assert result.summary == "This is not JSON at all"
        assert result.key_themes == []


class TestSearchResult:
    def test_dataclass_fields(self) -> None:
        sr = SearchResult(
            video_id="abc123",
            title="Test Video",
            url="https://youtube.com/watch?v=abc123",
            channel="TestChannel",
            duration_seconds=120,
            view_count=1000,
            description="A test video",
        )
        assert sr.video_id == "abc123"
        assert sr.duration_seconds == 120


class TestRunYtSearch:
    @patch("vaultmind.research.searcher.subprocess.run")
    def test_parses_yt_dlp_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "id": "vid123",
                    "title": "Found Video",
                    "url": "https://youtube.com/watch?v=vid123",
                    "channel": "Channel",
                    "duration": 300,
                    "view_count": 5000,
                    "description": "desc",
                }
            )
            + "\n",
            stderr="",
        )
        results = _run_yt_search("test query", max_results=1)
        assert len(results) == 1
        assert results[0].video_id == "vid123"
        assert results[0].title == "Found Video"

    @patch("vaultmind.research.searcher.subprocess.run")
    def test_raises_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error message",
        )
        with pytest.raises(RuntimeError, match="yt-dlp search failed"):
            _run_yt_search("fail query", max_results=1)
