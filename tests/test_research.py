"""Tests for research/ package — searcher, analyzer, pipeline."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaultmind.config import Settings
from vaultmind.knowledge.intake import CaptureResult, SourceIntake
from vaultmind.llm.client import LLMResponse
from vaultmind.research.analyzer import _parse_analysis
from vaultmind.research.pipeline import (
    PipelineResult,
    ResearchConfig,
    _render_source_note,
    run_research_pipeline,
)
from vaultmind.research.searcher import SearchResult, _run_yt_search
from vaultmind.vault.parser import VaultParser
from vaultmind.wiki.workspace import proposal_digest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from vaultmind.knowledge.intake import SourceRecord


class TestRenderSourceNote:
    def test_unapproved_source_projection_uses_safe_metadata(self) -> None:
        import frontmatter

        capture = CaptureResult("source-id", "_wiki/sources/source-id.md", "hash", "added")
        content = _render_source_note(
            'A "quoted": video',
            "Transcript text.",
            "https://example.invalid/video",
            "a, complex: query",
            capture,
            created="2026-10-24T00:00:00Z",
        )
        post = frontmatter.loads(content)
        assert post["title"] == 'A "quoted": video'
        assert post["authority"] == 1
        assert post["approved"] is False
        assert post["source_id"] == "source-id"
        assert post["tags"] == ["research", "a, complex: query"]


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


@pytest.fixture
def intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SourceIntake]:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    settings = Settings(vault={"path": tmp_path})
    with SourceIntake(settings) as selected:
        yield selected


def _records(intake: SourceIntake) -> dict[str, SourceRecord]:
    records = {}
    for path in (intake.workspace.root / "_wiki" / "sources").glob("*.intake.json"):
        record = intake.read(path.name.removesuffix(".intake.json"))
        assert record.provenance is not None
        records[record.provenance["role"]] = record
    return records


def _found(video_id: str = "video", title: str = "Shared title") -> SearchResult:
    return SearchResult(
        video_id,
        title,
        f"https://youtube.com/watch?v={video_id}",
        "Channel",
        120,
        10,
        "Available description",
    )


def _fetched(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[dict[str, object]],
    found: list[SearchResult] | None = None,
) -> AsyncMock:
    from youtube_transcript_api import YouTubeTranscriptApi

    search = AsyncMock(return_value=found if found is not None else [_found()])
    monkeypatch.setattr("vaultmind.research.pipeline.search_youtube", search)
    monkeypatch.setattr(
        YouTubeTranscriptApi, "get_transcript", MagicMock(return_value=segments), raising=False
    )
    return search


def _analysis_client() -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text=json.dumps(
            {
                "summary": "Parsed research summary.",
                "key_themes": ["A theme"],
                "comparative_insights": "Comparison.",
                "gaps": "",
                "recommendations": "Review it.",
            }
        ),
        model="scripted",
        usage={},
    )
    return client


async def _research(
    intake: SourceIntake,
    client: MagicMock,
    store: MagicMock | None = None,
    query: str = "topic",
) -> PipelineResult:
    selected_store = store if store is not None else MagicMock()
    selected_store.index_single_note.return_value = 1
    return await run_research_pipeline(
        query,
        intake.workspace.root,
        client,
        selected_store,
        VaultParser(intake.workspace.settings.vault),
        ResearchConfig(),
        "scripted",
        intake=intake,
    )


async def test_research_preserves_segments_full_text_and_actual_request_before_model(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    segments = [
        {"text": "x" * 9000 + "UNTRUNCATED-END\r\n", "start": 0.0, "duration": 1.5},
        {"text": "café", "start": 1.5, "duration": 2.0},
    ]
    _fetched(monkeypatch, segments)
    client = _analysis_client()
    response = client.complete.return_value
    transcript = " ".join(str(segment["text"]) for segment in segments)

    def complete(**kwargs: object) -> LLMResponse:
        records = _records(intake)
        original = json.loads(records["transcript-segments"].text or "")
        assert original["segments"] == segments
        assert original["search"]["video_id"] == "video"
        assert original["language"] == "en"
        assert records["transcript-text"].content == transcript.encode("utf-8")
        selected = json.loads(records["analysis-request"].text or "")
        messages = kwargs["messages"]
        assert messages[0].content == selected["messages"][0]["content"]  # type: ignore[index]
        assert selected["messages"][0]["content"] == records["analysis-input"].text
        assert "UNTRUNCATED-END" not in selected["messages"][0]["content"]
        for key in ("model", "max_tokens", "system"):
            assert kwargs[key] == selected[key]
        assert "analysis" not in records
        return response

    client.complete.side_effect = complete
    store = MagicMock()
    result = await _research(intake, client, store)
    assert result.error == ""
    assert result.summary_path is not None and result.summary_path.is_file()
    assert len(result.sources_created) == 1
    assert len(result.original_source_ids) == 1
    assert result.request_source_id is not None
    assert result.derivative_source_id is not None
    derivative = intake.read(result.derivative_source_id)
    assert derivative.provenance is not None
    assert derivative.provenance["original_retained"] is False
    assert derivative.provenance["parents"][0]["id"] == result.request_source_id
    assert result.summary_path.read_bytes() == derivative.content
    assert derivative.snapshot["id"] not in result.original_source_ids
    assert "approved: false" in (derivative.text or "")
    assert store.index_single_note.call_count == 2
    assert all(
        capture.indexed
        for capture in result.captures
        if capture.source_id
        in {
            result.derivative_source_id,
            _records(intake)["transcript-text"].source_id,
        }
    )
    assert not (intake.workspace.root / "wiki").exists()


async def test_research_failed_model_retains_originals_without_fake_summary(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(monkeypatch, [{"text": "Exact original", "start": 0.0, "duration": 1.0}])
    client = _analysis_client()
    client.complete.side_effect = RuntimeError("scripted provider failure")
    result = await _research(intake, client)
    assert result.error
    assert result.summary_path is None
    assert result.derivative_source_id is None
    assert result.request_source_id is not None
    assert result.original_source_ids
    assert "analysis" not in _records(intake)
    assert any(
        stage.stage == "provider" and stage.status == "failed" for stage in result.dispositions
    )
    assert len(result.sources_created) == 1
    assert all(path.is_file() for path in result.sources_created)
    client.complete.assert_called_once()


async def test_research_index_failure_is_not_capture_or_approval_success(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(monkeypatch, [{"text": "Original", "start": 0.0, "duration": 1.0}])
    store = MagicMock()
    store.index_single_note.side_effect = RuntimeError("scripted index failure")
    result = await _research(intake, _analysis_client(), store)
    assert result.summary_path is not None and result.summary_path.is_file()
    assert result.derivative_source_id is not None
    assert all(not capture.indexed and not capture.approved for capture in result.captures)
    assert (
        sum(stage.stage == "index" and stage.status == "failed" for stage in result.dispositions)
        == 2
    )
    intake.read(result.derivative_source_id)


@pytest.mark.parametrize(
    "original",
    [
        "---\ndo_not_retain: true\n---\nPrivate source",
        "---\nretention: do-not-retain\n---\nPrivate source",
        "---\nunterminated: true",
        "unsupported\x00control",
        "x" * (256 * 1024 + 1),
    ],
    ids=["do-not-retain", "retention-policy", "malformed-frontmatter", "null-byte", "oversized"],
)
async def test_research_original_policy_is_checked_before_json_envelope(
    original: str, intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(monkeypatch, [{"text": original, "start": 0.0, "duration": 1.0}])
    client = _analysis_client()
    result = await _research(intake, client)
    assert result.error
    assert result.summary_path is None
    assert result.original_source_ids == []
    assert set(_records(intake)) == {"query"}
    client.complete.assert_not_called()


async def test_research_query_retention_refusal_precedes_search(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    search = _fetched(monkeypatch, [])
    client = _analysis_client()
    result = await _research(
        intake,
        client,
        query="---\ndo_not_retain: true\n---\nPrivate query",
    )
    assert result.error
    assert not (intake.workspace.root / "_wiki" / "sources").exists()
    search.assert_not_called()
    client.complete.assert_not_called()


async def test_research_empty_search_has_no_summary_path(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(monkeypatch, [], found=[])
    client = _analysis_client()
    result = await _research(intake, client)
    assert result.summary_path is None
    assert result.sources_created == []
    assert result.analysis_summary == "No results found."
    client.complete.assert_not_called()


async def test_research_title_collisions_do_not_overwrite_discovery_notes(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(
        monkeypatch,
        [{"text": "Original", "start": 0.0, "duration": 1.0}],
        found=[_found("one"), _found("two")],
    )
    result = await _research(intake, _analysis_client())
    assert result.error == ""
    assert len(set(result.sources_created)) == 2
    assert len(set(result.original_source_ids)) == 2
    assert all(path.is_file() for path in result.sources_created)


async def test_research_dependency_limit_refuses_before_model(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(
        monkeypatch,
        [{"text": "Original", "start": 0.0, "duration": 1.0}],
        found=[_found(f"video-{index}") for index in range(30)],
    )
    client = _analysis_client()
    result = await _research(intake, client)
    assert result.error
    assert result.summary_path is None
    assert result.derivative_source_id is None
    assert len(result.original_source_ids) == 30
    client.complete.assert_not_called()


async def test_research_derivative_id_uses_explicit_existing_wiki_publication(
    intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetched(monkeypatch, [{"text": "Original evidence.", "start": 0.0, "duration": 1.0}])
    client = _analysis_client()
    result = await _research(intake, client)
    assert result.error == ""
    assert result.derivative_source_id is not None
    source_id = result.derivative_source_id
    root = intake.workspace.root
    draft = root / "reviewed-research.md"
    draft.write_text(
        "# Research evidence\n\n## Current understanding\n\n"
        f"An explicitly reviewed interpretation. [[_wiki/sources/{source_id}]]\n"
    )
    workspace = intake.workspace
    proposal = workspace.propose("wiki/research-evidence.md", [source_id], draft_file=draft)
    assert not (root / "wiki" / "research-evidence.md").exists()
    digest = proposal_digest(proposal.payload)
    assert digest in workspace.review(proposal.proposal_id)
    workspace.apply(proposal.proposal_id, digest)
    assert (root / "wiki" / "research-evidence.md").is_file()
    assert not intake.read(source_id).provenance["original_retained"]
    assert all(not capture.approved for capture in result.captures)
    client.complete.assert_called_once()
