"""Tests for sources/connectors/youtube_channel.py — channel-listing parsing,
cursor filtering, and fetch orchestration (no live yt-dlp/network calls)."""

from __future__ import annotations

import json

import pytest

from vaultmind.sources.connectors import youtube_channel
from vaultmind.sources.connectors.youtube_channel import fetch, parse_channel_listing
from vaultmind.sources.models import ConnectorState, SourceInstance
from vaultmind.vault.ingest import IngestResult, SourceType

_LISTING_NDJSON = "\n".join(
    json.dumps(entry)
    for entry in [
        {
            "id": "vid-newest",
            "title": "Newest upload",
            "webpage_url": "https://www.youtube.com/watch?v=vid-newest",
            "upload_date": "20260605",
        },
        {
            "id": "vid-middle",
            "title": "Middle upload",
            "webpage_url": "https://www.youtube.com/watch?v=vid-middle",
            "upload_date": "20260603",
        },
        {
            "id": "vid-oldest",
            "title": "Oldest upload",
            "webpage_url": "https://www.youtube.com/watch?v=vid-oldest",
            "upload_date": "20260601",
        },
    ]
)


class TestParseChannelListing:
    def test_parses_all_entries(self) -> None:
        items = parse_channel_listing(_LISTING_NDJSON)
        assert [i.item_id for i in items] == ["vid-newest", "vid-middle", "vid-oldest"]

    def test_parses_upload_date_to_iso(self) -> None:
        items = parse_channel_listing(_LISTING_NDJSON)
        assert items[0].published_at == "2026-06-05"

    def test_content_left_empty_pending_transcript_fetch(self) -> None:
        items = parse_channel_listing(_LISTING_NDJSON)
        assert all(item.content == "" for item in items)

    def test_skips_entries_without_id(self) -> None:
        raw = json.dumps({"title": "no id here"})
        assert parse_channel_listing(raw) == []

    def test_blank_lines_ignored(self) -> None:
        raw = _LISTING_NDJSON + "\n\n"
        assert len(parse_channel_listing(raw)) == 3


class TestFetch:
    @pytest.fixture
    def instance(self) -> SourceInstance:
        return SourceInstance(
            name="yt-fixture",
            kind="youtube-channel",
            target="https://www.youtube.com/@fixture-channel",
            enabled=True,
        )

    @pytest.fixture(autouse=True)
    def _stub_yt_dlp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            youtube_channel, "_run_channel_listing", lambda url, limit: _LISTING_NDJSON
        )

        async def _fake_fetch_youtube(url: str, language: str = "en") -> IngestResult:
            video_id = url.rsplit("=", 1)[-1]
            return IngestResult(
                source_type=SourceType.YOUTUBE,
                title=f"Transcript title for {video_id}",
                content=f"Full transcript text for {video_id}.",
                url=url,
            )

        monkeypatch.setattr(youtube_channel, "fetch_youtube", _fake_fetch_youtube)

    async def test_first_run_ingests_all_with_transcripts(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="yt-fixture")
        result = await fetch(instance, state)
        assert len(result.items) == 3
        assert result.next_cursor_id == "vid-newest"
        newest = next(i for i in result.items if i.item_id == "vid-newest")
        assert newest.content == "Full transcript text for vid-newest."
        assert newest.title == "Transcript title for vid-newest"

    async def test_rerun_with_cursor_ingests_only_newer(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="yt-fixture", last_seen_id="vid-middle")
        result = await fetch(instance, state)
        assert [i.item_id for i in result.items] == ["vid-newest"]
        assert result.next_cursor_id == "vid-newest"

    async def test_cursor_at_newest_ingests_nothing(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="yt-fixture", last_seen_id="vid-newest")
        result = await fetch(instance, state)
        assert result.items == []
        assert result.next_cursor_id == "vid-newest"

    async def test_unknown_cursor_falls_back_to_ingesting_everything(
        self, instance: SourceInstance
    ) -> None:
        state = ConnectorState(instance_name="yt-fixture", last_seen_id="vid-not-in-listing")
        result = await fetch(instance, state)
        assert len(result.items) == 3

    async def test_transcript_failure_falls_back_to_title_only_item(
        self, instance: SourceInstance, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _failing_fetch_youtube(url: str, language: str = "en") -> IngestResult:
            raise RuntimeError("transcript unavailable")

        monkeypatch.setattr(youtube_channel, "fetch_youtube", _failing_fetch_youtube)
        state = ConnectorState(instance_name="yt-fixture")
        result = await fetch(instance, state)
        # Every listed video is still ingested (title-only), never silently dropped.
        assert len(result.items) == 3
        assert all(item.content == "" for item in result.items)
