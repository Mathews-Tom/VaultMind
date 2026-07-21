"""Tests for sources/connectors/rss.py — feed parsing, cursor filtering,
and fixture-driven fetch (no live network calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vaultmind.sources.connectors.rss import (
    fetch,
    filter_new_items,
    parse_feed,
    parse_pub_date,
)
from vaultmind.sources.models import ConnectorState, SourceInstance, SourceItem

_FIXTURE_FEED = "tests/fixtures/sources/rss_feed.xml"

_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Fixture</title>
  <entry>
    <id>atom-entry-1</id>
    <title>Atom entry one</title>
    <link href="https://example.com/atom/1"/>
    <updated>2026-06-01T09:00:00Z</updated>
    <summary>First atom entry.</summary>
  </entry>
  <entry>
    <id>atom-entry-2</id>
    <title>Atom entry two</title>
    <link href="https://example.com/atom/2"/>
    <updated>2026-06-05T09:00:00Z</updated>
    <content>Second atom entry, via content instead of summary.</content>
  </entry>
</feed>
"""


class TestParsePubDate:
    def test_parses_rfc822_pubdate(self) -> None:
        parsed = parse_pub_date("Mon, 01 Jun 2026 09:00:00 GMT")
        assert parsed == datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

    def test_parses_iso8601_updated(self) -> None:
        parsed = parse_pub_date("2026-06-05T09:00:00Z")
        assert parsed == datetime(2026, 6, 5, 9, 0, tzinfo=UTC)

    def test_empty_string_returns_none(self) -> None:
        assert parse_pub_date("") is None

    def test_unparseable_string_returns_none(self) -> None:
        assert parse_pub_date("not a date") is None


class TestParseFeed:
    def test_parses_rss_items_from_fixture(self) -> None:
        raw = Path(_FIXTURE_FEED).read_text(encoding="utf-8")
        items = parse_feed(raw)
        assert len(items) == 3
        ids = {item.item_id for item in items}
        assert ids == {"fixture-post-1", "fixture-post-2", "fixture-post-3"}

    def test_rss_item_fields(self) -> None:
        raw = Path(_FIXTURE_FEED).read_text(encoding="utf-8")
        items = {item.item_id: item for item in parse_feed(raw)}
        newest = items["fixture-post-3"]
        assert newest.title == "Third fixture post"
        assert newest.url == "https://example.com/posts/third-fixture-post"
        assert "newest fixture post" in newest.content
        assert newest.published_at == "Fri, 05 Jun 2026 09:00:00 GMT"

    def test_parses_atom_entries(self) -> None:
        items = parse_feed(_ATOM_FEED)
        assert len(items) == 2
        ids = {item.item_id for item in items}
        assert ids == {"atom-entry-1", "atom-entry-2"}

    def test_atom_entry_falls_back_to_content_when_no_summary(self) -> None:
        items = {item.item_id: item for item in parse_feed(_ATOM_FEED)}
        assert items["atom-entry-2"].content == "Second atom entry, via content instead of summary."

    def test_item_without_guid_or_link_is_skipped(self) -> None:
        raw = """<?xml version="1.0"?>
        <rss version="2.0"><channel>
        <item><title>No identifier</title><description>orphan</description></item>
        </channel></rss>"""
        assert parse_feed(raw) == []


class TestFilterNewItems:
    def _items(self) -> list[SourceItem]:
        raw = Path(_FIXTURE_FEED).read_text(encoding="utf-8")
        return parse_feed(raw)

    def test_none_cursor_includes_everything(self) -> None:
        assert len(filter_new_items(self._items(), None)) == 3

    def test_cursor_excludes_items_not_after_it(self) -> None:
        cursor = datetime(2026, 6, 3, 9, 0, tzinfo=UTC)  # == fixture-post-2's pubDate
        new = filter_new_items(self._items(), cursor)
        assert {item.item_id for item in new} == {"fixture-post-3"}

    def test_cursor_after_newest_excludes_all(self) -> None:
        cursor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
        assert filter_new_items(self._items(), cursor) == []

    def test_item_with_unparseable_timestamp_always_included(self) -> None:
        items = [SourceItem(item_id="x", title="x", content="x", url="x", published_at="garbage")]
        cursor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
        assert filter_new_items(items, cursor) == items


class TestFetch:
    @pytest.fixture
    def instance(self) -> SourceInstance:
        return SourceInstance(name="rss-fixture", kind="rss", target=_FIXTURE_FEED, enabled=True)

    async def test_first_run_ingests_all_items(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="rss-fixture")
        result = await fetch(instance, state)
        assert len(result.items) == 3
        assert result.next_cursor_id == "fixture-post-3"
        assert result.next_cursor_at == datetime(2026, 6, 5, 9, 0, tzinfo=UTC)

    async def test_rerun_with_advanced_cursor_ingests_nothing_new(
        self, instance: SourceInstance
    ) -> None:
        first_state = ConnectorState(instance_name="rss-fixture")
        first_result = await fetch(instance, first_state)

        advanced_state = ConnectorState(
            instance_name="rss-fixture",
            last_seen_id=first_result.next_cursor_id or "",
            last_seen_at=first_result.next_cursor_at,
        )
        second_result = await fetch(instance, advanced_state)
        assert second_result.items == []
        assert second_result.next_cursor_at == first_result.next_cursor_at

    async def test_partial_cursor_ingests_only_newer_items(self, instance: SourceInstance) -> None:
        state = ConnectorState(
            instance_name="rss-fixture",
            last_seen_at=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),  # == fixture-post-1
        )
        result = await fetch(instance, state)
        assert {item.item_id for item in result.items} == {"fixture-post-2", "fixture-post-3"}

    async def test_local_file_target_no_network(self, instance: SourceInstance) -> None:
        """Fetching a local-path target never touches urllib — CI stays network-free."""
        assert not instance.target.startswith(("http://", "https://"))
        state = ConnectorState(instance_name="rss-fixture")
        result = await fetch(instance, state)
        assert len(result.items) == 3
