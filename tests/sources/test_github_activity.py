"""Tests for sources/connectors/github_activity.py — commit parsing, cursor
filtering, and token handling (no live network calls)."""

from __future__ import annotations

import pytest

from vaultmind.sources.connectors import github_activity
from vaultmind.sources.connectors.github_activity import fetch, parse_commits
from vaultmind.sources.models import ConnectorState, SourceInstance

_RAW_COMMITS = [
    {
        "sha": "sha-newest",
        "commit": {
            "message": "Newest commit\n\nBody text.",
            "author": {"date": "2026-06-05T09:00:00Z"},
        },
        "html_url": "https://github.com/acme/repo/commit/sha-newest",
    },
    {
        "sha": "sha-middle",
        "commit": {
            "message": "Middle commit",
            "author": {"date": "2026-06-03T09:00:00Z"},
        },
        "html_url": "https://github.com/acme/repo/commit/sha-middle",
    },
    {
        "sha": "sha-oldest",
        "commit": {
            "message": "Oldest commit",
            "author": {"date": "2026-06-01T09:00:00Z"},
        },
        "html_url": "https://github.com/acme/repo/commit/sha-oldest",
    },
]


class TestParseCommits:
    def test_parses_all_commits(self) -> None:
        items = parse_commits(_RAW_COMMITS, "acme/repo")
        assert [i.item_id for i in items] == ["sha-newest", "sha-middle", "sha-oldest"]

    def test_title_is_first_line_of_message(self) -> None:
        items = parse_commits(_RAW_COMMITS, "acme/repo")
        assert items[0].title == "acme/repo: Newest commit"

    def test_content_is_full_message(self) -> None:
        items = parse_commits(_RAW_COMMITS, "acme/repo")
        assert items[0].content == "Newest commit\n\nBody text."

    def test_skips_commits_without_sha(self) -> None:
        assert parse_commits([{"commit": {"message": "no sha"}}], "acme/repo") == []

    def test_published_at_from_author_date(self) -> None:
        items = parse_commits(_RAW_COMMITS, "acme/repo")
        assert items[0].published_at == "2026-06-05T09:00:00Z"


class TestFetch:
    @pytest.fixture
    def instance(self) -> SourceInstance:
        return SourceInstance(
            name="gh-fixture", kind="github-activity", target="acme/repo", enabled=True
        )

    @pytest.fixture(autouse=True)
    def _stub_github_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str]] = []

        def _fake_request(url: str, token: str) -> list[dict[str, object]]:
            calls.append((url, token))
            return list(_RAW_COMMITS)

        monkeypatch.setattr(github_activity, "_github_request", _fake_request)
        self._calls = calls

    async def test_first_run_ingests_all_commits(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="gh-fixture")
        result = await fetch(instance, state)
        assert len(result.items) == 3
        assert result.next_cursor_id == "sha-newest"

    async def test_rerun_with_cursor_ingests_only_newer(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="gh-fixture", last_seen_id="sha-middle")
        result = await fetch(instance, state)
        assert [i.item_id for i in result.items] == ["sha-newest"]

    async def test_cursor_at_newest_ingests_nothing(self, instance: SourceInstance) -> None:
        state = ConnectorState(instance_name="gh-fixture", last_seen_id="sha-newest")
        result = await fetch(instance, state)
        assert result.items == []

    async def test_unauthenticated_request_when_no_token_env(
        self, instance: SourceInstance
    ) -> None:
        state = ConnectorState(instance_name="gh-fixture")
        await fetch(instance, state)
        assert self._calls[-1][1] == ""

    async def test_token_read_from_named_env_var(
        self, instance: SourceInstance, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIXTURE_GH_TOKEN", "secret-token-value")
        instance_with_token = SourceInstance(
            name="gh-fixture",
            kind="github-activity",
            target="acme/repo",
            enabled=True,
            options={"token_env": "FIXTURE_GH_TOKEN"},
        )
        state = ConnectorState(instance_name="gh-fixture")
        await fetch(instance_with_token, state)
        assert self._calls[-1][1] == "secret-token-value"

    async def test_missing_token_env_var_falls_back_to_unauthenticated(
        self, instance: SourceInstance
    ) -> None:
        instance_with_token = SourceInstance(
            name="gh-fixture",
            kind="github-activity",
            target="acme/repo",
            enabled=True,
            options={"token_env": "NOT_A_REAL_ENV_VAR_12345"},
        )
        state = ConnectorState(instance_name="gh-fixture")
        await fetch(instance_with_token, state)
        assert self._calls[-1][1] == ""
