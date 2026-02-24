"""Tests for date resolution heuristics in CommandHandlers._resolve_date_range."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest


def _make_handlers() -> object:
    """Create a minimal CommandHandlers with mocked dependencies for date tests."""
    from vaultmind.bot.commands import CommandHandlers

    settings = MagicMock()
    settings.vault.path = MagicMock()
    settings.vault.excluded_folders = []
    settings.telegram.allowed_user_ids = []
    settings.routing.chat_model = ""
    settings.routing.chat_max_tokens = 100
    settings.llm.fast_model = "test-model"
    settings.graph.persist_path = MagicMock()

    store = MagicMock()
    graph = MagicMock()
    parser = MagicMock()
    thinking = MagicMock()
    thinking._sessions = {}
    llm_client = MagicMock()

    return CommandHandlers(
        settings=settings,
        store=store,
        graph=graph,
        parser=parser,
        thinking=thinking,
        llm_client=llm_client,
    )


@pytest.fixture
def handlers() -> object:
    return _make_handlers()


class TestExplicitDateFormats:
    """Test explicit date format parsing."""

    def test_iso_date(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("2026-02-20")  # type: ignore[attr-defined]
        assert start == "2026-02-20"
        assert end == "2026-02-20"

    def test_slash_date_dmy(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("20/02/2026")  # type: ignore[attr-defined]
        assert start == "2026-02-20"
        assert end == "2026-02-20"

    def test_slash_date_mdy(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("02/20/2026")  # type: ignore[attr-defined]
        assert start == "2026-02-20"
        assert end == "2026-02-20"


class TestKeywordDates:
    """Test keyword-based date resolution (no LLM)."""

    def test_today(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("today")  # type: ignore[attr-defined]
        expected = datetime.now().strftime("%Y-%m-%d")
        assert start == expected
        assert end == expected

    def test_yesterday(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("yesterday")  # type: ignore[attr-defined]
        expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert start == expected
        assert end == expected

    def test_this_week(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("this week")  # type: ignore[attr-defined]
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        assert start == monday.strftime("%Y-%m-%d")
        assert end == today.strftime("%Y-%m-%d")

    def test_last_week(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("last week")  # type: ignore[attr-defined]
        today = datetime.now()
        last_monday = today - timedelta(days=today.weekday() + 7)
        last_sunday = last_monday + timedelta(days=6)
        assert start == last_monday.strftime("%Y-%m-%d")
        assert end == last_sunday.strftime("%Y-%m-%d")

    def test_weekend(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("over the weekend")  # type: ignore[attr-defined]
        assert start is not None
        assert end is not None
        # Saturday is weekday 5, Sunday is 6
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        assert start_date.weekday() == 5  # Saturday
        assert end_date.weekday() == 6  # Sunday
        assert (end_date - start_date).days == 1

    def test_case_insensitive(self, handlers: object) -> None:
        start, end = handlers._resolve_date_range("Yesterday")  # type: ignore[attr-defined]
        expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert start == expected
