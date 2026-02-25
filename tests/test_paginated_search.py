"""Tests for paginated search — PaginatedSearch dataclass and recall pagination."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.bot.handlers.recall import (
    PaginatedSearch,
    _build_keyboard_for_user,
    _render_page,
    handle_recall_page_callback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(title: str, path: str, content: str, distance: float = 0.1) -> dict[str, Any]:
    return {
        "metadata": {"note_title": title, "note_path": path, "heading": ""},
        "content": content,
        "distance": distance,
    }


def _make_results(n: int) -> list[dict[str, Any]]:
    return [_make_hit(f"Note {i}", f"note-{i}.md", f"Content of note {i}") for i in range(n)]


def _make_session(
    n_results: int = 10,
    page_size: int = 5,
    page: int = 0,
    age_seconds: float = 0.0,
) -> PaginatedSearch:
    session = PaginatedSearch(
        query="test query",
        results=_make_results(n_results),
        page_size=page_size,
        current_page=page,
    )
    if age_seconds > 0:
        # Backdate created_at by subtracting from monotonic
        session.created_at = time.monotonic() - age_seconds
    return session


def _make_ctx(page_size: int = 5, max_results: int = 25, session_ttl: int = 300) -> MagicMock:
    ctx = MagicMock()
    ctx.settings.search.page_size = page_size
    ctx.settings.search.max_results = max_results
    ctx.settings.search.session_ttl = session_ttl
    return ctx


# ---------------------------------------------------------------------------
# PaginatedSearch dataclass
# ---------------------------------------------------------------------------


def test_paginated_search_defaults() -> None:
    session = PaginatedSearch(query="q", results=[], page_size=5)
    assert session.current_page == 0
    assert session.query == "q"
    assert session.results == []
    assert session.page_size == 5
    assert session.created_at <= time.monotonic()


def test_paginated_search_created_at_is_monotonic() -> None:
    before = time.monotonic()
    session = PaginatedSearch(query="q", results=[], page_size=5)
    after = time.monotonic()
    assert before <= session.created_at <= after


# ---------------------------------------------------------------------------
# _render_page — page rendering
# ---------------------------------------------------------------------------


def test_render_first_page_has_no_prev_button() -> None:
    session = _make_session(n_results=10, page_size=5, page=0)
    text, keyboard = _render_page(session)
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "\u25c0 Prev" not in labels
    assert "\u25b6 Next" in labels
    assert "Page 1/2" in labels


def test_render_last_page_has_no_next_button() -> None:
    session = _make_session(n_results=10, page_size=5, page=1)
    text, keyboard = _render_page(session)
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "\u25b6 Next" not in labels
    assert "\u25c0 Prev" in labels
    assert "Page 2/2" in labels


def test_render_middle_page_has_both_buttons() -> None:
    session = _make_session(n_results=15, page_size=5, page=1)
    text, keyboard = _render_page(session)
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "\u25c0 Prev" in labels
    assert "\u25b6 Next" in labels
    assert "Page 2/3" in labels


def test_render_single_page_no_prev_no_next() -> None:
    session = _make_session(n_results=3, page_size=5, page=0)
    text, keyboard = _render_page(session)
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "\u25c0 Prev" not in labels
    assert "\u25b6 Next" not in labels
    assert "Page 1/1" in labels


def test_render_first_page_text_contains_query() -> None:
    session = _make_session(n_results=5, page_size=5, page=0)
    text, _ = _render_page(session)
    assert "test query" in text


def test_render_page_shows_correct_result_slice() -> None:
    session = _make_session(n_results=10, page_size=5, page=1)
    text, _ = _render_page(session)
    # page 1 → results 5–9
    assert "Note 5" in text
    assert "Note 0" not in text


def test_render_page_callback_data_uses_uid_placeholder() -> None:
    session = _make_session(n_results=10, page_size=5, page=0)
    _, keyboard = _render_page(session)
    all_data = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
    next_data = next(d for d in all_data if d and "recall_page" in d)
    assert "__UID__" in next_data


# ---------------------------------------------------------------------------
# _build_keyboard_for_user — UID substitution
# ---------------------------------------------------------------------------


def test_build_keyboard_substitutes_user_id() -> None:
    session = _make_session(n_results=10, page_size=5, page=0)
    text, keyboard = _build_keyboard_for_user(session, user_id=42)
    all_data = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
    recall_data = [d for d in all_data if d and "recall_page" in d]
    assert all("42" in d for d in recall_data)
    assert all("__UID__" not in d for d in recall_data)


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_render_empty_results() -> None:
    session = PaginatedSearch(query="nothing", results=[], page_size=5, current_page=0)
    text, keyboard = _render_page(session)
    assert "nothing" in text
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "Page 1/1" in labels
    assert "\u25c0 Prev" not in labels
    assert "\u25b6 Next" not in labels


# ---------------------------------------------------------------------------
# handle_recall_page_callback — session expiry and navigation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_expired_session_answers_expiry_message() -> None:
    ctx = _make_ctx(session_ttl=300)
    search_sessions: dict[int, PaginatedSearch] = {}

    callback = MagicMock()
    callback.data = "recall_page:99:1"
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Session expired, run /recall again.")


@pytest.mark.asyncio
async def test_callback_ttl_expired_removes_session() -> None:
    ctx = _make_ctx(session_ttl=10)
    session = _make_session(n_results=10, page_size=5, page=0, age_seconds=20)
    search_sessions: dict[int, PaginatedSearch] = {7: session}

    callback = MagicMock()
    callback.data = "recall_page:7:1"
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Session expired, run /recall again.")
    assert 7 not in search_sessions


@pytest.mark.asyncio
async def test_callback_valid_page_edits_message() -> None:
    ctx = _make_ctx(session_ttl=300)
    session = _make_session(n_results=10, page_size=5, page=0)
    search_sessions: dict[int, PaginatedSearch] = {5: session}

    mock_message = AsyncMock()
    mock_message.edit_text = AsyncMock()

    callback = MagicMock()
    callback.data = "recall_page:5:1"
    callback.answer = AsyncMock()
    callback.message = mock_message

    await handle_recall_page_callback(ctx, callback, search_sessions)

    mock_message.edit_text.assert_called_once()
    call_kwargs = mock_message.edit_text.call_args
    assert call_kwargs.kwargs.get("parse_mode") == "Markdown"
    callback.answer.assert_called_once_with()
    assert search_sessions[5].current_page == 1


@pytest.mark.asyncio
async def test_callback_invalid_page_out_of_range() -> None:
    ctx = _make_ctx(session_ttl=300)
    session = _make_session(n_results=10, page_size=5, page=0)
    search_sessions: dict[int, PaginatedSearch] = {3: session}

    callback = MagicMock()
    callback.data = "recall_page:3:99"
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Page out of range.")


@pytest.mark.asyncio
async def test_callback_malformed_data() -> None:
    ctx = _make_ctx(session_ttl=300)
    search_sessions: dict[int, PaginatedSearch] = {}

    callback = MagicMock()
    callback.data = "recall_page:bad"
    callback.answer = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Invalid callback data.")


@pytest.mark.asyncio
async def test_callback_noop_not_registered_by_filter() -> None:
    """Verify the noop button data is never 'recall_page:' prefixed."""
    session = _make_session(n_results=5, page_size=5, page=0)
    _, keyboard = _build_keyboard_for_user(session, user_id=1)
    all_data = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
    # The page label button must have callback_data="noop"
    noop_buttons = [d for d in all_data if d == "noop"]
    assert len(noop_buttons) == 1
