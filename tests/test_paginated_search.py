"""Tests for paginated search — PaginatedSearch dataclass and recall pagination."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.bench.fixture_store import FixtureStore
from vaultmind.bot.handlers.recall import (
    PaginatedSearch,
    _build_keyboard_for_user,
    _render_page,
    handle_recall,
    handle_recall_page_callback,
)
from vaultmind.config import Settings, VaultConfig
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.knowledge.sources import search_sources
from vaultmind.memory.gaps import GapStore
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(title: str, path: str, content: str, distance: float = 0.1) -> dict[str, Any]:
    return {
        "path": path,
        "source_kind": "ordinary",
        "status": "source-only",
        "approved": False,
        "current_digest": hashlib.sha256(content.encode()).hexdigest(),
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


def _make_ctx(
    page_size: int = 5,
    max_results: int = 25,
    session_ttl: int = 300,
    *,
    vault_path: Path,
) -> MagicMock:
    ctx = MagicMock()
    ctx.settings = Settings.model_construct(vault=VaultConfig(path=vault_path))
    ctx.settings.search.page_size = page_size
    ctx.settings.search.max_results = max_results
    ctx.settings.search.session_ttl = session_ttl
    ctx.settings.telegram.allowed_user_ids = []
    ctx.store = FixtureStore({})
    return ctx


def _write_session(ctx: MagicMock, session: PaginatedSearch) -> None:
    for hit in session.results:
        path = ctx.settings.vault.path / hit["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(hit["content"], encoding="utf-8")


def _callback(uid: int, page: int) -> MagicMock:
    callback = MagicMock()
    callback.from_user.id = uid
    callback.data = f"recall_page:{uid}:{page}"
    callback.answer = AsyncMock()
    callback.message = AsyncMock()
    return callback


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


def test_render_page_mixed_evidence_shows_percent_only_for_vector_hits() -> None:
    vec = _make_hit("Vec", "vec.md", "vector", distance=0.25)
    lex_none = _make_hit("LexNone", "lex-none.md", "lexical")
    lex_none["distance"] = None
    lex_missing = _make_hit("LexMissing", "lex-missing.md", "lexical")
    del lex_missing["distance"]
    session = PaginatedSearch(
        query="q", results=[lex_none, vec, lex_missing], page_size=5, current_page=0
    )
    text, _ = _render_page(session)
    assert "<b>1. LexNone</b> (relevance unknown)" in text
    assert "<b>2. Vec</b> (75% match)" in text
    assert "<b>3. LexMissing</b> (relevance unknown)" in text
    assert "100%" not in text


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
async def test_callback_expired_session_answers_expiry_message(tmp_path: Path) -> None:
    ctx = _make_ctx(session_ttl=300, vault_path=tmp_path)
    search_sessions: dict[int, PaginatedSearch] = {}

    callback = MagicMock()
    callback.data = "recall_page:99:1"
    callback.from_user.id = 99
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Session expired, run /recall again.")


@pytest.mark.asyncio
async def test_callback_ttl_expired_removes_session(tmp_path: Path) -> None:
    ctx = _make_ctx(session_ttl=10, vault_path=tmp_path)
    session = _make_session(n_results=10, page_size=5, page=0, age_seconds=20)
    search_sessions: dict[int, PaginatedSearch] = {7: session}

    callback = MagicMock()
    callback.data = "recall_page:7:1"
    callback.from_user.id = 7
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Session expired, run /recall again.")
    assert 7 not in search_sessions


@pytest.mark.asyncio
async def test_callback_valid_page_edits_message(tmp_path: Path) -> None:
    ctx = _make_ctx(session_ttl=300, vault_path=tmp_path)
    session = _make_session(n_results=10, page_size=5, page=0)
    _write_session(ctx, session)
    search_sessions: dict[int, PaginatedSearch] = {5: session}

    mock_message = AsyncMock()
    mock_message.edit_text = AsyncMock()

    callback = MagicMock()
    callback.data = "recall_page:5:1"
    callback.from_user.id = 5
    callback.answer = AsyncMock()
    callback.message = mock_message

    await handle_recall_page_callback(ctx, callback, search_sessions)

    mock_message.edit_text.assert_called_once()
    call_kwargs = mock_message.edit_text.call_args
    assert call_kwargs.kwargs.get("parse_mode") == "HTML"
    callback.answer.assert_called_once_with()
    assert search_sessions[5].current_page == 1


@pytest.mark.asyncio
async def test_callback_invalid_page_out_of_range(tmp_path: Path) -> None:
    ctx = _make_ctx(session_ttl=300, vault_path=tmp_path)
    session = _make_session(n_results=10, page_size=5, page=0)
    search_sessions: dict[int, PaginatedSearch] = {3: session}

    callback = MagicMock()
    callback.data = "recall_page:3:99"
    callback.from_user.id = 3
    callback.answer = AsyncMock()
    callback.message = AsyncMock()

    await handle_recall_page_callback(ctx, callback, search_sessions)

    callback.answer.assert_called_once_with("Page out of range.")


@pytest.mark.asyncio
async def test_callback_malformed_data(tmp_path: Path) -> None:
    ctx = _make_ctx(session_ttl=300, vault_path=tmp_path)
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


def test_render_source_identity_digest_and_provenance_are_explicit() -> None:
    hit = _make_hit("Retained", "_wiki/sources/id.md", "Current text")
    hit.update(
        source_id="retained-id",
        source_kind="preserved",
        provenance={"adapter": "telegram", "origin": "actor:7", "role": "text", "revision": "9"},
    )
    text, _ = _render_page(PaginatedSearch(query="q", results=[hit], page_size=5))
    assert "Source-only / unapproved" in text
    assert hit["current_digest"] in text and "retained-id" in text
    assert "telegram" in text and "actor:7" in text and "revision: 9" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["changed", "missing"])
async def test_later_page_refuses_changed_or_missing_current_source(
    tmp_path: Path, mutation: str
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    session = _make_session(n_results=10)
    _write_session(ctx, session)
    target = tmp_path / "note-5.md"
    if mutation == "changed":
        target.write_text("replacement text", encoding="utf-8")
    else:
        target.unlink()
    callback = _callback(7, 1)
    sessions = {7: session}
    await handle_recall_page_callback(ctx, callback, sessions)
    assert 7 not in sessions
    displayed = callback.message.edit_text.call_args.args[0]
    assert "/recall again" in displayed
    assert "Content of note" not in displayed and "replacement text" not in displayed
    assert session.current_page == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["body", "provenance"])
async def test_preserved_page_revalidates_provenance_snapshot(
    tmp_path: Path, mutation: str
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    with SourceIntake(ctx.settings) as intake:
        capture = intake.preserve_text(
            "retained text", original_name="capture.txt", origin="local", adapter="test"
        )
    hits = search_sources(ctx.settings, "retained")
    sessions = {7: PaginatedSearch(query="retained", results=hits, page_size=1)}
    if mutation == "body":
        (tmp_path / capture.path).write_text("changed retained body", encoding="utf-8")
    else:
        manifest = tmp_path / "_wiki/sources" / f"{capture.source_id}.intake.json"
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        provenance["origin"] = "changed-origin"
        manifest.write_text(json.dumps(provenance), encoding="utf-8")
    callback = _callback(7, 0)
    await handle_recall_page_callback(ctx, callback, sessions)
    assert not sessions
    assert "/recall again" in callback.message.edit_text.call_args.args[0]
    assert "retained text" not in callback.message.edit_text.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("actor,allowed", [(99, [7]), (99, [])])
async def test_paging_authorizes_actor_and_ownership_before_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: int, allowed: list[int]
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    ctx.settings.telegram.allowed_user_ids = allowed
    hydrate = MagicMock()
    monkeypatch.setattr("vaultmind.bot.handlers.recall.revalidate_source_hit", hydrate)
    callback = _callback(7, 1)
    callback.from_user.id = actor
    session = _make_session()
    await handle_recall_page_callback(ctx, callback, {7: session})
    hydrate.assert_not_called()
    callback.message.edit_text.assert_not_called()
    assert session.current_page == 0


@pytest.mark.asyncio
async def test_recall_authorizes_before_source_scope_or_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    ctx.settings.telegram.allowed_user_ids = [7]
    search = MagicMock()
    monkeypatch.setattr("vaultmind.bot.handlers.recall.search_sources", search)
    message = MagicMock()
    message.from_user.id = 99
    message.answer = AsyncMock()
    await handle_recall(ctx, message, "query", {})
    search.assert_not_called()
    message.answer.assert_not_called()


@pytest.mark.asyncio
async def test_recall_displays_only_current_bytes_not_history_or_index_excerpt(
    tmp_path: Path,
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    (tmp_path / "current.md").write_text("Current body", encoding="utf-8")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki/answer.md").write_text("historical secret", encoding="utf-8")
    ctx.store = FixtureStore(
        {
            "query": [
                _make_hit("Old", "current.md", "stale index excerpt"),
                _make_hit("Approved", "wiki/answer.md", "historical secret"),
            ]
        }
    )
    message = MagicMock()
    message.from_user.id = 7
    message.answer = AsyncMock()
    sessions: dict[int, PaginatedSearch] = {}
    await handle_recall(ctx, message, "query", sessions)
    text = message.answer.call_args.args[0]
    assert "Current body" in text and "Source-only / unapproved" in text
    assert "stale index" not in text and "historical secret" not in text
    assert sessions[7].results[0]["distance"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("distance,expected_gaps", [(None, 0), (0.9, 1)])
async def test_current_raw_distance_preserves_unknown_and_weak_gap_semantics(
    tmp_path: Path, distance: float | None, expected_gaps: int
) -> None:
    ctx = _make_ctx(vault_path=tmp_path)
    note_path = tmp_path / "a.md"
    note_path.write_text("Current source", encoding="utf-8")
    parser = VaultParser(ctx.settings.vault)
    chunk = parser.chunk_note(parser.parse_file(note_path))[0]
    ctx.store = FixtureStore(
        {
            "query": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.to_chroma_metadata(),
                    "distance": distance,
                }
            ]
        }
    )
    ctx.gap_store = GapStore(tmp_path / "gaps.db")
    message = MagicMock()
    message.from_user.id = 7
    message.answer = AsyncMock()
    try:
        await handle_recall(ctx, message, "query", {})
        assert len(ctx.gap_store.list_open()) == expected_gaps
    finally:
        ctx.gap_store.close()


def test_render_source_text_and_identity_cannot_inject_telegram_markup() -> None:
    hit = _make_hit("<b>spoofed</b>", "path`with_ticks.md", "<script>body</script>")
    hit["provenance"] = {"origin": "<i>forged approval</i>"}
    session = PaginatedSearch(query="<b>query</b>", results=[hit], page_size=1)
    text, _ = _render_page(session)
    assert "<script>" not in text and "<b>spoofed</b>" not in text
    assert "&lt;script&gt;body&lt;/script&gt;" in text
    assert "&lt;i&gt;forged approval&lt;/i&gt;" in text
    assert "&lt;b&gt;query&lt;/b&gt;" in text
