"""Ordinary Telegram mutations cannot enter the selected vault's wiki namespaces."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.bot.handlers import bookmark, daily, delete, edit
from vaultmind.bot.handlers.bookmark import LastExchange
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.graph.evolution import LineageStore
from vaultmind.llm.client import LLMResponse
from vaultmind.vault.models import Note

BODY = "Reserved source body must not be read, changed, sent, or logged."


@pytest.fixture
def ctx(tmp_path: Path) -> Any:
    root = tmp_path / "selected-vault"
    root.mkdir()
    context = MagicMock(spec=HandlerContext)
    context.vault_root = root
    context.settings = MagicMock()
    context.settings.vault.path = tmp_path / "not-the-selected-vault"
    context.settings.vault.daily_folder = "daily"
    context.settings.telegram.allowed_user_ids = [7]
    context.settings.routing.chat_model = "test-model"
    context.settings.llm.fast_model = "test-model"
    context.settings.llm.max_tokens = 1024
    context.llm_client = MagicMock()
    context.llm_client.complete.return_value = LLMResponse(
        text="Edited ordinary body.", model="test-model", usage={}
    )
    context.store = MagicMock()
    context.store.search.return_value = []
    context.parser = MagicMock()
    context.thinking = MagicMock()
    context.thinking.has_active_session.return_value = False
    context.lineage_store = MagicMock(spec=LineageStore)
    context.review_queue = MagicMock()
    return context


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = MagicMock()
    clock.now.return_value = datetime(2026, 10, 24, 12, tzinfo=UTC)
    monkeypatch.setattr(bookmark, "datetime", clock)
    monkeypatch.setattr(daily, "datetime", clock)


def _message(user_id: int = 7) -> MagicMock:
    message = MagicMock()
    message.from_user = MagicMock(id=user_id)
    message.answer = AsyncMock()
    return message


def _callback(data: str, user_id: int = 7) -> MagicMock:
    callback = MagicMock()
    callback.from_user = MagicMock(id=user_id)
    callback.data = data
    callback.answer = AsyncMock()
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    return callback


def _reserved_note(root: Path, namespace: str, spelling: str) -> tuple[str, Path]:
    if spelling == "lexical-alias":
        folder = root / "ordinary"
        folder.mkdir()
        (root / namespace).symlink_to(folder, target_is_directory=True)
    else:
        folder = root / namespace
        folder.mkdir()
    note = folder / "private.md"
    note.write_text(BODY, encoding="utf-8")
    if spelling == "directory-alias":
        (root / "ordinary-alias").symlink_to(folder, target_is_directory=True)
        return "ordinary-alias/private.md", note
    if spelling == "file-alias":
        (root / "ordinary-alias.md").symlink_to(note)
        return "ordinary-alias.md", note
    if spelling == "case":
        return f"{namespace.upper()}/private.md", note
    return f"{namespace}/private.md", note


def _forbid_note_io(monkeypatch: pytest.MonkeyPatch) -> None:
    for method in ("read_text", "write_text", "write_bytes", "mkdir"):
        monkeypatch.setattr(
            Path,
            method,
            MagicMock(side_effect=AssertionError(f"Unexpected note I/O: {method}")),
        )
    monkeypatch.setattr(
        edit.frontmatter,
        "load",
        MagicMock(side_effect=AssertionError("Unexpected read for edit")),
    )
    for handler in (edit, delete):
        monkeypatch.setattr(
            handler,
            "write_superseded_block",
            MagicMock(side_effect=AssertionError("Unexpected supersede mutation")),
        )


def _assert_no_effects(ctx: Any) -> None:
    assert ctx.llm_client.mock_calls == []
    assert ctx.store.mock_calls == []
    assert ctx.parser.mock_calls == []
    assert ctx.thinking.mock_calls == []
    assert ctx.lineage_store.mock_calls == []
    assert ctx.review_queue.mock_calls == []


@pytest.mark.parametrize("command", ["edit", "delete"])
@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize(
    "spelling", ["direct", "case", "directory-alias", "file-alias", "lexical-alias"]
)
def test_initial_mutation_rejects_reserved_before_read_or_provider(
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    command: str,
    namespace: str,
    spelling: str,
) -> None:
    query, note = _reserved_note(ctx.vault_root, namespace, spelling)
    original = note.read_bytes()
    _forbid_note_io(monkeypatch)
    message = _message()
    pending: dict[int, dict[str, str]] = {}

    if command == "edit":
        asyncio.run(edit.handle_edit(ctx, message, f"{query} rewrite it", pending))
    else:
        asyncio.run(delete.handle_delete(ctx, message, query))

    message.answer.assert_awaited_once_with("Path not allowed.")
    assert pending == {}
    _assert_no_effects(ctx)
    assert note.read_bytes() == original
    assert BODY not in caplog.text


@pytest.mark.parametrize("command", ["edit", "delete"])
@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize(
    "spelling", ["direct", "case", "directory-alias", "file-alias", "lexical-alias"]
)
def test_forged_confirmation_rejects_reserved_without_effects(
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    command: str,
    namespace: str,
    spelling: str,
) -> None:
    query, note = _reserved_note(ctx.vault_root, namespace, spelling)
    original = note.read_bytes()
    _forbid_note_io(monkeypatch)

    if command == "edit":
        callback = _callback("edit_confirm:7")
        pending = {7: {"path": query, "original": BODY, "edited": "Changed", "instruction": "x"}}
        asyncio.run(edit.handle_edit_callback(ctx, callback, pending))
    else:
        callback = _callback(f"delete_confirm:{query}")
        asyncio.run(delete.handle_delete_callback(ctx, callback))

    callback.message.edit_text.assert_awaited_once_with("Path not allowed.")
    callback.answer.assert_awaited_once_with()
    _assert_no_effects(ctx)
    assert note.read_bytes() == original
    assert BODY not in caplog.text


@pytest.mark.parametrize("command", ["edit", "delete"])
@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
def test_confirmation_rechecks_note_replaced_by_reserved_symlink(
    ctx: Any, monkeypatch: pytest.MonkeyPatch, command: str, namespace: str
) -> None:
    note = ctx.vault_root / "ordinary.md"
    note.write_text("---\ntitle: Ordinary\n---\n\nOrdinary body.", encoding="utf-8")
    message = _message()
    pending: dict[int, dict[str, str]] = {}
    if command == "edit":
        asyncio.run(edit.handle_edit(ctx, message, "ordinary.md improve it", pending))
        assert pending[7]["path"] == "ordinary.md"
        ctx.llm_client.complete.assert_called_once()
    else:
        asyncio.run(delete.handle_delete(ctx, message, "ordinary.md"))
        assert message.answer.call_args.kwargs["reply_markup"] is not None

    _, reserved_note = _reserved_note(ctx.vault_root, namespace, "direct")
    original = reserved_note.read_bytes()
    note.unlink()
    note.symlink_to(reserved_note)
    ctx.llm_client.reset_mock()
    _forbid_note_io(monkeypatch)

    if command == "edit":
        callback = _callback("edit_confirm:7")
        asyncio.run(edit.handle_edit_callback(ctx, callback, pending))
    else:
        callback = _callback("delete_confirm:ordinary.md")
        asyncio.run(delete.handle_delete_callback(ctx, callback))

    callback.message.edit_text.assert_awaited_once_with("Path not allowed.")
    _assert_no_effects(ctx)
    assert reserved_note.read_bytes() == original


@pytest.mark.parametrize("command", ["edit", "delete"])
@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
def test_filename_resolution_cannot_select_reserved_note_for_mutation(
    ctx: Any, monkeypatch: pytest.MonkeyPatch, command: str, namespace: str
) -> None:
    _reserved_note(ctx.vault_root, namespace, "direct")
    _forbid_note_io(monkeypatch)
    message = _message()
    if command == "edit":
        asyncio.run(edit.handle_edit(ctx, message, "private improve it", {}))
    else:
        asyncio.run(delete.handle_delete(ctx, message, "private"))

    message.answer.assert_awaited_once_with("Path not allowed.")
    _assert_no_effects(ctx)


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
def test_bookmark_fixed_inbox_alias_is_rejected_before_session_lookup(
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: None,
    namespace: str,
) -> None:
    folder = ctx.vault_root / namespace
    folder.mkdir()
    (ctx.vault_root / bookmark.INBOX_FOLDER).symlink_to(folder, target_is_directory=True)
    _forbid_note_io(monkeypatch)
    message = _message()

    asyncio.run(
        bookmark.handle_bookmark(
            ctx, message, "My topic", {7: LastExchange(query=BODY, response=BODY, timestamp=1.0)}
        )
    )

    message.answer.assert_awaited_once_with("Path not allowed.")
    _assert_no_effects(ctx)
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("spelling", ["direct", "case", "directory-alias", "lexical-alias"])
@pytest.mark.parametrize("existing", [False, True])
def test_daily_reserved_destination_is_rejected_before_read_or_mkdir(
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: None,
    namespace: str,
    spelling: str,
    existing: bool,
) -> None:
    if spelling == "lexical-alias":
        folder = ctx.vault_root / "ordinary"
        folder.mkdir()
        (ctx.vault_root / namespace).symlink_to(folder, target_is_directory=True)
    else:
        folder = ctx.vault_root / namespace
        folder.mkdir()
    if spelling == "directory-alias":
        (ctx.vault_root / "daily").symlink_to(folder, target_is_directory=True)
        ctx.settings.vault.daily_folder = "daily"
    elif spelling == "case":
        ctx.settings.vault.daily_folder = namespace.upper()
    else:
        ctx.settings.vault.daily_folder = namespace
    note = folder / "2026-10-24.md"
    if existing:
        note.write_text(BODY, encoding="utf-8")
    _forbid_note_io(monkeypatch)
    message = _message()

    asyncio.run(daily.handle_daily(ctx, message))

    message.answer.assert_awaited_once_with("Path not allowed.")
    _assert_no_effects(ctx)
    assert note.exists() is existing
    if existing:
        assert note.read_bytes() == BODY.encode()


@pytest.mark.parametrize("command", ["bookmark", "daily"])
@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
def test_fixed_output_file_alias_cannot_overwrite_reserved_note(
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: None,
    command: str,
    namespace: str,
) -> None:
    _, reserved_note = _reserved_note(ctx.vault_root, namespace, "direct")
    folder = ctx.vault_root / (bookmark.INBOX_FOLDER if command == "bookmark" else "daily")
    folder.mkdir()
    filename = "2026-10-24-bookmark-my-topic.md" if command == "bookmark" else "2026-10-24.md"
    (folder / filename).symlink_to(reserved_note)
    original = reserved_note.read_bytes()
    _forbid_note_io(monkeypatch)
    message = _message()

    if command == "bookmark":
        exchanges = {7: LastExchange(query=BODY, response=BODY, timestamp=1.0)}
        asyncio.run(bookmark.handle_bookmark(ctx, message, "My topic", exchanges))
    else:
        asyncio.run(daily.handle_daily(ctx, message))

    message.answer.assert_awaited_once_with("Path not allowed.")
    _assert_no_effects(ctx)
    assert reserved_note.read_bytes() == original


@pytest.mark.parametrize("source", ["exchange", "thinking"])
def test_bookmark_ordinary_inbox_alias_uses_checked_path_and_preserves_content(
    ctx: Any, fixed_clock: None, source: str
) -> None:
    actual = ctx.vault_root / "ordinary"
    actual.mkdir()
    (ctx.vault_root / bookmark.INBOX_FOLDER).symlink_to(actual, target_is_directory=True)
    exchanges = {7: LastExchange(query="Question", response="Answer", timestamp=1.0)}
    if source == "thinking":
        ctx.thinking.has_active_session.return_value = True
        ctx.thinking._get_session.return_value.history = [
            {"user": "Question", "assistant": "Answer"}
        ]
    message = _message()

    asyncio.run(bookmark.handle_bookmark(ctx, message, "My topic", exchanges))

    expected = actual / "2026-10-24-bookmark-my-topic.md"
    content = expected.read_text(encoding="utf-8")
    assert "Question" in content
    assert "Answer" in content
    ctx.parser.parse_file.assert_called_once_with(expected)
    ctx.store.index_single_note.assert_called_once_with(
        ctx.parser.parse_file.return_value, ctx.parser
    )
    assert "ordinary/2026-10-24-bookmark-my-topic.md" in message.answer.call_args.args[0]
    ctx.llm_client.complete.assert_not_called()


def test_daily_ordinary_alias_still_creates_then_displays_daily_note(
    ctx: Any, fixed_clock: None
) -> None:
    actual = ctx.vault_root / "ordinary"
    actual.mkdir()
    (ctx.vault_root / "daily").symlink_to(actual, target_is_directory=True)
    message = _message()

    asyncio.run(daily.handle_daily(ctx, message))

    expected = actual / "2026-10-24.md"
    content = expected.read_text(encoding="utf-8")
    assert "type: daily" in content
    assert "## Captures" in content
    assert "## Tasks" in content
    assert "## Reflections" in content
    ctx.parser.parse_file.return_value = Note(
        path=Path("ordinary/2026-10-24.md"), title="Daily", content="Today's thoughts"
    )
    asyncio.run(daily.handle_daily(ctx, message))

    ctx.parser.parse_file.assert_called_once_with(expected)
    assert "Today's thoughts" in message.answer.call_args.args[0]
    assert expected.read_text(encoding="utf-8") == content
    ctx.llm_client.complete.assert_not_called()
    ctx.store.index_single_note.assert_not_called()


@pytest.mark.parametrize("user_id", [8, None])
def test_edit_confirmation_preserves_owner_gate_before_path_or_effects(
    ctx: Any, monkeypatch: pytest.MonkeyPatch, user_id: int | None
) -> None:
    ctx.settings.telegram.allowed_user_ids = []
    pending = {7: {"path": "wiki/private.md", "original": BODY, "edited": "x", "instruction": "y"}}
    callback = _callback("edit_confirm:7")
    callback.from_user = None if user_id is None else MagicMock(id=user_id)
    validator = MagicMock(side_effect=AssertionError("Ownership must precede destination lookup"))
    monkeypatch.setattr(edit, "validate_ordinary_write_path", validator)

    asyncio.run(edit.handle_edit_callback(ctx, callback, pending))

    assert 7 in pending
    callback.answer.assert_awaited_once_with("⛔ Not your edit")
    callback.message.edit_text.assert_not_called()
    validator.assert_not_called()
    _assert_no_effects(ctx)
