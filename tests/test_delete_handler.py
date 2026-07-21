"""Tests for the delete handler (`bot/handlers/delete.py`, M9 supersede wiring)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from vaultmind.graph.evolution import LineageStore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _write_note(tmp_path: Path, name: str = "note.md", body: str = "Original content.\n") -> Path:
    path = tmp_path / name
    path.write_text(f"---\ntitle: {name}\n---\n\n{body}", encoding="utf-8")
    return path


def _make_ctx(tmp_path: Path, *, lineage_store: LineageStore | None = None) -> Any:
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.vault_root = tmp_path
    ctx.store = MagicMock()
    ctx.store.delete_note = MagicMock(return_value=None)
    ctx.lineage_store = lineage_store
    return ctx


def _fake_callback(data: str) -> MagicMock:
    callback = MagicMock()
    callback.data = data
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()
    return callback


class TestHandleDeleteCallbackSupersede:
    def test_confirm_preserves_body_and_writes_dated_callout(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, body="Content that must survive.\n")
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("delete_confirm:note.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        assert path.exists(), "delete must never remove the file"
        content = path.read_text()
        assert "Content that must survive." in content
        assert "[!superseded] Deleted via bot" in content

    def test_confirm_removes_note_from_vector_store(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("delete_confirm:note.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        ctx.store.delete_note.assert_called_once_with("note.md")

    def test_confirm_records_one_lineage_edge(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        lineage_store = LineageStore(store_path=tmp_path / "lineage.json")
        ctx = _make_ctx(tmp_path, lineage_store=lineage_store)
        callback = _fake_callback("delete_confirm:note.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        history = lineage_store.get_lineage("note.md")
        assert len(history) == 1
        assert history[0].event == "deleted"

    def test_response_confirms_superseded_not_deleted(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("delete_confirm:note.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        text = callback.message.edit_text.call_args[0][0]
        assert "Superseded" in text
        assert "not deleted" in text


class TestHandleDeleteCallbackIdempotency:
    def test_confirming_twice_does_not_double_mark_or_double_delete(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        ctx = _make_ctx(tmp_path)

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, _fake_callback("delete_confirm:note.md")))
        second_callback = _fake_callback("delete_confirm:note.md")
        asyncio.run(handle_delete_callback(ctx, second_callback))

        ctx.store.delete_note.assert_called_once()
        text = second_callback.message.edit_text.call_args[0][0]
        assert "already superseded" in text

    def test_confirming_twice_records_only_one_lineage_edge(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        lineage_store = LineageStore(store_path=tmp_path / "lineage.json")
        ctx = _make_ctx(tmp_path, lineage_store=lineage_store)

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, _fake_callback("delete_confirm:note.md")))
        asyncio.run(handle_delete_callback(ctx, _fake_callback("delete_confirm:note.md")))

        assert len(lineage_store.get_lineage("note.md")) == 1


class TestHandleDeleteCallbackCancel:
    def test_cancel_leaves_file_untouched(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, body="Untouched.\n")
        original = path.read_text()
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("delete_cancel")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        assert path.read_text() == original
        ctx.store.delete_note.assert_not_called()


class TestHandleDeleteCallbackMissingNote:
    def test_confirm_on_missing_note_reports_not_found(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("delete_confirm:missing.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        text = callback.message.edit_text.call_args[0][0]
        assert "Note not found" in text
        ctx.store.delete_note.assert_not_called()


class TestHandleDeleteRequestMessage:
    def test_confirmation_prompt_mentions_supersede(self, tmp_path: Path) -> None:
        from vaultmind.bot.handlers.context import HandlerContext

        _write_note(tmp_path, name="target.md", body="Preview body.\n")

        ctx = MagicMock(spec=HandlerContext)
        ctx.vault_root = tmp_path
        ctx.settings = MagicMock()
        ctx.settings.telegram.allowed_user_ids = []
        ctx.store = MagicMock()
        ctx.store.search = MagicMock(return_value=[])

        message = MagicMock()
        message.from_user = MagicMock(id=1)
        message.answer = AsyncMock()

        from vaultmind.bot.handlers.delete import handle_delete

        asyncio.run(handle_delete(ctx, message, "target.md"))

        text = message.answer.call_args[0][0]
        assert "superseded" in text.lower()
