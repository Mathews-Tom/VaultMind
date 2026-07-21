"""Tests for the edit handler (`bot/handlers/edit.py`, M9 supersede wiring)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from vaultmind.graph.evolution import LineageStore
from vaultmind.llm.client import LLMResponse

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _write_note(tmp_path: Path, name: str = "note.md", body: str = "Original content.\n") -> Path:
    path = tmp_path / name
    path.write_text(f"---\ntitle: {name}\n---\n\n{body}", encoding="utf-8")
    return path


def _make_ctx(
    tmp_path: Path,
    *,
    lineage_store: LineageStore | None = None,
    llm_text: str = "Edited content.",
) -> Any:
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.vault_root = tmp_path
    ctx.settings = MagicMock()
    ctx.settings.telegram.allowed_user_ids = []
    ctx.settings.routing.chat_model = "test-model"
    ctx.settings.llm.fast_model = "test-model"
    ctx.settings.llm.max_tokens = 1024
    ctx.llm_client = MagicMock()
    ctx.llm_client.complete = MagicMock(
        return_value=LLMResponse(text=llm_text, model="test-model", usage={})
    )
    ctx.store = MagicMock()
    ctx.store.index_single_note = MagicMock(return_value=None)
    ctx.parser = MagicMock()
    ctx.parser.parse_file = MagicMock(return_value=MagicMock())
    ctx.lineage_store = lineage_store
    return ctx


def _fake_message() -> MagicMock:
    message = MagicMock()
    message.from_user = MagicMock(id=7)
    message.answer = AsyncMock()
    return message


def _fake_callback(data: str) -> MagicMock:
    callback = MagicMock()
    callback.data = data
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()
    return callback


class TestHandleEditShowsBodyOnlyToLLM:
    def test_llm_prompt_shows_body_not_raw_frontmatter(self, tmp_path: Path) -> None:
        _write_note(tmp_path, body="Only the body, not frontmatter.\n")
        ctx = _make_ctx(tmp_path)
        message = _fake_message()

        from vaultmind.bot.handlers.edit import handle_edit

        asyncio.run(handle_edit(ctx, message, "note.md add a line", {}))

        user_msg = ctx.llm_client.complete.call_args.kwargs["messages"][0].content
        assert "Only the body, not frontmatter." in user_msg
        assert "title: note.md" not in user_msg


class TestHandleEditCallbackSupersede:
    def test_confirm_writes_new_content_above_dated_callout_preserving_old_body(
        self, tmp_path: Path
    ) -> None:
        _write_note(tmp_path, body="Old body that must survive.\n")
        pending_edits: dict[int, dict[str, str]] = {
            7: {
                "path": "note.md",
                "original": "Old body that must survive.\n",
                "edited": "New body content.",
                "instruction": "make it better",
            }
        }
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, pending_edits))

        content = (tmp_path / "note.md").read_text()
        assert "New body content." in content
        assert "Old body that must survive." in content
        assert "[!superseded] Edited via bot" in content
        assert content.index("New body content.") < content.index("[!superseded]")
        assert content.index("[!superseded]") < content.index("Old body that must survive.")

    def test_confirm_reindexes_note(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        pending_edits: dict[int, dict[str, str]] = {
            7: {"path": "note.md", "original": "x", "edited": "y", "instruction": "z"}
        }
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, pending_edits))

        ctx.parser.parse_file.assert_called_once()
        ctx.store.index_single_note.assert_called_once()

    def test_confirm_records_one_lineage_edge_with_instruction_rationale(
        self, tmp_path: Path
    ) -> None:
        _write_note(tmp_path)
        pending_edits: dict[int, dict[str, str]] = {
            7: {
                "path": "note.md",
                "original": "x",
                "edited": "y",
                "instruction": "add a summary",
            }
        }
        lineage_store = LineageStore(store_path=tmp_path / "lineage.json")
        ctx = _make_ctx(tmp_path, lineage_store=lineage_store)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, pending_edits))

        history = lineage_store.get_lineage("note.md")
        assert len(history) == 1
        assert history[0].event == "edited"
        assert "add a summary" in history[0].detail

    def test_confirm_response_mentions_preserved_text(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        pending_edits: dict[int, dict[str, str]] = {
            7: {"path": "note.md", "original": "x", "edited": "y", "instruction": "z"}
        }
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, pending_edits))

        text = callback.message.edit_text.call_args[0][0]
        assert "preserved" in text.lower()


class TestHandleEditCallbackSequentialEdits:
    def test_two_sequential_edits_both_accumulate_history(self, tmp_path: Path) -> None:
        _write_note(tmp_path, body="v0 body.\n")
        lineage_store = LineageStore(store_path=tmp_path / "lineage.json")
        ctx = _make_ctx(tmp_path, lineage_store=lineage_store)

        from vaultmind.bot.handlers.edit import handle_edit_callback

        first_pending: dict[int, dict[str, str]] = {
            7: {
                "path": "note.md",
                "original": "v0 body.\n",
                "edited": "v1 body.",
                "instruction": "a",
            }
        }
        asyncio.run(handle_edit_callback(ctx, _fake_callback("edit_confirm:7"), first_pending))

        second_pending: dict[int, dict[str, str]] = {
            7: {"path": "note.md", "original": "v1 body.", "edited": "v2 body.", "instruction": "b"}
        }
        asyncio.run(handle_edit_callback(ctx, _fake_callback("edit_confirm:7"), second_pending))

        content = (tmp_path / "note.md").read_text()
        assert "v0 body." in content
        assert "v1 body." in content
        assert "v2 body." in content
        assert len(lineage_store.get_lineage("note.md")) == 2


class TestHandleEditCallbackCancel:
    def test_cancel_discards_pending_edit_and_leaves_file_untouched(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, body="Untouched.\n")
        original = path.read_text()
        pending_edits: dict[int, dict[str, str]] = {
            7: {"path": "note.md", "original": "x", "edited": "y", "instruction": "z"}
        }
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("edit_cancel:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, pending_edits))

        assert path.read_text() == original
        assert 7 not in pending_edits


class TestHandleEditCallbackExpired:
    def test_confirm_with_no_pending_edit_reports_expired(self, tmp_path: Path) -> None:
        _write_note(tmp_path)
        ctx = _make_ctx(tmp_path)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, {}))

        text = callback.message.edit_text.call_args[0][0]
        assert "expired" in text.lower()
