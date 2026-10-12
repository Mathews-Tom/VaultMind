"""Tests for the delete handler (`bot/handlers/delete.py`, M9 supersede + M14 retirement).

The note, parser, Chroma collection, SQLite BM25 index and lineage store are all
real (`tests/index_support.py`); only the Telegram objects and the embedding
provider are synthetic. Assertions are about the file on disk, what the vector,
keyword and hybrid retrieval paths return, the lineage history and what the
user is told.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.index_support import ALL_SURFACES, IndexEnv, make_env
from vaultmind.graph.evolution import LineageEdge, LineageStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

TOKEN = "kestrelquartz"
OTHER_TOKEN = "marmotquartz"

LEGACY_DELETED_NOTE = (
    "---\n"
    "superseded_at:\n"
    "- bot:delete\n"
    "title: note\n"
    "---\n\n"
    "> [!superseded] Deleted via bot — 2026-03-05\n"
    "> Requested via Telegram `/delete` — content preserved below.\n\n"
    f"Content that must survive {TOKEN}.\n"
)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[IndexEnv]:
    environment = make_env(tmp_path.resolve())
    try:
        yield environment
    finally:
        environment.close()


class FlakyLineage(LineageStore):
    """A real lineage store whose next `record` can fail like a full disk."""

    def __init__(self, store_path: Path) -> None:
        super().__init__(store_path=store_path)
        self.fail_next = False

    def record(
        self, note_path: str, event: Literal["deleted", "edited"], detail: str
    ) -> LineageEdge:
        if self.fail_next:
            self.fail_next = False
            raise OSError("no space left on device")
        return super().record(note_path, event, detail)


def _lineage(env: IndexEnv) -> LineageStore:
    return LineageStore(store_path=env.data / "lineage.json")


def _make_ctx(env: IndexEnv, *, lineage_store: LineageStore | None = None) -> Any:
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.vault_root = env.vault
    ctx.settings = MagicMock()
    ctx.settings.telegram.allowed_user_ids = []
    ctx.store = env.store
    ctx.parser = env.parser
    ctx.lineage_store = lineage_store
    return ctx


def _fake_callback(data: str, user_id: int = 7) -> MagicMock:
    callback = MagicMock()
    callback.data = data
    callback.from_user = MagicMock(id=user_id)
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()
    return callback


def _confirm(ctx: Any, data: str = "delete_confirm:note.md") -> str:
    """Run the real confirm handler and return the text the user was shown."""
    from vaultmind.bot.handlers.delete import handle_delete_callback

    callback = _fake_callback(data)
    asyncio.run(handle_delete_callback(ctx, callback))
    return str(callback.message.edit_text.call_args[0][0])


def _indexed_note(env: IndexEnv, body: str = f"Content that must survive {TOKEN}.") -> Path:
    path = env.write("note.md", body)
    assert env.index("note.md") > 0
    assert env.surfaces(TOKEN) == ALL_SURFACES
    return path


class TestHandleDeleteCallbackAuthorization:
    def test_unlisted_actor_confirm_leaves_note_index_and_lineage_unchanged(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        original = path.read_text(encoding="utf-8")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        ctx.settings.telegram.allowed_user_ids = [7]
        callback = _fake_callback("delete_confirm:note.md", user_id=99)

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        assert path.read_text(encoding="utf-8") == original
        assert env.surfaces(TOKEN) == ALL_SURFACES
        assert lineage_store.get_lineage("note.md") == []
        callback.message.edit_text.assert_not_called()

    def test_missing_actor_cannot_cancel_when_allowlist_set(self, env: IndexEnv) -> None:
        ctx = _make_ctx(env)
        ctx.settings.telegram.allowed_user_ids = [7]
        callback = _fake_callback("delete_cancel")
        callback.from_user = None

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        callback.message.edit_text.assert_not_called()


class TestHandleDeleteCallbackSupersede:
    def test_confirm_preserves_body_and_retires_every_retrieval_path(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        ctx = _make_ctx(env)

        reply = _confirm(ctx)

        assert path.exists(), "delete must never remove the file"
        content = path.read_text(encoding="utf-8")
        assert f"Content that must survive {TOKEN}." in content
        assert "[!superseded] Deleted via bot" in content
        assert env.surfaces(TOKEN) == set()
        assert "Superseded" in reply
        assert "not deleted" in reply

    def test_deleted_note_stays_retired_after_full_reindex_and_incremental_indexing(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        env.write("other.md", f"Unrelated live claim {OTHER_TOKEN}.")
        env.index("other.md")
        ctx = _make_ctx(env)

        _confirm(ctx)
        env.store.index_notes(env.parser.iter_notes(), env.parser)

        assert env.surfaces(TOKEN) == set()
        assert env.surfaces(OTHER_TOKEN) == ALL_SURFACES

        assert env.index("note.md") == 0
        assert env.surfaces(TOKEN) == set()
        assert f"Content that must survive {TOKEN}." in path.read_text(encoding="utf-8")

    def test_confirm_records_one_lineage_edge(self, env: IndexEnv) -> None:
        _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        _confirm(ctx)

        history = lineage_store.get_lineage("note.md")
        assert len(history) == 1
        assert history[0].event == "deleted"


class TestHandleDeleteCallbackIdempotency:
    def test_confirming_twice_changes_nothing_further(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        _confirm(ctx)
        after_first = path.read_text(encoding="utf-8")
        second_reply = _confirm(ctx)

        assert path.read_text(encoding="utf-8") == after_first
        assert "already superseded" in second_reply
        assert len(lineage_store.get_lineage("note.md")) == 1
        assert env.surfaces(TOKEN) == set()

    def test_replay_on_a_note_deleted_before_this_version_retires_it_and_records_lineage(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        path.write_text(LEGACY_DELETED_NOTE, encoding="utf-8")  # as the earlier bot wrote it
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        reply = _confirm(ctx)

        assert path.read_text(encoding="utf-8") == LEGACY_DELETED_NOTE
        assert "already superseded" in reply
        assert env.surfaces(TOKEN) == set()
        assert len(lineage_store.get_lineage("note.md")) == 1


class TestHandleDeleteCallbackFailureReporting:
    def test_retirement_failure_is_reported_and_a_repeat_confirmation_completes_it(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        env.bm25.fail_on = {"delete_note"}

        failed_reply = _confirm(ctx)

        marked = path.read_text(encoding="utf-8")
        assert "[!superseded] Deleted via bot" in marked
        assert f"Content that must survive {TOKEN}." in marked
        assert failed_reply.startswith("\u26a0\ufe0f")
        assert "removing it from search failed" in failed_reply
        assert "Superseded (not deleted)" not in failed_reply
        assert "keyword" in env.surfaces(TOKEN), "the injected failure left rows behind"
        assert len(lineage_store.get_lineage("note.md")) == 1

        env.bm25.fail_on = set()
        retry_reply = _confirm(ctx)

        assert path.read_text(encoding="utf-8") == marked
        assert "already superseded" in retry_reply
        assert env.surfaces(TOKEN) == set()
        assert len(lineage_store.get_lineage("note.md")) == 1

    def test_lineage_failure_is_reported_and_replay_records_exactly_one_edge(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        flaky = FlakyLineage(env.data / "lineage.json")
        ctx = _make_ctx(env, lineage_store=flaky)
        flaky.fail_next = True

        failed_reply = _confirm(ctx)

        marked = path.read_text(encoding="utf-8")
        assert failed_reply.startswith("\u26a0\ufe0f")
        assert "lineage edge was not recorded" in failed_reply
        assert env.surfaces(TOKEN) == set(), "retirement is attempted even when lineage fails"
        assert flaky.get_lineage("note.md") == []

        _confirm(ctx)
        _confirm(ctx)

        assert path.read_text(encoding="utf-8") == marked
        assert len(flaky.get_lineage("note.md")) == 1

    def test_malformed_history_is_reported_and_nothing_changes(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        path.write_text(
            "---\ntitle: note\nsupersede_boundaries:\n- not-a-record\n---\n\nBody.\n",
            encoding="utf-8",
        )
        original = path.read_text(encoding="utf-8")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        reply = _confirm(ctx)

        assert reply.startswith("Delete failed")
        assert "not applied" in reply
        assert path.read_text(encoding="utf-8") == original
        assert lineage_store.get_lineage("note.md") == []
        assert env.surfaces(TOKEN) == ALL_SURFACES

    @pytest.mark.parametrize(
        "metadata",
        [
            "superseded_at:\n- bot:delete\n",
            "superseded_at:\n  bot: delete\n",
            "superseded_at:\n- - bot:delete\n",
        ],
        ids=["marker-without-callout", "mapping", "nested-list"],
    )
    def test_ambiguous_bot_metadata_is_reported_not_treated_as_a_delete(
        self, env: IndexEnv, metadata: str
    ) -> None:
        path = _indexed_note(env)
        path.write_text(
            f"---\ntitle: note\n{metadata}---\n\nBody {TOKEN} without a generated callout.\n",
            encoding="utf-8",
        )
        original = path.read_text(encoding="utf-8")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        reply = _confirm(ctx)

        assert reply.startswith("Delete failed")
        assert path.read_text(encoding="utf-8") == original
        assert lineage_store.get_lineage("note.md") == []
        assert env.surfaces(TOKEN) == ALL_SURFACES


class TestHandleDeleteCallbackCancel:
    def test_cancel_leaves_file_and_index_untouched(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        original = path.read_text(encoding="utf-8")
        ctx = _make_ctx(env)

        _confirm(ctx, "delete_cancel")

        assert path.read_text(encoding="utf-8") == original
        assert env.surfaces(TOKEN) == ALL_SURFACES


class TestHandleDeleteCallbackMissingNote:
    def test_confirm_on_missing_note_reports_not_found_and_records_nothing(
        self, env: IndexEnv
    ) -> None:
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        reply = _confirm(ctx, "delete_confirm:missing.md")

        assert "Note not found" in reply
        assert lineage_store.get_lineage("missing.md") == []


class TestHandleDeleteRequestMessage:
    def test_confirmation_prompt_mentions_supersede(self, env: IndexEnv) -> None:
        env.write("target.md", "Preview body.")
        ctx = _make_ctx(env)
        message = MagicMock()
        message.from_user = MagicMock(id=1)
        message.answer = AsyncMock()

        from vaultmind.bot.handlers.delete import handle_delete

        asyncio.run(handle_delete(ctx, message, "target.md"))

        text = message.answer.call_args[0][0]
        assert "superseded" in text.lower()


class TestHandleDeleteCallbackUnresolvablePath:
    def test_confirm_on_symlink_loop_is_rejected_without_mutation(self, env: IndexEnv) -> None:
        note = _indexed_note(env)
        original = note.read_text(encoding="utf-8")
        (env.vault / "loop-a.md").symlink_to(env.vault / "loop-b.md")
        (env.vault / "loop-b.md").symlink_to(env.vault / "loop-a.md")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        callback = _fake_callback("delete_confirm:loop-a.md")

        from vaultmind.bot.handlers.delete import handle_delete_callback

        asyncio.run(handle_delete_callback(ctx, callback))

        callback.message.edit_text.assert_awaited_once()
        callback.answer.assert_awaited_once()
        assert lineage_store.get_lineage("loop-a.md") == []
        assert note.read_text(encoding="utf-8") == original
        assert env.surfaces(TOKEN) == ALL_SURFACES
