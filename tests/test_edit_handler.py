"""Tests for the edit handler (`bot/handlers/edit.py`, M9 supersede + M14 current content).

The note, parser, Chroma collection, SQLite BM25 index and lineage store are all
real (`tests/index_support.py`); only the Telegram objects, the language model and
the embedding provider are synthetic. Assertions are about the file on disk, what
the vector, keyword and hybrid retrieval paths return, the lineage history, the
text the model is shown and what the user is told.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.index_support import ALL_SURFACES, IndexEnv, make_env
from vaultmind.graph.evolution import LineageEdge, LineageStore
from vaultmind.llm.client import LLMResponse

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

T0 = "alphaquartz"
T1 = "betaquartz"
T2 = "gammaquartz"

AMBIGUOUS_NOTE = (
    "---\n"
    "title: note\n"
    "superseded_at:\n"
    "- '2026-03-01T10:00:00+00:00'\n"
    "---\n\n"
    f"Revised claim {T1}.\n\n"
    f"Old claim {T0}.\n"
)

Pending = dict[int, dict[str, str]]


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


def _make_ctx(
    env: IndexEnv,
    *,
    lineage_store: LineageStore | None = None,
    replies: Sequence[str] = (),
) -> Any:
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.vault_root = env.vault
    ctx.settings = MagicMock()
    ctx.settings.telegram.allowed_user_ids = []
    ctx.settings.routing.chat_model = "test-model"
    ctx.settings.llm.fast_model = "test-model"
    ctx.settings.llm.max_tokens = 1024
    ctx.llm_client = MagicMock()
    ctx.llm_client.complete = MagicMock(
        side_effect=[LLMResponse(text=text, model="test-model", usage={}) for text in replies]
    )
    ctx.store = env.store
    ctx.parser = env.parser
    ctx.lineage_store = lineage_store
    return ctx


def _fake_message() -> MagicMock:
    message = MagicMock()
    message.from_user = MagicMock(id=7)
    message.answer = AsyncMock()
    return message


def _fake_callback(data: str, user_id: int = 7) -> MagicMock:
    callback = MagicMock()
    callback.data = data
    callback.from_user = MagicMock(id=user_id)
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()
    return callback


def _request(ctx: Any, args: str, pending: Pending) -> list[str]:
    """Run the real /edit request handler; return everything the user was told."""
    from vaultmind.bot.handlers.edit import handle_edit

    message = _fake_message()
    asyncio.run(handle_edit(ctx, message, args, pending))
    return [str(call.args[0]) for call in message.answer.call_args_list]


def _confirm(ctx: Any, pending: Pending, data: str = "edit_confirm:7") -> str:
    """Run the real confirm handler; return the text the user was shown."""
    from vaultmind.bot.handlers.edit import handle_edit_callback

    callback = _fake_callback(data)
    asyncio.run(handle_edit_callback(ctx, callback, pending))
    return str(callback.message.edit_text.call_args[0][0])


def _delete(ctx: Any) -> None:
    from vaultmind.bot.handlers.delete import handle_delete_callback

    asyncio.run(handle_delete_callback(ctx, _fake_callback("delete_confirm:note.md")))


def _prompt(ctx: Any) -> str:
    return str(ctx.llm_client.complete.call_args.kwargs["messages"][0].content)


def _pending(edited: str, instruction: str = "z", path: str = "note.md") -> Pending:
    return {7: {"path": path, "original": "x", "edited": edited, "instruction": instruction}}


def _indexed_note(env: IndexEnv, body: str = f"Old body that must survive {T0}.") -> Path:
    path = env.write("note.md", body)
    assert env.index("note.md") > 0
    assert env.surfaces(T0) == ALL_SURFACES
    return path


class TestHandleEditShowsCurrentTextToLLM:
    def test_llm_prompt_shows_body_not_raw_frontmatter(self, env: IndexEnv) -> None:
        env.write("note.md", "Only the body, not frontmatter.")
        ctx = _make_ctx(env, replies=["Edited."])

        _request(ctx, "note.md add a line", {})

        assert "Only the body, not frontmatter." in _prompt(ctx)
        assert "title: note" not in _prompt(ctx)

    def test_second_edit_is_generated_from_the_first_edit_not_from_history(
        self, env: IndexEnv
    ) -> None:
        env.write("note.md", f"v0 claim {T0}.")
        ctx = _make_ctx(env, replies=[f"v1 claim {T1}.", f"v2 claim {T2}."])
        pending: Pending = {}

        _request(ctx, "note.md first change", pending)
        _confirm(ctx, pending)
        _request(ctx, "note.md second change", pending)

        prompt = _prompt(ctx)
        assert T1 in prompt
        assert T0 not in prompt
        assert "Edited via bot" not in prompt
        assert "vaultmind:supersede" not in prompt

    def test_ordinary_authored_supersede_callouts_are_part_of_the_text_the_model_sees(
        self, env: IndexEnv
    ) -> None:
        body = (
            "Claim kestrel.\n\n> [!superseded] Edited via bot — 2026-01-01\n"
            "> Instruction: an authored example\n\nTrailing claim."
        )
        env.write("note.md", body)
        ctx = _make_ctx(env, replies=["Edited."])

        _request(ctx, "note.md tidy", {})

        assert body in _prompt(ctx)


class TestHandleEditCallbackSupersede:
    def test_confirm_writes_new_content_above_dated_callout_preserving_old_body(
        self, env: IndexEnv
    ) -> None:
        path = env.write("note.md", "Old body that must survive.")
        ctx = _make_ctx(env)

        _confirm(ctx, _pending("New body content.", "make it better"))

        content = path.read_text(encoding="utf-8")
        assert "New body content." in content
        assert "Old body that must survive." in content
        assert "[!superseded] Edited via bot" in content
        assert content.index("New body content.") < content.index("[!superseded]")
        assert content.index("[!superseded]") < content.index("Old body that must survive.")

    def test_confirm_makes_only_the_new_text_retrievable(self, env: IndexEnv) -> None:
        _indexed_note(env)
        ctx = _make_ctx(env)

        _confirm(ctx, _pending(f"New body content {T1}."))

        assert env.surfaces(T0) == set()
        assert env.surfaces(T1) == ALL_SURFACES

    def test_confirm_records_one_lineage_edge_with_instruction_rationale(
        self, env: IndexEnv
    ) -> None:
        env.write("note.md", "Body.")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        _confirm(ctx, _pending("y", "add a summary"))

        history = lineage_store.get_lineage("note.md")
        assert len(history) == 1
        assert history[0].event == "edited"
        assert "add a summary" in history[0].detail

    def test_confirm_response_mentions_preserved_text(self, env: IndexEnv) -> None:
        env.write("note.md", "Body.")
        ctx = _make_ctx(env)

        reply = _confirm(ctx, _pending("y"))

        assert "preserved" in reply.lower()

    def test_replacement_text_containing_supersede_lookalikes_is_kept_whole(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        replacement = (
            f"Edited claim {T1}.\n\n"
            "> [!superseded] Edited via bot — 2026-01-01\n"
            "> Instruction: an authored example\n\n"
            f"Trailing claim {T2}."
        )
        ctx = _make_ctx(env)

        _confirm(ctx, _pending(replacement))

        assert env.note("note.md").current_content() == replacement
        assert env.surfaces(T1) == ALL_SURFACES
        assert env.surfaces(T2) == ALL_SURFACES
        assert env.surfaces(T0) == set()
        assert f"Old body that must survive {T0}." in path.read_text(encoding="utf-8")


class TestHandleEditCallbackMultilineInstruction:
    def test_blank_lines_in_the_instruction_stay_inside_the_callout_and_in_lineage(
        self, env: IndexEnv
    ) -> None:
        path = env.write("note.md", f"v0 claim {T0}.")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store, replies=[f"v1 claim {T1}."])
        pending: Pending = {}

        _request(ctx, "note.md first line\n\nsecond paragraph with kestrel", pending)
        _confirm(ctx, pending)

        raw = path.read_text(encoding="utf-8")
        assert "> Instruction: first line\n>\n> second paragraph with kestrel\n" in raw
        assert env.note("note.md").current_content() == f"v1 claim {T1}."
        detail = lineage_store.get_lineage("note.md")[0].detail
        assert "first line" in detail
        assert "second paragraph with kestrel" in detail


class TestHandleEditCallbackSequentialEdits:
    def test_two_sequential_edits_keep_history_and_retrieve_only_the_latest_claim(
        self, env: IndexEnv
    ) -> None:
        path = env.write("note.md", f"v0 claim {T0}.")
        env.index("note.md")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store, replies=[f"v1 claim {T1}.", f"v2 {T2}."])
        pending: Pending = {}

        _request(ctx, "note.md a", pending)
        _confirm(ctx, pending)
        _request(ctx, "note.md b", pending)
        _confirm(ctx, pending)

        content = path.read_text(encoding="utf-8")
        for token in (T0, T1, T2):
            assert token in content
        assert len(lineage_store.get_lineage("note.md")) == 2
        for token in (T0, T1):
            assert env.surfaces(token) == set()
        assert env.surfaces(T2) == ALL_SURFACES

        env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

        for token in (T0, T1):
            assert env.surfaces(token) == set()
        assert env.surfaces(T2) == ALL_SURFACES
        assert T0 in path.read_text(encoding="utf-8")


class TestHandleEditCallbackLeadingThematicRule:
    def test_authored_leading_rule_survives_two_edits_and_only_latest_is_current(
        self, env: IndexEnv
    ) -> None:
        rule = "---"
        original = f"{rule}\n\nv0 claim {T0}.\n\n{rule}\n\nOlder section after a rule."
        first = f"{rule}\n\nv1 claim {T1}.\n\n{rule}\n\nMiddle section after a rule."
        second = f"{rule}\n\nv2 claim {T2}.\n\n{rule}\n\nFinal section after a rule."
        path = env.write("note.md", original)
        assert env.index("note.md") > 0
        assert env.surfaces(T0) == ALL_SURFACES
        ctx = _make_ctx(env, lineage_store=_lineage(env), replies=[first, second])
        pending: Pending = {}

        _request(ctx, "note.md a", pending)
        _confirm(ctx, pending)
        _request(ctx, "note.md b", pending)
        _confirm(ctx, pending)

        raw = path.read_text(encoding="utf-8")
        for token in (T0, T1, T2, "Older section", "Middle section", "Final section"):
            assert token in raw

        note = env.note("note.md")
        assert note.current_content() == second
        chunk_text = "\n".join(c.content for c in env.parser.chunk_note(note))
        assert T2 in chunk_text
        assert T0 not in chunk_text
        assert T1 not in chunk_text

        for token in (T0, T1):
            assert env.surfaces(token) == set()
        assert env.surfaces(T2) == ALL_SURFACES

        env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

        for token in (T0, T1):
            assert env.surfaces(token) == set()
        assert env.surfaces(T2) == ALL_SURFACES
        assert path.read_text(encoding="utf-8") == raw


class TestHandleEditRefusals:
    def test_ambiguous_history_is_refused_before_the_model_is_asked(self, env: IndexEnv) -> None:
        path = env.vault / "note.md"
        path.write_text(AMBIGUOUS_NOTE, encoding="utf-8")
        ctx = _make_ctx(env)
        pending: Pending = {}

        replies = _request(ctx, "note.md tidy", pending)

        assert [reply.startswith("Edit not applied") for reply in replies] == [True]
        ctx.llm_client.complete.assert_not_called()
        assert pending == {}
        assert path.read_text(encoding="utf-8") == AMBIGUOUS_NOTE

    def test_bot_deleted_note_is_refused_before_the_model_is_asked(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        ctx = _make_ctx(env)
        _delete(ctx)
        deleted = path.read_text(encoding="utf-8")
        pending: Pending = {}

        replies = _request(ctx, "note.md tidy", pending)

        assert len(replies) == 1
        assert replies[0].startswith("Edit not applied")
        assert "deleted via bot" in replies[0]
        ctx.llm_client.complete.assert_not_called()
        assert pending == {}
        assert path.read_text(encoding="utf-8") == deleted
        assert env.surfaces(T0) == set()

    def test_note_deleted_after_the_request_is_not_edited_on_confirm(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store, replies=[f"Resurrected {T1}."])
        pending: Pending = {}
        _request(ctx, "note.md tidy", pending)
        _delete(ctx)
        deleted = path.read_text(encoding="utf-8")

        reply = _confirm(ctx, pending)

        assert reply.startswith("Edit not applied")
        assert "deleted via bot" in reply
        assert path.read_text(encoding="utf-8") == deleted
        assert [edge.event for edge in lineage_store.get_lineage("note.md")] == ["deleted"]
        assert env.surfaces(T0) == set()
        assert env.surfaces(T1) == set()

    def test_history_damaged_after_the_request_is_not_edited_on_confirm(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store, replies=[f"Edited {T1}."])
        pending: Pending = {}
        _request(ctx, "note.md tidy", pending)
        path.write_text(AMBIGUOUS_NOTE, encoding="utf-8")

        reply = _confirm(ctx, pending)

        assert reply.startswith("Edit not applied to note.md")
        assert "The note was not changed." in reply
        assert path.read_text(encoding="utf-8") == AMBIGUOUS_NOTE
        assert lineage_store.get_lineage("note.md") == []
        assert env.surfaces(T0) == ALL_SURFACES


class TestHandleEditCallbackFailureReporting:
    def test_embedding_failure_is_reported_and_leaves_the_last_good_evidence(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        env.embedder.failure = RuntimeError("provider down")

        reply = _confirm(ctx, _pending(f"New body {T1}."))

        assert reply.startswith("\u26a0\ufe0f")
        assert "indexing failed" in reply
        assert "Edit applied" not in reply
        assert "\u2705" not in reply
        content = path.read_text(encoding="utf-8")
        assert T0 in content
        assert T1 in content
        assert "[!superseded] Edited via bot" in content
        assert [edge.event for edge in lineage_store.get_lineage("note.md")] == ["edited"]
        assert env.surfaces(T0) == ALL_SURFACES
        assert env.surfaces(T1) == set()

        env.embedder.failure = None
        assert env.index("note.md") > 0
        assert env.surfaces(T0) == set()
        assert env.surfaces(T1) == ALL_SURFACES

    def test_keyword_index_failure_is_reported_and_a_retry_converges(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        env.bm25.fail_on = {"replace_note"}

        reply = _confirm(ctx, _pending(f"New body {T1}."))

        assert reply.startswith("\u26a0\ufe0f")
        assert "indexing failed" in reply
        assert "Edit applied" not in reply
        assert T0 in path.read_text(encoding="utf-8")
        assert len(lineage_store.get_lineage("note.md")) == 1

        env.bm25.fail_on = set()
        assert env.index("note.md") > 0
        assert env.surfaces(T0) == set()
        assert env.surfaces(T1) == ALL_SURFACES

    def test_lineage_failure_is_reported_but_the_edit_is_applied_and_indexed(
        self, env: IndexEnv
    ) -> None:
        path = _indexed_note(env)
        flaky = FlakyLineage(env.data / "lineage.json")
        ctx = _make_ctx(env, lineage_store=flaky)
        flaky.fail_next = True

        reply = _confirm(ctx, _pending(f"New body {T1}."))

        assert reply.startswith("\u26a0\ufe0f")
        assert "lineage edge was not recorded" in reply
        assert "Edit applied" not in reply
        assert T0 in path.read_text(encoding="utf-8")
        assert env.surfaces(T1) == ALL_SURFACES
        assert env.surfaces(T0) == set()
        assert flaky.get_lineage("note.md") == []


class TestHandleEditCallbackCancel:
    def test_cancel_discards_pending_edit_and_leaves_file_untouched(self, env: IndexEnv) -> None:
        path = env.write("note.md", "Untouched.")
        original = path.read_text(encoding="utf-8")
        pending = _pending("y")
        ctx = _make_ctx(env)

        _confirm(ctx, pending, "edit_cancel:7")

        assert path.read_text(encoding="utf-8") == original
        assert 7 not in pending


class TestHandleEditCallbackAuthorization:
    def test_spoofed_owner_payload_leaves_pending_and_note_untouched(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        original = path.read_text(encoding="utf-8")
        pending = _pending("y")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)

        from vaultmind.bot.handlers.edit import handle_edit_callback

        for data in ("edit_confirm:7", "edit_cancel:7"):
            asyncio.run(handle_edit_callback(ctx, _fake_callback(data, user_id=8), pending))

        assert 7 in pending
        assert path.read_text(encoding="utf-8") == original
        assert lineage_store.get_lineage("note.md") == []
        assert env.surfaces(T0) == ALL_SURFACES

    def test_unlisted_actor_cannot_consume_own_pending_edit(self, env: IndexEnv) -> None:
        path = _indexed_note(env)
        original = path.read_text(encoding="utf-8")
        pending: Pending = {8: _pending("y")[7]}
        ctx = _make_ctx(env)
        ctx.settings.telegram.allowed_user_ids = [7]

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, _fake_callback("edit_confirm:8", user_id=8), pending))

        assert 8 in pending
        assert path.read_text(encoding="utf-8") == original
        assert env.surfaces(T0) == ALL_SURFACES


class TestHandleEditCallbackExpired:
    def test_confirm_with_no_pending_edit_reports_expired(self, env: IndexEnv) -> None:
        env.write("note.md", "Body.")
        ctx = _make_ctx(env)

        reply = _confirm(ctx, {})

        assert "expired" in reply.lower()


class TestHandleEditCallbackUnresolvablePath:
    def test_confirm_on_symlink_loop_is_rejected_without_mutation(self, env: IndexEnv) -> None:
        note = _indexed_note(env)
        original = note.read_text(encoding="utf-8")
        (env.vault / "loop-a.md").symlink_to(env.vault / "loop-b.md")
        (env.vault / "loop-b.md").symlink_to(env.vault / "loop-a.md")
        lineage_store = _lineage(env)
        ctx = _make_ctx(env, lineage_store=lineage_store)
        callback = _fake_callback("edit_confirm:7")

        from vaultmind.bot.handlers.edit import handle_edit_callback

        asyncio.run(handle_edit_callback(ctx, callback, _pending("y", path="loop-a.md")))

        callback.message.edit_text.assert_awaited_once()
        callback.answer.assert_awaited_once()
        assert lineage_store.get_lineage("loop-a.md") == []
        assert note.read_text(encoding="utf-8") == original
        assert env.surfaces(T0) == ALL_SURFACES
