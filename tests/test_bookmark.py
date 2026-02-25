"""Tests for bookmark handler — session and Q&A bookmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vaultmind.bot.handlers.bookmark import (
    LastExchange,
    _format_session_body,
    _slugify,
    handle_bookmark,
)
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeVaultConfig:
    path: Path
    inbox_folder: str = "00-inbox"
    excluded_folders: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.excluded_folders is None:
            self.excluded_folders = []


@dataclass
class _FakeSettings:
    vault: _FakeVaultConfig
    telegram: Any = None

    def __post_init__(self) -> None:
        if self.telegram is None:
            self.telegram = _FakeTelegramConfig()


@dataclass
class _FakeTelegramConfig:
    allowed_user_ids: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allowed_user_ids is None:
            self.allowed_user_ids = []  # empty = allow all


def _make_from_user(user_id: int = 42) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    return user


def _make_message(user_id: int = 42) -> AsyncMock:
    msg = AsyncMock()
    msg.from_user = _make_from_user(user_id)
    msg.answer = AsyncMock()
    return msg


def _make_note_stub(path: Path, vault_root: Path) -> Note:
    rel = path.relative_to(vault_root)
    return Note(path=rel, title="Test", content="# Test\n\nContent")


def _make_ctx(tmp_path: Path, thinking: MagicMock, store: MagicMock) -> MagicMock:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()

    cfg = _FakeVaultConfig(path=vault_root)
    settings = _FakeSettings(vault=cfg)

    parser = MagicMock()
    parser.parse_file.side_effect = lambda fp: _make_note_stub(fp, vault_root)

    ctx = MagicMock()
    ctx.settings = settings
    ctx.vault_root = vault_root
    ctx.thinking = thinking
    ctx.store = store
    ctx.parser = parser
    return ctx


def _make_thinking(*, has_session: bool, history: list[dict[str, str]] | None = None) -> MagicMock:
    thinking = MagicMock()
    thinking.has_active_session.return_value = has_session
    session = MagicMock()
    session.history = history or []
    thinking._get_session.return_value = session
    return thinking


# ---------------------------------------------------------------------------
# Unit tests: _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercase_and_spaces_to_hyphens(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_strips_non_alphanumeric(self) -> None:
        assert _slugify("My Note! #1") == "my-note-1"

    def test_collapses_multiple_spaces(self) -> None:
        assert _slugify("a  b   c") == "a-b-c"

    def test_truncates_at_60(self) -> None:
        long = "a" * 80
        assert len(_slugify(long)) <= 60

    def test_strips_leading_trailing_hyphens(self) -> None:
        result = _slugify("  hello  ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_empty_string(self) -> None:
        assert _slugify("") == ""

    def test_keeps_hyphens(self) -> None:
        assert _slugify("already-hyphenated") == "already-hyphenated"

    def test_collapses_repeated_hyphens(self) -> None:
        assert _slugify("a--b") == "a-b"


# ---------------------------------------------------------------------------
# Unit tests: _format_session_body
# ---------------------------------------------------------------------------


class TestFormatSessionBody:
    def test_single_turn(self) -> None:
        history = [{"user": "What is PKM?", "assistant": "PKM stands for..."}]
        body = _format_session_body(history)
        assert "**User:** What is PKM?" in body
        assert "**Assistant:** PKM stands for..." in body

    def test_multiple_turns(self) -> None:
        history = [
            {"user": "Q1", "assistant": "A1"},
            {"user": "Q2", "assistant": "A2"},
        ]
        body = _format_session_body(history)
        assert body.index("Q1") < body.index("Q2")
        assert body.index("A1") < body.index("A2")

    def test_empty_history(self) -> None:
        assert _format_session_body([]) == ""


# ---------------------------------------------------------------------------
# Integration tests: handle_bookmark
# ---------------------------------------------------------------------------


class TestBookmarkFromThinkingSession:
    @pytest.mark.asyncio
    async def test_creates_note_file(self, tmp_path: Path) -> None:
        history = [{"user": "What is emergence?", "assistant": "Emergence is..."}]
        thinking = _make_thinking(has_session=True, history=history)
        store = MagicMock()
        ctx = _make_ctx(tmp_path, thinking, store)
        message = _make_message()

        await handle_bookmark(ctx, message, "Emergence Notes", {})

        inbox = ctx.vault_root / "00-inbox"
        files = list(inbox.glob("*.md"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_note_content_has_correct_tags(self, tmp_path: Path) -> None:
        history = [{"user": "topic", "assistant": "response"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "My Bookmark", {})

        inbox = ctx.vault_root / "00-inbox"
        note_file = list(inbox.glob("*.md"))[0]
        content = note_file.read_text()
        assert "tags: [bookmark, thinking]" in content
        assert "source: telegram-thinking" in content

    @pytest.mark.asyncio
    async def test_note_frontmatter_has_title(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "Session Title", {})

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        assert 'title: "Session Title"' in content

    @pytest.mark.asyncio
    async def test_note_body_contains_session_messages(self, tmp_path: Path) -> None:
        history = [{"user": "Hello?", "assistant": "World!"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "Exchange", {})

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        assert "**User:** Hello?" in content
        assert "**Assistant:** World!" in content

    @pytest.mark.asyncio
    async def test_indexes_note_immediately(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        store = MagicMock()
        ctx = _make_ctx(tmp_path, thinking, store)
        message = _make_message()

        await handle_bookmark(ctx, message, "Indexed Note", {})

        store.index_single_note.assert_called_once()

    @pytest.mark.asyncio
    async def test_reply_contains_relative_path(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "Path Test", {})

        call_args = message.answer.call_args
        reply_text = call_args[0][0] if call_args[0] else call_args.kwargs.get("text", "")
        assert "00-inbox" in reply_text
        assert "bookmark" in reply_text

    @pytest.mark.asyncio
    async def test_empty_session_history_replies_nothing(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=True, history=[])
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "Empty", {})

        call_args = message.answer.call_args
        reply = call_args[0][0] if call_args[0] else ""
        assert "Nothing to bookmark" in reply


class TestBookmarkFromLastExchange:
    @pytest.mark.asyncio
    async def test_creates_note_from_exchange(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=99)
        exchanges: dict[int, LastExchange] = {
            99: LastExchange(query="What is PKM?", response="PKM is...", timestamp=time.monotonic())
        }

        await handle_bookmark(ctx, message, "PKM Overview", exchanges)

        inbox = ctx.vault_root / "00-inbox"
        files = list(inbox.glob("*.md"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_note_content_has_chat_tags(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=99)
        exchanges: dict[int, LastExchange] = {
            99: LastExchange(query="Q", response="A", timestamp=time.monotonic())
        }

        await handle_bookmark(ctx, message, "Chat Bookmark", exchanges)

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        assert "tags: [bookmark, chat]" in content
        assert "source: telegram-chat" in content

    @pytest.mark.asyncio
    async def test_note_contains_query_and_response(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=5)
        exchanges: dict[int, LastExchange] = {
            5: LastExchange(
                query="How does attention work?",
                response="Attention mechanisms...",
                timestamp=time.monotonic(),
            )
        }

        await handle_bookmark(ctx, message, "Attention", exchanges)

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        assert "**Question:** How does attention work?" in content
        assert "**Answer:** Attention mechanisms..." in content

    @pytest.mark.asyncio
    async def test_indexes_exchange_note(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        store = MagicMock()
        ctx = _make_ctx(tmp_path, thinking, store)
        message = _make_message(user_id=7)
        exchanges: dict[int, LastExchange] = {
            7: LastExchange(query="Q", response="A", timestamp=time.monotonic())
        }

        await handle_bookmark(ctx, message, "Indexed", exchanges)

        store.index_single_note.assert_called_once()


class TestBookmarkNoContent:
    @pytest.mark.asyncio
    async def test_no_session_no_exchange_replies_nothing(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=1)

        await handle_bookmark(ctx, message, "Empty Bookmark", {})

        call_args = message.answer.call_args
        reply = call_args[0][0] if call_args[0] else ""
        assert "Nothing to bookmark" in reply

    @pytest.mark.asyncio
    async def test_no_file_created_when_nothing_to_bookmark(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=1)

        await handle_bookmark(ctx, message, "Empty", {})

        inbox = ctx.vault_root / "00-inbox"
        # Either inbox doesn't exist or is empty
        if inbox.exists():
            assert list(inbox.glob("*.md")) == []

    @pytest.mark.asyncio
    async def test_exchange_from_different_user_not_used(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=1)
        # Exchange belongs to user 2, not user 1
        exchanges: dict[int, LastExchange] = {
            2: LastExchange(query="Q", response="A", timestamp=time.monotonic())
        }

        await handle_bookmark(ctx, message, "Wrong user", exchanges)

        call_args = message.answer.call_args
        reply = call_args[0][0] if call_args[0] else ""
        assert "Nothing to bookmark" in reply


class TestSlugInFilename:
    @pytest.mark.asyncio
    async def test_filename_uses_slug(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "My Great Topic", {})

        inbox = ctx.vault_root / "00-inbox"
        files = list(inbox.glob("*.md"))
        assert len(files) == 1
        assert "my-great-topic" in files[0].name

    @pytest.mark.asyncio
    async def test_filename_includes_date_prefix(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "Test", {})

        inbox = ctx.vault_root / "00-inbox"
        files = list(inbox.glob("*.md"))
        # filename must start with YYYY-MM-DD pattern
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2}-bookmark-", files[0].name)

    @pytest.mark.asyncio
    async def test_inbox_created_if_missing(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        inbox = ctx.vault_root / "00-inbox"
        assert not inbox.exists()

        await handle_bookmark(ctx, message, "Auto-create inbox", {})

        assert inbox.exists()


class TestFrontmatterCorrectness:
    @pytest.mark.asyncio
    async def test_thinking_frontmatter_fields(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_bookmark(ctx, message, "FM Test", {})

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        import re

        assert re.search(r"date:\s+\"\d{4}-\d{2}-\d{2}\"", content)
        assert 'title: "FM Test"' in content

    @pytest.mark.asyncio
    async def test_chat_frontmatter_fields(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=3)
        exchanges: dict[int, LastExchange] = {
            3: LastExchange(query="Q", response="A", timestamp=time.monotonic())
        }

        await handle_bookmark(ctx, message, "Chat FM", exchanges)

        inbox = ctx.vault_root / "00-inbox"
        content = list(inbox.glob("*.md"))[0].read_text()
        import re

        assert re.search(r"date:\s+\"\d{4}-\d{2}-\d{2}\"", content)
        assert 'title: "Chat FM"' in content


class TestLastExchangeDataclass:
    def test_fields(self) -> None:
        ts = time.monotonic()
        ex = LastExchange(query="q", response="r", timestamp=ts)
        assert ex.query == "q"
        assert ex.response == "r"
        assert ex.timestamp == ts

    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(LastExchange)
