"""Tests for the manual text capture handler (`bot/handlers/capture.py::handle_capture`)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeVaultConfig:
    inbox_folder: str = "00-inbox"


class FakeIngestConfig:
    enabled: bool = False  # URL ingestion disabled -> exercises plain-text capture
    youtube_language: str = "en"
    max_content_length: int = 50_000


class FakeSettings:
    vault: FakeVaultConfig = FakeVaultConfig()
    ingest: FakeIngestConfig = FakeIngestConfig()
    telegram: Any = MagicMock(allowed_user_ids=[])


def _make_fake_store() -> MagicMock:
    store = MagicMock()
    store.index_single_note = MagicMock(return_value=None)
    return store


def _make_fake_parser(note: MagicMock | None = None) -> MagicMock:
    parser = MagicMock()
    parser.parse_file = MagicMock(return_value=note or MagicMock())
    return parser


def _make_ctx(tmp_path: Path) -> Any:
    """Build a minimal HandlerContext-like object for capture tests."""
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.settings = FakeSettings()
    ctx.vault_root = tmp_path
    ctx.store = _make_fake_store()
    ctx.parser = _make_fake_parser()
    return ctx


def _fake_message() -> MagicMock:
    message = MagicMock()
    message.from_user = MagicMock(id=42)
    message.answer = AsyncMock()
    return message


class TestHandleCaptureStampsAuthority:
    def test_manual_capture_stamped_with_authority_5(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        message = _fake_message()

        from vaultmind.bot.handlers.capture import handle_capture

        asyncio.run(handle_capture(ctx, message, "A quick thought worth keeping."))

        inbox = tmp_path / "00-inbox"
        notes = list(inbox.glob("*.md"))
        assert len(notes) == 1, f"Expected 1 note, found {len(notes)}"
        content = notes[0].read_text()
        assert "authority: 5" in content

    def test_manual_capture_note_indexed(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        message = _fake_message()

        from vaultmind.bot.handlers.capture import handle_capture

        asyncio.run(handle_capture(ctx, message, "Another quick capture."))

        ctx.store.index_single_note.assert_called_once()

    def test_empty_text_after_sanitization_writes_no_note(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        message = _fake_message()

        from vaultmind.bot.handlers.capture import handle_capture

        asyncio.run(handle_capture(ctx, message, "   "))

        inbox = tmp_path / "00-inbox"
        assert not inbox.exists() or len(list(inbox.glob("*.md"))) == 0
