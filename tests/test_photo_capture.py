"""Tests for photo capture handler and multimodal message types."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.llm.client import ContentPart, LLMResponse, Message, MultimodalMessage

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeImageConfig:
    enabled: bool = True
    vision_model: str = "gpt-4.1"
    max_image_size_bytes: int = 10_000_000
    save_originals: bool = True


class FakeVaultConfig:
    inbox_folder: str = "00-inbox"


class FakeLLMConfig:
    fast_model: str = "gpt-4.1"


class FakeSettings:
    image: FakeImageConfig = FakeImageConfig()
    vault: FakeVaultConfig = FakeVaultConfig()
    llm: FakeLLMConfig = FakeLLMConfig()
    telegram: Any = MagicMock(allowed_user_ids=[])


class FakeLLMClient:
    """Fake LLM client that records multimodal calls."""

    def __init__(self, description: str = "A whiteboard with notes.") -> None:
        self._description = description
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        raise AssertionError("complete() should not be called for photo capture")

    def complete_multimodal(
        self,
        messages: list[Message | MultimodalMessage],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        return LLMResponse(text=self._description, model=model, usage={})


def _make_fake_store() -> MagicMock:
    store = MagicMock()
    store.index_single_note = MagicMock(return_value=None)
    return store


def _make_fake_parser(note: MagicMock | None = None) -> MagicMock:
    parser = MagicMock()
    parser.parse_file = MagicMock(return_value=note or MagicMock())
    return parser


def _make_ctx(
    tmp_path: Path,
    save_originals: bool = True,
    vision_model: str = "gpt-4.1",
    llm_description: str = "A whiteboard.",
) -> Any:
    """Build a minimal HandlerContext-like object for photo capture tests."""
    from vaultmind.bot.handlers.context import HandlerContext

    settings = FakeSettings()
    settings.image = FakeImageConfig()
    settings.image.save_originals = save_originals
    settings.image.vision_model = vision_model

    ctx = MagicMock(spec=HandlerContext)
    ctx.settings = settings
    ctx.vault_root = tmp_path
    ctx.llm_client = FakeLLMClient(description=llm_description)
    ctx.store = _make_fake_store()
    ctx.parser = _make_fake_parser()
    return ctx


def _fake_photo(file_id: str = "FILE123", file_size: int = 1024) -> MagicMock:
    photo = MagicMock()
    photo.file_id = file_id
    photo.file_size = file_size
    return photo


def _fake_message(tmp_path: Path, fake_image_bytes: bytes = b"JPEG") -> MagicMock:
    """Build a fake aiogram Message with a bot that writes fake image bytes."""
    message = MagicMock()
    message.from_user = MagicMock(id=42)
    message.photo = [_fake_photo()]

    # Bot async fakes
    bot = MagicMock()
    fake_file = MagicMock()
    fake_file.file_path = "photos/file.jpg"
    bot.get_file = AsyncMock(return_value=fake_file)

    async def _download_file(file_path: str, destination: str) -> None:
        Path(destination).write_bytes(fake_image_bytes)

    bot.download_file = _download_file
    message.bot = bot
    message.answer = AsyncMock()
    return message


# ---------------------------------------------------------------------------
# Tests — multimodal message types
# ---------------------------------------------------------------------------


class TestMultimodalMessageStructure:
    def test_content_part_text(self) -> None:
        part = ContentPart(type="text", text="hello")
        assert part.type == "text"
        assert part.text == "hello"
        assert part.image_url == ""

    def test_content_part_image(self) -> None:
        uri = "data:image/jpeg;base64,abc=="
        part = ContentPart(type="image_url", image_url=uri)
        assert part.type == "image_url"
        assert part.image_url == uri

    def test_multimodal_message_parts(self) -> None:
        parts = [
            ContentPart(type="text", text="Describe this"),
            ContentPart(type="image_url", image_url="data:image/png;base64,xyz"),
        ]
        msg = MultimodalMessage(role="user", parts=parts)
        assert msg.role == "user"
        assert len(msg.parts) == 2
        assert msg.parts[0].type == "text"
        assert msg.parts[1].type == "image_url"

    def test_multimodal_message_frozen(self) -> None:
        msg = MultimodalMessage(role="user", parts=[])
        with pytest.raises((AttributeError, TypeError)):
            msg.role = "assistant"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — photo note creation
# ---------------------------------------------------------------------------


class TestPhotoCreatesNoteWithEmbed:
    def test_note_contains_image_embed(self, tmp_path: Path) -> None:
        """Written note must contain ![[images/..."""
        ctx = _make_ctx(tmp_path, save_originals=True)
        message = _fake_message(tmp_path)

        from vaultmind.bot.handlers.capture import handle_photo_capture

        asyncio.run(handle_photo_capture(ctx, message))

        inbox = tmp_path / "00-inbox"
        notes = list(inbox.glob("*.md"))
        assert len(notes) == 1, f"Expected 1 note, found {len(notes)}"
        content = notes[0].read_text()
        assert "![[images/" in content

    def test_note_contains_ai_description(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, llm_description="A cat on a keyboard.")
        message = _fake_message(tmp_path)

        from vaultmind.bot.handlers.capture import handle_photo_capture

        asyncio.run(handle_photo_capture(ctx, message))

        inbox = tmp_path / "00-inbox"
        notes = list(inbox.glob("*.md"))
        content = notes[0].read_text()
        assert "A cat on a keyboard." in content


class TestPhotoSavesOriginal:
    def test_image_file_written_when_save_originals_true(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, save_originals=True)
        fake_bytes = b"\xff\xd8\xff" + b"X" * 100  # minimal JPEG header
        message = _fake_message(tmp_path, fake_image_bytes=fake_bytes)

        from vaultmind.bot.handlers.capture import handle_photo_capture

        asyncio.run(handle_photo_capture(ctx, message))

        images_dir = tmp_path / "00-inbox" / "images"
        saved = list(images_dir.glob("*.jpg"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == fake_bytes

    def test_image_not_saved_when_save_originals_false(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, save_originals=False)
        message = _fake_message(tmp_path)

        from vaultmind.bot.handlers.capture import handle_photo_capture

        asyncio.run(handle_photo_capture(ctx, message))

        images_dir = tmp_path / "00-inbox" / "images"
        assert not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) == 0


class TestPhotoLargeImageRejected:
    def test_oversized_photo_rejected(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        ctx.settings.image.max_image_size_bytes = 100  # tiny limit

        message = _fake_message(tmp_path)
        # Override file_size to exceed limit
        message.photo = [_fake_photo(file_size=200)]

        from vaultmind.bot.handlers.capture import handle_photo_capture

        asyncio.run(handle_photo_capture(ctx, message))

        # No note should be written
        inbox = tmp_path / "00-inbox"
        assert not inbox.exists() or len(list(inbox.glob("*.md"))) == 0
        # Error message sent to user
        message.answer.assert_called_once()
        call_args = str(message.answer.call_args)
        assert "too large" in call_args.lower() or "max" in call_args.lower()
