"""Photo adapter preservation, retention and dependency boundaries."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from tests.test_capture import make_handlers, make_message, registered_callback, source_records
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.llm.client import ContentPart, LLMResponse, MultimodalMessage
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from pathlib import Path


def photo_message(data: bytes = b"JPEG-original") -> object:
    message = make_message()
    message.photo = [SimpleNamespace(file_id="photo-id", file_size=len(data))]
    message.bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="photo.jpg"))

    async def download(path: str, destination: io.BytesIO) -> None:
        assert isinstance(destination, io.BytesIO)
        destination.write(data)

    message.bot.download_file = AsyncMock(side_effect=download)
    return message


def test_multimodal_structure_remains_immutable() -> None:
    message = MultimodalMessage(
        role="user",
        parts=[
            ContentPart(type="text", text="describe"),
            ContentPart(type="image_url", image_url="data:image/jpeg;base64,eA=="),
        ],
    )
    assert message.parts[0].text == "describe"
    assert message.parts[1].image_url.endswith("eA==")
    with pytest.raises((AttributeError, TypeError)):
        message.role = "assistant"  # type: ignore[misc]


def test_original_precedes_vision_and_derivative_binds_exact_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    ctx = handlers._ctx
    ctx.llm_client.provider_name = "fake-provider"
    message = photo_message()

    def vision(*args: object) -> LLMResponse:
        records = source_records(handlers)
        assert len(records) == 1
        assert records[0].content == b"JPEG-original"
        return LLMResponse(text="  A whiteboard.\r\n", model="vision-model", usage={})

    ctx.llm_client.complete_multimodal.side_effect = vision
    callback = registered_callback(handlers, "handle_photo", monkeypatch)
    asyncio.run(callback(message))
    records = source_records(handlers)
    original = next(record for record in records if record.provenance["kind"] == "attachment")
    derived = next(record for record in records if record.provenance["role"] == "image-description")
    assert derived.text == "  A whiteboard.\r\n"
    assert original.source_id in {record.source_id for record in derived.dependencies}
    assert derived.provenance["attachments"][0]["id"] == original.source_id
    assert (ctx.settings.image.vision_model or ctx.settings.llm.fast_model) in derived.provenance[
        "adapter"
    ]
    assert next((tmp_path / "00-inbox/images").glob("*.jpg")).read_bytes() == b"JPEG-original"
    note = next((tmp_path / "00-inbox").glob("*.md")).read_text()
    assert "![[images/" in note and "A whiteboard." in note
    asyncio.run(callback(message))
    ctx.llm_client.complete_multimodal.assert_called_once()
    ctx.store.index_single_note.assert_called_once()
    (tmp_path / original.path).write_bytes(b"tampered")
    with SourceIntake(ctx.settings) as intake, pytest.raises(WikiError):
        intake.read(derived.source_id)


def test_originals_disabled_never_puts_binary_on_disk(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    ctx = handlers._ctx
    ctx.settings.image.save_originals = False
    ctx.llm_client.provider_name = "fake"
    ctx.llm_client.complete_multimodal.return_value = LLMResponse(
        text="Description", model="vision", usage={}
    )
    message = photo_message(b"UNIQUE-BINARY-ORIGINAL")
    asyncio.run(handlers.handle_photo(message))
    records = source_records(handlers)
    assert len(records) == 2
    assert all(record.provenance["original_retained"] is False for record in records)
    assert not list(tmp_path.rglob("*.bin"))
    assert not list(tmp_path.rglob("*.jpg"))
    assert all(
        b"UNIQUE-BINARY-ORIGINAL" not in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    asyncio.run(handlers.handle_photo(message))
    ctx.llm_client.complete_multimodal.assert_called_once()


def test_vision_failure_keeps_original_without_description_or_success(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    handlers._ctx.llm_client.provider_name = "fake"
    handlers._ctx.llm_client.complete_multimodal.side_effect = RuntimeError("private pixels")
    message = photo_message()
    asyncio.run(handlers.handle_photo(message))
    records = source_records(handlers)
    assert len(records) == 1 and records[0].content == b"JPEG-original"
    answer = message.answer.call_args.args[0]
    assert "binary original preserved" in answer and "Vision: failed" in answer
    assert "private pixels" not in answer
    assert not list((tmp_path / "00-inbox").glob("*.md"))
    handlers._ctx.store.index_single_note.assert_not_called()
    asyncio.run(handlers.handle_photo(message))
    handlers._ctx.llm_client.complete_multimodal.assert_called_once()


@pytest.mark.parametrize("mode", ["auth", "retention", "reserved", "declared-size", "actual-size"])
def test_photo_preconditions_prevent_provider_and_persistence(tmp_path: Path, mode: str) -> None:
    handlers = make_handlers(tmp_path)
    message = photo_message(b"12345")
    if mode == "auth":
        handlers._ctx.settings.telegram.allowed_user_ids = [999]
    elif mode == "retention":
        message.caption = "---\ndo_not_retain: true\n---\nprivate"
    elif mode == "reserved":
        handlers._ctx.settings.vault.inbox_folder = "_wiki"
    else:
        handlers._ctx.settings.image.max_image_size_bytes = 4
        if mode == "actual-size":
            message.photo[0].file_size = 1
    asyncio.run(handlers.handle_photo(message))
    handlers._ctx.llm_client.complete_multimodal.assert_not_called()
    assert not (tmp_path / "_wiki").exists()
    if mode != "actual-size":
        message.bot.download_file.assert_not_called()


@pytest.mark.parametrize("save_originals", [True, False])
def test_changed_caption_is_a_new_photo_revision_with_stable_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_originals: bool
) -> None:
    handlers = make_handlers(tmp_path)
    ctx = handlers._ctx
    ctx.settings.image.save_originals = save_originals
    ctx.llm_client.provider_name = "scripted"
    ctx.llm_client.complete_multimodal.return_value = LLMResponse(
        text="A whiteboard.", model="vision", usage={}
    )
    message = photo_message()
    message.caption = "First caption"
    callback = registered_callback(handlers, "handle_photo", monkeypatch)
    asyncio.run(callback(message))
    first = next(
        record
        for record in source_records(handlers)
        if record.provenance["role"] == "image-description"
    )
    before = {path: path.read_bytes() for path in (tmp_path / "_wiki/sources").iterdir()}

    message.caption = "Changed caption"
    asyncio.run(callback(message))
    second = next(
        record
        for record in source_records(handlers)
        if record.provenance["role"] == "image-description" and record.source_id != first.source_id
    )
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert second.provenance["event_id"] != first.provenance["event_id"]
    if save_originals:
        original = next(
            record for record in second.dependencies if record.content == b"JPEG-original"
        )
        assert original.dependencies[0].text == "Changed caption"
    else:
        assert second.dependencies[0].text == "Changed caption"
        assert not list(tmp_path.rglob("*.bin"))
        assert not list(tmp_path.rglob("*.jpg"))
    completed = {path: path.read_bytes() for path in (tmp_path / "_wiki/sources").iterdir()}
    asyncio.run(callback(message))
    assert {path: path.read_bytes() for path in (tmp_path / "_wiki/sources").iterdir()} == completed
    assert ctx.llm_client.complete_multimodal.call_count == 2
    assert ctx.store.index_single_note.call_count == 2
