"""Intake behavior through the public Telegram command facade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.bot.commands import CommandHandlers
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.config import Settings
from vaultmind.knowledge.intake import SourceIntake

if TYPE_CHECKING:
    from pathlib import Path


def make_handlers(tmp_path: Path) -> CommandHandlers:
    settings = Settings(vault={"path": tmp_path})
    settings.ingest.enabled = False
    ctx = HandlerContext(
        settings=settings,
        vault_root=tmp_path,
        parser=MagicMock(),
        store=MagicMock(),
        graph=MagicMock(),
        thinking=MagicMock(),
        llm_client=MagicMock(),
    )
    ctx.thinking.has_active_session.return_value = False
    handlers = CommandHandlers.__new__(CommandHandlers)
    handlers._ctx = ctx
    handlers._last_exchanges = {}
    return handlers


def make_message(text: str | None = None, event: int = 10) -> MagicMock:
    message = MagicMock()
    message.text = text
    message.caption = None
    message.chat = SimpleNamespace(id=12)
    message.message_id = event
    message.from_user = SimpleNamespace(id=42)
    message.answer = AsyncMock()
    return message


def source_records(handlers: CommandHandlers) -> list[object]:
    root = handlers._ctx.vault_root
    with SourceIntake(handlers._ctx.settings) as intake:
        return [
            intake.read(path.name.removesuffix(".intake.json"))
            for path in sorted((root / "_wiki/sources").glob("*.intake.json"))
        ]


def registered_callback(
    handlers: CommandHandlers, name: str, monkeypatch: pytest.MonkeyPatch
) -> object:
    from aiogram import Router

    from vaultmind.bot import telegram

    router = Router()
    monkeypatch.setattr(telegram, "router", router)
    telegram.register_handlers(handlers)
    return next(
        handler.callback for handler in router.message.handlers if handler.callback.__name__ == name
    )


def test_raw_routing_preserves_exact_bytes_and_full_capture_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    original = "note: \r\n" + "é" * 9000 + "\r\n  "
    message = make_message(original)
    callback = registered_callback(handlers, "handle_text", monkeypatch)
    asyncio.run(callback(message))
    records = source_records(handlers)
    assert len(records) == 1
    assert records[0].content == original.encode("utf-8")
    assert records[0].provenance["origin"] == "telegram:chat=12:actor=42"
    assert records[0].provenance["event_id"] == "12:10:42"
    note = next((tmp_path / "00-inbox").glob("*.md")).read_text()
    assert "é" * 9000 in note
    assert f"source_id: {records[0].source_id}" in note
    assert "authority: 5" in note
    handlers._ctx.store.index_single_note.assert_called_once()


@pytest.mark.parametrize(
    "text",
    [
        "note: " + "x" * 10001,
        "note: ---\ndo_not_retain: true\n---\nprivate",
        "note: ---\nretention: do-not-retain\n---\nprivate",
    ],
)
def test_routed_refusal_precedes_all_persistence(tmp_path: Path, text: str) -> None:
    handlers = make_handlers(tmp_path)
    message = make_message(text)
    asyncio.run(handlers.handle_message(message, text))
    assert not (tmp_path / "_wiki").exists()
    assert not (tmp_path / "00-inbox").exists()
    handlers._ctx.store.index_single_note.assert_not_called()


def test_event_replay_revision_and_actor_have_distinct_immutable_identity(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    first = make_message("exact\r\noriginal\r\n")
    asyncio.run(handlers.handle_capture(first, first.text))
    asyncio.run(handlers.handle_capture(first, first.text))
    assert len(source_records(handlers)) == 1
    handlers._ctx.store.index_single_note.assert_called_once()
    first.text = "changed bytes"
    asyncio.run(handlers.handle_capture(first, first.text))
    other = make_message("changed bytes", event=11)
    asyncio.run(handlers.handle_capture(other, other.text))
    third = make_message("changed bytes", event=11)
    third.from_user.id = 43
    asyncio.run(handlers.handle_capture(third, third.text))
    assert len(source_records(handlers)) == 4
    assert len(list((tmp_path / "00-inbox").glob("*.md"))) == 4


def test_index_failure_is_preserved_not_reported_as_indexed(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    handlers._ctx.store.index_single_note.side_effect = RuntimeError("secret-body")
    message = make_message("keep original")
    asyncio.run(handlers.handle_capture(message, message.text))
    answer = message.answer.call_args.args[0]
    assert "Preserved source" in answer and "index: failed" in answer
    assert "secret-body" not in answer
    asyncio.run(handlers.handle_capture(message, message.text))
    handlers._ctx.store.index_single_note.assert_called_once()
    assert source_records(handlers)[0].text == "keep original"


def test_auth_and_reserved_destination_precede_source_creation(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    message = make_message("private")
    handlers._ctx.settings.telegram.allowed_user_ids = [999]
    asyncio.run(handlers.handle_capture(message, message.text))
    assert not (tmp_path / "_wiki").exists()
    handlers._ctx.settings.telegram.allowed_user_ids = []
    handlers._ctx.settings.vault.inbox_folder = "wiki"
    asyncio.run(handlers.handle_capture(message, message.text))
    assert not (tmp_path / "_wiki").exists()


def test_url_original_and_extraction_are_related_not_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vaultmind.vault.ingest import IngestResult, SourceType

    handlers = make_handlers(tmp_path)
    handlers._ctx.settings.ingest.enabled = True
    fetch = AsyncMock(
        return_value=IngestResult(
            SourceType.ARTICLE, "Article", "extracted article", "https://example.com"
        )
    )
    monkeypatch.setattr("vaultmind.bot.handlers.capture.ingest_url", fetch)
    message = make_message("save: https://example.com\r\n")
    asyncio.run(handlers.handle_message(message, message.text))
    records = source_records(handlers)
    assert len(records) == 2
    original = next(item for item in records if item.provenance["role"] == "text")
    derived = next(item for item in records if item.provenance["role"] == "extracted-text")
    assert original.text == message.text
    assert derived.text == "extracted article"
    assert derived.dependencies[0].source_id == original.source_id
    asyncio.run(handlers.handle_message(message, message.text))
    fetch.assert_awaited_once()
    assert len(source_records(handlers)) == 2
    assert handlers._ctx.store.index_single_note.call_count == 2


def test_url_failure_reports_fetch_failure_with_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    handlers._ctx.settings.ingest.enabled = True
    fetch = AsyncMock(side_effect=RuntimeError("private URL body"))
    monkeypatch.setattr("vaultmind.bot.handlers.capture.ingest_url", fetch)
    message = make_message("https://example.com")
    asyncio.run(handlers.handle_capture(message, message.text))
    answer = message.answer.call_args.args[0]
    assert "URL fetch: failed" in answer and "original preserved" in answer
    assert "private URL body" not in answer
    assert len(source_records(handlers)) == 1
    asyncio.run(handlers.handle_capture(message, message.text))
    fetch.assert_awaited_once()


def test_registered_text_question_does_not_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    response = AsyncMock()
    monkeypatch.setattr("vaultmind.bot.handlers.routing.handle_smart_response", response)
    message = make_message("Why does the system work this way?")
    callback = registered_callback(handlers, "handle_text", monkeypatch)
    asyncio.run(callback(message))
    response.assert_awaited_once()
    assert not (tmp_path / "_wiki").exists()
    assert not (tmp_path / "00-inbox").exists()


def test_extraction_limit_is_failed_derivative_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vaultmind.vault.ingest import IngestResult, SourceType

    handlers = make_handlers(tmp_path)
    handlers._ctx.settings.ingest.enabled = True
    handlers._ctx.settings.ingest.max_content_length = 3
    fetch = AsyncMock(
        return_value=IngestResult(SourceType.ARTICLE, "Article", "too long", "https://example.com")
    )
    monkeypatch.setattr("vaultmind.bot.handlers.capture.ingest_url", fetch)
    message = make_message("https://example.com")
    asyncio.run(handlers.handle_capture(message, message.text))
    assert len(source_records(handlers)) == 1
    answer = message.answer.call_args.args[0]
    assert "URL fetch: succeeded; extraction: failed" in answer
    assert "Original preserved" in answer
