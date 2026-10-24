"""Registered Telegram routes consume the real approved-current query core."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_capture import make_message, registered_callback
from tests.test_knowledge_query import CACHE, Corpus, ScriptedProvider, cache_body, snapshot
from vaultmind.bot.commands import CommandHandlers
from vaultmind.bot.session_store import SessionStore
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.knowledge.query import query_knowledge
from vaultmind.knowledge.render import render_knowledge_result, split_knowledge_text
from vaultmind.vault.parser import VaultParser


@dataclass
class Transport:
    corpus: Corpus
    provider: ScriptedProvider
    sessions: SessionStore
    thinking: ThinkingPartner
    handlers: CommandHandlers


@pytest.fixture
def transport(corpus: Corpus) -> Transport:
    corpus.publish(CACHE, cache_body())
    approved = query_knowledge(corpus.settings, "cache retention", evidence_only=True)
    provider = ScriptedProvider(
        dict(
            status="supported",
            answer="The policy is 30 days.",
            citations=[row["id"] for row in approved["evidence"] if "30 days" in row["text"]],
        )
    )
    sessions = SessionStore(corpus.vault.parent / "sessions.db")
    thinking = ThinkingPartner(
        corpus.settings, llm_factory=lambda: provider, session_store=sessions
    )
    handlers = CommandHandlers(
        corpus.settings,
        MagicMock(),
        MagicMock(),
        VaultParser(corpus.settings.vault),
        thinking,
        provider,
    )
    return Transport(corpus, provider, sessions, thinking, handlers)


def delivered(message: Any) -> str:
    return "".join(
        call.args[0] for call in message.answer.call_args_list if call.args[0] != "🧠 Thinking..."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,text",
    [
        ("handle_text", "What is the cache retention policy?"),
        ("handle_text", "Explain the cache retention policy"),
        ("cmd_think", "/think critique: What is the cache retention policy?"),
        ("handle_text", "What is the cache\nretention policy?"),
        ("cmd_think", "/think What is the cache\nretention policy?"),
    ],
)
async def test_registered_answers_share_actual_outcome_and_do_not_mutate_knowledge(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    text: str,
) -> None:
    t = transport
    before = snapshot(t.corpus.vault)
    message = make_message(text)
    callback = registered_callback(t.handlers, route, monkeypatch)
    await callback(message)
    if "\n" in text:
        selected_question = text.removeprefix("/think ")
        assert json.loads(t.provider.calls[0][0].content)["question"] == selected_question
    result = query_knowledge(
        t.corpus.settings, "What is the cache retention policy?", llm_factory=lambda: t.provider
    )
    assert delivered(message) == render_knowledge_result(result)
    assert snapshot(t.corpus.vault) == before
    assert all(call.kwargs.get("parse_mode") is None for call in message.answer.call_args_list)
    t.handlers._ctx.store.search.assert_not_called()
    t.handlers._ctx.store.index_single_note.assert_not_called()
    t.handlers._ctx.graph.assert_not_called()
    assert t.thinking.has_active_session(42) is (route == "cmd_think")
    assert "section_mapping" in t.handlers._last_exchanges[42].response


@pytest.mark.asyncio
async def test_sticky_followup_reselects_updated_approved_revision(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = transport
    message = make_message("/think What is the cache retention policy?")
    await registered_callback(t.handlers, "cmd_think", monkeypatch)(message)
    first = json.loads(t.provider.calls[-1][0].content)
    t.corpus.publish(CACHE, cache_body().replace("30 days", "7 days"))
    t.provider.response["answer"] = "The current policy is 7 days."
    followup_text = (
        "What about now?\nCompare the retained policies.\nExplain the current revision.\n"
        "Keep the unresolved conflict explicit."
    )
    followup = make_message(followup_text, event=11)
    await registered_callback(t.handlers, "handle_text", monkeypatch)(followup)
    second = json.loads(t.provider.calls[-1][0].content)
    assert second["question"] == followup_text
    assert second["intent_questions"] == ["What is the cache retention policy?"]
    assert second["evidence"][0]["page_sha256"] != first["evidence"][0]["page_sha256"]
    assert any("7 days" in row["text"] for row in second["evidence"])
    assert "current policy is 7 days" in delivered(followup)
    assert len(t.sessions.load(42)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\n", "\r\n", "\t"])
async def test_legacy_multiline_user_turn_keeps_followups_usable_without_answer_authority(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    separator: str,
) -> None:
    t = transport
    prior = f"What is the cache{separator}retention policy?"
    t.sessions.save(42, [{"user": prior, "assistant": "The legacy answer was 90 days."}])
    message = make_message("What about now?", event=12)
    await registered_callback(t.handlers, "handle_text", monkeypatch)(message)
    assert "The policy is 30 days." in delivered(message)
    prompt = json.loads(t.provider.calls[-1][0].content)
    assert prompt["intent_questions"] == [prior]
    assert "90 days" not in json.dumps(prompt)
    history = t.sessions.load(42)
    assert history is not None
    assert [turn["user"] for turn in history] == [prior, "What about now?"]


@pytest.mark.asyncio
async def test_long_rendering_keeps_every_citation_and_qualification(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = transport
    t.provider.response["answer"] = "Literal _[*] text\n  " * 700 + "FINAL ANSWER"
    message = make_message("What is the cache retention policy?")
    await registered_callback(t.handlers, "handle_text", monkeypatch)(message)
    result = query_knowledge(t.corpus.settings, message.text, llm_factory=lambda: t.provider)
    expected = render_knowledge_result(result)
    assert delivered(message) == expected
    assert len(message.answer.call_args_list) > 2
    assert all(len(call.args[0]) <= 4000 for call in message.answer.call_args_list)
    assert "FINAL ANSWER" in delivered(message)
    assert "limitation" in delivered(message)
    assert "page_sha256" in delivered(message)
    assert "".join(split_knowledge_text(expected)) == expected


@pytest.mark.asyncio
async def test_real_core_runs_off_event_loop(transport: Transport) -> None:
    t = transport
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    original = t.provider.complete

    def complete(*args: Any, **kwargs: Any) -> Any:
        assert threading.get_ident() != owner
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    t.provider.complete = complete
    task = asyncio.create_task(t.thinking.think(42, "cache retention", use_session=False))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    assert (await task)["status"] == "supported"


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [False, True])
async def test_voice_question_registered_route_has_no_capture(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    oversized: bool,
) -> None:
    t = transport
    from vaultmind.bot.handlers import voice

    voice._voice_events.clear()
    transcriber = MagicMock()
    transcriber.transcribe.return_value = (
        "What is the cache retention policy?" if not oversized else "cache " + "x" * 4097 + "?"
    )
    t.handlers._ctx.transcriber = transcriber
    message = make_message(event=123)
    message.voice.file_size = 20
    message.bot.get_file = AsyncMock(return_value=type("File", (), {"file_path": "voice.ogg"})())

    async def download(path: str, destination: Any) -> None:
        destination.write(b"synthetic audio")

    message.bot.download_file = download
    before = snapshot(t.corpus.vault)
    await registered_callback(t.handlers, "handle_voice", monkeypatch)(message)
    assert ("Knowledge query refused" if oversized else "Status: supported") in delivered(message)
    assert snapshot(t.corpus.vault) == before
    assert not t.thinking.has_active_session(42)
    assert len(t.provider.calls) == int(not oversized)
