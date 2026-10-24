"""Real shared-core thinking, bounded session intent and non-query finalization."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from vaultmind.bot.session_store import SessionStore
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.llm.client import LLMResponse
from vaultmind.vault.parser import VaultParser
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from tests.test_knowledge_transports import Transport


@pytest.mark.asyncio
async def test_restart_uses_three_complete_user_turns_only(transport: Transport) -> None:
    t = transport
    history = [
        {"user": f"cache policy question {i}", "assistant": "90-day policy"} for i in range(5)
    ]
    t.sessions.save(42, history)
    partner = ThinkingPartner(
        t.corpus.settings, llm_factory=lambda: t.provider, session_store=t.sessions
    )
    result = await partner.think(42, "What is the policy now?")
    prompt = json.loads(t.provider.calls[-1][0].content)
    assert prompt["intent_questions"] == [turn["user"] for turn in history[-3:]]
    assert "90-day" not in t.provider.calls[-1][0].content
    assert result["status"] == "supported"
    assert len(t.sessions.load(42)) == 6
    partner.clear_session(42)
    assert t.sessions.load(42) is None
    assert not partner.has_active_session(42)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prior,current",
    [
        (None, "x" * 4097),
        (None, "\x00cache"),
        (None, "   "),
        ("cache " + "x" * 4097, "What is the cache policy?"),
    ],
)
async def test_invalid_current_or_selected_prior_is_not_truncated_or_completed(
    transport: Transport,
    prior: str | None,
    current: str,
) -> None:
    t = transport
    if prior is not None:
        t.sessions.save(42, [{"user": prior, "assistant": "legacy"}])
    factory = MagicMock(side_effect=AssertionError("Provider must stay lazy"))
    partner = ThinkingPartner(t.corpus.settings, llm_factory=factory, session_store=t.sessions)
    before = t.sessions.load(42)
    with pytest.raises(WikiError):
        await partner.think(42, current)
    factory.assert_not_called()
    assert t.sessions.load(42) == before


@pytest.mark.asyncio
async def test_evidence_only_and_unsupported_are_lazy_and_normal_has_no_session(
    transport: Transport,
) -> None:
    t = transport
    factory = MagicMock(side_effect=AssertionError("No provider"))
    partner = ThinkingPartner(t.corpus.settings, llm_factory=factory)
    evidence = await partner.think(42, "cache policy", use_session=False, evidence_only=True)
    unsupported = await partner.think(42, "unrelated xenobiology", use_session=False)
    assert evidence["status"] == "evidence"
    assert unsupported["status"] == "unsupported"
    assert not partner.has_active_session(42)
    factory.assert_not_called()


def expire(sessions: SessionStore, user_id: int = 42) -> None:
    sessions._conn.execute(
        "UPDATE thinking_sessions SET last_active = ? WHERE user_id = ?",
        (time.time() - 7200, user_id),
    )
    sessions._conn.commit()


class DistillProvider:
    provider_name = "scripted-idle-distillation"

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[Any] = []
        self.fail = fail

    def complete(self, messages: Any, model: str, **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        return LLMResponse(
            text=json.dumps(
                dict(
                    question="What is the cache policy?",
                    summary="Discussed policy.",
                    resolution="Use approved policy.",
                    systems=[],
                    participants=[],
                )
            ),
            model=model,
            usage={},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_restart_idle_finalization_is_real_claimed_once_and_preserves_original(
    transport: Transport,
    fail: bool,
) -> None:
    t = transport
    t.corpus.settings.distill.enabled = True
    t.corpus.settings.distill.min_turns = 1
    history = [{"user": "cache question", "assistant": "legacy answer"}]
    t.sessions.save(42, history)
    t.sessions.save_summary(42, "default", 0, 0, 0, "Original summary", ["cache"], [])
    expire(t.sessions)
    db_path = t.corpus.vault.parent / "sessions.db"
    t.sessions.close()
    sessions = SessionStore(db_path)
    provider = DistillProvider(fail)
    partner = ThinkingPartner(
        t.corpus.settings, llm_factory=lambda: provider, session_store=sessions
    )
    assert not partner.has_active_session(42)
    store = MagicMock()
    counts = await partner.finalize_idle_sessions(
        store=store, parser=VaultParser(t.corpus.settings.vault)
    )
    assert counts == {"completed": int(not fail), "failed": int(fail), "skipped": 0}
    row = sessions._conn.execute(
        "SELECT history, summaries, state FROM thinking_session_archives"
    ).fetchone()
    assert json.loads(row[0]) == history
    assert "Original summary" in row[1]
    assert row[2] == ("failed" if fail else "completed")
    assert len(provider.calls) == 1
    assert store.index_single_note.call_count == int(not fail)
    if not fail:
        assert list((t.corpus.vault / t.corpus.settings.distill.output_folder).glob("*.md"))
    sessions.close()
    restarted = SessionStore(db_path)
    partner = ThinkingPartner(
        t.corpus.settings, llm_factory=lambda: provider, session_store=restarted
    )
    assert await partner.finalize_idle_sessions(store=store, parser=MagicMock()) == {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("first_fails", [False, True])
async def test_overlapping_queries_crossing_ttl_keep_one_generation_until_idle(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    persistent: bool,
    first_fails: bool,
) -> None:
    t = transport
    settings = t.corpus.settings
    settings.distill.enabled = True
    settings.distill.min_turns = 1
    clock = [time.time()]
    monkeypatch.setattr("vaultmind.bot.thinking.time.time", lambda: clock[0])
    selected: list[Any] = [t.provider]
    partner = ThinkingPartner(
        settings,
        llm_factory=lambda: selected[0],
        session_store=t.sessions if persistent else None,
    )
    original = [{"user": "Original cache question", "assistant": "Original complete answer"}]
    if persistent:
        t.sessions.save(42, original)
        t.sessions.save_summary(42, "default", 0, 0, 0, "Original summary", ["cache"], [])
    else:
        partner._get_session(42).history.extend(original)
    questions = ["First cache policy question", "Second cache policy question"]
    entered = {question: threading.Event() for question in questions}
    release = {question: threading.Event() for question in questions}
    complete = t.provider.complete

    def blocked(messages: Any, model: str, **kwargs: Any) -> LLMResponse:
        question = json.loads(messages[0].content)["question"]
        entered[question].set()
        if not release[question].wait(10):
            raise TimeoutError("Test did not release the question")
        if first_fails and question == questions[0]:
            raise RuntimeError("Synthetic query failure")
        return complete(messages, model, **kwargs)

    monkeypatch.setattr(t.provider, "complete", blocked)
    store = MagicMock()
    parser = VaultParser(settings.vault)
    first = asyncio.create_task(partner.think(42, questions[0]))
    second: asyncio.Task[dict[str, Any]] | None = None
    try:
        assert await asyncio.to_thread(entered[questions[0]].wait, 10)
        clock[0] += partner.session_ttl + 1
        first_tick = await partner.finalize_idle_sessions(store=store, parser=parser)
        active_across_ttl = partner.has_active_session(42)
        second = asyncio.create_task(partner.think(42, questions[1]))
        assert await asyncio.to_thread(entered[questions[1]].wait, 10)
        release[questions[0]].set()
        if first_fails:
            with pytest.raises(WikiError):
                await first
        else:
            await first
        after_first = t.sessions.load(42) if persistent else partner._get_session(42).history[:]
        clock[0] += partner.session_ttl + 1
        second_tick = await partner.finalize_idle_sessions(store=store, parser=parser)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(
            *([first, second] if second is not None else [first]), return_exceptions=True
        )
    assert first_tick == second_tick == {"completed": 0, "failed": 0, "skipped": 0}
    assert active_across_ttl
    assert after_first is not None
    assert after_first[:1] == original
    assert [turn["user"] for turn in after_first] == [
        original[0]["user"],
        *([] if first_fails else [questions[0]]),
    ]
    expected_questions = [
        original[0]["user"],
        *([] if first_fails else [questions[0]]),
        questions[1],
    ]
    history = t.sessions.load(42) if persistent else partner._get_session(42).history
    assert history is not None
    assert [turn["user"] for turn in history] == expected_questions
    assert history[:1] == original
    finalizer = DistillProvider()
    selected[0] = finalizer
    if persistent:
        db_path = t.corpus.vault.parent / "sessions.db"
        t.sessions.close()
        sessions = SessionStore(db_path)
        partner = ThinkingPartner(settings, llm_factory=lambda: finalizer, session_store=sessions)
    clock[0] += partner.session_ttl + 1
    assert await partner.finalize_idle_sessions(store=store, parser=parser) == {
        "completed": 1,
        "failed": 0,
        "skipped": 0,
    }
    if persistent:
        archived = sessions._conn.execute(
            "SELECT history, summaries, state FROM thinking_session_archives"
        ).fetchall()
        assert len(archived) == 1
        assert json.loads(archived[0][0]) == history
        assert "Original summary" in archived[0][1]
        assert archived[0][2] == "completed"
        sessions.close()
        sessions = SessionStore(db_path)
        partner = ThinkingPartner(settings, llm_factory=lambda: finalizer, session_store=sessions)
    assert await partner.finalize_idle_sessions(store=store, parser=parser) == {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert len(finalizer.calls) == 1
    for question in expected_questions:
        assert question in finalizer.calls[0][0].content
    if persistent:
        sessions.close()


@pytest.mark.asyncio
async def test_explicit_clear_during_query_cannot_resurrect_deleted_history(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = transport
    t.sessions.save(42, [{"user": "Original cache question", "assistant": "Original answer"}])
    entered = threading.Event()
    release = threading.Event()
    complete = t.provider.complete

    def blocked(messages: Any, model: str, **kwargs: Any) -> LLMResponse:
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Test did not release the question")
        return complete(messages, model, **kwargs)

    monkeypatch.setattr(t.provider, "complete", blocked)
    pending = asyncio.create_task(t.thinking.think(42, "Cache retention policy"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        t.thinking.clear_session(42)
    finally:
        release.set()
        result = await pending
    assert result["status"] == "supported"
    assert t.sessions.load(42) is None
    assert not t.thinking.has_active_session(42)
    await t.thinking.think(42, "New cache question")
    history = t.sessions.load(42)
    assert history is not None
    assert [turn["user"] for turn in history] == ["New cache question"]
    assert json.loads(t.provider.calls[-1][0].content)["intent_questions"] == []


@pytest.mark.asyncio
async def test_query_preserves_expired_record_without_archival_dispatch_or_summary(
    transport: Transport,
) -> None:
    t = transport
    t.corpus.settings.distill.enabled = True
    original = [{"user": "old cache", "assistant": "old answer"}]
    t.sessions.save(42, original)
    expire(t.sessions)
    result = await t.thinking.think(42, "cache policy")
    assert result["status"] == "supported"
    assert len(t.provider.calls) == 1
    assert (
        t.sessions._conn.execute("SELECT COUNT(*) FROM thinking_session_archives").fetchone()[0]
        == 0
    )
    assert t.sessions.load(42) == original
    active = t.sessions.load_active(42, t.thinking.session_ttl)
    assert active["session_name"].startswith("thinking:")
    assert len(active["history"]) == 1
    assert json.loads(t.provider.calls[0][0].content)["intent_questions"] == []
    restarted = ThinkingPartner(
        t.corpus.settings, llm_factory=lambda: t.provider, session_store=t.sessions
    )
    assert restarted.has_active_session(42)
    assert restarted._get_session(42).history == active["history"]
    assert not (t.corpus.vault / t.corpus.settings.distill.output_folder).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [False, True])
async def test_explicit_bookmark_and_manual_distill_keep_complete_exchange(
    transport: Transport,
    session: bool,
) -> None:
    from tests.test_capture import make_message

    t = transport
    query = "What is the cache retention policy?"
    if session:
        await t.handlers.handle_think(make_message(), query)
    else:
        await t.handlers.handle_smart_response(make_message(), query, is_question=True)
    response = t.handlers._last_exchanges[42].response
    await t.handlers.handle_bookmark(make_message(), "Policy exchange")
    bookmarks = list((t.corpus.vault / "00-inbox").glob("*bookmark*.md"))
    assert len(bookmarks) == 1
    assert response in bookmarks[0].read_text()
    assert query in bookmarks[0].read_text()
    producer = DistillProvider()
    t.handlers._ctx.llm_client = producer
    await t.handlers.handle_distill(make_message())
    assert len(producer.calls) == 1
    assert response in producer.calls[0][0].content
    assert query in producer.calls[0][0].content
    assert list((t.corpus.vault / t.corpus.settings.distill.output_folder).glob("*.md"))
    assert t.handlers._ctx.store.index_single_note.call_count == 2


@pytest.mark.asyncio
async def test_nonpersistent_expired_history_is_retained_until_nonquery_finalizer(
    transport: Transport,
) -> None:
    t = transport
    t.corpus.settings.distill.enabled = True
    t.corpus.settings.distill.min_turns = 1
    selected: list[Any] = [t.provider]
    partner = ThinkingPartner(t.corpus.settings, llm_factory=lambda: selected[0])
    await partner.think(42, "cache retention")
    original = partner._get_session(42)
    original.last_active = time.time() - 7200
    await partner.think(42, "cache retention now")
    assert original.history[0]["user"] == "cache retention"
    assert len(original.history) == 1
    assert original.finalization_state == "active"
    assert len(t.provider.calls) == 2
    selected[0] = DistillProvider()
    counts = await partner.finalize_idle_sessions(
        store=MagicMock(), parser=VaultParser(t.corpus.settings.vault)
    )
    assert counts == {"completed": 1, "failed": 0, "skipped": 0}
    assert original.finalization_state == "completed"
    assert len(selected[0].calls) == 1
    assert partner.has_active_session(42)


@pytest.mark.asyncio
async def test_disabled_or_too_short_idle_producer_never_constructs_provider(
    transport: Transport,
) -> None:
    t = transport
    t.sessions.save(42, [{"user": "cache", "assistant": "answer"}])
    expire(t.sessions)
    factory = MagicMock(side_effect=AssertionError("Provider must stay lazy"))
    partner = ThinkingPartner(t.corpus.settings, llm_factory=factory, session_store=t.sessions)
    assert await partner.finalize_idle_sessions(store=MagicMock(), parser=MagicMock()) == {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert t.sessions.load(42) is not None
    t.corpus.settings.distill.enabled = True
    t.corpus.settings.distill.min_turns = 2
    assert await partner.finalize_idle_sessions(store=MagicMock(), parser=MagicMock()) == {
        "completed": 0,
        "failed": 0,
        "skipped": 1,
    }
    factory.assert_not_called()
