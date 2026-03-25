"""Tests for session summarization — store methods, thinking integration, and parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.bot.session_store import SessionStore
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.llm.client import LLMResponse, Message

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _FakeLLMConfig:
    thinking_model: str = "test-model"
    fast_model: str = "test-model"
    max_context_notes: int = 3
    max_tokens: int = 100
    graph_context_enabled: bool = False
    graph_hop_depth: int = 2
    graph_min_confidence: float = 0.6
    graph_max_relationships: int = 20
    single_pass_extraction_enabled: bool = False
    extraction_confidence_threshold: float = 0.7


@dataclass
class _FakeTelegramConfig:
    thinking_session_ttl: int = 3600
    thinking_summarization_enabled: bool = True
    thinking_message_count_threshold: int = 8
    thinking_recent_turns_to_keep: int = 2
    thinking_batch_size: int = 2
    thinking_summary_max_tokens: int = 400


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


@pytest.fixture()
def llm_client() -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(text="test reply", model="test-model", usage={})
    return client


@pytest.fixture()
def vault_store() -> MagicMock:
    s = MagicMock()
    s.search.return_value = []
    return s


@pytest.fixture()
def graph() -> MagicMock:
    g = MagicMock()
    g.get_neighbors.return_value = {
        "entity": None,
        "outgoing": [],
        "incoming": [],
        "neighbors": [],
    }
    return g


def _make_partner(
    llm_client: MagicMock,
    store: SessionStore | None = None,
    config: _FakeTelegramConfig | None = None,
) -> ThinkingPartner:
    return ThinkingPartner(
        llm_config=_FakeLLMConfig(),  # type: ignore[arg-type]
        telegram_config=config or _FakeTelegramConfig(),  # type: ignore[arg-type]
        llm_client=llm_client,
        session_store=store,
    )


# --- SessionStore summary methods ---


class TestSaveSummary:
    def test_save_and_retrieve_summary(self, store: SessionStore) -> None:
        store.save_summary(
            user_id=1,
            session_name="default",
            batch_number=0,
            start_turn_index=0,
            end_turn_index=3,
            summary_text="Discussed AI alignment.",
            key_topics=["AI", "alignment"],
            open_questions=["What are the key risks?"],
        )
        summaries = store.get_summaries(1, "default")
        assert len(summaries) == 1
        assert summaries[0]["batch_number"] == 0
        assert summaries[0]["summary"] == "Discussed AI alignment."
        assert summaries[0]["topics"] == ["AI", "alignment"]
        assert summaries[0]["questions"] == ["What are the key risks?"]

    def test_multiple_summaries_ordered(self, store: SessionStore) -> None:
        for i in range(3):
            store.save_summary(
                user_id=1,
                session_name="default",
                batch_number=i,
                start_turn_index=i * 4,
                end_turn_index=(i + 1) * 4 - 1,
                summary_text=f"Batch {i}",
                key_topics=[],
                open_questions=[],
            )
        summaries = store.get_summaries(1, "default")
        assert len(summaries) == 3
        assert summaries[0]["batch_number"] == 0
        assert summaries[2]["batch_number"] == 2

    def test_summaries_isolated_by_session_name(self, store: SessionStore) -> None:
        store.save_summary(1, "alpha", 0, 0, 1, "Alpha summary", [], [])
        store.save_summary(1, "beta", 0, 0, 1, "Beta summary", [], [])
        assert len(store.get_summaries(1, "alpha")) == 1
        assert len(store.get_summaries(1, "beta")) == 1
        assert store.get_summaries(1, "alpha")[0]["summary"] == "Alpha summary"

    def test_no_summaries_returns_empty_list(self, store: SessionStore) -> None:
        assert store.get_summaries(999) == []


class TestCountTurns:
    def test_count_turns_with_history(self, store: SessionStore) -> None:
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]
        store.save(1, history)
        assert store.count_turns(1) == 5

    def test_count_turns_no_session_returns_zero(self, store: SessionStore) -> None:
        assert store.count_turns(999) == 0


class TestGetUnsummarizedBatch:
    def test_returns_first_batch_when_no_summaries(self, store: SessionStore) -> None:
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(10)]
        store.save(1, history)

        batch = store.get_unsummarized_batch(
            user_id=1,
            session_name="default",
            recent_turns_to_keep=4,
            batch_size=2,
        )
        assert batch is not None
        turns, start, end = batch
        assert len(turns) == 2
        assert start == 0
        assert end == 1

    def test_returns_next_batch_after_existing_summary(self, store: SessionStore) -> None:
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(10)]
        store.save(1, history)
        store.save_summary(1, "default", 0, 0, 1, "First batch", [], [])

        batch = store.get_unsummarized_batch(1, "default", 4, 2)
        assert batch is not None
        _, start, end = batch
        assert start == 2
        assert end == 3

    def test_returns_none_when_too_few_turns(self, store: SessionStore) -> None:
        history = [{"user": "q", "assistant": "a"}]
        store.save(1, history)
        assert store.get_unsummarized_batch(1, "default", 4, 2) is None

    def test_returns_none_when_batch_would_eat_into_recent(self, store: SessionStore) -> None:
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(6)]
        store.save(1, history)
        # Keep 4 recent, batch of 2 → only turns 0-1 eligible
        store.save_summary(1, "default", 0, 0, 1, "Summarized", [], [])
        # Next batch would be 2-3, but 6-4=2, so next_end(3) >= 2 → None
        assert store.get_unsummarized_batch(1, "default", 4, 2) is None

    def test_returns_none_when_no_session(self, store: SessionStore) -> None:
        assert store.get_unsummarized_batch(999, "default", 4, 2) is None


class TestDeleteClearsSessionSummaries:
    def test_delete_removes_summaries(self, store: SessionStore) -> None:
        store.save(1, [{"user": "q", "assistant": "a"}])
        store.save_summary(1, "default", 0, 0, 0, "Summary", ["t"], ["q"])
        store.delete(1)
        assert store.get_summaries(1) == []


# --- ThinkingPartner summarization ---


class TestParseSummaryResponse:
    def test_parse_valid_response(self) -> None:
        text = (
            "SUMMARY: Discussed architecture and performance.\n"
            "TOPICS: architecture, performance, scaling\n"
            "QUESTIONS: What about caching?"
        )
        result = ThinkingPartner._parse_summary_response(text)
        assert result["summary"] == "Discussed architecture and performance."
        assert result["topics"] == ["architecture", "performance", "scaling"]
        assert result["questions"] == ["What about caching?"]

    def test_parse_questions_none(self) -> None:
        text = "SUMMARY: All resolved.\nTOPICS: topic1\nQUESTIONS: none"
        result = ThinkingPartner._parse_summary_response(text)
        assert result["questions"] == []

    def test_parse_empty_response_returns_defaults(self) -> None:
        result = ThinkingPartner._parse_summary_response("")
        assert result["summary"] == "No summary generated."
        assert result["topics"] == []
        assert result["questions"] == []

    def test_parse_case_insensitive(self) -> None:
        text = "summary: Lower case.\ntopics: t1, t2\nquestions: q1"
        result = ThinkingPartner._parse_summary_response(text)
        assert result["summary"] == "Lower case."
        assert result["topics"] == ["t1", "t2"]

    def test_parse_extra_whitespace_stripped(self) -> None:
        text = "  SUMMARY:  Spaced out.  \n  TOPICS:  a , b  \n  QUESTIONS: none  "
        result = ThinkingPartner._parse_summary_response(text)
        assert result["summary"] == "Spaced out."
        assert result["topics"] == ["a", "b"]


class TestBuildMessagesWithSummaries:
    def test_messages_include_summaries_before_recent_turns(
        self,
        llm_client: MagicMock,
        store: SessionStore,
    ) -> None:
        # Pre-populate session with 8 turns
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(8)]
        store.save(1, history)

        # Save a summary for the first 2 turns
        store.save_summary(1, "default", 0, 0, 1, "Early discussion", ["t1"], ["q1"])

        partner = _make_partner(llm_client, store)
        session = partner._get_session(1)

        messages = partner._build_messages(
            session, "new topic", "vault context", "explore", user_id=1
        )

        # First message should be summary (assistant)
        assert messages[0].role == "assistant"
        assert "Summary of earlier conversation" in messages[0].content
        assert "Early discussion" in messages[0].content

        # Last message should be the current turn (user)
        assert messages[-1].role == "user"
        assert "new topic" in messages[-1].content

    def test_messages_without_summaries_include_recent_only(
        self,
        llm_client: MagicMock,
        store: SessionStore,
    ) -> None:
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(4)]
        store.save(1, history)

        partner = _make_partner(llm_client, store)
        session = partner._get_session(1)

        # recent_turns_to_keep = 2, so only last 2 turns + current
        messages = partner._build_messages(session, "topic", "context", "explore", user_id=1)

        # 2 recent turns (4 messages) + 1 current user message = 5
        assert len(messages) == 5
        assert messages[0].role == "user"
        assert messages[0].content == "q2"


class TestSummarizeIfNeeded:
    @pytest.mark.asyncio()
    async def test_summarization_triggered_when_threshold_exceeded(
        self,
        store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        # Set up LLM to return different things for thinking vs summarization
        llm_client = MagicMock()
        call_count = 0

        def fake_complete(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            messages = kwargs.get("messages", [])
            # Summarization prompt contains "Summarize this"
            if messages and isinstance(messages, list):
                first_msg = messages[0]
                if isinstance(first_msg, Message) and "Summarize" in first_msg.content:
                    return LLMResponse(
                        text="SUMMARY: Test summary.\nTOPICS: t1\nQUESTIONS: none",
                        model="test-model",
                        usage={},
                    )
            return LLMResponse(text="test reply", model="test-model", usage={})

        llm_client.complete.side_effect = fake_complete

        # Config: threshold=8 (so 4 turns triggers summarization check),
        # recent_to_keep=2, batch_size=2
        config = _FakeTelegramConfig(
            thinking_message_count_threshold=8,
            thinking_recent_turns_to_keep=2,
            thinking_batch_size=2,
        )
        partner = _make_partner(llm_client, store, config)

        # Add 5 turns (exceeds threshold//2=4)
        for i in range(5):
            await partner.think(1, f"topic {i}", vault_store, graph)

        # Give async summarization time to complete
        import asyncio

        await asyncio.sleep(0.2)

        summaries = store.get_summaries(1)
        assert len(summaries) >= 1
        assert summaries[0]["summary"] == "Test summary."

    @pytest.mark.asyncio()
    async def test_no_summarization_when_disabled(
        self,
        store: SessionStore,
        llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        config = _FakeTelegramConfig(thinking_summarization_enabled=False)
        partner = _make_partner(llm_client, store, config)

        for i in range(6):
            await partner.think(1, f"topic {i}", vault_store, graph)

        import asyncio

        await asyncio.sleep(0.1)

        assert store.get_summaries(1) == []

    @pytest.mark.asyncio()
    async def test_no_summarization_below_threshold(
        self,
        store: SessionStore,
        llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        config = _FakeTelegramConfig(thinking_message_count_threshold=100)
        partner = _make_partner(llm_client, store, config)

        await partner.think(1, "single topic", vault_store, graph)

        import asyncio

        await asyncio.sleep(0.1)

        assert store.get_summaries(1) == []
