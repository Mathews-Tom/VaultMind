"""Tests for ThinkingPartner — session persistence integration."""

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


@dataclass
class _FakeTelegramConfig:
    thinking_session_ttl: int = 3600


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


@pytest.fixture
def llm_client() -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(text="test reply", model="test-model", usage={})
    return client


@pytest.fixture
def vault_store() -> MagicMock:
    s = MagicMock()
    s.search.return_value = []
    return s


@pytest.fixture
def graph() -> MagicMock:
    g = MagicMock()
    g.get_neighbors.return_value = {"entity": None, "outgoing": [], "incoming": [], "neighbors": []}
    return g


def _make_partner(
    llm_client: MagicMock,
    store: SessionStore | None = None,
    ttl: int = 3600,
) -> ThinkingPartner:
    return ThinkingPartner(
        llm_config=_FakeLLMConfig(),  # type: ignore[arg-type]
        telegram_config=_FakeTelegramConfig(thinking_session_ttl=ttl),  # type: ignore[arg-type]
        llm_client=llm_client,
        session_store=store,
    )


class TestPersistenceAfterThink:
    @pytest.mark.asyncio
    async def test_session_persisted_to_store(
        self,
        llm_client: MagicMock,
        store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, store)
        await partner.think(1, "test topic", vault_store, graph)
        loaded = store.load(1)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0]["assistant"] == "test reply"

    @pytest.mark.asyncio
    async def test_no_store_still_works(
        self,
        llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, store=None)
        result = await partner.think(1, "test topic", vault_store, graph)
        assert result == "test reply"


class TestLoadFromStoreOnCacheMiss:
    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_sqlite(
        self,
        llm_client: MagicMock,
        store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        # Pre-populate SQLite with history
        store.save(1, [{"user": "prev", "assistant": "prev reply"}])

        partner = _make_partner(llm_client, store)
        # Cache is empty, should load from SQLite
        await partner.think(1, "follow-up", vault_store, graph)

        loaded = store.load(1)
        assert loaded is not None
        assert len(loaded) == 2  # original + new turn

        # Verify LLM received prior history in messages
        call_args = llm_client.complete.call_args
        messages: list[Message] = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        # First two messages should be the prior turn
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].content == "prev reply"


class TestClearSession:
    @pytest.mark.asyncio
    async def test_clear_removes_from_cache_and_store(
        self,
        llm_client: MagicMock,
        store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, store)
        await partner.think(1, "topic", vault_store, graph)
        partner.clear_session(1)
        assert store.load(1) is None
        assert 1 not in partner._sessions


class TestHasActiveSession:
    @pytest.mark.asyncio
    async def test_true_when_in_cache(
        self,
        llm_client: MagicMock,
        store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, store)
        await partner.think(1, "topic", vault_store, graph)
        assert partner.has_active_session(1) is True

    def test_true_when_only_in_store(
        self,
        llm_client: MagicMock,
        store: SessionStore,
    ) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}])
        partner = _make_partner(llm_client, store)
        assert partner.has_active_session(1) is True

    def test_false_when_nowhere(
        self,
        llm_client: MagicMock,
        store: SessionStore,
    ) -> None:
        partner = _make_partner(llm_client, store)
        assert partner.has_active_session(999) is False

    def test_false_without_store(self, llm_client: MagicMock) -> None:
        partner = _make_partner(llm_client, store=None)
        assert partner.has_active_session(999) is False
