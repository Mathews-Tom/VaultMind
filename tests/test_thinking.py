"""Tests for ThinkingPartner — session persistence integration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.bot.session_store import SessionStore
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.config import VaultConfig
from vaultmind.llm.client import LLMResponse, Message
from vaultmind.memory.extractor import _SYSTEM_PROMPT as _EXTRACT_SYSTEM_PROMPT
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.distill import DISTILL_SYSTEM_PROMPT
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _FakeLLMConfig:
    thinking_model: str = "test-model"
    fast_model: str = "test-model"
    max_context_notes: int = 3
    max_tokens: int = 100
    single_pass_extraction_enabled: bool = False
    extraction_confidence_threshold: float = 0.7


@dataclass
class _FakeTelegramConfig:
    thinking_session_ttl: int = 3600
    thinking_summarization_enabled: bool = False
    thinking_message_count_threshold: int = 20
    thinking_recent_turns_to_keep: int = 6
    thinking_batch_size: int = 4
    thinking_summary_max_tokens: int = 400


@dataclass
class _FakeDistillConfig:
    enabled: bool = True
    min_turns: int = 1
    model: str = ""
    output_folder: str = "qa-artifacts"
    max_tokens: int = 800


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
    vault_root: Path | None = None,
    parser: VaultParser | None = None,
    distill_config: _FakeDistillConfig | None = None,
) -> ThinkingPartner:
    return ThinkingPartner(
        llm_config=_FakeLLMConfig(),  # type: ignore[arg-type]
        telegram_config=_FakeTelegramConfig(thinking_session_ttl=ttl),  # type: ignore[arg-type]
        llm_client=llm_client,
        session_store=store,
        vault_root=vault_root,
        parser=parser,
        distill_config=distill_config,  # type: ignore[arg-type]
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
        messages: list[Message] = call_args.kwargs.get(
            "messages", call_args.args[0] if call_args.args else []
        )
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


def _distill_side_effect(**kwargs: object) -> LLMResponse:
    system = kwargs.get("system", "")
    if system == DISTILL_SYSTEM_PROMPT:
        return LLMResponse(
            text=json.dumps(
                {
                    "question": "What storage should we use?",
                    "summary": "Discussed storage options and tradeoffs.",
                    "resolution": "Use ChromaDB now, migrate to pgvector past 1M chunks.",
                    "systems": ["ChromaDB"],
                    "participants": ["user"],
                }
            ),
            model="test-model",
            usage={},
        )
    if system == _EXTRACT_SYSTEM_PROMPT:
        return LLMResponse(
            text=json.dumps(
                [
                    {
                        "decision": "Use ChromaDB",
                        "context": "vector storage choice",
                        "outcome": "Adopted",
                        "outcome_status": "success",
                        "lessons": [],
                        "entities": ["ChromaDB"],
                    }
                ]
            ),
            model="test-model",
            usage={},
        )
    return LLMResponse(text="test reply", model="test-model", usage={})


@pytest.fixture
def distill_llm_client() -> MagicMock:
    client = MagicMock()
    client.complete.side_effect = _distill_side_effect
    return client


@pytest.fixture
def captured_tasks(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.Task[object]]:
    """Capture every `asyncio.ensure_future` call made from `bot.thinking` so
    fire-and-forget distillation tasks can be awaited to completion at the end
    of a test. Scoped to that module's `asyncio` reference only — patching the
    global `asyncio.ensure_future` would also intercept pytest-asyncio's own
    task scheduling for the test coroutine itself.
    """
    import vaultmind.bot.thinking as thinking_module

    tasks: list[asyncio.Task[object]] = []

    class _AsyncioProxy:
        def ensure_future(self, coro_or_future: object, **kwargs: object) -> asyncio.Task[object]:
            task = asyncio.ensure_future(coro_or_future, **kwargs)  # type: ignore[arg-type]
            tasks.append(task)
            return task

        def __getattr__(self, name: str) -> object:
            return getattr(asyncio, name)

    monkeypatch.setattr(thinking_module, "asyncio", _AsyncioProxy())
    return tasks


class TestDistillationTrigger:
    @pytest.mark.asyncio
    async def test_disabled_by_default_produces_no_pending_dispatch(
        self,
        distill_llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
        captured_tasks: list[asyncio.Task[object]],
    ) -> None:
        partner = _make_partner(distill_llm_client, ttl=0)
        await partner.think(1, "topic", vault_store, graph)
        await partner.think(2, "topic2", vault_store, graph)
        assert captured_tasks == []
        assert partner._pending_distill == []

    @pytest.mark.asyncio
    async def test_below_min_turns_not_distilled(
        self,
        distill_llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
        captured_tasks: list[asyncio.Task[object]],
        tmp_path: Path,
    ) -> None:
        config = _FakeDistillConfig(enabled=True, min_turns=5)
        parser = VaultParser(VaultConfig(path=tmp_path))
        partner = _make_partner(
            distill_llm_client, ttl=0, vault_root=tmp_path, parser=parser, distill_config=config
        )
        await partner.think(1, "topic", vault_store, graph)
        await partner.think(2, "topic2", vault_store, graph)
        assert captured_tasks == []

    @pytest.mark.asyncio
    async def test_idle_timeout_distills_and_indexes_note(
        self,
        distill_llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
        captured_tasks: list[asyncio.Task[object]],
        tmp_path: Path,
    ) -> None:
        config = _FakeDistillConfig(enabled=True, min_turns=1)
        parser = VaultParser(VaultConfig(path=tmp_path))
        partner = _make_partner(
            distill_llm_client, ttl=0, vault_root=tmp_path, parser=parser, distill_config=config
        )

        await partner.think(1, "topic", vault_store, graph, episode_store=None)
        await partner.think(2, "topic2", vault_store, graph, episode_store=None)

        assert captured_tasks
        await asyncio.gather(*captured_tasks)

        qa_dir = tmp_path / "qa-artifacts"
        assert qa_dir.exists()
        notes = list(qa_dir.glob("*.md"))
        assert len(notes) == 1
        content = notes[0].read_text()
        assert "type: qa-artifact" in content
        vault_store.index_single_note.assert_called_once()

    @pytest.mark.asyncio
    async def test_episode_extraction_runs_over_distilled_note(
        self,
        distill_llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
        captured_tasks: list[asyncio.Task[object]],
        tmp_path: Path,
    ) -> None:
        config = _FakeDistillConfig(enabled=True, min_turns=1)
        parser = VaultParser(VaultConfig(path=tmp_path))
        partner = _make_partner(
            distill_llm_client, ttl=0, vault_root=tmp_path, parser=parser, distill_config=config
        )
        episode_store = EpisodeStore(tmp_path / "episodes.db")

        await partner.think(1, "topic", vault_store, graph, episode_store=episode_store)
        await partner.think(2, "topic2", vault_store, graph, episode_store=episode_store)
        await asyncio.gather(*captured_tasks)

        resolved = episode_store.query_resolved()
        assert len(resolved) == 1
        assert resolved[0].decision == "Use ChromaDB"
        episode_store.close()

    @pytest.mark.asyncio
    async def test_missing_vault_root_skips_without_raising(
        self,
        distill_llm_client: MagicMock,
        vault_store: MagicMock,
        graph: MagicMock,
        captured_tasks: list[asyncio.Task[object]],
    ) -> None:
        config = _FakeDistillConfig(enabled=True, min_turns=1)
        partner = _make_partner(distill_llm_client, ttl=0, distill_config=config)

        await partner.think(1, "topic", vault_store, graph)
        await partner.think(2, "topic2", vault_store, graph)
        await asyncio.gather(*captured_tasks)

        vault_store.index_single_note.assert_not_called()
