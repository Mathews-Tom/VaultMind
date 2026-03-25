"""Tests for episodic memory integration in thinking partner context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.bot.session_store import SessionStore
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.llm.client import LLMResponse
from vaultmind.memory.models import Episode, OutcomeStatus
from vaultmind.memory.store import EpisodeStore

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
    graph_context_enabled: bool = False
    graph_hop_depth: int = 2
    graph_min_confidence: float = 0.6
    graph_max_relationships: int = 20


@dataclass
class _FakeTelegramConfig:
    thinking_session_ttl: int = 3600
    thinking_summarization_enabled: bool = False
    thinking_message_count_threshold: int = 20
    thinking_recent_turns_to_keep: int = 6
    thinking_batch_size: int = 4
    thinking_summary_max_tokens: int = 400


@pytest.fixture()
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


@pytest.fixture()
def episode_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


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
    session_store: SessionStore | None = None,
) -> ThinkingPartner:
    return ThinkingPartner(
        llm_config=_FakeLLMConfig(),  # type: ignore[arg-type]
        telegram_config=_FakeTelegramConfig(),  # type: ignore[arg-type]
        llm_client=llm_client,
        session_store=session_store,
    )


def _create_episode(
    store: EpisodeStore,
    decision: str,
    entities: list[str],
    outcome: str = "Successful",
    status: OutcomeStatus = OutcomeStatus.SUCCESS,
    lessons: list[str] | None = None,
) -> Episode:
    ep = store.create(decision=decision, entities=entities, tags=["test"])
    store.resolve(
        episode_id=ep.episode_id,
        outcome=outcome,
        status=status,
        lessons=lessons or [],
    )
    result = store.get(ep.episode_id)
    assert result is not None
    return result


class TestEpisodicContextInThinking:
    @pytest.mark.asyncio()
    async def test_episodes_surface_when_entity_matches(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Adopt Kubernetes for deployment",
            entities=["kubernetes", "deployment"],
            outcome="Reduced downtime by 50%",
            lessons=["Container orchestration simplifies scaling"],
        )

        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context(
            "kubernetes scaling strategy", vault_store, graph, episode_store
        )

        assert "Past Decisions" in context
        assert "Adopt Kubernetes for deployment" in context
        assert "Reduced downtime by 50%" in context
        assert "Container orchestration simplifies scaling" in context

    @pytest.mark.asyncio()
    async def test_no_episodes_no_past_decisions_section(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context(
            "quantum computing", vault_store, graph, episode_store
        )

        assert "Past Decisions" not in context

    @pytest.mark.asyncio()
    async def test_deduplication_across_entity_searches(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Use Python for backend services",
            entities=["python", "backend"],
        )

        partner = _make_partner(llm_client, session_store)
        # Both "python" and "backend" match the same episode
        context = partner._build_vault_context(
            "python backend architecture", vault_store, graph, episode_store
        )

        # Decision should appear exactly once
        count = context.count("Use Python for backend services")
        assert count == 1

    @pytest.mark.asyncio()
    async def test_episode_store_none_graceful_skip(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context("any topic", vault_store, graph, episode_store=None)

        assert "Past Decisions" not in context

    @pytest.mark.asyncio()
    async def test_multiple_episodes_all_shown(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Switch to PostgreSQL",
            entities=["database"],
        )
        _create_episode(
            episode_store,
            decision="Add Redis caching layer",
            entities=["database"],
        )

        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context(
            "database optimization", vault_store, graph, episode_store
        )

        assert "Switch to PostgreSQL" in context
        assert "Add Redis caching layer" in context

    @pytest.mark.asyncio()
    async def test_pending_episodes_show_pending_status(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        # Create episode without resolving
        episode_store.create(
            decision="Evaluate GraphQL migration",
            entities=["graphql"],
            tags=["test"],
        )

        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context(
            "graphql api design", vault_store, graph, episode_store
        )

        assert "Evaluate GraphQL migration" in context
        assert "pending" in context

    @pytest.mark.asyncio()
    async def test_lessons_limited_to_three(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Refactor auth module",
            entities=["auth"],
            lessons=["lesson1", "lesson2", "lesson3", "lesson4", "lesson5"],
        )

        partner = _make_partner(llm_client, session_store)
        context = partner._build_vault_context(
            "auth security review", vault_store, graph, episode_store
        )

        assert "lesson1" in context
        assert "lesson3" in context
        assert "lesson4" not in context  # truncated at 3

    @pytest.mark.asyncio()
    async def test_think_passes_episode_store_to_context(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Use RAG for search",
            entities=["search"],
        )

        partner = _make_partner(llm_client, session_store)
        await partner.think(
            user_id=1,
            topic="search improvement",
            store=vault_store,
            graph=graph,
            episode_store=episode_store,
        )

        # Verify LLM was called (context was built)
        assert llm_client.complete.called

    @pytest.mark.asyncio()
    async def test_short_words_skipped_in_entity_search(
        self,
        llm_client: MagicMock,
        session_store: SessionStore,
        episode_store: EpisodeStore,
        vault_store: MagicMock,
        graph: MagicMock,
    ) -> None:
        _create_episode(
            episode_store,
            decision="Adopt AI tooling",
            entities=["ai"],  # "ai" is only 2 chars
        )

        partner = _make_partner(llm_client, session_store)
        # "ai" is < 3 chars, so it won't be searched
        context = partner._build_vault_context(
            "ai ml deep learning", vault_store, graph, episode_store
        )

        # "ai" skipped (< 3 chars), but "deep" and "learning" don't match
        assert "Adopt AI tooling" not in context
