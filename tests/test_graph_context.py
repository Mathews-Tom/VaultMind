"""Tests for graph context builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vaultmind.graph.context import GraphContextBlock, GraphContextBuilder
from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.llm.client import LLMError, LLMResponse, Message


class FakeLLMClient:
    """Minimal LLM client for testing entity extraction."""

    def __init__(self, response_text: str = '["CAIRN", "MCP Server"]') -> None:
        self._response = response_text
        self.call_count = 0

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
        self.call_count += 1
        return LLMResponse(text=self._response, model=model, usage={"input": 10, "output": 5})


class FailingLLMClient:
    """LLM client that always raises."""

    @property
    def provider_name(self) -> str:
        return "fake"

    def complete(self, **kwargs: Any) -> LLMResponse:
        raise LLMError("API down", provider="fake")


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    from vaultmind.config import GraphConfig

    config = GraphConfig(persist_path=tmp_path / "graph.json")
    g = KnowledgeGraph(config)
    g.add_entity("CAIRN", "project", confidence=0.9)
    g.add_entity("MCP Server", "tool", confidence=0.85)
    g.add_entity("VaultMind", "project", confidence=0.95)
    g.add_relationship("CAIRN", "MCP Server", "depends_on", confidence=0.85)
    g.add_relationship("CAIRN", "VaultMind", "influences", confidence=0.72)
    g.add_relationship("MCP Server", "VaultMind", "part_of", confidence=0.9)
    return g


@pytest.fixture
def empty_graph(tmp_path: Path) -> KnowledgeGraph:
    from vaultmind.config import GraphConfig

    config = GraphConfig(persist_path=tmp_path / "empty_graph.json")
    return KnowledgeGraph(config)


@pytest.fixture
def builder(graph: KnowledgeGraph) -> GraphContextBuilder:
    return GraphContextBuilder(
        knowledge_graph=graph,
        llm_client=FakeLLMClient(),
        fast_model="test-model",
    )


class TestEntityExtraction:
    @pytest.mark.asyncio
    async def test_session_cached(self, builder: GraphContextBuilder) -> None:
        """Same session reuses cached entities."""
        await builder.extract_entities("Tell me about CAIRN", "session-1")
        entities2 = await builder.extract_entities("What about MCP?", "session-1")
        # Second call merges with cache, but LLM was called for new query
        assert "CAIRN" in entities2
        assert "MCP Server" in entities2

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self, builder: GraphContextBuilder) -> None:
        """Different sessions get independent caches."""
        await builder.extract_entities("CAIRN query", "session-1")
        await builder.extract_entities("Other query", "session-2")
        assert "session-1" in builder._session_cache
        assert "session-2" in builder._session_cache

    def test_clear_session(self, builder: GraphContextBuilder) -> None:
        """clear_session removes cached entities."""
        builder._session_cache["session-1"] = ["CAIRN"]
        builder.clear_session("session-1")
        assert "session-1" not in builder._session_cache

    @pytest.mark.asyncio
    async def test_extraction_failure_logs_and_continues(self, graph: KnowledgeGraph) -> None:
        """LLM failure returns empty list, doesn't crash."""
        failing_builder = GraphContextBuilder(
            knowledge_graph=graph,
            llm_client=FailingLLMClient(),
            fast_model="test",
        )
        entities = await failing_builder.extract_entities("test query", "s1")
        assert entities == []


class TestGraphContextBuilder:
    @pytest.mark.asyncio
    async def test_build_returns_context_block(self, builder: GraphContextBuilder) -> None:
        block = await builder.build("Tell me about CAIRN", "s1")
        assert block is not None
        assert len(block.entities) > 0

    @pytest.mark.asyncio
    async def test_build_includes_relationships(self, builder: GraphContextBuilder) -> None:
        block = await builder.build("CAIRN and MCP Server", "s1")
        assert block is not None
        assert len(block.relationships) > 0

    @pytest.mark.asyncio
    async def test_build_finds_cross_entity_paths(self, builder: GraphContextBuilder) -> None:
        block = await builder.build("CAIRN MCP Server", "s1")
        assert block is not None
        # Both entities exist, so paths should be found
        if len(block.entities) >= 2:
            assert len(block.paths) > 0

    @pytest.mark.asyncio
    async def test_empty_graph_returns_none(self, empty_graph: KnowledgeGraph) -> None:
        builder = GraphContextBuilder(
            knowledge_graph=empty_graph,
            llm_client=FakeLLMClient(),
            fast_model="test",
        )
        block = await builder.build("test query", "s1")
        assert block is None

    @pytest.mark.asyncio
    async def test_normalized_entity_matching(self, graph: KnowledgeGraph) -> None:
        """'cairn' matches 'CAIRN' in graph via normalization."""
        builder = GraphContextBuilder(
            knowledge_graph=graph,
            llm_client=FakeLLMClient(response_text='["cairn"]'),
            fast_model="test",
        )
        block = await builder.build("what about cairn?", "s1")
        assert block is not None
        assert len(block.entities) > 0


class TestGraphContextBlock:
    def test_render_empty_block(self) -> None:
        block = GraphContextBlock()
        assert block.render() == ""

    def test_render_with_entities(self) -> None:
        block = GraphContextBlock(
            entities=[{"label": "CAIRN", "type": "project"}],
            relationships=[
                {
                    "source": "CAIRN",
                    "target": "MCP",
                    "relation": "depends_on",
                    "confidence": 0.85,
                }
            ],
        )
        rendered = block.render(min_confidence=0.6)
        assert "CAIRN" in rendered
        assert "depends_on" in rendered

    def test_render_filters_low_confidence(self) -> None:
        block = GraphContextBlock(
            entities=[{"label": "X", "type": "concept"}],
            relationships=[
                {
                    "source": "X",
                    "target": "Y",
                    "relation": "related_to",
                    "confidence": 0.3,
                }
            ],
        )
        rendered = block.render(min_confidence=0.6)
        # Low confidence relationship should be filtered
        assert "Y" not in rendered
