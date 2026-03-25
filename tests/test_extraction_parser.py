"""Tests for extraction_parser module and thinking partner integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from vaultmind.bot.extraction_parser import (
    ExtractionResult,
    parse_extraction_tags,
)
from vaultmind.bot.thinking import ThinkingPartner
from vaultmind.config import GraphConfig
from vaultmind.graph.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# A. Parser unit tests
# ---------------------------------------------------------------------------


class TestParseExtractionTags:
    """Unit tests for parse_extraction_tags."""

    def test_parse_entity_valid_returns_entity(self) -> None:
        # Arrange
        raw = (
            'Some text <vm:entity name="Python" type="tool" confidence="0.9">'
            "A programming language</vm:entity> more text"
        )

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity.name == "Python"
        assert entity.type == "tool"
        assert entity.confidence == pytest.approx(0.9)
        assert entity.description == "A programming language"

    def test_parse_entity_missing_name_skipped(self) -> None:
        # Arrange
        raw = '<vm:entity type="tool" confidence="0.9">desc</vm:entity>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 0

    def test_parse_entity_invalid_type_skipped(self) -> None:
        # Arrange
        raw = '<vm:entity name="X" type="widget" confidence="0.9">desc</vm:entity>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 0

    def test_parse_entity_confidence_clamped(self) -> None:
        # Arrange
        raw = '<vm:entity name="X" type="concept" confidence="1.5">desc</vm:entity>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 1
        assert result.entities[0].confidence == pytest.approx(1.0)

    def test_parse_entity_default_confidence(self) -> None:
        # Arrange
        raw = '<vm:entity name="X" type="concept">desc</vm:entity>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 1
        assert result.entities[0].confidence == pytest.approx(1.0)

    def test_parse_relationship_valid(self) -> None:
        # Arrange
        raw = '<vm:relationship from="Python" to="Django" type="related_to" confidence="0.85" />'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.source == "Python"
        assert rel.target == "Django"
        assert rel.relation_type == "related_to"
        assert rel.confidence == pytest.approx(0.85)

    def test_parse_relationship_self_edge_skipped(self) -> None:
        # Arrange
        raw = '<vm:relationship from="Python" to="Python" type="related_to" />'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.relationships) == 0

    def test_parse_relationship_missing_from_skipped(self) -> None:
        # Arrange
        raw = '<vm:relationship to="Django" type="related_to" />'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.relationships) == 0

    def test_parse_relationship_invalid_type_skipped(self) -> None:
        # Arrange
        raw = '<vm:relationship from="A" to="B" type="loves" />'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.relationships) == 0

    def test_parse_episode_valid(self) -> None:
        # Arrange
        raw = (
            '<vm:episode decision="Use Rust" context="Performance needs" '
            'status="success" confidence="0.8">'
            "<lesson>Rust improved latency by 40%</lesson>"
            "<entity>Rust</entity>"
            "</vm:episode>"
        )

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.episodes) == 1
        ep = result.episodes[0]
        assert ep.decision == "Use Rust"
        assert ep.context == "Performance needs"
        assert ep.status == "success"
        assert ep.confidence == pytest.approx(0.8)
        assert ep.lessons == ["Rust improved latency by 40%"]
        assert ep.entities == ["Rust"]

    def test_parse_episode_missing_decision_skipped(self) -> None:
        # Arrange
        raw = '<vm:episode context="something" status="pending"><lesson>x</lesson></vm:episode>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.episodes) == 0

    def test_parse_no_tags_returns_original(self) -> None:
        # Arrange
        raw = "This is a plain response with no XML tags at all."

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert result.clean_response == raw
        assert result.entities == []
        assert result.relationships == []
        assert result.episodes == []

    def test_parse_mixed_tags_and_content(self) -> None:
        # Arrange
        raw = (
            "Here is some analysis.\n"
            '<vm:entity name="NetworkX" type="tool" confidence="0.9">graph lib</vm:entity>\n'
            "And more discussion.\n"
            '<vm:relationship from="NetworkX" to="Python" type="depends_on" />\n'
            "Final thoughts."
        )

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
        assert "Here is some analysis." in result.clean_response
        assert "And more discussion." in result.clean_response
        assert "Final thoughts." in result.clean_response

    def test_parse_tags_stripped_from_clean_response(self) -> None:
        # Arrange
        raw = (
            "Hello.\n"
            '<vm:entity name="X" type="concept">desc</vm:entity>\n'
            '<vm:relationship from="X" to="Y" type="related_to" />\n'
            "Goodbye."
        )

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert "<vm:" not in result.clean_response
        assert "</vm:" not in result.clean_response

    def test_parse_malformed_xml_skipped(self) -> None:
        # Arrange — unclosed attribute quotes
        raw = '<vm:entity name="broken type="concept">text</vm:entity>'

        # Act
        result = parse_extraction_tags(raw)

        # Assert — malformed tag is matched by regex but fails ET.fromstring
        assert len(result.entities) == 0

    def test_parse_self_closing_relationship(self) -> None:
        # Arrange
        raw = '<vm:relationship from="A" to="B" type="influences" confidence="0.7" />'

        # Act
        result = parse_extraction_tags(raw)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0].source == "A"
        assert result.relationships[0].target == "B"
        assert result.relationships[0].relation_type == "influences"


# ---------------------------------------------------------------------------
# B. ExtractionResult tests
# ---------------------------------------------------------------------------


class TestExtractionResult:
    """Tests for ExtractionResult dataclass defaults."""

    def test_empty_result_defaults(self) -> None:
        # Arrange / Act
        result = ExtractionResult(clean_response="")

        # Assert
        assert result.entities == []
        assert result.relationships == []
        assert result.episodes == []
        assert result.clean_response == ""

    def test_clean_response_no_excessive_newlines(self) -> None:
        # Arrange
        raw = 'Line one.\n\n\n\n<vm:entity name="X" type="concept">d</vm:entity>\n\n\n\nLine two.'

        # Act
        result = parse_extraction_tags(raw)

        # Assert — triple+ newlines collapsed to double
        assert "\n\n\n" not in result.clean_response
        assert "Line one." in result.clean_response
        assert "Line two." in result.clean_response


# ---------------------------------------------------------------------------
# C. Integration tests — _apply_extraction with real KnowledgeGraph
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMConfig:
    thinking_model: str = "test"
    fast_model: str = "test"
    max_context_notes: int = 3
    max_tokens: int = 100
    graph_context_enabled: bool = False
    graph_hop_depth: int = 2
    graph_min_confidence: float = 0.6
    graph_max_relationships: int = 20
    single_pass_extraction_enabled: bool = True
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
class _FakeLLMResponse:
    text: str = ""


class _FakeLLMClient:
    def complete(self, **kwargs: object) -> _FakeLLMResponse:
        return _FakeLLMResponse(text="")

    def complete_multimodal(self, **kwargs: object) -> _FakeLLMResponse:
        return _FakeLLMResponse(text="")


@pytest.fixture()
def graph_config(tmp_path: Path) -> GraphConfig:
    return GraphConfig(persist_path=tmp_path / "graph.json")


@pytest.fixture()
def graph(graph_config: GraphConfig) -> KnowledgeGraph:
    return KnowledgeGraph(graph_config)


@pytest.fixture()
def partner() -> ThinkingPartner:
    return ThinkingPartner(
        llm_config=_FakeLLMConfig(),  # type: ignore[arg-type]
        telegram_config=_FakeTelegramConfig(),  # type: ignore[arg-type]
        llm_client=_FakeLLMClient(),  # type: ignore[arg-type]
    )


class TestApplyExtraction:
    """Integration tests for ThinkingPartner._apply_extraction with real graph."""

    @pytest.mark.asyncio()
    async def test_apply_extraction_adds_entities_to_graph(
        self, partner: ThinkingPartner, graph: KnowledgeGraph
    ) -> None:
        # Arrange
        raw = (
            "Analysis here.\n"
            '<vm:entity name="FastAPI" type="tool" confidence="0.9">web framework</vm:entity>\n'
            '<vm:entity name="Pydantic" type="tool" confidence="0.85">validation</vm:entity>'
        )

        # Act
        clean = await partner._apply_extraction(raw, graph)

        # Assert
        assert graph.get_entity("FastAPI") is not None
        assert graph.get_entity("Pydantic") is not None
        assert "<vm:" not in clean

    @pytest.mark.asyncio()
    async def test_apply_extraction_adds_relationships_to_graph(
        self, partner: ThinkingPartner, graph: KnowledgeGraph
    ) -> None:
        # Arrange
        raw = (
            "Discussion.\n"
            '<vm:entity name="FastAPI" type="tool" confidence="0.9">fw</vm:entity>\n'
            '<vm:entity name="Pydantic" type="tool" confidence="0.9">val</vm:entity>\n'
            '<vm:relationship from="FastAPI" to="Pydantic" type="depends_on" confidence="0.8" />'
        )

        # Act
        await partner._apply_extraction(raw, graph)

        # Assert
        neighbors = graph.get_neighbors("FastAPI", depth=1)
        assert len(neighbors["outgoing"]) == 1
        assert neighbors["outgoing"][0]["target"] == "Pydantic"
        assert neighbors["outgoing"][0]["relation"] == "depends_on"

    @pytest.mark.asyncio()
    async def test_apply_extraction_returns_clean_response(
        self, partner: ThinkingPartner, graph: KnowledgeGraph
    ) -> None:
        # Arrange
        raw = (
            "Start of response.\n"
            '<vm:entity name="X" type="concept" confidence="0.8">d</vm:entity>\n'
            "End of response."
        )

        # Act
        clean = await partner._apply_extraction(raw, graph)

        # Assert
        assert "Start of response." in clean
        assert "End of response." in clean
        assert "<vm:" not in clean

    @pytest.mark.asyncio()
    async def test_apply_extraction_skips_below_confidence_threshold(
        self, graph: KnowledgeGraph
    ) -> None:
        # Arrange — threshold=0.7, entity confidence=0.5
        config = _FakeLLMConfig(extraction_confidence_threshold=0.7)
        tp = ThinkingPartner(
            llm_config=config,  # type: ignore[arg-type]
            telegram_config=_FakeTelegramConfig(),  # type: ignore[arg-type]
            llm_client=_FakeLLMClient(),  # type: ignore[arg-type]
        )

        raw = '<vm:entity name="LowConf" type="concept" confidence="0.5">d</vm:entity>'

        # Act
        await tp._apply_extraction(raw, graph)

        # Assert — entity below threshold not added
        assert graph.get_entity("LowConf") is None

    @pytest.mark.asyncio()
    async def test_apply_extraction_disabled_returns_raw(self, graph: KnowledgeGraph) -> None:
        # Arrange
        config = _FakeLLMConfig(single_pass_extraction_enabled=False)
        tp = ThinkingPartner(
            llm_config=config,  # type: ignore[arg-type]
            telegram_config=_FakeTelegramConfig(),  # type: ignore[arg-type]
            llm_client=_FakeLLMClient(),  # type: ignore[arg-type]
        )

        raw = 'Text with <vm:entity name="X" type="concept">d</vm:entity> tags.'

        # Act
        result = await tp._apply_extraction(raw, graph)

        # Assert — raw response returned unchanged, tags NOT stripped
        assert result == raw
        assert graph.get_entity("X") is None
