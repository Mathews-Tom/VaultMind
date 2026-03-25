"""Tests for composite ranking with connection density scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vaultmind.config import GraphConfig, RankingConfig
from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.indexer.connection_density import (
    ConnectionDensityCalculator,
    ConnectionDensityScore,
)
from vaultmind.indexer.ranking import (
    ARCHIVED_MULTIPLIER,
    MODE_MULTIPLIERS,
    RankedResult,
    composite_score,
    compute_note_type_score,
    compute_recency_score,
    compute_semantic_score,
    rank_results,
)

# --- Fixtures ---


@pytest.fixture
def graph_config(tmp_path: object) -> GraphConfig:
    return GraphConfig(persist_path=tmp_path / "graph.json")  # type: ignore[operator]


@pytest.fixture
def graph(graph_config: GraphConfig) -> KnowledgeGraph:
    return KnowledgeGraph(graph_config)


@pytest.fixture
def ranking_config() -> RankingConfig:
    return RankingConfig()


# --- Helpers ---


def _make_hit(
    chunk_id: str = "test::0",
    distance: float = 0.3,
    note_type: str = "fleeting",
    created: str = "",
    status: str = "active",
    content: str = "test content",
    entities: str = "",
    mode: str = "",
    activation_score: float = 0.0,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "distance": distance,
        "content": content,
        "metadata": {
            "note_type": note_type,
            "created": created,
            "status": status,
            "entities": entities,
            "mode": mode,
            "note_path": "test/note.md",
        },
        "activation_score": activation_score,
    }


def _add_star_graph(graph: KnowledgeGraph, center: str, n_neighbors: int) -> None:
    """Add a star graph: center entity connected to n_neighbors."""
    graph.add_entity(center, "concept", confidence=0.9)
    for i in range(n_neighbors):
        name = f"neighbor_{i}"
        graph.add_entity(name, "concept", confidence=0.9)
        graph.add_relationship(center, name, "related_to", confidence=0.9)


# --- A. ConnectionDensityCalculator tests ---


class TestConnectionDensityCalculator:
    def test_score_note_empty_entities_returns_zero(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        calc = ConnectionDensityCalculator(graph, ranking_config)

        result_none = calc.score_note("test.md", None)
        assert result_none.density_score == 0.0
        assert result_none.entity_count == 0

        result_empty = calc.score_note("test.md", [])
        assert result_empty.density_score == 0.0

    def test_score_note_empty_graph_returns_zero(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["nonexistent"])
        assert result.density_score == 0.0
        assert result.entity_count == 0

    def test_score_note_entity_not_in_graph_returns_zero(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        graph.add_entity("other_entity", "concept", confidence=0.9)
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["nonexistent_entity"])
        assert result.density_score == 0.0
        assert result.connected_entities == 0

    def test_score_note_entity_below_confidence_skipped(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        graph.add_entity("low_conf", "concept", confidence=0.1)
        graph.add_entity("neighbor", "concept", confidence=0.9)
        graph.add_relationship("low_conf", "neighbor", "related_to", confidence=0.9)
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["low_conf"])
        assert result.connected_entities == 0
        assert result.density_score == 0.0

    def test_score_note_single_connected_entity(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        _add_star_graph(graph, "center", 5)
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["center"])
        assert result.connected_entities == 1
        assert result.total_neighbors == 5
        assert result.density_score > 0.0
        # density = 5 / (5 + 10) = 1/3
        assert result.density_score == pytest.approx(5 / 15)

    def test_score_note_multiple_entities_union_neighbors(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        graph.add_entity("a", "concept", confidence=0.9)
        graph.add_entity("b", "concept", confidence=0.9)
        graph.add_entity("shared", "concept", confidence=0.9)
        graph.add_entity("only_a", "concept", confidence=0.9)
        graph.add_entity("only_b", "concept", confidence=0.9)
        graph.add_relationship("a", "shared", "related_to", confidence=0.9)
        graph.add_relationship("a", "only_a", "related_to", confidence=0.9)
        graph.add_relationship("b", "shared", "related_to", confidence=0.9)
        graph.add_relationship("b", "only_b", "related_to", confidence=0.9)

        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["a", "b"])
        assert result.connected_entities == 2
        # Neighbors: shared, only_a, only_b + b is neighbor of a and a is neighbor of b
        # "a" ego: {a, shared, only_a, b} minus a = {shared, only_a, b}
        # "b" ego: {b, shared, only_b, a} minus b = {shared, only_b, a}
        # Union = {shared, only_a, b, only_b, a} = 5
        assert result.total_neighbors >= 3  # at least shared + only_a + only_b

    def test_score_note_normalization_10_neighbors_half(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        _add_star_graph(graph, "hub", 10)
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["hub"])
        # density = 10 / (10 + 10) = 0.5
        assert result.density_score == pytest.approx(0.5)

    def test_score_note_normalization_zero_neighbors_zero(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        graph.add_entity("isolated", "concept", confidence=0.9)
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", ["isolated"])
        assert result.total_neighbors == 0
        assert result.density_score == 0.0

    def test_score_note_returns_connection_density_score_dataclass(
        self, graph: KnowledgeGraph, ranking_config: RankingConfig
    ) -> None:
        calc = ConnectionDensityCalculator(graph, ranking_config)
        result = calc.score_note("test.md", None)
        assert isinstance(result, ConnectionDensityScore)


# --- B. Composite scoring function tests ---


class TestCompositeScore:
    def test_composite_score_all_ones_returns_weighted_sum(self) -> None:
        result = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="active",
        )
        assert result == pytest.approx(1.0)

    def test_composite_score_all_zeros_returns_zero(self) -> None:
        result = composite_score(
            semantic=0.0,
            recency=0.0,
            connection_density=0.0,
            activation=0.0,
            note_type_normalized=0.0,
            status="active",
        )
        assert result == pytest.approx(0.0)

    def test_composite_score_semantic_only(self) -> None:
        result = composite_score(
            semantic=1.0,
            recency=0.0,
            connection_density=0.0,
            activation=0.0,
            note_type_normalized=0.0,
            status="active",
        )
        assert result == pytest.approx(0.40)

    def test_composite_score_archived_suppression(self) -> None:
        base = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="active",
        )
        archived = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="archived",
        )
        assert archived == pytest.approx(base * ARCHIVED_MULTIPLIER)

    def test_composite_score_mode_multiplier_operational(self) -> None:
        base = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="active",
            mode="",
        )
        operational = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="active",
            mode="operational",
        )
        assert operational == pytest.approx(base * MODE_MULTIPLIERS["operational"])

    def test_composite_score_default_config_used_when_none(self) -> None:
        result = composite_score(
            semantic=1.0,
            recency=0.0,
            connection_density=0.0,
            activation=0.0,
            note_type_normalized=0.0,
            status="active",
            config=None,
        )
        # Default semantic_weight = 0.40
        assert result == pytest.approx(0.40)

    def test_composite_score_custom_weights(self) -> None:
        config = RankingConfig(
            semantic_weight=0.50,
            recency_weight=0.20,
            connection_density_weight=0.15,
            activation_weight=0.05,
            note_type_weight=0.10,
        )
        result = composite_score(
            semantic=1.0,
            recency=0.0,
            connection_density=0.0,
            activation=0.0,
            note_type_normalized=0.0,
            status="active",
            config=config,
        )
        assert result == pytest.approx(0.50)

    def test_composite_score_completed_suppression(self) -> None:
        result = composite_score(
            semantic=1.0,
            recency=1.0,
            connection_density=1.0,
            activation=1.0,
            note_type_normalized=1.0,
            status="completed",
        )
        assert result == pytest.approx(ARCHIVED_MULTIPLIER)


# --- C. Component score function tests ---


class TestComponentScores:
    def test_compute_semantic_score_clamps_to_unit(self) -> None:
        assert compute_semantic_score(1.5) == pytest.approx(1.0)

    def test_compute_semantic_score_negative_clamped_zero(self) -> None:
        assert compute_semantic_score(-0.3) == pytest.approx(0.0)

    def test_compute_semantic_score_normal_passthrough(self) -> None:
        assert compute_semantic_score(0.7) == pytest.approx(0.7)

    def test_compute_recency_score_recent_note_near_one(self) -> None:
        now_iso = datetime.now(UTC).isoformat()
        result = compute_recency_score(now_iso, 30.0)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_compute_recency_score_old_note_decays(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        result = compute_recency_score(old, 30.0)
        # exp(-0.693 * 60 / 30) = exp(-1.386) ≈ 0.25
        assert result == pytest.approx(0.25, abs=0.05)

    def test_compute_recency_score_empty_date_returns_one(self) -> None:
        assert compute_recency_score("", 30.0) == pytest.approx(1.0)

    def test_compute_recency_score_invalid_date_returns_one(self) -> None:
        assert compute_recency_score("not-a-date", 30.0) == pytest.approx(1.0)

    def test_compute_recency_score_future_date_returns_one(self) -> None:
        future = (datetime.now(UTC) + timedelta(days=10)).isoformat()
        result = compute_recency_score(future, 30.0)
        assert result == pytest.approx(1.0)

    def test_compute_note_type_score_permanent_is_one(self) -> None:
        assert compute_note_type_score("permanent") == pytest.approx(1.0)

    def test_compute_note_type_score_fleeting_is_zero(self) -> None:
        assert compute_note_type_score("fleeting") == pytest.approx(0.0)

    def test_compute_note_type_score_concept_is_high(self) -> None:
        assert compute_note_type_score("concept") == pytest.approx(0.8)

    def test_compute_note_type_score_literature(self) -> None:
        assert compute_note_type_score("literature") == pytest.approx(0.6)

    def test_compute_note_type_score_daily(self) -> None:
        assert compute_note_type_score("daily") == pytest.approx(0.2)

    def test_compute_note_type_score_unknown_uses_default(self) -> None:
        # Default multiplier is 1.0 → (1.0 - 0.8) / (1.3 - 0.8) = 0.4
        assert compute_note_type_score("unknown_type") == pytest.approx(0.4)


# --- D. rank_results integration tests ---


class TestRankResultsComposite:
    def test_rank_results_without_graph_backward_compat(self) -> None:
        hits = [_make_hit()]
        results = rank_results(hits, enabled=True, knowledge_graph=None)
        assert len(results) == 1
        assert results[0].connection_density_score == 0.0

    def test_rank_results_disabled_passthrough(self) -> None:
        hits = [_make_hit(distance=0.4), _make_hit(chunk_id="b::0", distance=0.6)]
        results = rank_results(hits, enabled=False)
        for r in results:
            assert r.final_score == r.raw_score

    def test_rank_results_empty_hits_returns_empty(self) -> None:
        assert rank_results([]) == []

    def test_rank_results_populates_component_scores(self) -> None:
        hits = [_make_hit(note_type="permanent", activation_score=0.5)]
        results = rank_results(hits, enabled=True)
        r = results[0]
        assert r.semantic_score > 0.0
        assert r.recency_score > 0.0
        assert r.note_type_score > 0.0

    def test_rank_results_sorted_descending_by_final_score(self) -> None:
        hits = [
            _make_hit(chunk_id="low::0", distance=0.8, note_type="fleeting"),
            _make_hit(chunk_id="high::0", distance=0.1, note_type="permanent"),
        ]
        results = rank_results(hits, enabled=True)
        assert results[0].chunk_id == "high::0"
        assert results[0].final_score >= results[1].final_score

    def test_rank_results_returns_ranked_result_instances(self) -> None:
        hits = [_make_hit()]
        results = rank_results(hits, enabled=True)
        assert isinstance(results[0], RankedResult)

    def test_rank_results_entities_parsed_from_metadata(self) -> None:
        hits = [_make_hit(entities="foo,bar,baz")]
        # No graph, so density stays 0, but parsing should not error
        results = rank_results(hits, enabled=True, knowledge_graph=None)
        assert len(results) == 1

    def test_rank_results_with_ranking_config(self) -> None:
        config = RankingConfig(
            semantic_weight=0.50,
            recency_weight=0.20,
            connection_density_weight=0.15,
            activation_weight=0.05,
            note_type_weight=0.10,
        )
        hits = [_make_hit()]
        results = rank_results(hits, enabled=True, ranking_config=config)
        assert len(results) == 1
        assert results[0].final_score > 0.0


# --- E. RankingConfig validation tests ---


class TestRankingConfig:
    def test_default_weights_sum_to_one(self) -> None:
        config = RankingConfig()
        total = (
            config.semantic_weight
            + config.recency_weight
            + config.connection_density_weight
            + config.activation_weight
            + config.note_type_weight
        )
        assert total == pytest.approx(1.0)

    def test_invalid_weights_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to"):
            RankingConfig(
                semantic_weight=0.2,
                recency_weight=0.1,
                connection_density_weight=0.1,
                activation_weight=0.05,
                note_type_weight=0.05,
            )

    def test_custom_valid_weights_accepted(self) -> None:
        config = RankingConfig(
            semantic_weight=0.30,
            recency_weight=0.30,
            connection_density_weight=0.20,
            activation_weight=0.10,
            note_type_weight=0.10,
        )
        assert config.semantic_weight == pytest.approx(0.30)

    def test_weights_near_one_within_tolerance(self) -> None:
        # 0.97 total is within ±0.05 of 1.0
        config = RankingConfig(
            semantic_weight=0.37,
            recency_weight=0.20,
            connection_density_weight=0.25,
            activation_weight=0.05,
            note_type_weight=0.10,
        )
        assert config.semantic_weight == pytest.approx(0.37)

    def test_default_connection_max_hops(self) -> None:
        config = RankingConfig()
        assert config.connection_max_hops == 2

    def test_default_entity_confidence_threshold(self) -> None:
        config = RankingConfig()
        assert config.entity_confidence_threshold == pytest.approx(0.6)

    def test_default_recency_half_life_days(self) -> None:
        config = RankingConfig()
        assert config.recency_half_life_days == pytest.approx(30.0)
