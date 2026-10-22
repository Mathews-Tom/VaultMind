"""Tests for MCP introspection tools (PR #29)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path  # noqa: TC003 — used at runtime in tmp_path fixture signatures

import networkx as nx
import pytest

from vaultmind.config import GraphConfig
from vaultmind.graph.evolution import EvolutionDetector
from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.mcp.auth import ProfileEnforcer, ProfileError, ProfilePolicy
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool
from vaultmind.memory.procedural import ProceduralMemory
from vaultmind.memory.store import EpisodeStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Create a minimal vault with a few .md files."""
    inbox = tmp_path / "00-inbox"
    inbox.mkdir()
    projects = tmp_path / "02-projects"
    projects.mkdir()

    (inbox / "note1.md").write_text(
        "---\ntype: fleeting\ntags: [test]\ncreated: 2026-01-01\n---\n\n# Note 1\n\nContent.",
        encoding="utf-8",
    )
    (inbox / "note2.md").write_text(
        "---\ntype: literature\ntags: [test]\ncreated: 2026-01-02\n---\n\n# Note 2\n\nContent.",
        encoding="utf-8",
    )
    (projects / "project.md").write_text(
        "---\ntype: permanent\ntags: [project]\ncreated: 2026-01-03\n---\n\n# Project.\n\nContent.",
        encoding="utf-8",
    )
    # Hidden folder — should be excluded
    hidden = tmp_path / ".obsidian"
    hidden.mkdir()
    (hidden / "config.md").write_text("---\n---\nhidden", encoding="utf-8")

    return tmp_path


@pytest.fixture
def knowledge_graph(tmp_path: Path) -> KnowledgeGraph:
    config = GraphConfig(persist_path=tmp_path / "graph.json")
    return KnowledgeGraph(config)


@pytest.fixture
def episode_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


@pytest.fixture
def procedural_memory(tmp_path: Path) -> ProceduralMemory:
    return ProceduralMemory(tmp_path / "workflows.db")


@pytest.fixture
def evolution_detector(tmp_path: Path, knowledge_graph: KnowledgeGraph) -> EvolutionDetector:
    return EvolutionDetector(
        knowledge_graph,
        store_path=tmp_path / "evolution_dismissed.json",
    )


# Minimal stubs required by _dispatch_tool but not exercised in these tests
class _StubStore:
    pass


class _StubParser:
    pass


def _dispatch(
    name: str,
    args: dict,
    vault_path: Path,
    graph: KnowledgeGraph,
    **kwargs,
) -> dict:
    """Thin wrapper that fills in unused positional stubs and the full-policy enforcer."""
    return _dispatch_tool(
        name,
        args,
        vault_path,
        _StubStore(),  # type: ignore[arg-type]
        graph,
        _StubParser(),  # type: ignore[arg-type]
        enforcer=kwargs.pop("enforcer", ProfileEnforcer(load_profile("full"), vault_path)),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# vault_stats
# ---------------------------------------------------------------------------


class TestVaultStats:
    def test_total_count(self, vault_root: Path, knowledge_graph: KnowledgeGraph) -> None:
        result = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
        assert result["total_notes"] == 3

    def test_by_folder_structure(self, vault_root: Path, knowledge_graph: KnowledgeGraph) -> None:
        result = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
        assert result["by_folder"]["00-inbox"] == 2
        assert result["by_folder"]["02-projects"] == 1

    def test_by_type_parsed_from_frontmatter(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
        assert result["by_type"]["fleeting"] == 1
        assert result["by_type"]["literature"] == 1
        assert result["by_type"]["permanent"] == 1

    def test_hidden_folders_excluded(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
        # .obsidian folder should not appear
        assert ".obsidian" not in result["by_folder"]

    def test_graph_info_present(self, vault_root: Path, knowledge_graph: KnowledgeGraph) -> None:
        knowledge_graph.add_entity("Python", "tool")
        knowledge_graph.add_entity("VaultMind", "project")
        knowledge_graph.add_relationship("VaultMind", "Python", "depends_on")
        result = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
        assert result["graph"]["entities"] == 2
        assert result["graph"]["edges"] == 1

    def test_empty_vault(self, tmp_path: Path, knowledge_graph: KnowledgeGraph) -> None:
        result = _dispatch("vault_stats", {}, tmp_path, knowledge_graph)
        assert result["total_notes"] == 0
        assert result["by_type"] == {}
        assert result["by_folder"] == {}


# ---------------------------------------------------------------------------
# episode_query
# ---------------------------------------------------------------------------


class TestEpisodeQuery:
    def test_returns_error_when_store_none(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("episode_query", {}, vault_root, knowledge_graph)
        assert "error" in result
        assert "not configured" in result["error"]

    def test_returns_empty_when_no_episodes(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        episode_store: EpisodeStore,
    ) -> None:
        result = _dispatch(
            "episode_query", {}, vault_root, knowledge_graph, episode_store=episode_store
        )
        assert result["count"] == 0
        assert result["episodes"] == []

    def test_query_pending_status(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        episode_store: EpisodeStore,
    ) -> None:
        episode_store.create("Should I migrate to Rust?", context="performance concerns")
        result = _dispatch(
            "episode_query",
            {"status": "pending"},
            vault_root,
            knowledge_graph,
            episode_store=episode_store,
        )
        assert result["count"] == 1
        ep = result["episodes"][0]
        assert ep["decision"] == "Should I migrate to Rust?"
        assert ep["status"] == "pending"

    def test_query_by_entity(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        episode_store: EpisodeStore,
    ) -> None:
        from vaultmind.memory.models import OutcomeStatus

        ep = episode_store.create(
            "Use Python for data pipeline",
            context="project start",
            entities=["python", "data-pipeline"],
        )
        episode_store.resolve(
            ep.episode_id,
            outcome="Worked well",
            status=OutcomeStatus.SUCCESS,
            lessons=["Python is fast enough"],
        )
        result = _dispatch(
            "episode_query",
            {"entity": "python"},
            vault_root,
            knowledge_graph,
            episode_store=episode_store,
        )
        assert result["count"] == 1
        assert "python" in result["episodes"][0]["entities"]

    def test_episode_fields_present(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        episode_store: EpisodeStore,
    ) -> None:
        from vaultmind.memory.models import OutcomeStatus

        ep = episode_store.create("Deploy to prod", context="release day")
        episode_store.resolve(
            ep.episode_id,
            outcome="Success",
            status=OutcomeStatus.SUCCESS,
            lessons=["Always test first"],
        )
        result = _dispatch(
            "episode_query",
            {"status": "resolved"},
            vault_root,
            knowledge_graph,
            episode_store=episode_store,
        )
        assert result["count"] == 1
        ep_data = result["episodes"][0]
        for field in (
            "episode_id",
            "decision",
            "context",
            "outcome",
            "status",
            "lessons",
            "entities",
            "created",
        ):
            assert field in ep_data


# ---------------------------------------------------------------------------
# workflow_suggest
# ---------------------------------------------------------------------------


class TestWorkflowSuggest:
    def test_returns_error_when_memory_none(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch(
            "workflow_suggest", {"context": "deploy app"}, vault_root, knowledge_graph
        )
        assert "error" in result
        assert "not configured" in result["error"]

    def test_returns_null_when_no_workflows(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        procedural_memory: ProceduralMemory,
    ) -> None:
        result = _dispatch(
            "workflow_suggest",
            {"context": "deploy app"},
            vault_root,
            knowledge_graph,
            procedural_memory=procedural_memory,
        )
        assert result["workflow"] is None
        assert "No matching workflow" in result["message"]

    def test_returns_matching_workflow(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        procedural_memory: ProceduralMemory,
    ) -> None:
        procedural_memory.create_workflow(
            name="Deploy Checklist",
            description="Steps for deploying to production",
            steps=["Run tests", "Build artifact", "Deploy"],
            trigger_pattern="deploy production release",
        )
        result = _dispatch(
            "workflow_suggest",
            {"context": "deploy to production"},
            vault_root,
            knowledge_graph,
            procedural_memory=procedural_memory,
        )
        assert result["workflow"] is not None
        wf = result["workflow"]
        assert wf["name"] == "Deploy Checklist"
        assert "steps" in wf
        assert "trigger_pattern" in wf
        assert "success_rate" in wf
        assert "usage_count" in wf


# ---------------------------------------------------------------------------
# graph_evolution
# ---------------------------------------------------------------------------


class TestGraphEvolution:
    def test_returns_error_when_detector_none(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("graph_evolution", {}, vault_root, knowledge_graph)
        assert "error" in result
        assert "not configured" in result["error"]

    def test_returns_empty_signals_on_empty_graph(
        self,
        vault_root: Path,
        knowledge_graph: KnowledgeGraph,
        evolution_detector: EvolutionDetector,
    ) -> None:
        result = _dispatch(
            "graph_evolution",
            {},
            vault_root,
            knowledge_graph,
            evolution_detector=evolution_detector,
        )
        assert result["count"] == 0
        assert result["signals"] == []

    def test_severity_filter_on_engaged_multisource_signals(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(GraphConfig(persist_path=tmp_path / "graph.json"))
        kg._graph = nx.MultiDiGraph()
        kg.add_entity("A", "concept")
        kg.add_entity("B", "concept")
        kg._graph.add_edge("a", "b", relation="related_to", confidence=0.2, source_notes=["a.md"])
        kg._graph.add_edge("a", "b", relation="depends_on", confidence=0.9, source_notes=["b.md"])
        detector = EvolutionDetector(kg, store_path=tmp_path / "dismissed.json")
        before = deepcopy(kg._graph)
        result = _dispatch("graph_evolution", {}, tmp_path, kg, evolution_detector=detector)
        assert {s["signal_type"] for s in result["signals"]} == {
            "confidence_drift",
            "relationship_shift",
        }
        assert result["count"] == 2
        filtered = _dispatch(
            "graph_evolution",
            {"min_severity": 0.9},
            tmp_path,
            kg,
            evolution_detector=detector,
        )
        assert [s["signal_type"] for s in filtered["signals"]] == ["confidence_drift"]
        assert filtered["count"] == 1
        assert nx.utils.graphs_equal(before, kg._graph)
        assert not (tmp_path / "dismissed.json").exists()


# ---------------------------------------------------------------------------
# recent_activity
# ---------------------------------------------------------------------------


class TestRecentActivity:
    def test_newly_created_files_appear(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("recent_activity", {"days": 365 * 5}, vault_root, knowledge_graph)
        # All 3 test notes were just created, so they should be in created
        assert result["created_count"] + result["modified_count"] == 3

    def test_hidden_folder_excluded(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("recent_activity", {"days": 365 * 5}, vault_root, knowledge_graph)
        all_paths = result["created"] + result["modified"]
        assert not any(".obsidian" in p for p in all_paths)

    def test_zero_days_returns_empty(
        self, vault_root: Path, knowledge_graph: KnowledgeGraph
    ) -> None:
        result = _dispatch("recent_activity", {"days": 0}, vault_root, knowledge_graph)
        assert result["created_count"] == 0
        assert result["modified_count"] == 0


def _restricted(vault: Path) -> ProfileEnforcer:
    return ProfileEnforcer(
        ProfilePolicy("scoped", "Scoped", frozenset({"*"}), ("02-projects",), False),
        vault,
    )


@pytest.mark.parametrize("unknown_first", [True, False])
def test_unknown_graph_contributions_deny_restricted_consumers(
    vault_root: Path, knowledge_graph: KnowledgeGraph, unknown_first: bool
) -> None:
    graph = knowledge_graph
    source = "02-projects/project.md"
    for origin in ("", source) if unknown_first else (source, ""):
        confidence = 0.7 if origin else 0.95
        for label in ("A", "B"):
            graph.add_entity(
                label,
                "concept",
                source_note=origin,
                confidence=confidence,
                **({} if origin else {"private_context": "UNPROVENANCED"}),
            )
        graph.add_relationship(
            "A",
            "B",
            source_note=origin,
            confidence=confidence,
            relation="related_to" if origin else "depends_on",
        )
    # An all-allowed source list does not account for the retained unknown data.
    full = _dispatch("graph_query", {"entity": "A"}, vault_root, graph)
    assert full["entity"]["source_notes"] == [source]
    assert full["entity"]["private_context"] == "UNPROVENANCED"
    assert full["entity"]["confidence"] == 0.95
    assert full["outgoing"] == [
        {
            "target": "B",
            "relation": "depends_on" if unknown_first else "related_to",
            "confidence": 0.95,
        }
    ]
    assert _dispatch("graph_path", {"source": "A", "target": "B"}, vault_root, graph) == {
        "path": ["A", "B"],
        "length": 1,
    }
    assert _dispatch("vault_stats", {}, vault_root, graph)["graph"] == {"entities": 2, "edges": 1}
    for tool, arguments in (
        ("graph_query", {"entity": "A"}),
        ("graph_path", {"source": "A", "target": "B"}),
        ("vault_stats", {}),
    ):
        with pytest.raises(ProfileError, match="complete note provenance"):
            _dispatch(tool, arguments, vault_root, graph, enforcer=_restricted(vault_root))


def test_aggregate_notes_authorize_and_deduplicate_aliases(
    vault_root: Path, knowledge_graph: KnowledgeGraph
) -> None:
    allowed = vault_root / "02-projects" / "project.md"
    (allowed.parent / "alias.md").symlink_to(allowed)
    (allowed.parent / "private-alias.md").symlink_to(vault_root / "00-inbox" / "note1.md")
    (allowed.parent / "missing.md").symlink_to(vault_root / "missing.md")
    stats = _dispatch("vault_stats", {}, vault_root, knowledge_graph)
    assert stats == {
        "total_notes": 3,
        "by_type": {"fleeting": 1, "literature": 1, "permanent": 1},
        "by_folder": {"00-inbox": 2, "02-projects": 1},
        "graph": {"entities": 0, "edges": 0},
    }
    activity = _dispatch(
        "recent_activity",
        {"days": 365},
        vault_root,
        knowledge_graph,
        enforcer=_restricted(vault_root),
    )
    assert activity["created"] + activity["modified"] == ["02-projects/project.md"]
    assert activity["created_count"] + activity["modified_count"] == 1


@pytest.mark.parametrize(
    "tool",
    [
        "graph_query",
        "graph_path",
        "vault_stats",
        "episode_query",
        "workflow_suggest",
        "graph_evolution",
    ],
)
def test_unscopable_capabilities_deny_before_dependency_lookup(
    tool: str, vault_root: Path, knowledge_graph: KnowledgeGraph
) -> None:
    with pytest.raises(ProfileError, match="complete note provenance"):
        _dispatch(tool, {}, vault_root, knowledge_graph, enforcer=_restricted(vault_root))


@pytest.mark.parametrize(
    "tool",
    [
        "graph_query",
        "graph_path",
        "vault_stats",
        "recent_activity",
        "episode_query",
        "workflow_suggest",
        "graph_evolution",
    ],
)
def test_derived_dispatch_without_enforcement_fails_closed(
    tool: str, vault_root: Path, knowledge_graph: KnowledgeGraph
) -> None:
    with pytest.raises(ProfileError, match="requires profile enforcement"):
        _dispatch(tool, {}, vault_root, knowledge_graph, enforcer=None)
