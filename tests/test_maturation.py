"""Tests for Zettelkasten maturation pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import frontmatter
import pytest

from vaultmind.config import Settings, VaultConfig
from vaultmind.errors import VaultMindError
from vaultmind.knowledge.intake import SourceIntake, StageDisposition
from vaultmind.llm.client import LLMResponse
from vaultmind.pipeline.clustering import NoteCluster
from vaultmind.pipeline.maturation import (
    MaturationPipeline,
    MaturationState,
    load_state,
    save_state,
)
from vaultmind.pipeline.synthesis import generate_synthesis
from vaultmind.wiki.workspace import WikiError, proposal_digest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def mat_config() -> MagicMock:
    config = MagicMock()
    config.enabled = True
    config.min_cluster_size = 3
    config.max_clusters_per_digest = 3
    config.cluster_eps = 0.25
    config.synthesis_max_tokens = 1500
    config.synthesis_model = "test-model"
    config.target_note_types = ["fleeting", "literature"]
    config.dismissed_cluster_expiry_days = 90
    config.inbox_folder = "00-inbox"
    return config


@pytest.fixture
def pipeline(mat_config: MagicMock, tmp_path: Path) -> Iterator[MaturationPipeline]:
    collection = MagicMock()
    graph = MagicMock()
    llm = MagicMock()
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    with SourceIntake(Settings(vault=VaultConfig(path=vault_root))) as intake:
        yield MaturationPipeline(
            config=mat_config,
            collection=collection,
            knowledge_graph=graph,
            llm=llm,
            intake=intake,
            state_path=tmp_path / "state.json",
        )


def test_synthesis_failure_raises_without_recording_completion(
    pipeline: MaturationPipeline,
) -> None:
    cluster = NoteCluster(
        note_paths=["missing-source.md"],
        note_titles=["Missing"],
        top_entity="Missing source",
        score=0.8,
        fingerprint="missing-fingerprint",
    )
    before = list(pipeline._state.synthesized)
    with pytest.raises(VaultMindError):
        pipeline.synthesize(cluster)
    assert pipeline._state.synthesized == before
    assert pipeline._state.last_run == ""


class TestMaturationState:
    def test_load_empty(self, tmp_path: Path) -> None:
        state = load_state(tmp_path / "nonexistent.json")
        assert state.last_run == ""
        assert state.dismissed_clusters == {}
        assert state.synthesized == []

    def test_save_and_load(self, tmp_path: Path) -> None:
        state = MaturationState(
            last_run="2024-01-01T00:00:00",
            dismissed_clusters={"abc123": "2024-01-01T00:00:00"},
            synthesized=["def456"],
        )
        path = tmp_path / "state.json"
        save_state(state, path)

        loaded = load_state(path)
        assert loaded.last_run == "2024-01-01T00:00:00"
        assert "abc123" in loaded.dismissed_clusters
        assert "def456" in loaded.synthesized


class TestDismiss:
    def test_dismissed_cluster_not_reoffered(self, pipeline: MaturationPipeline) -> None:
        pipeline.dismiss("test_fingerprint")
        assert "test_fingerprint" in pipeline._state.dismissed_clusters

    def test_dismissed_cluster_expires(self, pipeline: MaturationPipeline) -> None:
        old_date = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        pipeline._state.dismissed_clusters["old_fp"] = old_date
        pipeline._expire_dismissed()
        assert "old_fp" not in pipeline._state.dismissed_clusters

    def test_recent_dismissed_not_expired(self, pipeline: MaturationPipeline) -> None:
        recent_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        pipeline._state.dismissed_clusters["recent_fp"] = recent_date
        pipeline._expire_dismissed()
        assert "recent_fp" in pipeline._state.dismissed_clusters


class TestDiscover:
    @patch("vaultmind.pipeline.maturation.discover_clusters")
    def test_filters_dismissed(
        self,
        mock_discover: MagicMock,
        pipeline: MaturationPipeline,
    ) -> None:
        from vaultmind.pipeline.clustering import NoteCluster

        cluster1 = NoteCluster(
            note_paths=["a.md"],
            note_titles=["A"],
            top_entity="Alpha",
            score=0.9,
            fingerprint="fp1",
        )
        cluster2 = NoteCluster(
            note_paths=["b.md"],
            note_titles=["B"],
            top_entity="Beta",
            score=0.8,
            fingerprint="fp2",
        )
        mock_discover.return_value = [cluster1, cluster2]

        pipeline.dismiss("fp1")
        clusters = pipeline.discover()

        assert len(clusters) == 1
        assert clusters[0].fingerprint == "fp2"

    @patch("vaultmind.pipeline.maturation.discover_clusters")
    def test_filters_synthesized(
        self,
        mock_discover: MagicMock,
        pipeline: MaturationPipeline,
    ) -> None:
        from vaultmind.pipeline.clustering import NoteCluster

        cluster1 = NoteCluster(
            note_paths=["a.md"],
            note_titles=["A"],
            top_entity="Alpha",
            score=0.9,
            fingerprint="fp_synth",
        )
        mock_discover.return_value = [cluster1]
        pipeline._state.synthesized.append("fp_synth")

        clusters = pipeline.discover()
        assert len(clusters) == 0

    @patch("vaultmind.pipeline.maturation.discover_clusters")
    def test_limits_to_max_clusters(
        self,
        mock_discover: MagicMock,
        pipeline: MaturationPipeline,
    ) -> None:
        from vaultmind.pipeline.clustering import NoteCluster

        clusters = [
            NoteCluster(
                note_paths=[f"{i}.md"],
                note_titles=[f"Note{i}"],
                top_entity=f"Entity{i}",
                score=float(i),
                fingerprint=f"fp{i}",
            )
            for i in range(10)
        ]
        mock_discover.return_value = clusters

        result = pipeline.discover()
        assert len(result) <= pipeline._config.max_clusters_per_digest


class TestMarkRun:
    def test_mark_run_persists(self, pipeline: MaturationPipeline) -> None:
        pipeline.mark_run()
        assert pipeline._state.last_run != ""
        # Verify it was saved
        loaded = load_state(pipeline._state_path)
        assert loaded.last_run != ""


def _selected_cluster(pipeline: MaturationPipeline) -> NoteCluster:
    pipeline._vault_root.joinpath("a.md").write_bytes(b"# First\r\n\r\nOriginal first source.\r\n")
    pipeline._vault_root.joinpath("b.md").write_text("# Second\n\nOriginal second source.\n")
    pipeline._llm.complete.return_value = LLMResponse(
        text="---\ntype: permanent\ntags: [synthesis]\n---\n# Insight\n\nSee [[a]] and [[b]].",
        model="test-model",
        usage={},
    )
    return NoteCluster(
        note_paths=["a.md", "b.md"],
        note_titles=["First", "Second"],
        top_entity="Insight",
        score=0.8,
        fingerprint="selected-cluster",
    )


def test_generator_is_filesystem_free_and_explicitly_unapproved(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text="---\ntype: permanent\napproved: true\n---\n# Insight\n\nGenerated.",
        model="test",
        usage={},
    )
    result = generate_synthesis("complete retained prompt", llm, "test", 123)
    assert result.success
    assert list(tmp_path.iterdir()) == []
    metadata = frontmatter.loads(result.content).metadata
    assert metadata["approved"] is False and metadata["status"] == "unapproved"
    assert llm.complete.call_args.kwargs["max_tokens"] == 123


def test_maturation_retains_all_bytes_and_exact_request_before_generation(
    pipeline: MaturationPipeline,
) -> None:
    cluster = _selected_cluster(pipeline)
    llm = pipeline._llm
    response = llm.complete.return_value

    def complete(**kwargs):
        manifests = [
            json.loads(path.read_text())
            for path in (pipeline._vault_root / "_wiki/sources").glob("*.intake.json")
        ]
        originals = [item for item in manifests if item["role"] == "source-note"]
        assert len(originals) == 2
        assert any(
            pipeline._intake.read(item["id"]).content
            == b"# First\r\n\r\nOriginal first source.\r\n"
            for item in originals
        )
        request = next(item for item in manifests if item["role"] == "model-request")
        values = json.loads(pipeline._intake.read(request["id"]).text)
        assert values["messages"][0]["content"] == kwargs["messages"][0].content
        assert values["model"] == kwargs["model"]
        assert values["max_tokens"] == kwargs["max_tokens"]
        pipeline._vault_root.joinpath("a.md").write_text("Changed after selection")
        assert "Original first source" in kwargs["messages"][0].content
        assert "Changed after selection" not in kwargs["messages"][0].content
        return response

    llm.complete.side_effect = complete
    output = pipeline.synthesize(cluster)
    assert isinstance(output, str)
    path = pipeline._vault_root / output
    retained = pipeline._intake.read(path.stem)
    assert retained.content == path.read_bytes()
    assert retained.provenance["original_retained"] is False
    assert len(retained.dependencies[0].dependencies) == 2
    assert frontmatter.loads(retained.text).metadata["approved"] is False
    assert cluster.fingerprint in pipeline._state.synthesized
    assert pipeline._state.last_run


@pytest.mark.parametrize("failure", ["missing", "retention", "oversized", "too-many"])
def test_maturation_refuses_incomplete_or_unretainable_selection_before_model(
    pipeline: MaturationPipeline, failure: str
) -> None:
    cluster = _selected_cluster(pipeline)
    if failure == "missing":
        pipeline._vault_root.joinpath("b.md").unlink()
    elif failure == "retention":
        pipeline._vault_root.joinpath("b.md").write_text("---\ndo_not_retain: true\n---\nprivate")
    elif failure == "oversized":
        pipeline._vault_root.joinpath("b.md").write_bytes(b"x" * (256 * 1024 + 1))
    else:
        paths = [f"source-{i}.md" for i in range(31)]
        for path in paths:
            pipeline._vault_root.joinpath(path).write_text(f"Original {path}")
        cluster = NoteCluster(paths, paths, "Too many", 0.8, "too-many")
    with pytest.raises(VaultMindError):
        pipeline.synthesize(cluster)
    pipeline._llm.complete.assert_not_called()
    assert pipeline._state.synthesized == []


@pytest.mark.parametrize("failure", ["provider", "invalid", "projection"])
def test_maturation_failure_retains_originals_and_never_records_completion(
    pipeline: MaturationPipeline, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cluster = _selected_cluster(pipeline)
    if failure == "provider":
        pipeline._llm.complete.side_effect = RuntimeError("provider unavailable")
    elif failure == "invalid":
        pipeline._llm.complete.return_value = LLMResponse(text="invalid", model="test", usage={})
    else:
        monkeypatch.setattr(
            pipeline._intake, "project_text", lambda *_: StageDisposition("projection", "failed")
        )
    with pytest.raises(VaultMindError):
        pipeline.synthesize(cluster)
    manifests = [
        json.loads(path.read_text())
        for path in (pipeline._vault_root / "_wiki/sources").glob("*.intake.json")
    ]
    originals = [item for item in manifests if item["role"] == "source-note"]
    assert len(originals) == 2
    assert all(
        pipeline._intake.read(item["id"]).provenance["original_retained"] for item in originals
    )
    assert pipeline._state.synthesized == [] and not pipeline._state.last_run
    assert not (pipeline._vault_root / "00-inbox").exists()
    with pytest.raises(VaultMindError):
        pipeline.synthesize(cluster)
    assert pipeline._llm.complete.call_count == 1


def test_maturation_generated_id_obeys_wiki_review_and_parent_validation(
    pipeline: MaturationPipeline, tmp_path: Path
) -> None:
    output = pipeline.synthesize(_selected_cluster(pipeline))
    source_id = output.rsplit("/", 1)[-1].removesuffix(".md")
    workspace = pipeline._intake.workspace
    draft = tmp_path / "wiki-draft.md"
    draft.write_text(
        "# Insight\n\n## Current understanding\n\n"
        f"Reviewed insight. [[_wiki/sources/{source_id}]]\n"
    )
    proposal = workspace.propose("wiki/insight.md", [source_id], draft_file=draft)
    assert "Exact before/after unified diff" in workspace.review(proposal.proposal_id)
    parent = pipeline._intake.read(source_id).dependencies[0].dependencies[0]
    (pipeline._vault_root / parent.path).unlink()
    with pytest.raises(WikiError):
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    assert not (pipeline._vault_root / "wiki/insight.md").exists()


def test_absolute_maturation_folder_keeps_plain_relative_applier_result(
    pipeline: MaturationPipeline,
) -> None:
    pipeline._config.inbox_folder = str(pipeline._vault_root / "nested")
    result = pipeline.synthesize(_selected_cluster(pipeline))
    assert result.startswith("nested/")
    assert (pipeline._vault_root / result).is_file()
