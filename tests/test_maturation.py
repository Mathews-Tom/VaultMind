"""Tests for Zettelkasten maturation pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from vaultmind.pipeline.maturation import (
    MaturationPipeline,
    MaturationState,
    load_state,
    save_state,
)

if TYPE_CHECKING:
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
def pipeline(mat_config: MagicMock, tmp_path: Path) -> MaturationPipeline:
    collection = MagicMock()
    graph = MagicMock()
    llm = MagicMock()
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    return MaturationPipeline(
        config=mat_config,
        collection=collection,
        knowledge_graph=graph,
        llm=llm,
        vault_root=vault_root,
        state_path=tmp_path / "state.json",
    )


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
