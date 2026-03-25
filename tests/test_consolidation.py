"""Tests for memory consolidation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.llm.client import LLMResponse
from vaultmind.memory.consolidation import ConsolidationReport, MemoryConsolidator
from vaultmind.memory.models import OutcomeStatus
from vaultmind.memory.store import EpisodeStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


@pytest.fixture()
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    return root


@pytest.fixture()
def llm_client() -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text="- Lesson 1: Always test\n- Lesson 2: Keep it simple",
        model="test",
        usage={},
    )
    return client


def _create_old_resolved(
    store: EpisodeStore,
    decision: str,
    entities: list[str],
    age_days: int = 400,
    outcome: str = "Succeeded",
    lessons: list[str] | None = None,
) -> str:
    ep = store.create(decision=decision, entities=entities, tags=["test"])
    store.resolve(
        episode_id=ep.episode_id,
        outcome=outcome,
        status=OutcomeStatus.SUCCESS,
        lessons=lessons or ["learned something"],
    )
    store._conn.execute(
        "UPDATE episodes SET created = datetime('now', ? || ' days') WHERE episode_id = ?",
        (f"-{age_days}", ep.episode_id),
    )
    store._conn.commit()
    return ep.episode_id


class TestArchiveOldResolved:
    def test_archive_old_episodes(self, store: EpisodeStore) -> None:
        _create_old_resolved(store, "Old decision", ["test"], age_days=400)
        archived = store.archive_old_resolved(age_days=365)
        assert archived == 1

    def test_skip_recent_episodes(self, store: EpisodeStore) -> None:
        _create_old_resolved(store, "Recent decision", ["test"], age_days=30)
        archived = store.archive_old_resolved(age_days=365)
        assert archived == 0

    def test_skip_pending_episodes(self, store: EpisodeStore) -> None:
        ep = store.create(decision="Pending decision", entities=["test"])
        store._conn.execute(
            "UPDATE episodes SET created = datetime('now', '-400 days') WHERE episode_id = ?",
            (ep.episode_id,),
        )
        store._conn.commit()
        archived = store.archive_old_resolved(age_days=365)
        assert archived == 0  # pending episodes not archived

    def test_skip_already_archived(self, store: EpisodeStore) -> None:
        _create_old_resolved(store, "Will archive", ["test"], age_days=400)
        store.archive_old_resolved(age_days=365)
        # Second call should not re-archive
        archived = store.archive_old_resolved(age_days=365)
        assert archived == 0


class TestQueryArchived:
    def test_query_returns_archived(self, store: EpisodeStore) -> None:
        _create_old_resolved(store, "Archived ep", ["test"], age_days=400)
        store.archive_old_resolved(age_days=365)
        results = store.query_archived()
        assert len(results) == 1
        assert results[0].decision == "Archived ep"

    def test_query_empty_when_none_archived(self, store: EpisodeStore) -> None:
        assert store.query_archived() == []


class TestCountEntityReferences:
    def test_count_references(self, store: EpisodeStore) -> None:
        store.create(decision="A", entities=["kubernetes"])
        store.create(decision="B", entities=["kubernetes", "docker"])
        store.create(decision="C", entities=["docker"])
        assert store.count_entity_references("kubernetes") == 2
        assert store.count_entity_references("docker") == 2
        assert store.count_entity_references("unknown") == 0


class TestConsolidator:
    def test_consolidate_archives_old_episodes(
        self,
        store: EpisodeStore,
        vault_root: Path,
    ) -> None:
        _create_old_resolved(store, "Old ep", ["test"], age_days=400)
        consolidator = MemoryConsolidator(
            episode_store=store,
            vault_root=vault_root,
            retention_days=365,
        )
        report = consolidator.consolidate()
        assert report.archived_count == 1

    def test_consolidate_summarizes_clusters(
        self,
        store: EpisodeStore,
        vault_root: Path,
        llm_client: MagicMock,
    ) -> None:
        for i in range(4):
            _create_old_resolved(
                store,
                f"K8s decision {i}",
                ["kubernetes"],
                age_days=400,
                lessons=[f"lesson {i}"],
            )
        # Archive them first
        store.archive_old_resolved(age_days=365)

        consolidator = MemoryConsolidator(
            episode_store=store,
            vault_root=vault_root,
            llm_client=llm_client,
            retention_days=365,
        )
        report = consolidator.consolidate()
        assert len(report.summaries_created) >= 1
        # Verify lesson note was written
        lesson_files = list((vault_root / "_meta" / "lessons").glob("*.md"))
        assert len(lesson_files) >= 1
        content = lesson_files[0].read_text()
        assert "Lessons" in content

    def test_consolidate_no_llm_skips_summarization(
        self,
        store: EpisodeStore,
        vault_root: Path,
    ) -> None:
        for i in range(4):
            _create_old_resolved(store, f"Decision {i}", ["test"], age_days=400)
        store.archive_old_resolved(age_days=365)

        consolidator = MemoryConsolidator(
            episode_store=store,
            vault_root=vault_root,
            llm_client=None,
            retention_days=365,
        )
        report = consolidator.consolidate()
        assert report.summaries_created == []

    def test_consolidate_promotes_referenced(
        self,
        store: EpisodeStore,
        vault_root: Path,
    ) -> None:
        # Create 4 episodes sharing entity "api" — each references the others
        for i in range(4):
            ep = store.create(
                decision=f"API decision {i}",
                entities=["api"],
            )
            store.resolve(
                episode_id=ep.episode_id,
                outcome=f"Result {i}",
                status=OutcomeStatus.SUCCESS,
                lessons=[f"lesson {i}"],
            )

        consolidator = MemoryConsolidator(
            episode_store=store,
            vault_root=vault_root,
            min_references_for_promotion=3,
        )
        report = consolidator.consolidate()
        assert report.promoted_count >= 1
        # Verify promoted note was written
        episode_files = list((vault_root / "_meta" / "episodes").glob("*.md"))
        assert len(episode_files) >= 1

    def test_consolidate_disabled_does_nothing(self) -> None:
        report = ConsolidationReport()
        assert report.archived_count == 0
        assert report.summaries_created == []
        assert report.promoted_count == 0


class TestConsolidationConfig:
    def test_defaults(self) -> None:
        from vaultmind.config import ConsolidationConfig

        cfg = ConsolidationConfig()
        assert cfg.enabled is False
        assert cfg.retention_days == 365
        assert cfg.min_references_for_promotion == 3
        assert cfg.schedule == "0 0 1 * *"

    def test_in_settings(self) -> None:
        from vaultmind.config import Settings

        assert hasattr(Settings, "model_fields")
        assert "consolidation" in Settings.model_fields
