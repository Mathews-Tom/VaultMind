"""Reserved storage cannot be reached through ordinary report/state aliases."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.config import DigestConfig, MaturationConfig, Settings
from vaultmind.indexer.digest import DigestGenerator, DigestReport
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.memory.consolidation import MemoryConsolidator
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.maturation import MaturationPipeline
from vaultmind.vault.security import ReservedNamespaceError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("reserved", ["wiki", "_wiki"])
def test_lesson_alias_refuses_before_generation(tmp_path: Path, reserved: str) -> None:
    vault = tmp_path / "vault"
    protected = vault / reserved
    protected.mkdir(parents=True)
    retained = protected / "existing.md"
    retained.write_bytes(b"exact retained page")
    (vault / "_meta").symlink_to(protected, target_is_directory=True)
    provider = MagicMock()
    with closing(EpisodeStore(tmp_path / "episodes.db")) as store:
        consolidator = MemoryConsolidator(store, vault, llm_client=provider)
        with pytest.raises(ReservedNamespaceError):
            consolidator._write_lesson_note("topic", [])
    provider.complete.assert_not_called()
    assert {path.name for path in protected.iterdir()} == {"existing.md"}
    assert retained.read_bytes() == b"exact retained page"


@pytest.mark.parametrize("reserved", ["wiki", "_wiki"])
def test_digest_alias_cannot_create_report(tmp_path: Path, reserved: str) -> None:
    vault = tmp_path / "vault"
    protected = vault / reserved
    protected.mkdir(parents=True)
    (vault / "_meta").symlink_to(protected, target_is_directory=True)
    generator = DigestGenerator(MagicMock(), MagicMock(), MagicMock(), DigestConfig())
    report = DigestReport(datetime(2026, 10, 24, tzinfo=UTC), 7)
    with pytest.raises(ReservedNamespaceError):
        generator.save_to_vault(report, vault)
    assert list(protected.iterdir()) == []


def test_maturation_state_alias_is_rejected_before_loading(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    protected = vault / "_wiki"
    protected.mkdir(parents=True)
    retained = protected / "state.json"
    retained.write_bytes(b"retained non-state bytes")
    alias = vault / "ordinary-state.json"
    alias.symlink_to(retained)
    with (
        SourceIntake(Settings(vault={"path": vault})) as intake,
        pytest.raises(ReservedNamespaceError),
    ):
        MaturationPipeline(
            MaturationConfig(), MagicMock(), MagicMock(), MagicMock(), intake, state_path=alias
        )
    assert retained.read_bytes() == b"retained non-state bytes"
    assert {path.name for path in protected.iterdir()} == {"state.json"}
