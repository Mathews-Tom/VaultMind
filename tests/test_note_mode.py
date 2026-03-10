"""Tests for the note mode system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.config import VaultConfig
from vaultmind.indexer.ranking import MODE_MULTIPLIERS, score
from vaultmind.vault.models import NoteMode
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def parser(tmp_vault: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=tmp_vault))  # type: ignore[arg-type]


class TestNoteModeParsingFixture:
    def test_mode_parsed_from_frontmatter(self, parser: VaultParser, tmp_vault: Path) -> None:
        """mode: operational in frontmatter is parsed as NoteMode.OPERATIONAL."""
        note_file = tmp_vault / "ops-guide.md"
        note_file.write_text("---\nmode: operational\ntitle: Ops Guide\n---\n\nContent here.\n")
        note = parser.parse_file(note_file)
        assert note.mode == NoteMode.OPERATIONAL

    def test_mode_defaults_to_learning(self, parser: VaultParser, tmp_vault: Path) -> None:
        """Notes without mode frontmatter default to NoteMode.LEARNING."""
        note_file = tmp_vault / "random.md"
        note_file.write_text("---\ntitle: Random\n---\n\nNo mode set.\n")
        note = parser.parse_file(note_file)
        assert note.mode == NoteMode.LEARNING

    def test_invalid_mode_falls_back_to_learning(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        """An unrecognized mode value falls back to NoteMode.LEARNING."""
        note_file = tmp_vault / "weird.md"
        note_file.write_text("---\nmode: unknown_value\n---\n\nContent.\n")
        note = parser.parse_file(note_file)
        assert note.mode == NoteMode.LEARNING

    def test_mode_in_chunk(self, parser: VaultParser, tmp_vault: Path) -> None:
        """Mode value propagates from Note to every NoteChunk."""
        note_file = tmp_vault / "ops.md"
        note_file.write_text(
            "---\nmode: operational\n---\n\n## Section One\n\nContent for section.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.mode == "operational"

    def test_mode_in_chroma_metadata(self, parser: VaultParser, tmp_vault: Path) -> None:
        """Mode is stored in ChromaDB-compatible metadata dict."""
        note_file = tmp_vault / "learn.md"
        note_file.write_text("---\nmode: learning\n---\n\n## Section\n\nLearning content.\n")
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert len(chunks) > 0
        meta = chunks[0].to_chroma_metadata()
        assert "mode" in meta
        assert meta["mode"] == "learning"


class TestNoteModeRanking:
    def test_mode_multiplier_in_ranking(self) -> None:
        """Operational notes score higher than learning notes at same raw score."""
        s_operational = score(1.0, "permanent", "", "active", mode="operational")
        s_learning = score(1.0, "permanent", "", "active", mode="learning")
        assert s_operational > s_learning
        assert s_operational == pytest.approx(1.3 * MODE_MULTIPLIERS["operational"])
        assert s_learning == pytest.approx(1.3 * MODE_MULTIPLIERS["learning"])

    def test_unknown_mode_uses_neutral_multiplier(self) -> None:
        """Unrecognized mode string gets 1.0x multiplier."""
        s_unknown = score(1.0, "permanent", "", "active", mode="")
        s_known = score(1.0, "permanent", "", "active", mode="learning")
        assert s_unknown == pytest.approx(s_known)

    def test_activation_boost_stacks_with_mode(self) -> None:
        """Activation boost and mode multiplier both apply."""
        base = score(1.0, "permanent", "", "active", mode="learning", activation_score=0.0)
        boosted = score(1.0, "permanent", "", "active", mode="operational", activation_score=1.0)
        assert boosted > base
