"""Tests for the OKF bundle exporter (M10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from vaultmind.config import VaultConfig
from vaultmind.export.okf import OkfExportResult, export_okf_bundle
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Fixture vault: permanent/concept notes to export, plus non-exported types."""
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    physics = permanent / "physics"
    physics.mkdir()
    concept = tmp_path / "concept"
    concept.mkdir()
    inbox = tmp_path / "00-inbox"
    inbox.mkdir()

    (permanent / "entropy.md").write_text(
        "---\n"
        "type: permanent\n"
        "title: Entropy\n"
        "description: A measure of disorder in a system.\n"
        "tags: [physics, thermodynamics]\n"
        "authority: 5\n"
        "---\n\n"
        "# Entropy\n\n"
        "Related to [[Information]] theory and distinct from [[Randomness]] "
        "as commonly (mis)used.\n",
        encoding="utf-8",
    )
    (physics / "thermo.md").write_text(
        "---\ntype: permanent\ntitle: Thermodynamics\n---\n\n# Thermodynamics\n\nBody text.\n",
        encoding="utf-8",
    )
    (concept / "information.md").write_text(
        "---\n"
        "type: concept\n"
        "title: Information\n"
        "authority: 3\n"
        "---\n\n"
        "# Information\n\nSee also [[Entropy]].\n",
        encoding="utf-8",
    )
    # Not exported: a fleeting note, even though its title collides with an
    # in-body wikilink target above — must not be linked to.
    (inbox / "randomness.md").write_text(
        "---\ntype: fleeting\ntitle: Randomness\n---\n\nA fleeting capture.\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def notes(vault_root: Path) -> list:  # noqa: ANN201 - Note import kept local to fixture use
    parser = VaultParser(VaultConfig(path=vault_root))  # type: ignore[arg-type]
    return parser.iter_notes()


def _read_frontmatter(text: str) -> tuple[dict, str]:
    assert text.startswith("---\n")
    _, fm_and_body = text.split("---\n", 1)
    fm_text, body = fm_and_body.split("---\n", 1)
    return yaml.safe_load(fm_text), body.lstrip("\n")


class TestExportOkfBundle:
    def test_exports_only_permanent_and_concept_notes(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        result = export_okf_bundle(notes, output_dir)

        assert isinstance(result, OkfExportResult)
        assert result.concept_count == 3
        assert (output_dir / "permanent" / "entropy.md").is_file()
        assert (output_dir / "permanent" / "physics" / "thermo.md").is_file()
        assert (output_dir / "concept" / "information.md").is_file()
        assert not (output_dir / "00-inbox").exists()

    def test_concept_frontmatter_carries_authority_and_type(
        self, notes: list, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "entropy.md").read_text(encoding="utf-8")
        fm, _ = _read_frontmatter(text)
        assert fm["type"] == "permanent"
        assert fm["title"] == "Entropy"
        assert fm["description"] == "A measure of disorder in a system."
        assert fm["tags"] == ["physics", "thermodynamics"]
        assert fm["authority"] == 5

    def test_unstamped_authority_exports_as_zero(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "physics" / "thermo.md").read_text(encoding="utf-8")
        fm, _ = _read_frontmatter(text)
        assert fm["authority"] == 0

    def test_wikilink_to_exported_note_becomes_standard_link(
        self, notes: list, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "entropy.md").read_text(encoding="utf-8")
        _, body = _read_frontmatter(text)
        assert "[Information](/concept/information.md)" in body
        assert "[[Information]]" not in text

    def test_wikilink_to_non_exported_note_is_flattened_to_plain_text(
        self, notes: list, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "entropy.md").read_text(encoding="utf-8")
        assert "[[Randomness]]" not in text
        assert "[[" not in text
        assert "]]" not in text
        assert "Randomness" in text

    def test_root_index_carries_only_okf_version(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir, okf_version="0.1")

        text = (output_dir / "index.md").read_text(encoding="utf-8")
        fm, body = _read_frontmatter(text)
        assert fm == {"okf_version": "0.1"}
        assert f"# {output_dir.name}" in body
        assert "* [concept](/concept/index.md)" in body
        assert "* [permanent](/permanent/index.md)" in body

    def test_non_root_index_has_no_frontmatter(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "index.md").read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert "* [Entropy](/permanent/entropy.md) - A measure of disorder in a system." in text
        assert "* [physics](/permanent/physics/index.md)" in text

    def test_nested_index_lists_only_direct_children(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "permanent" / "physics" / "index.md").read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert "* [Thermodynamics](/permanent/physics/thermo.md)" in text
        assert "Entropy" not in text

    def test_log_md_has_one_dated_heading(self, notes: list, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        export_okf_bundle(notes, output_dir)

        text = (output_dir / "log.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        assert len(headings) == 1
        assert headings[0].startswith("## 20")
        assert "3 concepts exported" in text

    def test_empty_vault_produces_empty_but_valid_bundle(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        result = export_okf_bundle([], output_dir)

        assert result.concept_count == 0
        assert (output_dir / "index.md").is_file()
        assert (output_dir / "log.md").is_file()
        text = (output_dir / "log.md").read_text(encoding="utf-8")
        assert "0 concepts exported" in text
