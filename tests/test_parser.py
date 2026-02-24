"""Tests for the vault parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.config import VaultConfig
from vaultmind.vault.models import NoteType
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with sample notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "00-inbox").mkdir()
    (vault / "02-projects").mkdir()
    (vault / ".obsidian").mkdir()

    # Sample note with frontmatter
    note = vault / "02-projects" / "test-project.md"
    note.write_text(
        "---\n"
        "type: project\n"
        "tags: [python, ai]\n"
        "created: 2026-01-15\n"
        "entities: [TestProject, Python]\n"
        "---\n\n"
        "# Test Project\n\n"
        "## Overview\n\n"
        "This is a test project for VaultMind.\n\n"
        "## Architecture\n\n"
        "Uses [[Python]] and [[NetworkX]] for graph operations.\n"
        "Also links to [[another-note]].\n\n"
        "## Tasks\n\n"
        "- [ ] Write tests\n"
        "- [ ] Deploy\n"
    )

    # Inbox note without frontmatter
    inbox_note = vault / "00-inbox" / "quick-thought.md"
    inbox_note.write_text("Just a quick thought about #ideas and #ai\n")

    # Note in excluded folder (should be skipped)
    obsidian_note = vault / ".obsidian" / "config.md"
    obsidian_note.write_text("This should be ignored\n")

    return vault


@pytest.fixture
def parser(tmp_vault: Path) -> VaultParser:
    config = VaultConfig(path=tmp_vault)
    return VaultParser(config)


def test_iter_notes_finds_markdown(parser: VaultParser) -> None:
    notes = parser.iter_notes()
    assert len(notes) == 2  # project + inbox, not .obsidian


def test_iter_notes_excludes_obsidian_folder(parser: VaultParser) -> None:
    notes = parser.iter_notes()
    paths = [str(n.path) for n in notes]
    assert not any(".obsidian" in p for p in paths)


def test_parse_frontmatter(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "02-projects" / "test-project.md")
    assert note.note_type == NoteType.PROJECT
    assert "python" in note.tags
    assert "ai" in note.tags
    assert "TestProject" in note.entities


def test_extract_wikilinks(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "02-projects" / "test-project.md")
    assert "Python" in note.wikilinks
    assert "NetworkX" in note.wikilinks
    assert "another-note" in note.wikilinks


def test_infer_type_from_folder(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "00-inbox" / "quick-thought.md")
    assert note.note_type == NoteType.FLEETING


def test_inline_tag_extraction(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "00-inbox" / "quick-thought.md")
    assert "ideas" in note.tags
    assert "ai" in note.tags


def test_chunking_by_headings(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "02-projects" / "test-project.md")
    chunks = parser.chunk_note(note)
    assert len(chunks) >= 3  # Overview, Architecture, Tasks sections
    # Each chunk should have proper metadata
    for chunk in chunks:
        assert chunk.note_path
        assert chunk.note_title
        assert chunk.chunk_id


def test_chunk_preserves_heading_context(parser: VaultParser, tmp_vault: Path) -> None:
    note = parser.parse_file(tmp_vault / "02-projects" / "test-project.md")
    chunks = parser.chunk_note(note)
    headings = [c.heading for c in chunks if c.heading]
    assert any("Architecture" in h for h in headings)


def test_empty_note_produces_no_chunks(parser: VaultParser, tmp_vault: Path) -> None:
    empty = tmp_vault / "00-inbox" / "empty.md"
    empty.write_text("---\ntype: fleeting\n---\n\n")
    note = parser.parse_file(empty)
    chunks = parser.chunk_note(note)
    assert len(chunks) == 0
