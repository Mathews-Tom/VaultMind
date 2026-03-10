"""Tests for contextual chunk header prefixes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.config import VaultConfig
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "00-inbox").mkdir()
    (vault / "02-projects").mkdir()
    return vault


@pytest.fixture
def parser(tmp_vault: Path) -> VaultParser:
    config = VaultConfig(path=tmp_vault)
    return VaultParser(config)


class TestContextualPrefix:
    def test_chunk_content_has_prefix(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "00-inbox" / "test-note.md"
        note_file.write_text(
            "---\ntype: fleeting\ntags: [python]\n---\n\nSome content here.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert len(chunks) == 1
        assert chunks[0].content.startswith("note: ")

    def test_prefix_includes_heading(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "02-projects" / "headed.md"
        note_file.write_text(
            "---\ntype: project\n---\n\n"
            "## Architecture\n\nDesign details here.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert len(chunks) == 1
        assert "section: Architecture" in chunks[0].content

    def test_prefix_includes_tags(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "00-inbox" / "tagged.md"
        note_file.write_text(
            "---\ntype: fleeting\ntags: [rust, wasm]\n---\n\nContent.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert len(chunks) == 1
        assert "tags: " in chunks[0].content

    def test_prefix_truncates_tags_at_five(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        tags = "[a, b, c, d, e, f, g]"
        note_file = tmp_vault / "00-inbox" / "many-tags.md"
        note_file.write_text(
            f"---\ntype: fleeting\ntags: {tags}\n---\n\nContent.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        prefix_line = chunks[0].content.split("\n")[0]
        # Extract the tags portion and count
        tags_part = prefix_line.split("tags: ")[1]
        tag_list = [t.strip() for t in tags_part.split(",")]
        assert len(tag_list) <= 5

    def test_chunk_id_unchanged(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "00-inbox" / "id-check.md"
        note_file.write_text(
            "---\ntype: fleeting\n---\n\nSome content.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert chunks[0].chunk_id == f"{note.path}::0"

    def test_prefix_no_heading_no_section(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "00-inbox" / "no-heading.md"
        note_file.write_text(
            "---\ntype: fleeting\n---\n\nPlain paragraph without heading.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert "section:" not in chunks[0].content

    def test_prefix_no_tags_no_tags_part(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        note_file = tmp_vault / "00-inbox" / "no-tags.md"
        note_file.write_text(
            "---\ntype: fleeting\n---\n\nNo tags on this note.\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note)
        assert "tags:" not in chunks[0].content

    def test_prefix_on_paragraph_split_chunks(
        self, parser: VaultParser, tmp_vault: Path
    ) -> None:
        """Large sections split by paragraphs still get the prefix."""
        long_content = "\n\n".join([f"Paragraph {i}. " + "word " * 200 for i in range(5)])
        note_file = tmp_vault / "02-projects" / "long.md"
        note_file.write_text(
            f"---\ntype: project\ntags: [big]\n---\n\n## Big Section\n\n{long_content}\n"
        )
        note = parser.parse_file(note_file)
        chunks = parser.chunk_note(note, max_tokens=300)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content.startswith("note: ")
