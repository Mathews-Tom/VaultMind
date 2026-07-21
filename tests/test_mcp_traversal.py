"""Tests for traversal-first MCP tools: read_frontmatter, list_folder_index, follow_links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.config import VaultConfig
from vaultmind.mcp.profiles import DEFAULT_PROFILES, load_profile
from vaultmind.mcp.server import _dispatch_tool
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Create a fixture vault with frontmatter variety and a nested folder."""
    inbox = tmp_path / "00-inbox"
    inbox.mkdir()
    projects = tmp_path / "02-projects"
    projects.mkdir()
    sub = projects / "sub"
    sub.mkdir()

    (inbox / "fleeting-note.md").write_text(
        "---\n"
        "type: fleeting\n"
        "tags: [test, capture]\n"
        "authority: 2\n"
        "status: active\n"
        "source: manual\n"
        "created: 2026-01-01\n"
        "---\n\n"
        "# Fleeting Note\n\n"
        "First body line as the description fallback.\n\nMore content below.\n",
        encoding="utf-8",
    )
    (projects / "project-alpha.md").write_text(
        "---\n"
        "title: Project Alpha\n"
        "type: project\n"
        "tags: [proj]\n"
        "authority: 4\n"
        "description: Explicit frontmatter description.\n"
        "created: 2026-01-02\n"
        "---\n\n"
        "# Project Alpha\n\n"
        "Body text that must never leak into read_frontmatter or list_folder_index output.\n",
        encoding="utf-8",
    )
    (sub / "nested-note.md").write_text(
        "---\ntype: literature\ntags: [nested]\ncreated: 2026-01-03\n---\n\nNested content.\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def parser(vault_root: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=vault_root))  # type: ignore[arg-type]


class _StubStore:
    pass


class _StubGraph:
    pass


def _dispatch(name: str, args: dict, vault_root: Path, parser: VaultParser) -> dict:
    return _dispatch_tool(
        name,
        args,
        vault_root,
        _StubStore(),  # type: ignore[arg-type]
        _StubGraph(),  # type: ignore[arg-type]
        parser,
    )


# ---------------------------------------------------------------------------
# read_frontmatter
# ---------------------------------------------------------------------------


class TestReadFrontmatter:
    def test_returns_parsed_frontmatter_fields(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch(
            "read_frontmatter", {"path": "02-projects/project-alpha.md"}, vault_root, parser
        )
        assert result["path"] == "02-projects/project-alpha.md"
        assert result["title"] == "Project Alpha"
        assert result["note_type"] == "project"
        assert result["tags"] == ["proj"]
        assert result["authority"] == 4
        assert result["frontmatter"]["description"] == "Explicit frontmatter description."
        assert "created" in result and "modified" in result

    def test_never_returns_body_text(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch(
            "read_frontmatter", {"path": "02-projects/project-alpha.md"}, vault_root, parser
        )
        assert "content" not in result
        assert "body" not in result
        serialized = str(result)
        assert "Body text that must never leak" not in serialized

    def test_missing_note_returns_error(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch("read_frontmatter", {"path": "00-inbox/nope.md"}, vault_root, parser)
        assert "error" in result

    def test_path_traversal_blocked(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch("read_frontmatter", {"path": "../../etc/passwd"}, vault_root, parser)
        assert "error" in result
        assert "not allowed" in result["error"]


class TestReadFrontmatterProfileWiring:
    def test_in_researcher_allowed_tools(self) -> None:
        assert "read_frontmatter" in DEFAULT_PROFILES["researcher"]["allowed_tools"]

    def test_in_planner_allowed_tools(self) -> None:
        assert "read_frontmatter" in DEFAULT_PROFILES["planner"]["allowed_tools"]

    def test_load_profile_includes_read_frontmatter(self) -> None:
        policy = load_profile("researcher")
        assert "read_frontmatter" in policy.allowed_tools

