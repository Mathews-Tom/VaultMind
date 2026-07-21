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


# ---------------------------------------------------------------------------
# list_folder_index
# ---------------------------------------------------------------------------


class TestListFolderIndex:
    def test_folder_index_lists_titles_and_descriptions(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "02-projects"}, vault_root, parser)
        by_path = {entry["path"]: entry for entry in result["notes"]}
        alpha = by_path["02-projects/project-alpha.md"]
        assert alpha["title"] == "Project Alpha"
        assert alpha["description"] == "Explicit frontmatter description."
        assert alpha["note_type"] == "project"
        assert "content" not in alpha
        assert "Body text that must never leak" not in str(alpha)

    def test_folder_index_description_falls_back_to_first_body_line(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "00-inbox"}, vault_root, parser)
        entry = result["notes"][0]
        assert entry["description"] == "First body line as the description fallback."

    def test_folder_index_recursive_listing_includes_subfolders(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "02-projects"}, vault_root, parser)
        paths = {entry["path"] for entry in result["notes"]}
        assert "02-projects/sub/nested-note.md" in paths
        assert result["count"] == len(paths)

    def test_folder_index_default_folder_lists_whole_vault(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {}, vault_root, parser)
        assert result["count"] >= 3

    def test_folder_index_not_found_returns_error(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "99-missing"}, vault_root, parser)
        assert "error" in result

    def test_folder_index_path_traversal_blocked(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "../../etc"}, vault_root, parser)
        assert "error" in result
        assert "not allowed" in result["error"]


class TestFolderIndexProfileWiring:
    def test_folder_index_in_researcher_allowed_tools(self) -> None:
        assert "list_folder_index" in DEFAULT_PROFILES["researcher"]["allowed_tools"]

    def test_folder_index_in_planner_allowed_tools(self) -> None:
        assert "list_folder_index" in DEFAULT_PROFILES["planner"]["allowed_tools"]


# ---------------------------------------------------------------------------
# follow_links
# ---------------------------------------------------------------------------


@pytest.fixture
def linked_vault_root(tmp_path: Path) -> Path:
    """Create a fixture vault with wikilinked notes for follow_links tests."""
    notes_dir = tmp_path / "02-projects"
    notes_dir.mkdir(parents=True)

    (notes_dir / "hub.md").write_text(
        "---\ntitle: Hub Note\ntype: permanent\ncreated: 2026-01-01\n---\n\n"
        "# Hub Note\n\n"
        "Links to [[Spoke A]], [[Spoke B]], and [[Missing Target]].\n",
        encoding="utf-8",
    )
    (notes_dir / "spoke-a.md").write_text(
        "---\ntitle: Spoke A\ntype: permanent\ncreated: 2026-01-02\n---\n\n"
        "# Spoke A\n\nLinks back to [[Hub Note]].\n",
        encoding="utf-8",
    )
    (notes_dir / "spoke-b.md").write_text(
        "---\ntitle: Spoke B\ntype: permanent\ncreated: 2026-01-03\n---\n\n"
        "# Spoke B\n\nNo backlink to the hub.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def linked_parser(linked_vault_root: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=linked_vault_root))  # type: ignore[arg-type]


class TestFollowLinks:
    def test_follow_links_returns_out_links(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        assert result["title"] == "Hub Note"
        out_titles = {link["title"] for link in result["out_links"]}
        assert out_titles == {"Spoke A", "Spoke B", "Missing Target"}
        assert result["out_link_count"] == 3

    def test_follow_links_resolves_out_link_paths(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        by_title = {link["title"]: link["path"] for link in result["out_links"]}
        assert by_title["Spoke A"] == "02-projects/spoke-a.md"
        assert by_title["Missing Target"] is None

    def test_follow_links_returns_backlinks(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        backlink_titles = {link["title"] for link in result["backlinks"]}
        assert backlink_titles == {"Spoke A"}
        assert result["backlink_count"] == 1
        backlink = result["backlinks"][0]
        assert backlink["path"] == "02-projects/spoke-a.md"

    def test_follow_links_never_returns_body_text(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        assert "content" not in result
        assert "No backlink to the hub" not in str(result)

    def test_follow_links_missing_note_returns_error(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/nope.md"}, linked_vault_root, linked_parser
        )
        assert "error" in result

    def test_follow_links_path_traversal_blocked(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "../../etc/passwd"}, linked_vault_root, linked_parser
        )
        assert "error" in result
        assert "not allowed" in result["error"]


class TestFollowLinksProfileWiring:
    def test_follow_links_in_researcher_allowed_tools(self) -> None:
        assert "follow_links" in DEFAULT_PROFILES["researcher"]["allowed_tools"]

    def test_follow_links_in_planner_allowed_tools(self) -> None:
        assert "follow_links" in DEFAULT_PROFILES["planner"]["allowed_tools"]
