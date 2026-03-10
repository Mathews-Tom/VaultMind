"""Tests for MCP capture_note tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaultmind.mcp.profiles import DEFAULT_PROFILES, load_profile
from vaultmind.mcp.server import _dispatch_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_store() -> MagicMock:
    store = MagicMock()
    store.index_single_note.return_value = None
    return store


def _make_fake_parser(vault_root: Path) -> MagicMock:
    parser = MagicMock()
    parser.vault_root = vault_root

    def _parse_file(filepath: Path) -> MagicMock:
        note = MagicMock()
        note.path = filepath.relative_to(vault_root)
        note.title = filepath.stem
        return note

    parser.parse_file.side_effect = _parse_file
    return parser


def _make_fake_graph() -> MagicMock:
    return MagicMock()


def _call_capture_note(
    vault_root: Path,
    args: dict,
) -> dict:
    store = _make_fake_store()
    graph = _make_fake_graph()
    parser = _make_fake_parser(vault_root)
    return _dispatch_tool(
        "capture_note",
        args,
        vault_root,
        store,
        graph,
        parser,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureNoteCreatesFile:
    def test_capture_note_creates_file(self, tmp_path: Path) -> None:
        """File is written to vault with correct frontmatter."""
        result = _call_capture_note(
            tmp_path,
            {
                "content": "This is the note body.",
                "title": "My Test Note",
                "tags": ["test", "mcp"],
                "note_type": "fleeting",
                "folder": "00-inbox",
            },
        )

        assert result["status"] == "ok"
        note_path = tmp_path / result["path"]
        assert note_path.exists()

        content = note_path.read_text()
        assert "title: My Test Note" in content
        assert "type: fleeting" in content
        assert "test" in content
        assert "mcp" in content
        assert "source: mcp" in content
        assert "status: active" in content
        assert "This is the note body." in content

    def test_capture_note_frontmatter_structure(self, tmp_path: Path) -> None:
        """Frontmatter delimiters are present and well-formed."""
        result = _call_capture_note(
            tmp_path,
            {
                "content": "Body text here.",
                "title": "Structured Note",
            },
        )

        note_path = tmp_path / result["path"]
        content = note_path.read_text()
        parts = content.split("---")
        # At least 3 parts: empty, frontmatter, body
        assert len(parts) >= 3

    def test_capture_note_indexes_note(self, tmp_path: Path) -> None:
        """index_single_note is called after writing."""
        store = _make_fake_store()
        graph = _make_fake_graph()
        parser = _make_fake_parser(tmp_path)

        _dispatch_tool(
            "capture_note",
            {"content": "Some content", "title": "Indexed Note"},
            tmp_path,
            store,
            graph,
            parser,
        )

        store.index_single_note.assert_called_once()


class TestCaptureNoteCustomFolder:
    def test_capture_note_custom_folder(self, tmp_path: Path) -> None:
        """Non-inbox target folder is respected."""
        result = _call_capture_note(
            tmp_path,
            {
                "content": "Project note content.",
                "title": "Project Note",
                "folder": "02-projects",
            },
        )

        assert result["status"] == "ok"
        assert result["path"].startswith("02-projects/")
        note_path = tmp_path / result["path"]
        assert note_path.exists()

    def test_capture_note_creates_folder_if_missing(self, tmp_path: Path) -> None:
        """Target folder is created if it does not exist."""
        result = _call_capture_note(
            tmp_path,
            {
                "content": "Nested content.",
                "folder": "99-new-folder",
            },
        )

        assert result["status"] == "ok"
        note_path = tmp_path / result["path"]
        assert note_path.exists()


class TestCaptureNoteAutoTitle:
    def test_capture_note_auto_title(self, tmp_path: Path) -> None:
        """Title is generated from first content line when omitted."""
        content = "# My Auto Title\n\nBody of the note."
        result = _call_capture_note(
            tmp_path,
            {"content": content},
        )

        assert result["status"] == "ok"
        assert result["title"] == "My Auto Title"

    def test_capture_note_auto_title_truncated(self, tmp_path: Path) -> None:
        """Auto-generated title is truncated to 50 characters."""
        long_line = "A " * 40  # 80 chars
        result = _call_capture_note(
            tmp_path,
            {"content": long_line},
        )

        assert len(result["title"]) <= 50

    def test_capture_note_auto_title_from_plain_text(self, tmp_path: Path) -> None:
        """First non-empty plain line used as title when no heading present."""
        content = "\nFirst sentence of this note.\n\nMore body."
        result = _call_capture_note(
            tmp_path,
            {"content": content},
        )

        assert result["title"] == "First sentence of this note."

    def test_capture_note_explicit_title_used_when_provided(self, tmp_path: Path) -> None:
        """Explicit title overrides auto-generation."""
        result = _call_capture_note(
            tmp_path,
            {
                "content": "# Ignored Heading\n\nBody.",
                "title": "Explicit Title",
            },
        )

        assert result["title"] == "Explicit Title"


class TestCaptureNoteProfileEnforced:
    def test_capture_note_in_planner_allowed_tools(self) -> None:
        """capture_note appears in planner profile's allowed_tools."""
        assert "capture_note" in DEFAULT_PROFILES["planner"]["allowed_tools"]

    def test_load_planner_profile_includes_capture_note(self) -> None:
        """load_profile returns planner policy that allows capture_note."""
        policy = load_profile("planner")
        assert "capture_note" in policy.allowed_tools

    def test_capture_note_not_in_researcher_profile(self) -> None:
        """capture_note is not in the researcher (read-only) profile."""
        policy = load_profile("researcher")
        assert "capture_note" not in policy.allowed_tools

    def test_full_profile_allows_all_tools(self) -> None:
        """Full profile with wildcard allows capture_note implicitly."""
        policy = load_profile("full")
        assert policy.allows_all_tools

    def test_capture_note_note_type_default_is_fleeting(self, tmp_path: Path) -> None:
        """When note_type is omitted, defaults to fleeting."""
        result = _call_capture_note(
            tmp_path,
            {"content": "Some content without type."},
        )

        note_path = tmp_path / result["path"]
        content = note_path.read_text()
        assert "type: fleeting" in content
