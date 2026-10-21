"""Tests for MCP capture_note tool."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.config import VaultConfig
from vaultmind.mcp.auth import ProfileEnforcer, ProfileError, ProfilePolicy
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_store() -> MagicMock:
    store = MagicMock()
    store.index_single_note.return_value = None
    return store


def _make_parser(vault_root: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=vault_root))  # type: ignore[arg-type]


def _make_fake_graph() -> MagicMock:
    return MagicMock()


def _full_enforcer(vault_root: Path) -> ProfileEnforcer:
    return ProfileEnforcer(load_profile("full"), vault_root)


def _call_capture_note(
    vault_root: Path,
    args: dict,
) -> dict:
    store = _make_fake_store()
    graph = _make_fake_graph()
    parser = _make_parser(vault_root)
    return _dispatch_tool(
        "capture_note",
        args,
        vault_root,
        store,
        graph,
        parser,
        enforcer=_full_enforcer(vault_root),
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


def _capture_policy(
    root: Path,
    *,
    folders: tuple[str, ...] = ("02-projects",),
    write: bool = True,
    limit: int | None = None,
) -> ProfileEnforcer:
    return ProfileEnforcer(
        ProfilePolicy(
            name="capture-test",
            description="Capture boundary",
            allowed_tools=frozenset({"*"}),
            folder_scope=folders,
            write_enabled=write,
            max_note_size_bytes=limit,
        ),
        root,
    )


def _capture_dispatch(
    tool: str, args: dict, root: Path, scope: ProfileEnforcer, store: MagicMock
) -> dict:
    return _dispatch_tool(
        tool,
        args,
        root,
        store,
        _make_fake_graph(),
        _make_parser(root),
        enforcer=scope,
    )


class TestCaptureMutationBoundary:
    @pytest.mark.parametrize("tool", ["capture", "capture_note"])
    def test_forbidden_destination_is_denied_without_side_effects(
        self, tmp_path: Path, tool: str
    ) -> None:
        store = _make_fake_store()
        with pytest.raises(ProfileError):
            _capture_dispatch(tool, {"content": "body"}, tmp_path, _capture_policy(tmp_path), store)
        assert list(tmp_path.iterdir()) == []
        store.index_single_note.assert_not_called()

    @pytest.mark.parametrize("tool", ["capture", "capture_note"])
    def test_write_disabled_is_denied_without_side_effects(self, tmp_path: Path, tool: str) -> None:
        store = _make_fake_store()
        with pytest.raises(ProfileError):
            _capture_dispatch(
                tool,
                {"content": "body"},
                tmp_path,
                _capture_policy(tmp_path, folders=("*",), write=False),
                store,
            )
        assert list(tmp_path.iterdir()) == []
        store.index_single_note.assert_not_called()

    @pytest.mark.parametrize("tool", ["capture", "capture_note"])
    @pytest.mark.parametrize("profile", ["restricted", "wildcard"])
    def test_symlink_escape_is_denied_before_creating_outside_directory(
        self, tmp_path: Path, tool: str, profile: str
    ) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        outside = tmp_path / "outside-not-created"
        (root / "00-inbox").symlink_to(outside, target_is_directory=True)
        store = _make_fake_store()
        folders = ("*",) if profile == "wildcard" else ("00-inbox",)
        with pytest.raises(ProfileError):
            _capture_dispatch(
                tool,
                {"content": "body"},
                root,
                _capture_policy(root, folders=folders),
                store,
            )
        assert not outside.exists()
        assert sorted(p.name for p in root.iterdir()) == ["00-inbox"]
        store.index_single_note.assert_not_called()

    @pytest.mark.parametrize("tool", ["capture", "capture_note"])
    @pytest.mark.parametrize(
        "args",
        [
            {"content": "é" * 200},
            {"content": "x", "title": "T" * 400},
            {"content": "x", "tags": ["tag" * 150]},
            {"content": "x"},
        ],
    )
    def test_serialized_size_is_checked_before_mkdir_write_or_index(
        self, tmp_path: Path, tool: str, args: dict
    ) -> None:
        store = _make_fake_store()
        # Even the one-byte body exceeds this limit after required frontmatter is serialized.
        with pytest.raises(ProfileError):
            _capture_dispatch(
                tool,
                args,
                tmp_path,
                _capture_policy(tmp_path, folders=("*",), limit=64),
                store,
            )
        assert list(tmp_path.iterdir()) == []
        store.index_single_note.assert_not_called()

    @pytest.mark.parametrize("folder", ["04-private/new", "../outside", "/outside"])
    def test_capture_note_explicit_destination_denial(self, tmp_path: Path, folder: str) -> None:
        store = _make_fake_store()
        with pytest.raises(ProfileError):
            _capture_dispatch(
                "capture_note",
                {"content": "body", "folder": folder},
                tmp_path,
                _capture_policy(tmp_path),
                store,
            )
        assert list(tmp_path.iterdir()) == []
        store.index_single_note.assert_not_called()

    @pytest.mark.parametrize("tool", ["capture", "capture_note"])
    @pytest.mark.parametrize("folders", [("00-inbox",), ("*",)])
    def test_allowed_capture_remains_positive(
        self, tmp_path: Path, tool: str, folders: tuple[str, ...]
    ) -> None:
        store = _make_fake_store()
        result = _capture_dispatch(
            tool,
            {"content": "Allowed body", "title": "Allowed Capture"},
            tmp_path,
            _capture_policy(tmp_path, folders=folders, limit=50000),
            store,
        )
        assert result["status"] == "ok"
        path = tmp_path / result["path"]
        assert path.is_file()
        assert "Allowed body" in path.read_text(encoding="utf-8")
        note = _make_parser(tmp_path).parse_file(path)
        assert note.path.as_posix() == result["path"]

    @pytest.mark.parametrize("title", ["Q3/Q4 planning", "x/../../02-projects/y"])
    @pytest.mark.parametrize("folders", [("00-inbox",), ("*",)])
    def test_capture_title_cannot_change_the_inbox_destination(
        self, tmp_path: Path, title: str, folders: tuple[str, ...]
    ) -> None:
        result = _capture_dispatch(
            "capture",
            {"content": "Captured body", "title": title},
            tmp_path,
            _capture_policy(tmp_path, folders=folders),
            _make_fake_store(),
        )
        path = tmp_path / result["path"]
        assert path.parent == tmp_path / "00-inbox"
        content = path.read_text(encoding="utf-8")
        assert f"# {title}\n" in content
        assert "Captured body" in content

    def test_capture_note_can_create_allowed_nested_folder(self, tmp_path: Path) -> None:
        store = _make_fake_store()
        result = _capture_dispatch(
            "capture_note",
            {"content": "body", "folder": "02-projects/new/nested"},
            tmp_path,
            _capture_policy(tmp_path),
            store,
        )
        assert result["status"] == "ok"
        assert (tmp_path / result["path"]).is_file()

    @pytest.mark.parametrize("absolute", [False, True])
    def test_wildcard_capture_note_rejects_explicit_vault_escape(
        self, tmp_path: Path, absolute: bool
    ) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        outside = tmp_path / "outside"
        folder = str(outside / "new") if absolute else "../outside/new"
        store = _make_fake_store()
        with pytest.raises(ProfileError):
            _capture_dispatch(
                "capture_note",
                {"content": "body", "folder": folder},
                root,
                _capture_policy(root, folders=("*",)),
                store,
            )
        assert not outside.exists()
        assert list(root.iterdir()) == []
        store.index_single_note.assert_not_called()
