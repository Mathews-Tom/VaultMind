"""Tests for vault path traversal protection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.vault.security import PathTraversalError, validate_vault_path


@pytest.fixture()
def vault_root(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "02-projects").mkdir()
    (vault / "02-projects" / "vaultmind.md").write_text("# VaultMind")
    (vault / "subdir").mkdir()
    (vault / "subdir" / "note.md").write_text("# Note")
    return vault


class TestValidateVaultPath:
    def test_valid_relative_path(self, vault_root: Path) -> None:
        result = validate_vault_path("02-projects/vaultmind.md", vault_root)
        assert result == (vault_root / "02-projects" / "vaultmind.md").resolve()

    def test_valid_subdirectory(self, vault_root: Path) -> None:
        result = validate_vault_path("subdir/note.md", vault_root)
        assert result == (vault_root / "subdir" / "note.md").resolve()

    def test_empty_string_returns_vault_root(self, vault_root: Path) -> None:
        result = validate_vault_path("", vault_root)
        assert result == vault_root.resolve()

    def test_parent_traversal_etc_passwd(self, vault_root: Path) -> None:
        with pytest.raises(PathTraversalError) as exc_info:
            validate_vault_path("../../etc/passwd", vault_root)
        assert exc_info.value.user_path == "../../etc/passwd"
        assert exc_info.value.vault_root == vault_root

    def test_parent_traversal_dotenv(self, vault_root: Path) -> None:
        with pytest.raises(PathTraversalError):
            validate_vault_path("../../../.env", vault_root)

    def test_embedded_parent_traversal(self, vault_root: Path) -> None:
        with pytest.raises(PathTraversalError):
            validate_vault_path("subdir/../../escape", vault_root)

    def test_absolute_path_outside_vault(self, vault_root: Path) -> None:
        with pytest.raises(PathTraversalError):
            validate_vault_path("/etc/passwd", vault_root)

    def test_dot_current_dir_is_valid(self, vault_root: Path) -> None:
        result = validate_vault_path(".", vault_root)
        assert result == vault_root.resolve()

    def test_normal_nested_path(self, vault_root: Path) -> None:
        result = validate_vault_path("02-projects/vaultmind.md", vault_root)
        assert result.is_relative_to(vault_root.resolve())

    def test_error_is_value_error_subclass(self) -> None:
        assert issubclass(PathTraversalError, ValueError)
