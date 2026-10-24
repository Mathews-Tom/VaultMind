"""Tests for vault path traversal protection."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import pytest

from vaultmind.vault.security import (
    PathResolutionError,
    PathTraversalError,
    ReservedNamespaceError,
    validate_ordinary_output_path,
    validate_ordinary_write_path,
    validate_vault_path,
)


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


# ---------------------------------------------------------------------------
# Structural resolution boundary — real filesystem, no stubbed resolve()
# ---------------------------------------------------------------------------

PRIVATE_DIR = "private-store-9c1e"
PRIVATE_FILE = "classified-9c1e.md"


def _rendered(exc: BaseException) -> str:
    """Everything a handler could surface or log for this exception."""
    return f"{exc}|{exc!r}|{exc.args}|" + "".join(traceback.format_exception(exc))


def _assert_no_private_leak(exc: BaseException) -> None:
    rendered = _rendered(exc)
    assert PRIVATE_DIR not in rendered
    assert PRIVATE_FILE not in rendered


@pytest.fixture()
def private_dir(tmp_path: Path) -> Path:
    """A directory outside the vault holding a file the vault must never reveal."""
    private = tmp_path / PRIVATE_DIR
    private.mkdir()
    (private / PRIVATE_FILE).write_text("secret")
    # Neutral-named alias so callers can address the secret without echoing its real name.
    (private / "public.md").symlink_to(private / PRIVATE_FILE)
    return private


@pytest.fixture()
def aliased_vault(vault_root: Path, private_dir: Path) -> Path:
    """Vault with in-vault and escaping symlink aliases."""
    projects = vault_root / "02-projects"
    (projects / "public-alias.md").symlink_to(projects / "vaultmind.md")
    (projects / "private-file-alias.md").symlink_to(private_dir / PRIVATE_FILE)
    (projects / "private-dir-alias").symlink_to(private_dir, target_is_directory=True)
    return vault_root


class TestFileChildFallback:
    """A path below a regular file cannot resolve strictly; it must behave like missing."""

    def test_child_of_in_vault_file_resolves_inside_vault(self, vault_root: Path) -> None:
        result = validate_vault_path("02-projects/vaultmind.md/child", vault_root)
        assert result == (vault_root / "02-projects" / "vaultmind.md" / "child").resolve()
        assert not result.exists()
        assert not result.is_file()

    def test_child_of_in_vault_alias_to_in_vault_file_resolves_to_target(
        self, aliased_vault: Path
    ) -> None:
        result = validate_vault_path("02-projects/public-alias.md/child", aliased_vault)
        target = (aliased_vault / "02-projects" / "vaultmind.md").resolve()
        assert result == target / "child"

    @pytest.mark.parametrize("suffix", ["", "/x", "/x/y"])
    def test_alias_to_private_file_is_denied_without_leaking_target(
        self, aliased_vault: Path, suffix: str
    ) -> None:
        with pytest.raises(PathTraversalError) as exc_info:
            validate_vault_path(f"02-projects/private-file-alias.md{suffix}", aliased_vault)
        _assert_no_private_leak(exc_info.value)

    @pytest.mark.parametrize("suffix", ["/missing.md", "/public.md/x"])
    def test_alias_to_outside_dir_is_denied_without_leaking_target(
        self, aliased_vault: Path, suffix: str
    ) -> None:
        with pytest.raises(PathTraversalError) as exc_info:
            validate_vault_path(f"02-projects/private-dir-alias{suffix}", aliased_vault)
        _assert_no_private_leak(exc_info.value)


class TestOrdinaryStrictResolutionFailures:
    def test_missing_file_resolves_to_would_be_path(self, vault_root: Path) -> None:
        result = validate_vault_path("02-projects/missing.md", vault_root)
        assert result == (vault_root / "02-projects" / "missing.md").resolve()
        assert not result.exists()

    def test_missing_nested_directories_resolve_inside_vault(self, vault_root: Path) -> None:
        result = validate_vault_path("nope/deeper/missing.md", vault_root)
        assert result == (vault_root / "nope" / "deeper" / "missing.md").resolve()

    def test_overlong_component_resolves_inside_vault(self, vault_root: Path) -> None:
        name = "a" * 300 + ".md"
        result = validate_vault_path(f"02-projects/{name}", vault_root)
        assert result == (vault_root / "02-projects").resolve() / name

    def test_overlong_component_below_outside_alias_is_denied_without_leak(
        self, aliased_vault: Path
    ) -> None:
        name = "a" * 300 + ".md"
        with pytest.raises(PathTraversalError) as exc_info:
            validate_vault_path(f"02-projects/private-dir-alias/{name}", aliased_vault)
        _assert_no_private_leak(exc_info.value)

    def test_unreadable_in_vault_directory_resolves_inside_vault(self, vault_root: Path) -> None:
        locked = vault_root / "locked"
        locked.mkdir()
        (locked / "note.md").write_text("hidden")
        locked.chmod(0o000)
        try:
            if os.access(locked, os.R_OK | os.X_OK):
                pytest.skip("directory permissions are not enforced for this user")
            result = validate_vault_path("locked/note.md", vault_root)
            assert result == (vault_root / "locked" / "note.md").resolve()
        finally:
            locked.chmod(0o700)

    def test_unreadable_outside_directory_behind_alias_is_denied_without_leak(
        self, aliased_vault: Path, private_dir: Path
    ) -> None:
        private_dir.chmod(0o000)
        try:
            if os.access(private_dir, os.R_OK | os.X_OK):
                pytest.skip("directory permissions are not enforced for this user")
            with pytest.raises(PathTraversalError) as exc_info:
                validate_vault_path("02-projects/private-dir-alias/public.md", aliased_vault)
            _assert_no_private_leak(exc_info.value)
        finally:
            private_dir.chmod(0o700)


class TestSymlinkLoops:
    @pytest.fixture()
    def looped_vault(self, vault_root: Path) -> Path:
        (vault_root / "loop-a").symlink_to(vault_root / "loop-b")
        (vault_root / "loop-b").symlink_to(vault_root / "loop-a")
        (vault_root / "self-loop").symlink_to(vault_root / "self-loop")
        return vault_root

    @pytest.mark.parametrize(
        "user_path",
        ["loop-a", "loop-b/x", "self-loop", "self-loop/x/y"],
    )
    def test_loop_raises_typed_error_carrying_only_caller_input(
        self, looped_vault: Path, user_path: str
    ) -> None:
        with pytest.raises(PathResolutionError) as exc_info:
            validate_vault_path(user_path, looped_vault)
        exc = exc_info.value
        assert exc.user_path == user_path
        assert isinstance(exc, ValueError)
        assert exc.__cause__ is None
        assert exc.__suppress_context__
        assert str(looped_vault) not in _rendered(exc)

    def test_loop_is_not_a_containment_violation(self, looped_vault: Path) -> None:
        with pytest.raises(PathResolutionError) as exc_info:
            validate_vault_path("loop-a", looped_vault)
        assert not isinstance(exc_info.value, PathTraversalError)

    def test_loop_does_not_disturb_neighbouring_valid_paths(self, looped_vault: Path) -> None:
        result = validate_vault_path("subdir/note.md", looped_vault)
        assert result == (looped_vault / "subdir" / "note.md").resolve()

    def test_containment_denial_still_wins_for_escaping_loop_free_path(
        self, looped_vault: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            validate_vault_path("../outside.md", looped_vault)


@pytest.mark.parametrize("name", ["wiki", "_wiki", "WiKi", "_WiKi"])
@pytest.mark.parametrize("suffix", ["", "/source.md", "/metadata.json", "/image.png", "/state.db"])
def test_ordinary_writes_reject_reserved_roots(vault_root: Path, name: str, suffix: str) -> None:
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path(name + suffix, vault_root)
    assert not (vault_root / name).exists()


def test_reserved_errors_redact_requested_filename_and_body(vault_root: Path) -> None:
    sensitive = "private-body-with-newline\nand-token"
    with pytest.raises(ReservedNamespaceError) as error:
        validate_ordinary_write_path(f"_wiki/{sensitive}", vault_root)
    assert sensitive not in _rendered(error.value)
    assert str(vault_root) not in _rendered(error.value)


@pytest.mark.parametrize("name", ["wiki", "_wiki"])
def test_ordinary_writes_reject_lexical_reserved_escape(vault_root: Path, name: str) -> None:
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path(f"{name}/../ordinary.json", vault_root)


@pytest.mark.parametrize("name", ["wiki", "_wiki"])
def test_ordinary_alias_into_reserved_tree_is_denied(vault_root: Path, name: str) -> None:
    reserved = vault_root / name
    reserved.mkdir()
    original = reserved / "source.json"
    original.write_bytes(b'{"immutable":true}')
    (vault_root / "ordinary").symlink_to(reserved, target_is_directory=True)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path("ordinary/source.json", vault_root)
    assert original.read_bytes() == b'{"immutable":true}'
    # The general containment API remains available to authorized wiki writers.
    assert validate_vault_path(f"{name}/source.json", vault_root) == original


def test_reserved_tree_symlink_cannot_launder_ordinary_output(
    vault_root: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "retained"
    destination.mkdir()
    (vault_root / "_wiki").symlink_to(destination, target_is_directory=True)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path("_wiki/new.json", vault_root)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_output_path(destination / "new.json", vault_root)
    assert not (destination / "new.json").exists()


def test_checked_ordinary_path_resolves_safe_alias(vault_root: Path, tmp_path: Path) -> None:
    (vault_root / "ordinary").symlink_to(vault_root / "subdir", target_is_directory=True)
    alias_root = tmp_path / "vault-alias"
    alias_root.symlink_to(vault_root, target_is_directory=True)
    assert validate_ordinary_write_path("ordinary/state.json", alias_root) == (
        vault_root / "subdir" / "state.json"
    )
    assert validate_ordinary_write_path("subdir/wiki/image.png", vault_root) == (
        vault_root / "subdir" / "wiki" / "image.png"
    )


def test_ordinary_containment_and_loop_failures_remain_closed(vault_root: Path) -> None:
    with pytest.raises(PathTraversalError):
        validate_ordinary_write_path("../outside.json", vault_root)
    (vault_root / "cycle").symlink_to(vault_root / "cycle")
    with pytest.raises(PathResolutionError):
        validate_ordinary_write_path("cycle/data.json", vault_root)
    with pytest.raises(PathResolutionError):
        validate_ordinary_output_path(vault_root / "cycle" / "data.json", vault_root)


def test_external_outputs_are_allowed_but_aliases_into_wiki_are_not(
    vault_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.chdir(external)
    assert validate_ordinary_output_path(Path("state.db"), vault_root) == external / "state.db"
    reserved = vault_root / "_wiki"
    reserved.mkdir()
    (external / "alias").symlink_to(reserved, target_is_directory=True)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_output_path(external / "alias" / "state.db", vault_root)
    assert not (reserved / "state.db").exists()


def test_absolute_case_alias_output_is_reserved(vault_root: Path) -> None:
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_output_path(vault_root / "WIKI" / "export.json", vault_root)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path(str(vault_root / "_WIKI" / "image.png"), vault_root)


def test_lexical_reserved_request_through_separate_vault_alias_is_denied(
    vault_root: Path, tmp_path: Path
) -> None:
    alias = tmp_path / "selected-vault-alias"
    alias.symlink_to(vault_root, target_is_directory=True)
    request = alias / "_wiki" / ".." / "ordinary.json"
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_output_path(request, vault_root)
    with pytest.raises(ReservedNamespaceError):
        validate_ordinary_write_path(str(request), vault_root)
    assert not (vault_root / "ordinary.json").exists()
