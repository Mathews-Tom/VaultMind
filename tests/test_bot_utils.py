"""Tests for bot handler helpers (`bot/handlers/utils.py`) — note path resolution."""

from __future__ import annotations

import errno
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from vaultmind.vault.security import PathTraversalError

if TYPE_CHECKING:
    from pathlib import Path


def _make_ctx(vault: Path, *, search_note_path: str | None = None) -> Any:
    from vaultmind.bot.handlers.context import HandlerContext

    ctx = MagicMock(spec=HandlerContext)
    ctx.vault_root = vault
    ctx.store = MagicMock()
    if search_note_path is None:
        ctx.store.search = MagicMock(return_value=[])
    else:
        ctx.store.search = MagicMock(
            return_value=[{"metadata": {"note_path": search_note_path}, "distance": 0.1}]
        )
    return ctx


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "02-projects").mkdir()
    (root / "02-projects" / "hub.md").write_text("# Hub")
    (root / "02-projects" / "target.md").write_text("# Target")
    (root / "loop-a").symlink_to(root / "loop-b")
    (root / "loop-b").symlink_to(root / "loop-a")
    return root


@pytest.fixture()
def private_file(tmp_path: Path) -> Path:
    private = tmp_path / "private-store"
    private.mkdir()
    secret = private / "classified.md"
    secret.write_text("secret")
    return secret


class TestResolveNotePathDirectMatches:
    def test_exact_relative_path(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault)
        assert (
            _resolve_note_path(ctx, "02-projects/hub.md")
            == (vault / "02-projects" / "hub.md").resolve()
        )

    def test_path_without_md_extension(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault)
        assert (
            _resolve_note_path(ctx, "02-projects/hub")
            == (vault / "02-projects" / "hub.md").resolve()
        )

    def test_unknown_query_without_search_hit_returns_none(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault)
        assert _resolve_note_path(ctx, "nothing-like-this") is None
        ctx.store.search.assert_called_once()


class TestResolveNotePathUnresolvableCandidates:
    @pytest.mark.parametrize("strategy", ["query", "filename", "semantic"])
    def test_unreadable_alias_does_not_abort_lookup(self, vault: Path, strategy: str) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        locked = vault / "locked"
        locked.mkdir()
        hidden = locked / "hidden.md"
        hidden.write_text("# Inaccessible")
        alias = vault / "02-projects" / "unreadable-note.md"
        alias.symlink_to(hidden)
        query = {
            "query": "02-projects/unreadable-note.md",
            "filename": "unreadable-note",
            "semantic": "no-filename-match",
        }[strategy]
        ctx = _make_ctx(
            vault,
            search_note_path=(
                "02-projects/unreadable-note.md"
                if strategy == "semantic"
                else "02-projects/target.md"
            ),
        )
        locked.chmod(0)
        try:
            result = _resolve_note_path(ctx, query)
        finally:
            locked.chmod(0o700)

        expected = None if strategy == "semantic" else (vault / "02-projects/target.md").resolve()
        assert result == expected

    @pytest.mark.parametrize("strategy", ["query", "filename"])
    def test_overlong_candidate_falls_through_to_semantic_search(
        self, vault: Path, strategy: str
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        too_long = "x" * 300
        (vault / "02-projects" / "overlong-note.md").symlink_to(vault / too_long)
        query = too_long if strategy == "query" else "overlong-note"
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")

        assert _resolve_note_path(ctx, query) == (vault / "02-projects/target.md").resolve()

    def test_unexpected_stat_failure_is_not_suppressed(
        self, vault: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        def fail_stat(path: Path) -> bool:
            raise OSError(errno.EIO, "fixture I/O failure")

        monkeypatch.setattr(type(vault), "is_file", fail_stat)
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        with pytest.raises(OSError) as caught:
            _resolve_note_path(ctx, "02-projects/hub.md")
        assert caught.value.errno == errno.EIO

    def test_nondirectory_query_falls_through_to_semantic_search(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        result = _resolve_note_path(ctx, "02-projects/hub.md/child")

        assert result == (vault / "02-projects" / "target.md").resolve()
        ctx.store.search.assert_called_once()

    def test_nondirectory_query_without_search_hit_is_not_found(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault)
        assert _resolve_note_path(ctx, "02-projects/hub.md/child") is None

    def test_symlink_loop_query_falls_through_to_semantic_search(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        result = _resolve_note_path(ctx, "loop-a")

        assert result == (vault / "02-projects" / "target.md").resolve()

    def test_md_symlink_loop_filename_match_falls_through_to_semantic_search(
        self, vault: Path
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        (vault / "cycle-a.md").symlink_to(vault / "cycle-b.md")
        (vault / "cycle-b.md").symlink_to(vault / "cycle-a.md")
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")

        result = _resolve_note_path(ctx, "cycle-a")

        assert result == (vault / "02-projects" / "target.md").resolve()
        ctx.store.search.assert_called_once()

    def test_md_symlink_loop_does_not_hide_later_valid_filename_match(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        (vault / "cycle-a.md").symlink_to(vault / "cycle-b.md")
        (vault / "cycle-b.md").symlink_to(vault / "cycle-a.md")
        (vault / "02-projects" / "cycle-notes.md").write_text("# Cycle notes")
        ctx = _make_ctx(vault)

        result = _resolve_note_path(ctx, "cycle")

        assert result == (vault / "02-projects" / "cycle-notes.md").resolve()

    @pytest.mark.parametrize("bad_candidate", ["loop-a", "02-projects/hub.md/child", "missing.md"])
    def test_invalid_search_candidate_is_not_found_rather_than_raising(
        self, vault: Path, bad_candidate: str
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault, search_note_path=bad_candidate)
        assert _resolve_note_path(ctx, "no-filename-match") is None


class TestResolveNotePathPreservesDenials:
    def test_parent_traversal_query_is_denied(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        with pytest.raises(PathTraversalError):
            _resolve_note_path(ctx, "../outside.md")
        ctx.store.search.assert_not_called()

    def test_alias_to_outside_file_query_is_denied(self, vault: Path, private_file: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        (vault / "02-projects" / "alias.md").symlink_to(private_file)
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        with pytest.raises(PathTraversalError):
            _resolve_note_path(ctx, "02-projects/alias.md")
        ctx.store.search.assert_not_called()

    def test_file_child_of_outside_alias_query_is_denied(
        self, vault: Path, private_file: Path
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        (vault / "02-projects" / "alias.md").symlink_to(private_file)
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        with pytest.raises(PathTraversalError):
            _resolve_note_path(ctx, "02-projects/alias.md/child")
        ctx.store.search.assert_not_called()

    def test_filename_match_aliasing_outside_file_is_denied(
        self, vault: Path, private_file: Path
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        (vault / "02-projects" / "mirror.md").symlink_to(private_file)
        ctx = _make_ctx(vault, search_note_path="02-projects/target.md")
        with pytest.raises(PathTraversalError):
            _resolve_note_path(ctx, "mirror")
        ctx.store.search.assert_not_called()

    def test_search_candidate_outside_vault_is_denied(
        self, vault: Path, private_file: Path
    ) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault, search_note_path=str(private_file))
        with pytest.raises(PathTraversalError):
            _resolve_note_path(ctx, "no-filename-match")

    def test_unexpected_search_failure_is_not_suppressed(self, vault: Path) -> None:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        ctx = _make_ctx(vault)
        ctx.store.search = MagicMock(side_effect=RuntimeError("store down"))
        with pytest.raises(RuntimeError):
            _resolve_note_path(ctx, "no-filename-match")
