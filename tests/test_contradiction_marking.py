"""Tests for non-destructive contradiction marking (M6 PR-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import frontmatter

from vaultmind.contradiction.marking import mark_contradicted

if TYPE_CHECKING:
    from pathlib import Path


def _write_note(tmp_path: Path, name: str = "loser.md", body: str = "Original content.\n") -> Path:
    path = tmp_path / name
    path.write_text(f"---\ntitle: {name}\n---\n\n{body}", encoding="utf-8")
    return path


class TestMarkContradictedFirstWrite:
    def test_adds_contradicted_by_frontmatter(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path)

        result = mark_contradicted(path, "winner.md", "Winner Note", "newer note supersedes")

        assert result is True
        post = frontmatter.load(str(path))
        assert post.metadata["contradicted_by"] == ["winner.md"]

    def test_prepends_callout_without_removing_body(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, body="Original content that must survive.\n")

        mark_contradicted(path, "winner.md", "Winner Note", "newer note supersedes")

        post = frontmatter.load(str(path))
        assert "Original content that must survive." in post.content
        assert "> [!warning]" in post.content
        assert "[[Winner Note]]" in post.content
        assert post.content.index("[!warning]") < post.content.index("Original content")

    def test_callout_includes_rationale(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path)

        mark_contradicted(path, "winner.md", "Winner Note", "specific rationale text")

        post = frontmatter.load(str(path))
        assert "specific rationale text" in post.content

    def test_never_touches_other_frontmatter_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "loser.md"
        path.write_text(
            "---\ntitle: Loser\ntags: [foo, bar]\nauthority: 3\n---\n\nBody text.\n",
            encoding="utf-8",
        )

        mark_contradicted(path, "winner.md", "Winner Note", "rationale")

        post = frontmatter.load(str(path))
        assert post.metadata["title"] == "Loser"
        assert post.metadata["tags"] == ["foo", "bar"]
        assert post.metadata["authority"] == 3


class TestMarkContradictedIdempotency:
    def test_same_winner_twice_is_a_noop(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path)

        first = mark_contradicted(path, "winner.md", "Winner Note", "rationale")
        content_after_first = path.read_text(encoding="utf-8")
        second = mark_contradicted(path, "winner.md", "Winner Note", "rationale")
        content_after_second = path.read_text(encoding="utf-8")

        assert first is True
        assert second is False
        assert content_after_first == content_after_second

    def test_different_winners_both_accumulate(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path)

        mark_contradicted(path, "winner-a.md", "Winner A", "rationale a")
        mark_contradicted(path, "winner-b.md", "Winner B", "rationale b")

        post = frontmatter.load(str(path))
        assert post.metadata["contradicted_by"] == ["winner-a.md", "winner-b.md"]
        assert "[[Winner A]]" in post.content
        assert "[[Winner B]]" in post.content


class TestMarkContradictedStringFrontmatterCoercion:
    def test_legacy_string_contradicted_by_is_coerced_to_list(self, tmp_path: Path) -> None:
        path = tmp_path / "loser.md"
        path.write_text(
            "---\ntitle: Loser\ncontradicted_by: old-winner.md\n---\n\nBody.\n",
            encoding="utf-8",
        )

        mark_contradicted(path, "new-winner.md", "New Winner", "rationale")

        post = frontmatter.load(str(path))
        assert post.metadata["contradicted_by"] == ["old-winner.md", "new-winner.md"]
