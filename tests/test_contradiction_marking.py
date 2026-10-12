"""Tests for non-destructive contradiction marking (M6 PR-3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import frontmatter
import pytest

from vaultmind.contradiction.marking import mark_contradicted, write_superseded_block

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


class TestWriteSupersededBlockDated:
    def test_dated_callout_includes_iso_date(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md")
        ts = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)

        write_superseded_block(
            path,
            callout_type="superseded",
            title="Deleted via bot",
            rationale="Requested via /delete",
            frontmatter_key="superseded_at",
            frontmatter_value="bot:delete",
            dated=True,
            timestamp=ts,
        )

        post = frontmatter.load(str(path))
        assert "[!superseded] Deleted via bot — 2026-07-22" in post.content

    def test_undated_callout_has_no_date_suffix(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md")

        write_superseded_block(
            path,
            callout_type="warning",
            title="Some title",
            rationale="rationale text",
            frontmatter_key="marker",
            frontmatter_value="v1",
        )

        post = frontmatter.load(str(path))
        assert "[!warning] Some title\n" in post.content


class TestWriteSupersededBlockNewContent:
    def test_new_content_placed_above_callout_and_preserves_old_body(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md", body="Original body that must survive.\n")

        write_superseded_block(
            path,
            callout_type="superseded",
            title="Edited via bot",
            rationale="Instruction: add a summary",
            frontmatter_key="superseded_at",
            frontmatter_value="2026-07-22T12:00:00+00:00",
            dated=True,
            new_content="Edited replacement content.",
        )

        post = frontmatter.load(str(path))
        assert "Edited replacement content." in post.content
        assert "Original body that must survive." in post.content
        assert post.content.index("Edited replacement content.") < post.content.index(
            "[!superseded]"
        )
        assert post.content.index("[!superseded]") < post.content.index(
            "Original body that must survive."
        )

    def test_without_new_content_body_is_unchanged_except_callout(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md", body="Untouched body.\n")

        write_superseded_block(
            path,
            callout_type="superseded",
            title="Deleted via bot",
            rationale="Requested via /delete",
            frontmatter_key="superseded_at",
            frontmatter_value="bot:delete",
            dated=True,
        )

        post = frontmatter.load(str(path))
        assert "Untouched body." in post.content


class TestWriteSupersededBlockIdempotency:
    def test_same_frontmatter_value_twice_is_a_noop(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md")
        kwargs = {
            "callout_type": "superseded",
            "title": "Deleted via bot",
            "rationale": "Requested via /delete",
            "frontmatter_key": "superseded_at",
            "frontmatter_value": "bot:delete",
            "dated": True,
        }

        first = write_superseded_block(path, **kwargs)
        content_after_first = path.read_text()
        second = write_superseded_block(path, **kwargs)
        content_after_second = path.read_text()

        assert first is True
        assert second is False
        assert content_after_first == content_after_second


class TestMarkContradictedUsesSharedWriter:
    def test_mark_contradicted_output_unchanged_by_generalization(self, tmp_path: Path) -> None:
        """Regression guard: extracting `write_superseded_block()` must not
        change `mark_contradicted()`'s callout format, frontmatter key, or
        idempotency contract."""
        path = _write_note(tmp_path, body="Body.\n")

        result = mark_contradicted(path, "winner.md", "Winner Note", "specific rationale text")

        post = frontmatter.load(str(path))
        assert result is True
        assert post.metadata["contradicted_by"] == ["winner.md"]
        assert post.content.startswith("> [!warning] Contradicted by [[Winner Note]]\n")
        assert "— 20" not in post.content  # undated: no ISO-date suffix


TS = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
DELETE_ARGS: dict[str, Any] = {
    "callout_type": "superseded",
    "title": "Deleted via bot",
    "rationale": "Requested via Telegram `/delete` — content preserved below.",
    "frontmatter_key": "superseded_at",
    "frontmatter_value": "bot:delete",
    "dated": True,
    "timestamp": TS,
    "boundary": "delete",
}
EDIT_ARGS: dict[str, Any] = {
    "callout_type": "superseded",
    "title": "Edited via bot",
    "rationale": "Instruction: add a summary",
    "frontmatter_key": "superseded_at",
    "frontmatter_value": TS.isoformat(),
    "dated": True,
    "timestamp": TS,
    "new_content": "Replacement text.",
    "boundary": "edit",
}


def _boundary_id_and_value(path: Path) -> tuple[str, str]:
    records = frontmatter.load(str(path)).metadata["supersede_boundaries"]
    assert len(records) == 1
    boundary_id, value = records[0].split("=", 1)
    return boundary_id, value


class TestBoundaryWrites:
    def test_delete_pairs_an_id_marker_with_metadata_and_keeps_the_prior_body(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "note.md"
        prior = "First paragraph.\n\nSecond paragraph."
        path.write_text(
            f"---\ntitle: Note\ntags: [a, b]\nauthority: 3\n---\n\n{prior}\n", encoding="utf-8"
        )

        assert write_superseded_block(path, **DELETE_ARGS) is True

        post = frontmatter.load(str(path))
        boundary_id, value = _boundary_id_and_value(path)
        assert value == "bot:delete"
        assert post.metadata["superseded_at"] == ["bot:delete"]
        assert post.content.startswith(
            f"<!-- vaultmind:supersede id={boundary_id} -->\n"
            "> [!superseded] Deleted via bot — 2026-07-22\n"
            "> Requested via Telegram `/delete` — content preserved below.\n\n"
        )
        assert post.content.endswith(prior)
        assert (post.metadata["title"], post.metadata["tags"], post.metadata["authority"]) == (
            "Note",
            ["a", "b"],
            3,
        )

    def test_edit_places_the_new_text_above_the_marker_and_preserves_the_prior_body(
        self, tmp_path: Path
    ) -> None:
        path = _write_note(tmp_path, name="note.md", body="Prior body.\n")

        assert write_superseded_block(path, **EDIT_ARGS) is True

        content = frontmatter.load(str(path)).content
        boundary_id, value = _boundary_id_and_value(path)
        assert value == TS.isoformat()
        assert content == (
            "Replacement text.\n\n"
            f"<!-- vaultmind:supersede id={boundary_id} -->\n"
            "> [!superseded] Edited via bot — 2026-07-22\n"
            "> Instruction: add a summary\n\n"
            "Prior body."
        )

    def test_each_event_gets_its_own_unguessable_boundary(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md")
        later = TS + timedelta(minutes=5)

        write_superseded_block(path, **EDIT_ARGS)
        write_superseded_block(
            path, **{**EDIT_ARGS, "frontmatter_value": later.isoformat(), "timestamp": later}
        )

        records = frontmatter.load(str(path)).metadata["supersede_boundaries"]
        ids = [record.split("=", 1)[0] for record in records]
        assert len(ids) == 2
        assert len(set(ids)) == 2
        assert all(len(boundary_id) == 32 for boundary_id in ids)

    def test_multiline_instruction_stays_inside_one_callout(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md", body="Prior body.\n")

        write_superseded_block(path, **{**EDIT_ARGS, "rationale": "Instruction: one\n\ntwo\nthree"})

        content = frontmatter.load(str(path)).content
        assert (
            "> [!superseded] Edited via bot — 2026-07-22\n"
            "> Instruction: one\n>\n> two\n> three\n\n"
            "Prior body."
        ) in content

    def test_repeating_a_delete_is_a_noop(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path, name="note.md")

        assert write_superseded_block(path, **DELETE_ARGS) is True
        after_first = path.read_text(encoding="utf-8")
        assert write_superseded_block(path, **DELETE_ARGS) is False

        assert path.read_text(encoding="utf-8") == after_first

    def test_contradiction_marking_never_writes_a_boundary(self, tmp_path: Path) -> None:
        path = _write_note(tmp_path)

        mark_contradicted(path, "winner.md", "Winner Note", "rationale")

        post = frontmatter.load(str(path))
        assert "supersede_boundaries" not in post.metadata
        assert "<!-- vaultmind" not in post.content


class TestBoundaryWriteRejections:
    @pytest.mark.parametrize(
        "override",
        [
            {"frontmatter_key": "contradicted_by"},
            {"frontmatter_value": "bot:other"},
            {"new_content": "text that a delete must not carry"},
        ],
        ids=["wrong-key", "wrong-marker-value", "delete-with-new-content"],
    )
    def test_invalid_delete_arguments_leave_the_file_untouched(
        self, tmp_path: Path, override: dict[str, Any]
    ) -> None:
        path = _write_note(tmp_path, name="note.md")
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(ValueError, match=r"."):
            write_superseded_block(path, **{**DELETE_ARGS, **override})

        assert path.read_text(encoding="utf-8") == raw

    @pytest.mark.parametrize(
        "override",
        [
            {"new_content": None},
            {"timestamp": None},
            {"timestamp": datetime(2026, 7, 22, 12, 0)},
            {"timestamp": datetime(2026, 7, 22, 14, 0, tzinfo=timezone(timedelta(hours=2)))},
            {"frontmatter_value": "2026-07-22T12:00:01+00:00"},
            {"frontmatter_value": "yesterday"},
        ],
        ids=[
            "no-new-content",
            "no-timestamp",
            "naive-timestamp",
            "non-utc-timestamp",
            "stamp-disagrees-with-timestamp",
            "not-a-stamp",
        ],
    )
    def test_invalid_edit_arguments_leave_the_file_untouched(
        self, tmp_path: Path, override: dict[str, Any]
    ) -> None:
        path = _write_note(tmp_path, name="note.md")
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(ValueError, match=r"."):
            write_superseded_block(path, **{**EDIT_ARGS, **override})

        assert path.read_text(encoding="utf-8") == raw
