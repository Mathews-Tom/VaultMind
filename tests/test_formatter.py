"""Tests for TelegramFormatter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from vaultmind.bot.formatter import TelegramFormatter, _escape
from vaultmind.vault.models import Note, NoteType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(
    content: str = "",
    title: str = "Test Note",
    path: str = "inbox/test-note.md",
    tags: list[str] | None = None,
    note_type: NoteType = NoteType.FLEETING,
    frontmatter: dict | None = None,
    created: datetime | None = None,
) -> Note:
    return Note(
        path=Path(path),
        title=title,
        content=content,
        note_type=note_type,
        tags=tags or [],
        frontmatter=frontmatter or {},
        created=created or datetime(2024, 1, 15, 9, 0, 0),
    )


# ---------------------------------------------------------------------------
# Escape helper
# ---------------------------------------------------------------------------


class TestEscape:
    def test_ampersand(self) -> None:
        assert _escape("a & b") == "a &amp; b"

    def test_less_than(self) -> None:
        assert _escape("a < b") == "a &lt; b"

    def test_greater_than(self) -> None:
        assert _escape("a > b") == "a &gt; b"

    def test_all_three(self) -> None:
        assert _escape("<a>&") == "&lt;a&gt;&amp;"

    def test_no_special(self) -> None:
        assert _escape("hello world") == "hello world"

    def test_empty(self) -> None:
        assert _escape("") == ""


# ---------------------------------------------------------------------------
# Header rendering
# ---------------------------------------------------------------------------


class TestHeaderRendering:
    def test_title_is_bold(self) -> None:
        note = _make_note(title="My Note")
        result = TelegramFormatter.format_note(note)
        assert "<b>My Note</b>" in result

    def test_path_is_code(self) -> None:
        note = _make_note(path="projects/alpha.md")
        result = TelegramFormatter.format_note(note)
        assert "<code>projects/alpha.md</code>" in result

    def test_tags_are_italic(self) -> None:
        note = _make_note(tags=["python", "learning"])
        result = TelegramFormatter.format_note(note)
        assert "<i>#python #learning</i>" in result

    def test_no_tags_no_tag_line(self) -> None:
        note = _make_note(tags=[])
        result = TelegramFormatter.format_note(note)
        assert "#" not in result or "<i>#" not in result

    def test_date_from_frontmatter(self) -> None:
        note = _make_note(frontmatter={"created": "2024-03-10"})
        result = TelegramFormatter.format_note(note)
        assert "2024-03-10" in result

    def test_date_trimmed_from_datetime_string(self) -> None:
        note = _make_note(frontmatter={"created": "2024-03-10T14:30:00"})
        result = TelegramFormatter.format_note(note)
        assert "2024-03-10" in result
        assert "T14" not in result

    def test_note_type_fleeting_not_shown(self) -> None:
        note = _make_note(note_type=NoteType.FLEETING)
        result = TelegramFormatter.format_note(note)
        assert "fleeting" not in result

    def test_note_type_permanent_shown(self) -> None:
        note = _make_note(
            note_type=NoteType.PERMANENT,
            frontmatter={"type": "permanent"},
        )
        result = TelegramFormatter.format_note(note)
        assert "permanent" in result

    def test_title_html_escaped(self) -> None:
        note = _make_note(title="A & B <Topic>")
        result = TelegramFormatter.format_note(note)
        assert "<b>A &amp; B &lt;Topic&gt;</b>" in result


# ---------------------------------------------------------------------------
# Body conversion rules
# ---------------------------------------------------------------------------


class TestBodyConversion:
    def test_heading_level1(self) -> None:
        note = _make_note(content="# Hello World")
        result = TelegramFormatter.format_note(note)
        assert "<b>Hello World</b>" in result

    def test_heading_level3(self) -> None:
        note = _make_note(content="### Deep Section")
        result = TelegramFormatter.format_note(note)
        assert "<b>Deep Section</b>" in result

    def test_bold_double_star(self) -> None:
        note = _make_note(content="This is **important**.")
        result = TelegramFormatter.format_note(note)
        assert "<b>important</b>" in result

    def test_bold_double_underscore(self) -> None:
        note = _make_note(content="This is __critical__.")
        result = TelegramFormatter.format_note(note)
        assert "<b>critical</b>" in result

    def test_italic_single_star(self) -> None:
        note = _make_note(content="This is *emphasized*.")
        result = TelegramFormatter.format_note(note)
        assert "<i>emphasized</i>" in result

    def test_italic_single_underscore(self) -> None:
        note = _make_note(content="This is _subtle_.")
        result = TelegramFormatter.format_note(note)
        assert "<i>subtle</i>" in result

    def test_inline_code(self) -> None:
        note = _make_note(content="Run `pytest` now.")
        result = TelegramFormatter.format_note(note)
        assert "<code>pytest</code>" in result

    def test_code_block_plain(self) -> None:
        note = _make_note(content="```\nprint('hello')\n```")
        result = TelegramFormatter.format_note(note)
        assert "<pre>" in result
        assert "print" in result

    def test_code_block_python(self) -> None:
        note = _make_note(content="```python\ndef foo(): pass\n```")
        result = TelegramFormatter.format_note(note)
        assert "<pre>" in result
        assert "def foo" in result

    def test_dataview_block_replaced(self) -> None:
        note = _make_note(content="```dataview\nTABLE file.name\n```")
        result = TelegramFormatter.format_note(note)
        assert "[Dataview block]" in result
        assert "TABLE" not in result

    def test_mermaid_block_replaced(self) -> None:
        note = _make_note(content="```mermaid\ngraph TD\n```")
        result = TelegramFormatter.format_note(note)
        assert "[Mermaid block]" in result
        assert "graph TD" not in result

    def test_wikilink_plain(self) -> None:
        note = _make_note(content="See [[My Note]] for details.")
        result = TelegramFormatter.format_note(note)
        assert "<b>My Note</b>" in result

    def test_wikilink_aliased(self) -> None:
        note = _make_note(content="See [[some/path|Display Text]] here.")
        result = TelegramFormatter.format_note(note)
        assert "<b>Display Text</b>" in result
        assert "some/path" not in result

    def test_blockquote_plain(self) -> None:
        note = _make_note(content="> This is quoted.")
        result = TelegramFormatter.format_note(note)
        assert "┃" in result
        assert "<i>This is quoted.</i>" in result

    def test_image_markdown(self) -> None:
        note = _make_note(content="![a cat](https://example.com/cat.png)")
        result = TelegramFormatter.format_note(note)
        assert "[🖼 a cat]" in result

    def test_image_wikilink(self) -> None:
        note = _make_note(content="![[photo.jpg]]")
        result = TelegramFormatter.format_note(note)
        assert "[🖼" in result
        assert "photo" in result

    def test_image_markdown_no_alt(self) -> None:
        note = _make_note(content="![](https://example.com/x.png)")
        result = TelegramFormatter.format_note(note)
        assert "[🖼 Image]" in result

    def test_task_open(self) -> None:
        note = _make_note(content="- [ ] Write tests")
        result = TelegramFormatter.format_note(note)
        assert "☐ Write tests" in result

    def test_task_done(self) -> None:
        note = _make_note(content="- [x] Deploy fix")
        result = TelegramFormatter.format_note(note)
        assert "☑ Deploy fix" in result

    def test_unordered_list_dash(self) -> None:
        note = _make_note(content="- item one")
        result = TelegramFormatter.format_note(note)
        assert "• item one" in result

    def test_unordered_list_star(self) -> None:
        note = _make_note(content="* item two")
        result = TelegramFormatter.format_note(note)
        assert "• item two" in result

    def test_horizontal_rule(self) -> None:
        note = _make_note(content="before\n---\nafter")
        result = TelegramFormatter.format_note(note)
        assert "─────" in result

    def test_html_tags_stripped(self) -> None:
        note = _make_note(content="Hello <span>world</span>.")
        result = TelegramFormatter.format_note(note)
        assert "<span>" not in result
        assert "world" in result


# ---------------------------------------------------------------------------
# Callout rendering
# ---------------------------------------------------------------------------


class TestCalloutRendering:
    @pytest.mark.parametrize(
        ("callout_type", "expected_emoji"),
        [
            ("note", "📝"),
            ("tip", "💡"),
            ("warning", "⚠️"),
            ("danger", "🔴"),
            ("info", "ℹ️"),
            ("unknown", "📌"),
        ],
    )
    def test_callout_emoji(self, callout_type: str, expected_emoji: str) -> None:
        note = _make_note(content=f"> [!{callout_type}] Title Here")
        result = TelegramFormatter.format_note(note)
        assert expected_emoji in result
        assert "<b>Title Here</b>" in result

    def test_callout_no_title_uses_type(self) -> None:
        note = _make_note(content="> [!tip]")
        result = TelegramFormatter.format_note(note)
        assert "💡" in result
        # Defaults to capitalised type name
        assert "<b>Tip</b>" in result

    def test_callout_case_insensitive(self) -> None:
        note = _make_note(content="> [!WARNING] Be careful")
        result = TelegramFormatter.format_note(note)
        assert "⚠️" in result


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------


class TestHTMLEscaping:
    def test_ampersand_in_body(self) -> None:
        note = _make_note(content="apples & oranges")
        result = TelegramFormatter.format_note(note)
        assert "&amp;" in result
        assert "apples &amp; oranges" in result

    def test_less_than_in_body(self) -> None:
        note = _make_note(content="a < b means less")
        result = TelegramFormatter.format_note(note)
        assert "&lt;" in result

    def test_greater_than_in_body(self) -> None:
        note = _make_note(content="b > a means greater")
        result = TelegramFormatter.format_note(note)
        assert "&gt;" in result

    def test_code_block_content_escaped(self) -> None:
        note = _make_note(content="```\n<script>alert('xss')</script>\n```")
        result = TelegramFormatter.format_note(note)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escape_before_formatting_tags(self) -> None:
        # The & must become &amp; not interfere with <b> tags
        note = _make_note(content="**a & b**")
        result = TelegramFormatter.format_note(note)
        assert "<b>a &amp; b</b>" in result


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_note_not_truncated(self) -> None:
        note = _make_note(content="Short content.")
        result = TelegramFormatter.format_note(note, max_length=4000)
        assert "truncated" not in result

    def test_long_note_truncated(self) -> None:
        long_body = "Word " * 1000  # ~5000 chars
        note = _make_note(content=long_body)
        result = TelegramFormatter.format_note(note, max_length=500)
        assert "truncated" in result
        assert len(result) <= 600  # some slack for the notice itself

    def test_truncation_includes_total_chars(self) -> None:
        long_body = "X" * 2000
        note = _make_note(content=long_body, title="T")
        result = TelegramFormatter.format_note(note, max_length=200)
        assert "chars total" in result

    def test_truncation_at_newline_boundary(self) -> None:
        # Build content with clear newline boundaries
        lines = ["Line number " + str(i) for i in range(200)]
        note = _make_note(content="\n".join(lines))
        result = TelegramFormatter.format_note(note, max_length=300)
        # The truncated result must not cut in the middle of a word
        main_part = result.split("\n\n<i>…truncated")[0]
        # Every line in the main part must be complete (ends cleanly)
        assert (
            main_part.endswith("Line number " + main_part.rstrip().split()[-1]) or "\n" in main_part
        )


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_content(self) -> None:
        note = _make_note(content="")
        result = TelegramFormatter.format_note(note)
        # Should still return the header
        assert "<b>Test Note</b>" in result

    def test_content_only_frontmatter(self) -> None:
        # body_without_frontmatter will return empty string
        note = _make_note(content="---\ntitle: test\n---\n")
        result = TelegramFormatter.format_note(note)
        # Header is still present
        assert "<b>Test Note</b>" in result

    def test_note_no_title_uses_stem(self) -> None:
        note = _make_note(title="", path="inbox/my-file.md")
        result = TelegramFormatter.format_note(note)
        # Falls back to path.stem
        assert "my-file" in result

    def test_nested_bold_italic(self) -> None:
        note = _make_note(content="This is **bold and *italic* inside**.")
        result = TelegramFormatter.format_note(note)
        # Bold wrapper should be present
        assert "<b>" in result

    def test_multiple_wikilinks_on_one_line(self) -> None:
        note = _make_note(content="See [[NoteA]] and [[NoteB|Second]] here.")
        result = TelegramFormatter.format_note(note)
        assert "<b>NoteA</b>" in result
        assert "<b>Second</b>" in result


# ---------------------------------------------------------------------------
# format_search_result
# ---------------------------------------------------------------------------


class TestFormatSearchResult:
    def _hit(
        self,
        title: str = "My Note",
        note_path: str = "inbox/my-note.md",
        heading: str = "",
        content: str = "Some content here.",
        distance: float = 0.1,
    ) -> dict:
        return {
            "metadata": {
                "note_title": title,
                "note_path": note_path,
                "heading": heading,
            },
            "content": content,
            "distance": distance,
        }

    def test_title_is_bold(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(), 1)
        assert "<b>1. My Note</b>" in result

    def test_relevance_score(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(distance=0.2), 1)
        assert "80% match" in result

    def test_path_in_code(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(), 1)
        assert "<code>inbox/my-note.md</code>" in result

    def test_heading_appended_when_present(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(heading="## Section"), 1)
        assert "## Section" in result

    def test_content_in_italic(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(content="A snippet."), 1)
        assert "<i>A snippet.</i>" in result

    def test_content_truncated_at_200(self) -> None:
        long = "W" * 300
        result = TelegramFormatter.format_search_result(self._hit(content=long), 1)
        assert "..." in result

    def test_html_escaped_in_title(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(title="A & B"), 1)
        assert "&amp;" in result
        assert "A & B" not in result

    def test_index_shown(self) -> None:
        result = TelegramFormatter.format_search_result(self._hit(), 7)
        assert "<b>7." in result
