"""Current-content projection of notes carrying bot supersede history (M14).

Every case drives real files through the real parser (`parse_file`,
`current_content()`, `chunk_note`) and, for new bot writes, through the real
`write_superseded_block` writer. Legacy fixtures are the literal text the
pre-boundary bot wrote; nothing here asserts on how the code is wired.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import frontmatter
import pytest

from vaultmind.config import VaultConfig
from vaultmind.contradiction.marking import write_superseded_block
from vaultmind.vault.parser import VaultParser
from vaultmind.vault.projection import AmbiguousHistoryError, RetiredNoteError

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.vault.models import Note

ID_A = "a" * 32
ID_B = "b" * 32
STAMP_1 = "2026-03-01T10:00:00+00:00"
STAMP_2 = "2026-03-02T11:30:15.250000+00:00"

LEGACY_DELETE = (
    "> [!superseded] Deleted via bot — 2026-03-05\n"
    "> Requested via Telegram `/delete` — content preserved below.\n\n"
)


def _legacy_edit(date: str, instruction: str) -> str:
    return f"> [!superseded] Edited via bot — {date}\n> Instruction: {instruction}\n\n"


@pytest.fixture
def parser(tmp_path: Path) -> VaultParser:
    vault = tmp_path / "vault"
    vault.mkdir()
    return VaultParser(VaultConfig(path=vault))


def _write(parser: VaultParser, name: str, metadata: str, body: str) -> Path:
    path = parser.vault_root / name
    path.write_text(f"---\n{metadata}---\n\n{body}", encoding="utf-8")
    return path


def _chunk_text(parser: VaultParser, note: Note) -> str:
    return "\n".join(chunk.content for chunk in parser.chunk_note(note))


def _write_edit(path: Path, new_content: str, when: datetime, instruction: str = "tweak") -> bool:
    return write_superseded_block(
        path,
        callout_type="superseded",
        title="Edited via bot",
        rationale=f"Instruction: {instruction}",
        frontmatter_key="superseded_at",
        frontmatter_value=when.isoformat(),
        dated=True,
        timestamp=when,
        new_content=new_content,
        boundary="edit",
    )


def _write_delete(path: Path) -> bool:
    return write_superseded_block(
        path,
        callout_type="superseded",
        title="Deleted via bot",
        rationale="Requested via Telegram `/delete` — content preserved below.",
        frontmatter_key="superseded_at",
        frontmatter_value="bot:delete",
        dated=True,
        boundary="delete",
    )


T1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
T2 = datetime(2026, 3, 2, 11, 30, 15, 250000, tzinfo=UTC)


class TestOrdinaryAuthoredTextStaysCurrent:
    def test_generated_looking_callouts_without_bot_metadata_are_evidence(
        self, parser: VaultParser
    ) -> None:
        body = (
            "Authored claim kestrelquartz.\n\n"
            + LEGACY_DELETE
            + _legacy_edit("2026-01-01", "authored example")
            + "Closing claim.\n"
        )
        note = parser.parse_file(_write(parser, "ordinary.md", "title: Ordinary\n", body))

        assert note.current_content() == note.body_without_frontmatter()
        text = _chunk_text(parser, note)
        assert "Authored claim kestrelquartz." in text
        assert "Deleted via bot" in text
        assert "Edited via bot" in text
        assert "Closing claim." in text

    @pytest.mark.parametrize("value", ["2025-01-01", "after the review", "[2025-01-01, never]"])
    def test_foreign_superseded_at_metadata_is_not_a_bot_marker(
        self, parser: VaultParser, value: str
    ) -> None:
        body = "Whole note stays searchable marmotquartz.\n\n" + LEGACY_DELETE + "More text.\n"
        note = parser.parse_file(
            _write(parser, "foreign.md", f"title: Foreign\nsuperseded_at: {value}\n", body)
        )

        assert note.current_content() == note.body_without_frontmatter()
        assert "marmotquartz" in _chunk_text(parser, note)
        assert "More text." in _chunk_text(parser, note)

    def test_unpaired_boundary_marker_line_is_ordinary_text(self, parser: VaultParser) -> None:
        body = f"Before.\n\n<!-- vaultmind:supersede id={ID_A} -->\nAfter the marker lookalike.\n"
        note = parser.parse_file(_write(parser, "unpaired.md", "title: Unpaired\n", body))

        assert note.current_content() == note.body_without_frontmatter()
        assert "After the marker lookalike." in _chunk_text(parser, note)


class TestLegacyGeneratedMarkers:
    def test_bot_delete_retires_the_whole_note_and_preserves_raw_content(
        self, parser: VaultParser
    ) -> None:
        metadata = "superseded_at:\n- bot:delete\ntitle: Ledger\n"
        path = _write(parser, "deleted.md", metadata, LEGACY_DELETE + "Original claim tidewrack.\n")
        raw = path.read_text(encoding="utf-8")
        note = parser.parse_file(path)

        assert note.current_content() == ""
        assert parser.chunk_note(note) == []
        assert "Original claim tidewrack." in note.content
        assert "Deleted via bot" in note.content
        assert path.read_text(encoding="utf-8") == raw

    def test_single_legacy_edit_exposes_only_the_new_body(self, parser: VaultParser) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n"
        body = (
            "Revised claim ironweft.\n\n"
            + _legacy_edit("2026-03-01", "revise")
            + "Old tidewrack.\n"
        )
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == "Revised claim ironweft."
        text = _chunk_text(parser, note)
        assert "ironweft" in text
        assert "tidewrack" not in text
        assert "Old tidewrack." in note.content

    def test_repeated_legacy_edits_keep_history_but_expose_latest_claim(
        self, parser: VaultParser
    ) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\n- '{STAMP_2}'\ntitle: Log\n"
        body = (
            "v2 claim gammaquartz.\n\n"
            + _legacy_edit("2026-03-02", "second change")
            + "v1 claim betaquartz.\n\n"
            + _legacy_edit("2026-03-01", "first change")
            + "v0 claim alphaquartz.\n"
        )
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == "v2 claim gammaquartz."
        text = _chunk_text(parser, note)
        assert "gammaquartz" in text
        assert "betaquartz" not in text
        assert "alphaquartz" not in text
        for claim in ("alphaquartz", "betaquartz", "gammaquartz"):
            assert claim in note.content

    def test_multiline_instruction_with_blank_lines_does_not_confuse_the_boundary(
        self, parser: VaultParser
    ) -> None:
        instruction = "first line\n\nsecond paragraph\n- a bullet\n\n> quoted advice"
        metadata = f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n"
        body = (
            "Current claim sandwort.\n\n"
            + _legacy_edit("2026-03-01", instruction)
            + "Preserved claim driftglass.\n"
        )
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == "Current claim sandwort."
        assert "driftglass" not in _chunk_text(parser, note)

    def test_other_callouts_in_the_current_text_are_kept(self, parser: VaultParser) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n"
        current = (
            "Current claim.\n\n> [!note] Keep this\n> an ordinary callout\n\n"
            "> [!superseded] Custom title — 2026-01-01\n> authored, not generated\n"
        )
        body = current + "\n" + _legacy_edit("2026-03-01", "revise") + "Old claim.\n"
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == current.strip()

    def test_delete_after_edits_is_terminal(self, parser: VaultParser) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\n- bot:delete\ntitle: Log\n"
        body = (
            LEGACY_DELETE
            + "Edited claim ironweft.\n\n"
            + _legacy_edit("2026-03-01", "revise")
            + "Original tidewrack.\n"
        )
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == ""
        assert parser.chunk_note(note) == []

    def test_empty_replacement_body_projects_to_no_current_text(self, parser: VaultParser) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n"
        body = _legacy_edit("2026-03-01", "clear it") + "Old claim.\n"
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == ""
        assert parser.chunk_note(note) == []


AMBIGUOUS_LEGACY = {
    "delete marker without callout": (
        "superseded_at:\n- bot:delete\ntitle: Log\n",
        "Original claim.\n",
    ),
    "delete callout with altered rationale": (
        "superseded_at:\n- bot:delete\ntitle: Log\n",
        "> [!superseded] Deleted via bot — 2026-03-05\n> something else entirely\n\nOriginal.\n",
    ),
    "duplicated delete callout": (
        "superseded_at:\n- bot:delete\ntitle: Log\n",
        LEGACY_DELETE + "Original.\n\n" + LEGACY_DELETE + "Quoted copy.\n",
    ),
    "edit stamp without callout": (
        f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n",
        "Revised claim.\n\nOld claim.\n",
    ),
    "two stamps, one callout": (
        f"superseded_at:\n- '{STAMP_1}'\n- '{STAMP_2}'\ntitle: Log\n",
        "v2.\n\n" + _legacy_edit("2026-03-02", "b") + "v1.\n",
    ),
    "authored lookalike callout in the new text": (
        f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n",
        "Claim quoting a lookalike:\n\n"
        + _legacy_edit("2026-02-14", "example")
        + "still the new text\n\n"
        + _legacy_edit("2026-03-01", "revise")
        + "Old claim.\n",
    ),
    "callout date disagrees with the stamp": (
        f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n",
        "Revised.\n\n" + _legacy_edit("2026-03-09", "revise") + "Old.\n",
    ),
}

AMBIGUOUS_EXPLICIT = {
    "record that is not id=value": (
        f"superseded_at:\n- '{STAMP_1}'\nsupersede_boundaries:\n- not-a-record\ntitle: Log\n",
        "Revised.\n\n" + _legacy_edit("2026-03-01", "revise") + "Old.\n",
    ),
    "records are a mapping": (
        f"superseded_at:\n- '{STAMP_1}'\n"
        f"supersede_boundaries:\n  {ID_A}: '{STAMP_1}'\ntitle: Log\n",
        "Revised.\n",
    ),
    "record value is not a bot value": (
        f"superseded_at:\n- '{STAMP_1}'\nsupersede_boundaries:\n- {ID_A}=sometime\ntitle: Log\n",
        f"Revised.\n\n<!-- vaultmind:supersede id={ID_A} -->\nOld.\n",
    ),
    "duplicate record ids": (
        f"superseded_at:\n- '{STAMP_1}'\n- '{STAMP_2}'\n"
        f"supersede_boundaries:\n- {ID_A}={STAMP_1}\n- {ID_A}={STAMP_2}\ntitle: Log\n",
        f"New.\n\n<!-- vaultmind:supersede id={ID_A} -->\nOld.\n",
    ),
    "record not backed by superseded_at": (
        f"supersede_boundaries:\n- {ID_A}={STAMP_1}\ntitle: Log\n",
        f"Revised.\n\n<!-- vaultmind:supersede id={ID_A} -->\nOld.\n",
    ),
    "record without its marker in the body": (
        f"superseded_at:\n- '{STAMP_1}'\nsupersede_boundaries:\n- {ID_A}={STAMP_1}\ntitle: Log\n",
        "Revised.\n\nOld.\n",
    ),
    "marker present twice": (
        f"superseded_at:\n- '{STAMP_1}'\nsupersede_boundaries:\n- {ID_A}={STAMP_1}\ntitle: Log\n",
        f"Revised.\n\n<!-- vaultmind:supersede id={ID_A} -->\nOld.\n\n"
        f"<!-- vaultmind:supersede id={ID_A} -->\nCopy.\n",
    ),
    "markers out of newest-first order": (
        f"superseded_at:\n- '{STAMP_1}'\n- '{STAMP_2}'\n"
        f"supersede_boundaries:\n- {ID_A}={STAMP_1}\n- {ID_B}={STAMP_2}\ntitle: Log\n",
        f"v2.\n\n<!-- vaultmind:supersede id={ID_A} -->\nv1.\n\n"
        f"<!-- vaultmind:supersede id={ID_B} -->\nv0.\n",
    ),
    "delete record without its marker": (
        f"superseded_at:\n- bot:delete\nsupersede_boundaries:\n- {ID_A}=bot:delete\ntitle: Log\n",
        LEGACY_DELETE + "Original.\n",
    ),
    "delete record while an older record lacks its marker": (
        f"superseded_at:\n- '{STAMP_1}'\n- bot:delete\n"
        f"supersede_boundaries:\n- {ID_A}={STAMP_1}\n- {ID_B}=bot:delete\ntitle: Log\n",
        f"<!-- vaultmind:supersede id={ID_B} -->\n" + LEGACY_DELETE + "Original.\n",
    ),
    "delete record with markers out of newest-first order": (
        f"superseded_at:\n- '{STAMP_1}'\n- bot:delete\n"
        f"supersede_boundaries:\n- {ID_A}={STAMP_1}\n- {ID_B}=bot:delete\ntitle: Log\n",
        f"<!-- vaultmind:supersede id={ID_A} -->\nv1.\n\n"
        f"<!-- vaultmind:supersede id={ID_B} -->\n" + LEGACY_DELETE + "v0.\n",
    ),
}

# Malformed superseded_at next to an exact generated delete callout: the intended bot:delete
# stamp must never degrade to "the whole retired body is current".
AMBIGUOUS_SUPERSEDED_AT_TYPES = {
    "mapping": "superseded_at:\n  bot: delete\ntitle: Log\n",
    "nested list": "superseded_at:\n- - bot:delete\ntitle: Log\n",
    "bare number": "superseded_at: 42\ntitle: Log\n",
    "bare boolean": "superseded_at: true\ntitle: Log\n",
    "list holding a number": "superseded_at:\n- bot:delete\n- 42\ntitle: Log\n",
    "list holding a mapping": "superseded_at:\n- bot:delete\n- when: now\ntitle: Log\n",
    "list holding a null": "superseded_at:\n- bot:delete\n-\ntitle: Log\n",
}


class TestAmbiguousHistoryIsReportedNotGuessed:
    @pytest.mark.parametrize(
        ("metadata", "body"),
        [pytest.param(m, b, id=name) for name, (m, b) in AMBIGUOUS_LEGACY.items()]
        + [pytest.param(m, b, id=name) for name, (m, b) in AMBIGUOUS_EXPLICIT.items()]
        + [
            pytest.param(m, LEGACY_DELETE + "Original claim.\n", id=f"superseded_at {name}")
            for name, m in AMBIGUOUS_SUPERSEDED_AT_TYPES.items()
        ],
    )
    def test_chunking_refuses_and_nothing_is_rewritten(
        self, parser: VaultParser, metadata: str, body: str
    ) -> None:
        path = _write(parser, "ambiguous.md", metadata, body)
        raw = path.read_text(encoding="utf-8")

        note = parser.parse_file(path)  # reading the note never fails on ambiguity
        assert note.content == body.strip()

        with pytest.raises(AmbiguousHistoryError):
            note.current_content()
        with pytest.raises(ValueError, match=r"."):
            parser.chunk_note(note)
        assert path.read_text(encoding="utf-8") == raw


class TestExplicitBoundaryWrittenByTheBot:
    def test_hand_authored_pairing_projects_to_the_text_above_the_boundary(
        self, parser: VaultParser
    ) -> None:
        metadata = (
            f"superseded_at:\n- '{STAMP_1}'\nsupersede_boundaries:\n- {ID_A}={STAMP_1}\n"
            "title: Log\n"
        )
        body = (
            "Newest claim.\n\n"
            f"<!-- vaultmind:supersede id={ID_A} -->\n"
            "> [!superseded] Edited via bot — 2026-03-01\n> Instruction: tweak\n\n"
            "Older claim.\n"
        )
        note = parser.parse_file(_write(parser, "log.md", metadata, body))

        assert note.current_content() == "Newest claim."

    def test_two_edits_keep_every_revision_but_expose_only_the_latest(
        self, parser: VaultParser
    ) -> None:
        path = _write(parser, "log.md", "title: Log\n", "v0 claim alphaquartz.\n")
        assert _write_edit(path, "v1 claim betaquartz.", T1, "first")
        assert _write_edit(path, "v2 claim gammaquartz.", T2, "second\n\nwith a blank line")

        note = parser.parse_file(path)
        assert note.current_content() == "v2 claim gammaquartz."
        text = _chunk_text(parser, note)
        assert "gammaquartz" in text
        assert "alphaquartz" not in text
        assert "betaquartz" not in text
        positions = [note.content.index(c) for c in ("gammaquartz", "betaquartz", "alphaquartz")]
        assert positions == sorted(positions)

    def test_authored_lookalikes_inside_the_new_text_do_not_truncate_it(
        self, parser: VaultParser
    ) -> None:
        path = _write(parser, "log.md", "title: Log\n", "Old claim.\n")
        new_text = (
            "New claim lookalikequartz.\n\n"
            "> [!superseded] Edited via bot — 2026-01-01\n> Instruction: an example\n"
            "<!-- vaultmind:supersede id=" + "c" * 32 + " -->\n"
            "Text after the lookalikes."
        )
        assert _write_edit(path, new_text, T1)

        note = parser.parse_file(path)
        assert note.current_content() == new_text
        text = _chunk_text(parser, note)
        assert "Text after the lookalikes." in text
        assert "Old claim." not in text

    def test_delete_is_terminal_even_after_earlier_edits(self, parser: VaultParser) -> None:
        path = _write(parser, "log.md", "title: Log\n", "v0 claim alphaquartz.\n")
        assert _write_edit(path, "v1 claim betaquartz.", T1)
        assert _write_delete(path)

        note = parser.parse_file(path)
        assert note.current_content() == ""
        assert parser.chunk_note(note) == []
        assert "alphaquartz" in note.content
        assert "betaquartz" in note.content

    def test_new_edit_on_top_of_legacy_history_wins(self, parser: VaultParser) -> None:
        metadata = f"superseded_at:\n- '{STAMP_1}'\ntitle: Log\n"
        body = "Legacy claim ironweft.\n\n" + _legacy_edit("2026-03-01", "old") + "Origin.\n"
        path = _write(parser, "log.md", metadata, body)

        assert _write_edit(path, "Fresh claim saltreed.", T2)

        note = parser.parse_file(path)
        assert note.current_content() == "Fresh claim saltreed."
        text = _chunk_text(parser, note)
        assert "ironweft" not in text
        assert "Origin." not in text
        assert "ironweft" in note.content

    def test_projection_survives_a_save_reload_cycle_of_the_frontmatter(
        self, parser: VaultParser
    ) -> None:
        path = _write(parser, "log.md", "title: Log\n", "Old claim.\n")
        assert _write_edit(path, "New claim.", T1)

        reloaded = frontmatter.load(str(path))
        path.write_text(frontmatter.dumps(reloaded), encoding="utf-8")

        assert parser.parse_file(path).current_content() == "New claim."

    def test_authored_leading_thematic_rules_are_body_text_not_frontmatter(
        self, parser: VaultParser
    ) -> None:
        original = "---\n\nv0 claim alphaquartz.\n\n---\n\nOlder rule section."
        first = "---\n\nv1 claim betaquartz.\n\n---\n\nMiddle rule section."
        second = "---\n\nv2 claim gammaquartz.\n\n---\n\nFinal rule section."
        path = _write(parser, "log.md", "title: Log\n", original + "\n")
        assert parser.parse_file(path).current_content() == original

        assert _write_edit(path, first, T1)
        assert _write_edit(path, second, T2)

        note = parser.parse_file(path)
        assert note.current_content() == second
        text = _chunk_text(parser, note)
        assert "gammaquartz" in text
        assert "Final rule section." in text
        assert "alphaquartz" not in text
        assert "betaquartz" not in text
        for kept in ("alphaquartz", "Older rule section.", "betaquartz", "Middle rule section."):
            assert kept in note.content


class TestWriterRefusesToCorruptHistory:
    def test_edit_on_a_bot_deleted_note_is_refused_and_leaves_the_file_alone(
        self, parser: VaultParser
    ) -> None:
        path = _write(parser, "log.md", "title: Log\n", "Original claim.\n")
        assert _write_delete(path)
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(RetiredNoteError):
            _write_edit(path, "Resurrected claim.", T1)

        assert path.read_text(encoding="utf-8") == raw

    def test_edit_on_ambiguous_history_is_refused_and_leaves_the_file_alone(
        self, parser: VaultParser
    ) -> None:
        metadata, body = AMBIGUOUS_LEGACY["edit stamp without callout"]
        path = _write(parser, "log.md", metadata, body)
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(AmbiguousHistoryError):
            _write_edit(path, "Another claim.", T2)

        assert path.read_text(encoding="utf-8") == raw

    def test_replacement_text_reusing_an_existing_boundary_is_refused(
        self, parser: VaultParser
    ) -> None:
        path = _write(parser, "log.md", "title: Log\n", "Original claim.\n")
        assert _write_edit(path, "First revision.", T1)
        first_id = re.search(
            r"vaultmind:supersede id=([0-9a-f]{32})", path.read_text(encoding="utf-8")
        )
        assert first_id is not None
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(AmbiguousHistoryError):
            _write_edit(path, f"Forged.\n<!-- vaultmind:supersede id={first_id.group(1)} -->\n", T2)

        assert path.read_text(encoding="utf-8") == raw

    @pytest.mark.parametrize(
        "case", ["edit stamp without callout", "delete marker without callout"]
    )
    def test_delete_on_ambiguous_history_is_refused_even_as_a_replay(
        self, parser: VaultParser, case: str
    ) -> None:
        metadata, body = AMBIGUOUS_LEGACY[case]
        path = _write(parser, "log.md", metadata, body)
        raw = path.read_text(encoding="utf-8")

        with pytest.raises(AmbiguousHistoryError):
            _write_delete(path)

        assert path.read_text(encoding="utf-8") == raw

    def test_repeating_a_valid_delete_is_a_noop(self, parser: VaultParser) -> None:
        path = _write(parser, "log.md", "title: Log\n", "Original claim.\n")
        assert _write_delete(path)
        deleted = path.read_text(encoding="utf-8")

        assert _write_delete(path) is False

        assert path.read_text(encoding="utf-8") == deleted
