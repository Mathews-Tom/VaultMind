"""Tests for inbox triage feature in DigestGenerator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from vaultmind.indexer.digest import DigestGenerator, DigestReport
from vaultmind.vault.models import Note


# ---------------------------------------------------------------------------
# Helpers — reuse same Fake pattern as test_digest.py
# ---------------------------------------------------------------------------


class FakeDigestConfig:
    def __init__(
        self,
        inbox_folder: str = "00-inbox",
        inbox_age_warning_days: int = 7,
        max_inbox_shown: int = 10,
        period_days: int = 7,
        max_trending: int = 10,
        max_suggestions: int = 5,
        connection_threshold_low: float = 0.70,
        connection_threshold_high: float = 0.85,
    ) -> None:
        self.enabled = True
        self.period_days = period_days
        self.schedule_hour = 8
        self.timezone = "UTC"
        self.save_to_vault = True
        self.send_telegram = True
        self.max_trending = max_trending
        self.max_suggestions = max_suggestions
        self.connection_threshold_low = connection_threshold_low
        self.connection_threshold_high = connection_threshold_high
        self.inbox_folder = inbox_folder
        self.inbox_age_warning_days = inbox_age_warning_days
        self.max_inbox_shown = max_inbox_shown


class FakeStore:
    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        return []


class FakeGraph:
    def __init__(self) -> None:
        import networkx as nx

        self._graph = nx.DiGraph()

    @property
    def stats(self) -> dict:
        return {"nodes": 0, "edges": 0}


class FakeParser:
    def __init__(self, notes: list[Note], vault_root: Path) -> None:
        self._notes = notes
        self.vault_root = vault_root

    def iter_notes(self) -> list[Note]:
        return list(self._notes)


def _make_note(path: str = "test/note.md", title: str = "Test Note") -> Note:
    content = "---\ntype: fleeting\n---\n\nSome content " * 5
    return Note(path=path, title=title, content=content)


def _make_generator(
    notes: list[Note],
    vault_root: Path,
    config: FakeDigestConfig | None = None,
) -> DigestGenerator:
    cfg = config or FakeDigestConfig()
    return DigestGenerator(
        store=FakeStore(),  # type: ignore[arg-type]
        graph=FakeGraph(),  # type: ignore[arg-type]
        parser=FakeParser(notes, vault_root),  # type: ignore[arg-type]
        config=cfg,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInboxTriageCounts:
    def test_inbox_triage_counts_inbox_notes(self, tmp_path: Path) -> None:
        """Notes under inbox_folder are counted correctly."""
        notes = [
            _make_note(path="00-inbox/note1.md", title="Inbox Note 1"),
            _make_note(path="00-inbox/note2.md", title="Inbox Note 2"),
            _make_note(path="03-permanent/permanent.md", title="Permanent Note"),
        ]
        generator = _make_generator(notes, tmp_path)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        old = datetime.now(tz=UTC) - timedelta(days=30)

        with (
            patch.object(generator, "_file_ctime", return_value=recent),
            patch.object(generator, "_file_mtime", return_value=old),
        ):
            report = generator.generate()

        assert report.inbox_count == 2

    def test_non_inbox_notes_excluded_from_triage(self, tmp_path: Path) -> None:
        """Notes outside inbox folder do not appear in inbox triage."""
        notes = [
            _make_note(path="03-permanent/perm.md", title="Permanent"),
            _make_note(path="02-projects/project.md", title="Project"),
        ]
        generator = _make_generator(notes, tmp_path)

        old = datetime.now(tz=UTC) - timedelta(days=30)
        with (
            patch.object(generator, "_file_ctime", return_value=old),
            patch.object(generator, "_file_mtime", return_value=old),
        ):
            report = generator.generate()

        assert report.inbox_count == 0
        assert report.inbox_notes == []


class TestInboxTriageOldestAge:
    def test_inbox_triage_oldest_age(self, tmp_path: Path) -> None:
        """Oldest note age is calculated relative to now."""
        notes = [
            _make_note(path="00-inbox/new.md", title="New Inbox"),
            _make_note(path="00-inbox/old.md", title="Old Inbox"),
        ]
        generator = _make_generator(notes, tmp_path)

        # We'll patch _file_ctime per-path — use side_effect
        def ctime_side_effect(rel_path: Path) -> datetime:
            if "new.md" in str(rel_path):
                return datetime.now(tz=UTC) - timedelta(days=3)
            return datetime.now(tz=UTC) - timedelta(days=20)

        old_mtime = datetime.now(tz=UTC) - timedelta(days=30)

        with (
            patch.object(generator, "_file_ctime", side_effect=ctime_side_effect),
            patch.object(generator, "_file_mtime", return_value=old_mtime),
        ):
            report = generator.generate()

        assert report.oldest_inbox_note == "Old Inbox"
        assert report.oldest_inbox_age_days >= 19

    def test_inbox_triage_oldest_first_ordering(self, tmp_path: Path) -> None:
        """inbox_notes list is ordered oldest first."""
        notes = [
            _make_note(path="00-inbox/a.md", title="Note A"),
            _make_note(path="00-inbox/b.md", title="Note B"),
            _make_note(path="00-inbox/c.md", title="Note C"),
        ]
        generator = _make_generator(notes, tmp_path)

        ages = {"a.md": 10, "b.md": 30, "c.md": 5}

        def ctime_side_effect(rel_path: Path) -> datetime:
            for stem, days in ages.items():
                if stem in str(rel_path):
                    return datetime.now(tz=UTC) - timedelta(days=days)
            return datetime.now(tz=UTC)

        old_mtime = datetime.now(tz=UTC) - timedelta(days=40)

        with (
            patch.object(generator, "_file_ctime", side_effect=ctime_side_effect),
            patch.object(generator, "_file_mtime", return_value=old_mtime),
        ):
            report = generator.generate()

        assert report.oldest_inbox_note == "Note B"
        # inbox_notes should be ordered: B (30d), A (10d), C (5d)
        assert report.inbox_notes[0] == "Note B"


class TestInboxTriageTelegramFormat:
    def test_inbox_triage_telegram_format(self, tmp_path: Path) -> None:
        """HTML output contains inbox section when notes present."""
        gen = _make_generator([], tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            inbox_count=3,
            oldest_inbox_note="My Old Note",
            oldest_inbox_age_days=14,
            inbox_notes=["My Old Note", "Second Note", "Third Note"],
        )

        msg = gen.format_telegram(report)

        assert "📥 Inbox Triage" in msg
        assert "3 notes awaiting processing" in msg
        assert "⚠️ Oldest: My Old Note (14 days)" in msg
        assert "Second Note" in msg

    def test_inbox_triage_no_warning_when_below_threshold(self, tmp_path: Path) -> None:
        """Warning line absent when oldest age <= inbox_age_warning_days."""
        config = FakeDigestConfig(inbox_age_warning_days=14)
        gen = _make_generator([], tmp_path, config=config)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            inbox_count=1,
            oldest_inbox_note="Fresh Note",
            oldest_inbox_age_days=5,
            inbox_notes=["Fresh Note"],
        )

        msg = gen.format_telegram(report)

        assert "📥 Inbox Triage" in msg
        assert "⚠️" not in msg

    def test_inbox_triage_max_inbox_shown_respected(self, tmp_path: Path) -> None:
        """Only max_inbox_shown titles rendered."""
        config = FakeDigestConfig(max_inbox_shown=3, inbox_age_warning_days=100)
        gen = _make_generator([], tmp_path, config=config)
        now = datetime.now(tz=UTC)
        titles = [f"Note {i}" for i in range(10)]
        report = DigestReport(
            generated_at=now,
            period_days=7,
            inbox_count=10,
            oldest_inbox_note="Note 0",
            oldest_inbox_age_days=1,
            inbox_notes=titles,
        )

        msg = gen.format_telegram(report)

        # Only the first 3 titles should appear
        assert "Note 0" in msg
        assert "Note 1" in msg
        assert "Note 2" in msg
        assert "Note 9" not in msg


class TestInboxTriageEmptyInbox:
    def test_inbox_triage_empty_inbox(self, tmp_path: Path) -> None:
        """Zero count, no warning, no section when inbox is empty."""
        gen = _make_generator([], tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            new_notes=["A Note"],  # ensure report is non-empty overall
            inbox_count=0,
            oldest_inbox_note="",
            oldest_inbox_age_days=0,
            inbox_notes=[],
        )

        msg = gen.format_telegram(report)

        assert "📥 Inbox Triage" not in msg
        assert "⚠️" not in msg

    def test_generate_with_empty_inbox(self, tmp_path: Path) -> None:
        """generate() sets inbox_count=0 when no inbox notes exist."""
        notes = [_make_note(path="03-permanent/perm.md", title="Permanent")]
        generator = _make_generator(notes, tmp_path)

        old = datetime.now(tz=UTC) - timedelta(days=30)
        with (
            patch.object(generator, "_file_ctime", return_value=old),
            patch.object(generator, "_file_mtime", return_value=old),
        ):
            report = generator.generate()

        assert report.inbox_count == 0
        assert report.oldest_inbox_note == ""
        assert report.oldest_inbox_age_days == 0
        assert report.inbox_notes == []
