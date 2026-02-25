"""Tests for DigestGenerator — Smart Daily Digest (1.7)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from vaultmind.indexer.digest import (
    DigestGenerator,
    DigestReport,
    SuggestedConnection,
    TrendingEntity,
)
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDigestConfig:
    """Minimal DigestConfig stand-in."""

    def __init__(
        self,
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


class FakeStore:
    """Minimal VaultStore stand-in."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []

    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        return self._results


class FakeGraph:
    """Minimal KnowledgeGraph stand-in."""

    def __init__(
        self,
        nodes: dict[str, dict] | None = None,
        stats: dict | None = None,
    ) -> None:
        import networkx as nx

        self._graph = nx.DiGraph()
        for node_id, data in (nodes or {}).items():
            self._graph.add_node(node_id, **data)
        self._stats = stats or {"nodes": len(nodes or {}), "edges": 0}

    @property
    def stats(self) -> dict:
        return self._stats


class FakeParser:
    """Minimal VaultParser stand-in."""

    def __init__(self, notes: list[Note], vault_root: Path) -> None:
        self._notes = notes
        self.vault_root = vault_root

    def iter_notes(self) -> list[Note]:
        return list(self._notes)


def _make_note(
    path: str = "test/note.md",
    title: str = "Test Note",
    content: str = "",
    wikilinks: list[str] | None = None,
) -> Note:
    if not content:
        content = "---\ntype: fleeting\n---\n\n" + "Some content " * 20
    note = Note(path=path, title=title, content=content)
    return note


def _make_hit(
    note_path: str,
    title: str = "Other Note",
    distance: float = 0.20,
) -> dict:
    return {
        "chunk_id": f"{note_path}::0",
        "content": "some content",
        "metadata": {
            "note_path": note_path,
            "note_title": title,
        },
        "distance": distance,
    }


def _make_generator(
    notes: list[Note],
    vault_root: Path,
    store_results: list[dict] | None = None,
    graph_nodes: dict | None = None,
    config: FakeDigestConfig | None = None,
) -> DigestGenerator:
    cfg = config or FakeDigestConfig()
    store = FakeStore(store_results)
    graph = FakeGraph(nodes=graph_nodes)
    parser = FakeParser(notes, vault_root)
    return DigestGenerator(
        store=store,  # type: ignore[arg-type]
        graph=graph,  # type: ignore[arg-type]
        parser=parser,  # type: ignore[arg-type]
        config=cfg,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests — DigestReport dataclass
# ---------------------------------------------------------------------------


class TestDigestReport:
    def test_defaults(self) -> None:
        now = datetime.now(tz=UTC)
        report = DigestReport(generated_at=now, period_days=7)
        assert report.new_notes == []
        assert report.modified_notes == []
        assert report.trending_entities == []
        assert report.suggested_connections == []
        assert report.orphan_notes == []
        assert report.total_notes == 0
        assert report.total_entities == 0

    def test_fields(self) -> None:
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=3,
            new_notes=["A"],
            modified_notes=["B"],
            total_notes=10,
            total_entities=5,
        )
        assert report.new_notes == ["A"]
        assert report.modified_notes == ["B"]
        assert report.total_notes == 10
        assert report.total_entities == 5


# ---------------------------------------------------------------------------
# Tests — TrendingEntity / SuggestedConnection
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_trending_entity(self) -> None:
        e = TrendingEntity(name="Python", current_count=5, previous_count=2, delta=3)
        assert e.delta == 3

    def test_suggested_connection(self) -> None:
        c = SuggestedConnection(note_a="A", note_b="B", similarity=0.75)
        assert c.similarity == 0.75


# ---------------------------------------------------------------------------
# Tests — period window filtering (new_notes / modified_notes)
# ---------------------------------------------------------------------------


class TestActivityWindow:
    def test_note_in_period_classified_as_new(self, tmp_path: Path) -> None:
        note_file = tmp_path / "note.md"
        note_file.write_text("content")

        # Set mtime to 2 days ago (inside 7-day window)
        two_days_ago = time.time() - 2 * 86400
        import os

        os.utime(note_file, (two_days_ago, two_days_ago))

        note = _make_note(path="note.md", title="Recent Note")
        generator = _make_generator([note], tmp_path)

        # Patch _file_ctime to return recent time (simulating new file)
        recent = datetime.now(tz=UTC) - timedelta(days=2)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        assert "Recent Note" in report.new_notes

    def test_note_outside_period_excluded(self, tmp_path: Path) -> None:
        note_file = tmp_path / "old.md"
        note_file.write_text("content")

        # Set mtime to 30 days ago (outside 7-day window)
        old_time = time.time() - 30 * 86400
        import os

        os.utime(note_file, (old_time, old_time))

        note = _make_note(path="old.md", title="Old Note")
        generator = _make_generator([note], tmp_path, config=FakeDigestConfig(period_days=7))

        old_dt = datetime.now(tz=UTC) - timedelta(days=30)
        with (
            patch.object(generator, "_file_ctime", return_value=old_dt),
            patch.object(generator, "_file_mtime", return_value=old_dt),
        ):
            report = generator.generate()

        assert "Old Note" not in report.new_notes
        assert "Old Note" not in report.modified_notes

    def test_modified_note_not_in_new(self, tmp_path: Path) -> None:
        note = _make_note(path="mod.md", title="Modified Note")
        generator = _make_generator([note], tmp_path)

        old_ctime = datetime.now(tz=UTC) - timedelta(days=20)
        recent_mtime = datetime.now(tz=UTC) - timedelta(days=3)

        with (
            patch.object(generator, "_file_ctime", return_value=old_ctime),
            patch.object(generator, "_file_mtime", return_value=recent_mtime),
        ):
            report = generator.generate()

        assert "Modified Note" not in report.new_notes
        assert "Modified Note" in report.modified_notes

    def test_empty_vault(self, tmp_path: Path) -> None:
        generator = _make_generator([], tmp_path)
        report = generator.generate()

        assert report.new_notes == []
        assert report.modified_notes == []
        assert report.total_notes == 0


# ---------------------------------------------------------------------------
# Tests — Trending entity calculation
# ---------------------------------------------------------------------------


class TestTrendingEntities:
    def test_delta_computation(self, tmp_path: Path) -> None:
        # Entity mentioned in a note modified in the current window
        note_file = tmp_path / "current_note.md"
        note_file.write_text("content")

        note = _make_note(path="current_note.md", title="Current Note")

        # Graph has entity "Python" sourced from "current_note.md"
        graph_nodes = {
            "python": {
                "label": "Python",
                "type": "concept",
                "source_notes": ["current_note.md"],
                "confidence": 1.0,
            }
        }
        generator = _make_generator([note], tmp_path, graph_nodes=graph_nodes)

        # current_note.md mtime is in the current window
        recent = datetime.now(tz=UTC) - timedelta(days=2)
        with patch.object(generator, "_source_note_mtime", return_value=recent):
            report = generator.generate()

        assert any(e.name == "Python" for e in report.trending_entities)
        python_entity = next(e for e in report.trending_entities if e.name == "Python")
        assert python_entity.delta > 0

    def test_no_trending_when_graph_empty(self, tmp_path: Path) -> None:
        note = _make_note(path="note.md")
        generator = _make_generator([note], tmp_path, graph_nodes={})
        report = generator.generate()
        assert report.trending_entities == []

    def test_entities_sorted_by_delta_descending(self, tmp_path: Path) -> None:
        # Two entities, one with higher delta
        graph_nodes = {
            "python": {
                "label": "Python",
                "type": "concept",
                "source_notes": ["a.md", "b.md", "c.md"],
                "confidence": 1.0,
            },
            "rust": {
                "label": "Rust",
                "type": "concept",
                "source_notes": ["d.md"],
                "confidence": 1.0,
            },
        }
        note_a = _make_note(path="a.md", title="A")
        note_b = _make_note(path="b.md", title="B")
        note_c = _make_note(path="c.md", title="C")
        note_d = _make_note(path="d.md", title="D")
        generator = _make_generator(
            [note_a, note_b, note_c, note_d], tmp_path, graph_nodes=graph_nodes
        )

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_source_note_mtime", return_value=recent):
            report = generator.generate()

        if len(report.trending_entities) >= 2:
            assert report.trending_entities[0].delta >= report.trending_entities[1].delta

    def test_max_trending_respected(self, tmp_path: Path) -> None:
        graph_nodes = {
            f"entity_{i}": {
                "label": f"Entity {i}",
                "type": "concept",
                "source_notes": [f"note_{i}.md"],
                "confidence": 1.0,
            }
            for i in range(20)
        }
        notes = [_make_note(path=f"note_{i}.md", title=f"Note {i}") for i in range(20)]
        config = FakeDigestConfig(max_trending=5)
        generator = _make_generator(notes, tmp_path, graph_nodes=graph_nodes, config=config)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_source_note_mtime", return_value=recent):
            report = generator.generate()

        assert len(report.trending_entities) <= 5


# ---------------------------------------------------------------------------
# Tests — Suggested connections
# ---------------------------------------------------------------------------


class TestSuggestedConnections:
    def test_unlinked_pair_included(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Alpha", content="---\n---\n\n" + "Alpha " * 30)
        note_b = _make_note(path="b.md", title="Beta")

        # Hit for b.md at distance 0.20 (similarity 0.80 — within 0.70-0.85 band)
        store_results = [_make_hit("b.md", "Beta", distance=0.20)]
        generator = _make_generator([note_a, note_b], tmp_path, store_results=store_results)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        assert any(
            (c.note_a == "Alpha" and c.note_b == "Beta")
            or (c.note_a == "Beta" and c.note_b == "Alpha")
            for c in report.suggested_connections
        )

    def test_already_linked_pair_excluded(self, tmp_path: Path) -> None:
        # note_a links to note_b via wikilink
        content_a = "---\n---\n\n[[Beta]] is related. " * 20
        note_a = _make_note(path="a.md", title="Alpha", content=content_a)
        note_b = _make_note(path="b.md", title="Beta")

        store_results = [_make_hit("b.md", "Beta", distance=0.20)]
        generator = _make_generator([note_a, note_b], tmp_path, store_results=store_results)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        # Should not suggest since they're already linked
        assert not any(
            (c.note_a == "Alpha" and c.note_b == "Beta")
            or (c.note_a == "Beta" and c.note_b == "Alpha")
            for c in report.suggested_connections
        )

    def test_self_pair_excluded(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Alpha")
        store_results = [_make_hit("a.md", "Alpha", distance=0.20)]
        generator = _make_generator([note_a], tmp_path, store_results=store_results)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        assert report.suggested_connections == []

    def test_out_of_band_excluded(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Alpha", content="---\n---\n\n" + "A " * 30)
        note_b = _make_note(path="b.md", title="Beta")

        # distance 0.05 → similarity 0.95, above threshold_high=0.85 → out of band (duplicate)
        store_results = [_make_hit("b.md", "Beta", distance=0.05)]
        generator = _make_generator([note_a, note_b], tmp_path, store_results=store_results)

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        assert report.suggested_connections == []

    def test_max_suggestions_respected(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Alpha", content="---\n---\n\n" + "A " * 30)
        other_notes = [_make_note(path=f"other_{i}.md", title=f"Note {i}") for i in range(10)]
        store_results = [_make_hit(f"other_{i}.md", f"Note {i}", distance=0.20) for i in range(10)]

        config = FakeDigestConfig(max_suggestions=3)
        generator = _make_generator(
            [note_a] + other_notes, tmp_path, store_results=store_results, config=config
        )

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        assert len(report.suggested_connections) <= 3

    def test_duplicate_pair_deduplicated(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Alpha", content="---\n---\n\n" + "A " * 30)
        note_b = _make_note(path="b.md", title="Beta", content="---\n---\n\n" + "B " * 30)

        store_results_a = [_make_hit("b.md", "Beta", distance=0.20)]
        store_results_b = [_make_hit("a.md", "Alpha", distance=0.20)]

        # Both are "active" — both would produce the same pair
        store = FakeStore(store_results_a + store_results_b)
        config = FakeDigestConfig()
        parser = FakeParser([note_a, note_b], tmp_path)
        from vaultmind.indexer.digest import DigestGenerator

        generator = DigestGenerator(
            store=store,  # type: ignore[arg-type]
            graph=FakeGraph(),  # type: ignore[arg-type]
            parser=parser,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
        )

        recent = datetime.now(tz=UTC) - timedelta(days=1)
        with patch.object(generator, "_file_ctime", return_value=recent):
            report = generator.generate()

        # Same pair should appear only once
        pairs = [frozenset([c.note_a, c.note_b]) for c in report.suggested_connections]
        assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# Tests — Orphan note detection
# ---------------------------------------------------------------------------


class TestOrphanNotes:
    def test_note_with_no_links_is_orphan(self, tmp_path: Path) -> None:
        note_a = _make_note(path="a.md", title="Isolated")
        note_b = _make_note(path="b.md", title="Linked")
        # b links to itself — not to a
        content_c = "---\n---\n\n[[Linked]] is here. " * 5
        note_c = _make_note(path="c.md", title="Connector", content=content_c)

        generator = _make_generator([note_a, note_b, note_c], tmp_path)

        with (
            patch.object(generator, "_file_mtime", return_value=None),
            patch.object(generator, "_file_ctime", return_value=None),
        ):
            report = generator.generate()

        assert "Isolated" in report.orphan_notes

    def test_linked_note_not_orphan(self, tmp_path: Path) -> None:
        content_a = "---\n---\n\n[[Beta]] is related. " * 5
        note_a = _make_note(path="a.md", title="Alpha", content=content_a)
        note_b = _make_note(path="b.md", title="Beta")

        generator = _make_generator([note_a, note_b], tmp_path)

        with (
            patch.object(generator, "_file_mtime", return_value=None),
            patch.object(generator, "_file_ctime", return_value=None),
        ):
            report = generator.generate()

        assert "Alpha" not in report.orphan_notes
        assert "Beta" not in report.orphan_notes

    def test_orphan_list_sorted(self, tmp_path: Path) -> None:
        notes = [
            _make_note(path="z.md", title="Zulu"),
            _make_note(path="a.md", title="Alpha"),
            _make_note(path="m.md", title="Mike"),
        ]
        generator = _make_generator(notes, tmp_path)

        with (
            patch.object(generator, "_file_mtime", return_value=None),
            patch.object(generator, "_file_ctime", return_value=None),
        ):
            report = generator.generate()

        assert report.orphan_notes == sorted(report.orphan_notes)

    def test_empty_vault_no_orphans(self, tmp_path: Path) -> None:
        generator = _make_generator([], tmp_path)
        report = generator.generate()
        assert report.orphan_notes == []


# ---------------------------------------------------------------------------
# Tests — Telegram HTML formatting
# ---------------------------------------------------------------------------


class TestFormatTelegram:
    def _generator(self, tmp_path: Path) -> DigestGenerator:
        return _make_generator([], tmp_path)

    def test_empty_report_message(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(generated_at=now, period_days=7)

        msg = gen.format_telegram(report)
        assert "No activity" in msg
        assert "last 7 days" in msg

    def test_activity_section(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            new_notes=["Note A", "Note B"],
            modified_notes=["Note C"],
        )

        msg = gen.format_telegram(report)
        assert "📝 Activity" in msg
        assert "2 new notes" in msg
        assert "1 modified" in msg

    def test_trending_section(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            trending_entities=[
                TrendingEntity(name="Python", current_count=5, previous_count=2, delta=3),
            ],
        )

        msg = gen.format_telegram(report)
        assert "🔥 Trending Topics" in msg
        assert "Python" in msg
        assert "+3 mentions" in msg

    def test_suggested_connections_section(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            suggested_connections=[
                SuggestedConnection(note_a="Alpha", note_b="Beta", similarity=0.78),
            ],
        )

        msg = gen.format_telegram(report)
        assert "🔗 Suggested Connections" in msg
        assert "Alpha" in msg
        assert "Beta" in msg
        assert "↔" in msg
        assert "78%" in msg

    def test_orphan_notes_section(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            orphan_notes=["Isolated Note"],
        )

        msg = gen.format_telegram(report)
        assert "🏝 Orphan Notes" in msg
        assert "Isolated Note" in msg
        assert "(no links)" in msg

    def test_empty_sections_omitted(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            new_notes=["Only Note"],
        )

        msg = gen.format_telegram(report)
        assert "📝 Activity" in msg
        assert "🔥 Trending Topics" not in msg
        assert "🔗 Suggested Connections" not in msg
        assert "🏝 Orphan Notes" not in msg

    def test_html_escaping(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            trending_entities=[
                TrendingEntity(name="<Script>", current_count=1, previous_count=0, delta=1),
            ],
        )

        msg = gen.format_telegram(report)
        assert "<Script>" not in msg
        assert "&lt;Script&gt;" in msg

    def test_footer_contains_stats(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            new_notes=["A"],
            total_notes=42,
            total_entities=15,
        )

        msg = gen.format_telegram(report)
        assert "42 total notes" in msg
        assert "15 entities" in msg

    def test_date_in_header(self, tmp_path: Path) -> None:
        gen = self._generator(tmp_path)
        now = datetime(2026, 2, 25, 8, 0, 0, tzinfo=UTC)
        report = DigestReport(generated_at=now, period_days=7, new_notes=["A"])

        msg = gen.format_telegram(report)
        assert "2026-02-25" in msg

    def test_max_trending_respected_in_format(self, tmp_path: Path) -> None:
        config = FakeDigestConfig(max_trending=2)
        gen = _make_generator([], tmp_path, config=config)
        now = datetime.now(tz=UTC)
        entities = [
            TrendingEntity(name=f"Entity{i}", current_count=i, previous_count=0, delta=i)
            for i in range(5, 0, -1)
        ]
        report = DigestReport(generated_at=now, period_days=7, trending_entities=entities)

        msg = gen.format_telegram(report)
        # Only first 2 should appear
        assert "Entity5" in msg
        assert "Entity4" in msg
        assert "Entity1" not in msg


# ---------------------------------------------------------------------------
# Tests — Vault markdown save
# ---------------------------------------------------------------------------


class TestSaveToVault:
    def test_creates_directory_and_file(self, tmp_path: Path) -> None:
        gen = _make_generator([], tmp_path)
        now = datetime(2026, 2, 25, 8, 0, 0, tzinfo=UTC)
        report = DigestReport(generated_at=now, period_days=7)

        dest = gen.save_to_vault(report, tmp_path)

        assert dest.exists()
        assert dest.parent == tmp_path / "_meta" / "digests"
        assert dest.name == "2026-02-25.md"

    def test_frontmatter_present(self, tmp_path: Path) -> None:
        gen = _make_generator([], tmp_path)
        now = datetime(2026, 2, 25, 8, 0, 0, tzinfo=UTC)
        report = DigestReport(generated_at=now, period_days=7)

        dest = gen.save_to_vault(report, tmp_path)
        content = dest.read_text()

        assert "---" in content
        assert "title:" in content
        assert "date: 2026-02-25" in content
        assert "tags: [digest, auto-generated]" in content
        assert "type: digest" in content

    def test_sections_in_markdown(self, tmp_path: Path) -> None:
        gen = _make_generator([], tmp_path)
        now = datetime(2026, 2, 25, 8, 0, 0, tzinfo=UTC)
        report = DigestReport(
            generated_at=now,
            period_days=7,
            new_notes=["Alpha"],
            modified_notes=["Beta"],
            trending_entities=[
                TrendingEntity(name="Python", current_count=3, previous_count=1, delta=2)
            ],
            suggested_connections=[SuggestedConnection(note_a="X", note_b="Y", similarity=0.72)],
            orphan_notes=["Loner"],
        )

        dest = gen.save_to_vault(report, tmp_path)
        content = dest.read_text()

        assert "## Activity" in content
        assert "## Trending Topics" in content
        assert "## Suggested Connections" in content
        assert "## Orphan Notes" in content
        assert "[[X]]" in content
        assert "[[Loner]]" in content

    def test_returns_path(self, tmp_path: Path) -> None:
        gen = _make_generator([], tmp_path)
        now = datetime.now(tz=UTC)
        report = DigestReport(generated_at=now, period_days=7)

        dest = gen.save_to_vault(report, tmp_path)
        assert isinstance(dest, Path)
        assert dest.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        gen = _make_generator([], tmp_path)
        now = datetime(2026, 2, 25, 8, 0, 0, tzinfo=UTC)
        report = DigestReport(generated_at=now, period_days=7)

        dest1 = gen.save_to_vault(report, tmp_path)
        dest2 = gen.save_to_vault(report, tmp_path)

        assert dest1 == dest2
        assert dest2.exists()


# ---------------------------------------------------------------------------
# Tests — generate() integration (mocked file times)
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_total_notes_count(self, tmp_path: Path) -> None:
        notes = [_make_note(path=f"{i}.md", title=f"Note {i}") for i in range(5)]
        generator = _make_generator(notes, tmp_path)

        with (
            patch.object(generator, "_file_ctime", return_value=None),
            patch.object(generator, "_file_mtime", return_value=None),
        ):
            report = generator.generate()

        assert report.total_notes == 5

    def test_period_days_in_report(self, tmp_path: Path) -> None:
        config = FakeDigestConfig(period_days=14)
        generator = _make_generator([], tmp_path, config=config)
        report = generator.generate()
        assert report.period_days == 14

    def test_generated_at_is_utc(self, tmp_path: Path) -> None:
        generator = _make_generator([], tmp_path)
        report = generator.generate()
        assert report.generated_at.tzinfo is not None
        assert report.generated_at.tzinfo == UTC

    def test_no_activity_all_old(self, tmp_path: Path) -> None:
        notes = [_make_note(path="old.md", title="Old")]
        generator = _make_generator(notes, tmp_path)

        old = datetime.now(tz=UTC) - timedelta(days=30)
        with (
            patch.object(generator, "_file_ctime", return_value=old),
            patch.object(generator, "_file_mtime", return_value=old),
        ):
            report = generator.generate()

        assert report.new_notes == []
        assert report.modified_notes == []

    def test_total_entities_from_graph(self, tmp_path: Path) -> None:
        graph_nodes = {
            "python": {"label": "Python", "type": "concept", "source_notes": []},
            "rust": {"label": "Rust", "type": "concept", "source_notes": []},
        }
        generator = _make_generator([], tmp_path, graph_nodes=graph_nodes)
        report = generator.generate()
        assert report.total_entities == 2
