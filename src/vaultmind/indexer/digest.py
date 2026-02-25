"""Smart Daily Digest — metadata-only vault activity report.

Zero LLM cost: all computation from file mtimes, graph node counts, and
ChromaDB similarity queries.  No API calls beyond what the store already has
indexed.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.config import DigestConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.models import Note
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)

# Similarity band for suggested connections
# Using cosine distance: low distance = high similarity
_CONNECTION_DEFAULT_LOW_DISTANCE = 0.15  # similarity ≥ 0.85
_CONNECTION_DEFAULT_HIGH_DISTANCE = 0.30  # similarity ≥ 0.70


@dataclass
class TrendingEntity:
    """An entity whose mention frequency is rising in the digest period."""

    name: str
    current_count: int
    previous_count: int
    delta: int


@dataclass
class SuggestedConnection:
    """A high-similarity pair of notes that are not yet wikilinked."""

    note_a: str
    note_b: str
    similarity: float


@dataclass
class DigestReport:
    """Full vault activity digest for a time period."""

    generated_at: datetime
    period_days: int
    new_notes: list[str] = field(default_factory=list)
    modified_notes: list[str] = field(default_factory=list)
    trending_entities: list[TrendingEntity] = field(default_factory=list)
    suggested_connections: list[SuggestedConnection] = field(default_factory=list)
    orphan_notes: list[str] = field(default_factory=list)
    total_notes: int = 0
    total_entities: int = 0


class DigestGenerator:
    """Generates a vault digest report from metadata, graph stats, and ChromaDB queries.

    All computation is local — zero LLM API cost.
    """

    def __init__(
        self,
        store: VaultStore,
        graph: KnowledgeGraph,
        parser: VaultParser,
        config: DigestConfig,
    ) -> None:
        self._store = store
        self._graph = graph
        self._parser = parser
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> DigestReport:
        """Generate a digest report for the configured period."""
        now = datetime.now(tz=UTC)
        period_start = now - timedelta(days=self._config.period_days)
        prev_start = period_start - timedelta(days=self._config.period_days)

        notes = self._parser.iter_notes()

        new_notes: list[str] = []
        modified_notes: list[str] = []

        for note in notes:
            mtime = self._file_mtime(note.path)
            ctime = self._file_ctime(note.path)

            if ctime is not None and ctime >= period_start:
                new_notes.append(note.title)
            elif mtime is not None and mtime >= period_start:
                modified_notes.append(note.title)

        # Trending entities: compare current window vs previous window
        trending = self._compute_trending(notes, period_start, prev_start)

        # Suggested connections for notes active in the period
        active_titles = set(new_notes) | set(modified_notes)
        active_notes = [n for n in notes if n.title in active_titles]
        suggested = self._compute_suggested_connections(active_notes, notes)

        # Orphan notes: notes with zero wikilinks in or out
        orphan_notes = self._compute_orphans(notes)

        graph_stats = self._graph.stats

        return DigestReport(
            generated_at=now,
            period_days=self._config.period_days,
            new_notes=new_notes,
            modified_notes=modified_notes,
            trending_entities=trending,
            suggested_connections=suggested,
            orphan_notes=orphan_notes,
            total_notes=len(notes),
            total_entities=graph_stats["nodes"],
        )

    def format_telegram(self, report: DigestReport) -> str:
        """Format the report as a Telegram HTML message."""
        date_str = report.generated_at.strftime("%Y-%m-%d")

        if (
            not report.new_notes
            and not report.modified_notes
            and not report.trending_entities
            and not report.suggested_connections
            and not report.orphan_notes
        ):
            return (
                f"<b>📊 Daily Digest — {date_str}</b>\n\n"
                f"<i>No activity in the last {report.period_days} days.</i>"
            )

        lines: list[str] = [f"<b>📊 Daily Digest — {date_str}</b>"]

        # Activity section
        if report.new_notes or report.modified_notes:
            lines.append("")
            lines.append("<b>📝 Activity</b>")
            activity_parts: list[str] = []
            if report.new_notes:
                activity_parts.append(f"{len(report.new_notes)} new notes")
            if report.modified_notes:
                activity_parts.append(f"{len(report.modified_notes)} modified")
            lines.append("• " + ", ".join(activity_parts))

        # Trending topics
        if report.trending_entities:
            lines.append("")
            lines.append("<b>🔥 Trending Topics</b>")
            for entity in report.trending_entities[: self._config.max_trending]:
                name = html.escape(entity.name)
                lines.append(f"• {name} (+{entity.delta} mentions)")

        # Suggested connections
        if report.suggested_connections:
            lines.append("")
            lines.append("<b>🔗 Suggested Connections</b>")
            for conn in report.suggested_connections[: self._config.max_suggestions]:
                note_a = html.escape(conn.note_a)
                note_b = html.escape(conn.note_b)
                pct = int(conn.similarity * 100)
                lines.append(f"• {note_a} ↔ {note_b} ({pct}%)")

        # Orphan notes
        if report.orphan_notes:
            lines.append("")
            lines.append("<b>🏝 Orphan Notes</b>")
            for title in report.orphan_notes:
                lines.append(f"• {html.escape(title)} (no links)")

        # Footer
        lines.append("")
        lines.append(
            f"<i>Period: last {report.period_days} days"
            f" • {report.total_notes} total notes"
            f" • {report.total_entities} entities</i>"
        )

        return "\n".join(lines)

    def save_to_vault(self, report: DigestReport, vault_root: Path) -> Path:
        """Write the digest as a markdown file to {vault_root}/_meta/digests/YYYY-MM-DD.md."""
        date_str = report.generated_at.strftime("%Y-%m-%d")
        digest_dir = vault_root / "_meta" / "digests"
        digest_dir.mkdir(parents=True, exist_ok=True)
        dest = digest_dir / f"{date_str}.md"

        content = self._format_markdown(report)
        dest.write_text(content, encoding="utf-8")
        logger.info("Digest saved to %s", dest)
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _file_mtime(self, rel_path: Path) -> datetime | None:
        """Return mtime of a vault file as timezone-aware UTC datetime."""
        abs_path = self._parser.vault_root / rel_path
        if not abs_path.exists():
            return None
        return datetime.fromtimestamp(abs_path.stat().st_mtime, tz=UTC)

    def _file_ctime(self, rel_path: Path) -> datetime | None:
        """Return ctime (creation time on most systems) as timezone-aware UTC datetime."""
        abs_path = self._parser.vault_root / rel_path
        if not abs_path.exists():
            return None
        return datetime.fromtimestamp(abs_path.stat().st_ctime, tz=UTC)

    def _compute_trending(
        self,
        notes: list[Note],
        period_start: datetime,
        prev_start: datetime,
    ) -> list[TrendingEntity]:
        """Compare entity mention counts between current and previous window."""
        # Build per-window mention counts from graph node source_notes
        current: dict[str, int] = {}
        previous: dict[str, int] = {}

        for node_id in self._graph._graph.nodes:
            data = self._graph._graph.nodes[node_id]
            label = data.get("label", node_id)
            source_notes: list[str] = data.get("source_notes", [])

            for source_path in source_notes:
                # Resolve mtime for the source note path
                mtime = self._source_note_mtime(source_path, notes)
                if mtime is None:
                    continue
                if mtime >= period_start:
                    current[label] = current.get(label, 0) + 1
                elif mtime >= prev_start:
                    previous[label] = previous.get(label, 0) + 1

        trending: list[TrendingEntity] = []
        for entity, cur_count in current.items():
            prev_count = previous.get(entity, 0)
            delta = cur_count - prev_count
            if delta > 0:
                trending.append(
                    TrendingEntity(
                        name=entity,
                        current_count=cur_count,
                        previous_count=prev_count,
                        delta=delta,
                    )
                )

        trending.sort(key=lambda e: e.delta, reverse=True)
        return trending[: self._config.max_trending]

    def _source_note_mtime(self, source_path: str, notes: list[Note]) -> datetime | None:
        """Get mtime for a graph source note path string."""
        # source_path is stored as the relative path string
        for note in notes:
            if str(note.path) == source_path or note.title == source_path:
                return self._file_mtime(note.path)
        return None

    def _compute_suggested_connections(
        self,
        active_notes: list[Note],
        all_notes: list[Note],
    ) -> list[SuggestedConnection]:
        """Find high-similarity unlinked pairs for notes active in the period."""
        # Build wikilink map: note title → set of titles it links to
        wikilink_map: dict[str, set[str]] = {}
        for note in all_notes:
            wikilink_map[note.title] = set(note.wikilinks)

        low_dist = 1.0 - self._config.connection_threshold_high
        high_dist = 1.0 - self._config.connection_threshold_low

        seen_pairs: set[frozenset[str]] = set()
        suggestions: list[SuggestedConnection] = []

        for note in active_notes:
            body = note.body_without_frontmatter().strip()
            if not body:
                continue

            raw_results = self._store.search(
                query=body[:2000],
                n_results=20,
            )

            for hit in raw_results:
                meta = hit.get("metadata", {})
                hit_path = meta.get("note_path", "")
                hit_title = meta.get("note_title", "Untitled")
                distance = hit.get("distance", 1.0)

                if hit_path == str(note.path):
                    continue

                if distance < low_dist or distance > high_dist:
                    continue

                pair = frozenset([note.title, hit_title])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Skip if already linked in either direction
                if hit_title in wikilink_map.get(note.title, set()):
                    continue
                if note.title in wikilink_map.get(hit_title, set()):
                    continue

                similarity = round(1.0 - distance, 4)
                suggestions.append(
                    SuggestedConnection(
                        note_a=note.title,
                        note_b=hit_title,
                        similarity=similarity,
                    )
                )

        suggestions.sort(key=lambda c: c.similarity, reverse=True)
        return suggestions[: self._config.max_suggestions]

    def _compute_orphans(self, notes: list[Note]) -> list[str]:
        """Find notes with zero wikilinks in or out across all vault notes."""
        # Build a set of all note titles
        all_titles = {note.title for note in notes}

        # Build outgoing links per note
        outgoing: dict[str, set[str]] = {}
        incoming: dict[str, set[str]] = {}

        for note in notes:
            outgoing[note.title] = set()
            for link in note.wikilinks:
                # Match against known titles (exact match)
                if link in all_titles:
                    outgoing[note.title].add(link)
                    incoming.setdefault(link, set()).add(note.title)

        orphans: list[str] = []
        for note in notes:
            has_out = bool(outgoing.get(note.title))
            has_in = bool(incoming.get(note.title))
            if not has_out and not has_in:
                orphans.append(note.title)

        return sorted(orphans)

    def _format_markdown(self, report: DigestReport) -> str:
        """Format the report as a markdown document with frontmatter."""
        date_str = report.generated_at.strftime("%Y-%m-%d")
        generated_iso = report.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        lines: list[str] = [
            "---",
            f"title: Daily Digest — {date_str}",
            f"date: {date_str}",
            "tags: [digest, auto-generated]",
            "type: digest",
            f"generated_at: {generated_iso}",
            f"period_days: {report.period_days}",
            "---",
            "",
            f"# Daily Digest — {date_str}",
            "",
        ]

        # Activity
        if report.new_notes or report.modified_notes:
            lines.append("## Activity")
            lines.append("")
            if report.new_notes:
                lines.append(f"**New notes ({len(report.new_notes)})**")
                lines.append("")
                for title in report.new_notes:
                    lines.append(f"- {title}")
                lines.append("")
            if report.modified_notes:
                lines.append(f"**Modified notes ({len(report.modified_notes)})**")
                lines.append("")
                for title in report.modified_notes:
                    lines.append(f"- {title}")
                lines.append("")

        # Trending entities
        if report.trending_entities:
            lines.append("## Trending Topics")
            lines.append("")
            for entity in report.trending_entities:
                lines.append(f"- **{entity.name}** (+{entity.delta} mentions)")
            lines.append("")

        # Suggested connections
        if report.suggested_connections:
            lines.append("## Suggested Connections")
            lines.append("")
            for conn in report.suggested_connections:
                pct = int(conn.similarity * 100)
                lines.append(f"- [[{conn.note_a}]] ↔ [[{conn.note_b}]] ({pct}% similarity)")
            lines.append("")

        # Orphan notes
        if report.orphan_notes:
            lines.append("## Orphan Notes")
            lines.append("")
            for title in report.orphan_notes:
                lines.append(f"- [[{title}]]")
            lines.append("")

        # Stats footer
        lines.append("---")
        lines.append("")
        lines.append(
            f"*{report.total_notes} total notes — "
            f"{report.total_entities} entities — "
            f"last {report.period_days} days*"
        )

        return "\n".join(lines)
