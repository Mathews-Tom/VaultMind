"""Memory consolidation pipeline for episodic memory lifecycle management.

Archives old resolved episodes, summarizes clusters of related episodes
into lesson notes, and promotes frequently-referenced episodes to
permanent vault notes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.memory.store import EpisodeStore

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    """Results of a consolidation run."""

    archived_count: int = 0
    summaries_created: list[str] = field(default_factory=list)
    promoted_count: int = 0
    errors: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """Periodic consolidation of episodic memory."""

    def __init__(
        self,
        episode_store: EpisodeStore,
        vault_root: Path,
        llm_client: Any = None,
        retention_days: int = 365,
        min_references_for_promotion: int = 3,
        synthesis_model: str = "",
    ) -> None:
        self._store = episode_store
        self._vault_root = vault_root
        self._llm = llm_client
        self._retention_days = retention_days
        self._min_refs = min_references_for_promotion
        self._model = synthesis_model

    def consolidate(self) -> ConsolidationReport:
        """Run full consolidation: archive, summarize, promote."""
        report = ConsolidationReport()

        # Step 1: Archive old resolved episodes
        try:
            report.archived_count = self._archive_old_episodes()
        except Exception as e:
            logger.warning("Archival failed: %s", e)
            report.errors.append(f"archive: {e}")

        # Step 2: Summarize clusters of archived episodes into lesson notes
        try:
            report.summaries_created = self._summarize_clusters()
        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            report.errors.append(f"summarize: {e}")

        # Step 3: Promote frequently-referenced episodes
        try:
            report.promoted_count = self._promote_referenced()
        except Exception as e:
            logger.warning("Promotion failed: %s", e)
            report.errors.append(f"promote: {e}")

        logger.info(
            "Consolidation complete: %d archived, %d summaries, %d promoted",
            report.archived_count,
            len(report.summaries_created),
            report.promoted_count,
        )
        return report

    def _archive_old_episodes(self) -> int:
        """Mark old resolved episodes as archived."""
        return self._store.archive_old_resolved(age_days=self._retention_days)

    def _summarize_clusters(self) -> list[str]:
        """Group archived episodes by shared entities, write lesson notes.

        Returns list of created note paths.
        """
        if self._llm is None:
            return []

        archived = self._store.query_archived(limit=200)
        if not archived:
            return []

        # Group by shared entities
        entity_groups: dict[str, list[Any]] = defaultdict(list)
        for ep in archived:
            for entity in ep.entities:
                entity_groups[entity.lower()].append(ep)

        # Only summarize clusters with 3+ episodes
        created_paths: list[str] = []
        for entity, episodes in entity_groups.items():
            if len(episodes) < 3:
                continue

            # Deduplicate episodes (same episode may appear in multiple entity groups)
            seen_ids: set[str] = set()
            unique_episodes = []
            for ep in episodes:
                if ep.episode_id not in seen_ids:
                    seen_ids.add(ep.episode_id)
                    unique_episodes.append(ep)

            if len(unique_episodes) < 3:
                continue

            try:
                note_path = self._write_lesson_note(entity, unique_episodes[:10])
                created_paths.append(note_path)
            except Exception:
                logger.warning("Failed to create lesson note for '%s'", entity, exc_info=True)

        return created_paths

    def _write_lesson_note(self, entity: str, episodes: list[Any]) -> str:
        """Generate and write a lesson note from a cluster of episodes."""
        from vaultmind.llm.client import Message

        episode_text = "\n".join(
            f"- Decision: {ep.decision}\n  Outcome: {ep.outcome} ({ep.outcome_status.value})\n"
            f"  Lessons: {'; '.join(ep.lessons[:3])}"
            for ep in episodes
        )

        prompt = (
            f"Synthesize lessons from these related decisions about '{entity}':\n\n"
            f"{episode_text}\n\n"
            "Write a concise lesson note (3-5 bullet points) capturing the key patterns "
            "and actionable takeaways. Format as markdown."
        )

        response = self._llm.complete(
            messages=[Message(role="user", content=prompt)],
            model=self._model or "default",
            max_tokens=500,
            system="You synthesize decision-outcome patterns into actionable lessons.",
        )

        # Write to vault
        slug = entity.replace(" ", "-").lower()[:40]
        now = datetime.now()
        filename = f"lesson-{slug}-{now.strftime('%Y%m%d')}.md"
        rel_path = f"_meta/lessons/{filename}"
        filepath = self._vault_root / rel_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        content = (
            f"---\n"
            f"title: Lessons — {entity}\n"
            f"type: permanent\n"
            f"tags: [lesson, consolidation]\n"
            f"created: {now.strftime('%Y-%m-%d %H:%M')}\n"
            f"source: consolidation\n"
            f"status: active\n"
            f"---\n\n"
            f"# Lessons — {entity}\n\n"
            f"{response.text}\n\n"
            f"---\n\n"
            f"*Synthesized from {len(episodes)} episodes.*\n"
        )
        filepath.write_text(content, encoding="utf-8")
        logger.info("Created lesson note: %s", rel_path)
        return rel_path

    def _promote_referenced(self) -> int:
        """Promote episodes referenced by 3+ other episodes to permanent notes.

        Finds episodes whose entities appear in many other episodes,
        writes them as permanent vault notes.
        """
        resolved = self._store.query_resolved(limit=200)
        promoted = 0

        for ep in resolved:
            # Count how many other episodes share entities with this one
            total_refs = 0
            for entity in ep.entities:
                total_refs += self._store.count_entity_references(entity)

            # Subtract self-references (this episode counts itself)
            total_refs = max(0, total_refs - len(ep.entities))

            if total_refs >= self._min_refs:
                try:
                    self._write_promoted_note(ep)
                    promoted += 1
                except Exception:
                    logger.warning("Failed to promote episode %s", ep.episode_id, exc_info=True)

        if promoted > 0:
            logger.info("Promoted %d episodes to permanent notes", promoted)
        return promoted

    def _write_promoted_note(self, ep: Any) -> str:
        """Write a promoted episode as a permanent vault note."""
        slug = ep.decision[:40].replace(" ", "-").lower()
        now = datetime.now()
        filename = f"episode-{slug}-{now.strftime('%Y%m%d')}.md"
        rel_path = f"_meta/episodes/{filename}"
        filepath = self._vault_root / rel_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        lessons_md = "\n".join(f"- {lesson}" for lesson in ep.lessons) if ep.lessons else "- (none)"
        entities_md = ", ".join(ep.entities) if ep.entities else "(none)"

        content = (
            f"---\n"
            f"title: {ep.decision[:60]}\n"
            f"type: permanent\n"
            f"tags: [episode, promoted]\n"
            f"created: {now.strftime('%Y-%m-%d %H:%M')}\n"
            f"source: consolidation\n"
            f"status: active\n"
            f"---\n\n"
            f"# {ep.decision}\n\n"
            f"**Context:** {ep.context or '(none)'}\n\n"
            f"**Outcome:** {ep.outcome} ({ep.outcome_status.value})\n\n"
            f"## Lessons\n\n{lessons_md}\n\n"
            f"**Entities:** {entities_md}\n"
        )
        filepath.write_text(content, encoding="utf-8")
        logger.info("Promoted episode to note: %s", rel_path)
        return rel_path
