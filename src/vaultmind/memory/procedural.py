"""Procedural memory — self-evolving workflow patterns mined from episodic memory.

This module is DISABLED by default (experimental). Enable via config:
    [procedural]
    enabled = true
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime for mkdir/connect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaultmind.llm.client import LLMClient
    from vaultmind.memory.models import Episode
    from vaultmind.memory.store import EpisodeStore

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    steps TEXT NOT NULL DEFAULT '[]',
    trigger_pattern TEXT NOT NULL DEFAULT '',
    success_rate REAL NOT NULL DEFAULT 0.0,
    usage_count INTEGER NOT NULL DEFAULT 0,
    source_episodes TEXT NOT NULL DEFAULT '[]',
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_workflows_active ON workflows(active);
"""

_SYNTHESIS_PROMPT = """\
Given these resolved decision-outcome episodes, identify a reusable workflow pattern.

Episodes:
{episodes}

Extract a workflow with these fields (return as JSON object):
- name: short workflow name
- description: what this workflow achieves
- steps: ordered list of action steps
- trigger_pattern: when to apply this workflow (natural language)

Return ONLY valid JSON. Return null if no clear pattern exists.\
"""


@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    steps: list[str]
    trigger_pattern: str
    success_rate: float
    usage_count: int
    source_episodes: list[str]
    created: datetime
    updated: datetime
    active: bool = True


def _row_to_workflow(row: sqlite3.Row) -> Workflow:
    return Workflow(
        workflow_id=row["workflow_id"],
        name=row["name"],
        description=row["description"],
        steps=json.loads(row["steps"]),
        trigger_pattern=row["trigger_pattern"],
        success_rate=row["success_rate"],
        usage_count=row["usage_count"],
        source_episodes=json.loads(row["source_episodes"]),
        created=datetime.fromisoformat(row["created"]),
        updated=datetime.fromisoformat(row["updated"]),
        active=bool(row["active"]),
    )


class ProceduralMemory:
    """SQLite-backed store for workflow patterns synthesized from episodic memory."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        name: str,
        description: str,
        steps: list[str],
        trigger_pattern: str,
        source_episodes: list[str] | None = None,
    ) -> Workflow:
        """Create and persist a workflow directly (useful for testing and manual creation)."""
        now = datetime.now()
        workflow = Workflow(
            workflow_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            steps=steps,
            trigger_pattern=trigger_pattern,
            success_rate=0.0,
            usage_count=0,
            source_episodes=source_episodes or [],
            created=now,
            updated=now,
            active=True,
        )
        self._persist(workflow)
        return workflow

    def synthesize_workflows(
        self,
        episode_store: EpisodeStore,
        llm_client: LLMClient,
        model: str,
        min_episodes: int = 3,
    ) -> list[Workflow]:
        """Mine resolved episodes for recurring patterns and create Workflow objects."""
        resolved = episode_store.query_resolved()
        if not resolved:
            logger.info("No resolved episodes to synthesize workflows from.")
            return []

        # Group episodes by shared entities
        groups: dict[str, list[str]] = {}  # entity -> [episode_id]
        for ep in resolved:
            for entity in ep.entities:
                key = entity.lower()
                if key not in groups:
                    groups[key] = []
                groups[key].append(ep.episode_id)

        # Build episode index for quick lookup
        ep_index = {ep.episode_id: ep for ep in resolved}

        # Collect groups with enough episodes (deduplicate groups by frozenset of IDs)
        seen_groups: list[frozenset[str]] = []
        candidate_groups: list[list[str]] = []
        for ep_ids in groups.values():
            unique_ids = list(set(ep_ids))
            if len(unique_ids) < min_episodes:
                continue
            group_key: frozenset[str] = frozenset(unique_ids)
            if group_key in seen_groups:
                continue
            seen_groups.append(group_key)
            candidate_groups.append(unique_ids)

        if not candidate_groups:
            logger.info("No entity groups with >= %d resolved episodes.", min_episodes)
            return []

        new_workflows: list[Workflow] = []
        for ep_ids in candidate_groups:
            episodes = [ep_index[eid] for eid in ep_ids if eid in ep_index]
            formatted = _format_episodes(episodes)

            prompt = _SYNTHESIS_PROMPT.format(episodes=formatted)
            from vaultmind.llm.client import Message

            try:
                response = llm_client.complete(
                    messages=[Message(role="user", content=prompt)],
                    model=model,
                )
                raw = response.text.strip()
            except Exception:
                logger.exception("LLM synthesis failed for group of %d episodes.", len(episodes))
                continue

            if not raw or raw.lower() in {"null", "none", "{}"}:
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON LLM response for workflow synthesis: %.120s", raw)
                continue

            if not isinstance(data, dict) or not data:
                continue

            name = str(data.get("name", "Unnamed Workflow"))
            description = str(data.get("description", ""))
            steps_raw = data.get("steps", [])
            steps = [str(s) for s in steps_raw] if isinstance(steps_raw, list) else []
            trigger_pattern = str(data.get("trigger_pattern", ""))

            workflow = self.create_workflow(
                name=name,
                description=description,
                steps=steps,
                trigger_pattern=trigger_pattern,
                source_episodes=ep_ids,
            )
            new_workflows.append(workflow)
            logger.info("Synthesized workflow '%s' from %d episodes.", name, len(ep_ids))

        return new_workflows

    def suggest_workflow(self, context: str) -> Workflow | None:
        """Return the highest-usage active workflow whose trigger_pattern overlaps with context."""
        context_words = set(context.lower().split())
        if not context_words:
            return None

        workflows = self.list_active()
        best: Workflow | None = None
        best_overlap = 0

        for wf in workflows:
            trigger_words = set(wf.trigger_pattern.lower().split())
            overlap = len(context_words & trigger_words)
            if overlap == 0:
                continue
            if overlap > best_overlap or (
                overlap == best_overlap and best is not None and wf.usage_count > best.usage_count
            ):
                best = wf
                best_overlap = overlap

        return best

    def record_usage(self, workflow_id: str, success: bool) -> None:
        """Update running success rate and increment usage count."""
        row = self._conn.execute(
            "SELECT success_rate, usage_count FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            logger.warning("record_usage: workflow %s not found.", workflow_id)
            return

        old_rate: float = row["success_rate"]
        old_count: int = row["usage_count"]
        new_count = old_count + 1
        new_rate = ((old_rate * old_count) + (1.0 if success else 0.0)) / new_count

        self._conn.execute(
            "UPDATE workflows"
            " SET success_rate = ?, usage_count = ?, updated = ?"
            " WHERE workflow_id = ?",
            (new_rate, new_count, datetime.now().isoformat(), workflow_id),
        )
        self._conn.commit()

    def list_active(self) -> list[Workflow]:
        """Return all active workflows ordered by usage_count desc."""
        rows = self._conn.execute(
            "SELECT * FROM workflows WHERE active = 1 ORDER BY usage_count DESC"
        ).fetchall()
        return [_row_to_workflow(r) for r in rows]

    def deactivate(self, workflow_id: str) -> None:
        """Mark a workflow as inactive (soft-delete)."""
        self._conn.execute(
            "UPDATE workflows SET active = 0, updated = ? WHERE workflow_id = ?",
            (datetime.now().isoformat(), workflow_id),
        )
        self._conn.commit()

    def get(self, workflow_id: str) -> Workflow | None:
        """Retrieve a single workflow by ID."""
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        return _row_to_workflow(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, workflow: Workflow) -> None:
        self._conn.execute(
            """
            INSERT INTO workflows
                (workflow_id, name, description, steps, trigger_pattern,
                 success_rate, usage_count, source_episodes, created, updated, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                json.dumps(workflow.steps),
                workflow.trigger_pattern,
                workflow.success_rate,
                workflow.usage_count,
                json.dumps(workflow.source_episodes),
                workflow.created.isoformat(),
                workflow.updated.isoformat(),
                1 if workflow.active else 0,
            ),
        )
        self._conn.commit()


# ------------------------------------------------------------------
# Private formatting helper
# ------------------------------------------------------------------


def _format_episodes(episodes: list[Episode]) -> str:
    """Serialize episodes into a readable block for the LLM prompt."""
    lines: list[str] = []
    for i, ep in enumerate(episodes, 1):
        lines.append(f"Episode {i}:")
        lines.append(f"  Decision: {ep.decision}")
        lines.append(f"  Context: {ep.context}")
        lines.append(f"  Outcome: {ep.outcome}")
        lines.append(f"  Status: {ep.outcome_status}")
        if ep.lessons:
            lines.append(f"  Lessons: {', '.join(ep.lessons)}")
        lines.append("")
    return "\n".join(lines)
