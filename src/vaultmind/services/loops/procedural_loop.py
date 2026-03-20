"""Procedural synthesis loop — mines episodic memory for workflow patterns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vaultmind.llm.client import LLMClient
    from vaultmind.memory.procedural import ProceduralMemory
    from vaultmind.memory.store import EpisodeStore

logger = logging.getLogger(__name__)


def create_procedural_executor(
    procedural_memory: ProceduralMemory,
    episode_store: EpisodeStore,
    llm_client: LLMClient,
    model: str,
    min_episodes: int = 3,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Create a stateful executor for the procedural synthesis loop."""

    async def execute(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        resolved = episode_store.query_resolved()
        last_count = state.get("last_episode_count", 0)
        surfaced: list[str] = list(state.get("surfaced_workflows", []))

        # Skip if fewer than min_episodes new resolved episodes
        new_count = len(resolved) - last_count
        if new_count < min_episodes:
            logger.info(
                "Procedural loop: %d new episodes (need %d), skipping.",
                new_count,
                min_episodes,
            )
            return {
                "last_episode_count": len(resolved),
                "surfaced_workflows": surfaced,
                "notification": None,
            }

        # Run synthesis in thread (blocking SQLite + LLM calls)
        workflows = await asyncio.to_thread(
            procedural_memory.synthesize_workflows,
            episode_store,
            llm_client,
            model,
            min_episodes,
        )

        # Filter out already-surfaced workflows
        new_workflows = [w for w in workflows if w.workflow_id not in surfaced]

        notification = None
        if new_workflows:
            lines = ["*VaultMind Procedural Loop*", ""]
            lines.append(f"Discovered {len(new_workflows)} new workflow pattern(s):")
            for wf in new_workflows:
                lines.append(f"  - *{wf.name}*: {wf.description}")
                if wf.steps:
                    for i, step in enumerate(wf.steps[:3], 1):
                        lines.append(f"    {i}. {step}")
            notification = "\n".join(lines)
            surfaced.extend(w.workflow_id for w in new_workflows)

        return {
            "last_episode_count": len(resolved),
            "surfaced_workflows": surfaced,
            "notification": notification,
        }

    return execute
