"""Insight accumulation loop — detects shifts in usage patterns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vaultmind.tracking.preferences import PreferenceStore

logger = logging.getLogger(__name__)


def create_insight_executor(
    preference_store: PreferenceStore,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Create a stateful executor for the insight loop."""

    async def execute(state: dict[str, Any]) -> dict[str, Any]:
        from vaultmind.tracking.analyzer import analyze_preferences

        insights = analyze_preferences(preference_store, days=30)

        prior_top_searches = state.get("top_searches", [])
        prior_acceptance = state.get("acceptance_rate", 0.0)
        prior_total = state.get("total_interactions", 0)

        # Detect shifts
        shifts: list[str] = []

        # New top searches
        current_searches = [q for q, _ in insights.top_searches[:5]]
        if prior_top_searches:
            new_searches = [s for s in current_searches if s not in prior_top_searches]
            if new_searches:
                shifts.append(f"New trending searches: {', '.join(new_searches)}")

        # Acceptance rate shift (>15% change)
        if (
            prior_acceptance > 0
            and abs(insights.suggestions_acceptance_rate - prior_acceptance) > 0.15
        ):
            direction = "up" if insights.suggestions_acceptance_rate > prior_acceptance else "down"
            pct = int(insights.suggestions_acceptance_rate * 100)
            shifts.append(f"Suggestion acceptance rate shifted {direction} to {pct}%")

        # Interaction volume shift (>50% change)
        if prior_total > 0:
            change = (insights.total_interactions - prior_total) / prior_total
            if abs(change) > 0.5:
                direction = "increased" if change > 0 else "decreased"
                shifts.append(
                    f"Interaction volume {direction}: "
                    f"{prior_total} -> {insights.total_interactions}"
                )

        # Build notification
        notification = None
        if shifts:
            lines = ["*VaultMind Insight Loop*", ""]
            lines.extend(f"- {s}" for s in shifts)
            if insights.recommendations:
                lines.append("")
                lines.append("*Recommendations:*")
                lines.extend(f"- {r}" for r in insights.recommendations[:3])
            notification = "\n".join(lines)

        # Update state for next run
        return {
            "top_searches": current_searches,
            "acceptance_rate": insights.suggestions_acceptance_rate,
            "total_interactions": insights.total_interactions,
            "recommendations": insights.recommendations[:5],
            "notification": notification,
        }

    return execute
