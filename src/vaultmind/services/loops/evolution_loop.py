"""Belief evolution accumulation loop — tracks entity drift across scans."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vaultmind.graph.evolution import EvolutionDetector

logger = logging.getLogger(__name__)


def create_evolution_executor(
    detector: EvolutionDetector,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Create a stateful executor for the evolution loop."""

    async def execute(state: dict[str, Any]) -> dict[str, Any]:
        signals = detector.scan()

        prior_ids: set[str] = set(state.get("signal_ids", []))
        trend_counts: dict[str, int] = dict(state.get("trend_counts", {}))

        new_signals = [s for s in signals if s.evolution_id not in prior_ids]
        current_ids = [s.evolution_id for s in signals]

        # Track escalating trends (entities appearing in multiple consecutive scans)
        for signal in signals:
            key = f"{signal.entity_a}:{signal.entity_b}"
            trend_counts[key] = trend_counts.get(key, 0) + 1

        # Escalating = appeared in 3+ consecutive scans
        escalating = {k: v for k, v in trend_counts.items() if v >= 3}

        # Prune trends for entities no longer in signals
        active_keys = {f"{s.entity_a}:{s.entity_b}" for s in signals}
        trend_counts = {k: v for k, v in trend_counts.items() if k in active_keys}

        # Build notification
        notification = None
        notify_parts: list[str] = []

        if new_signals:
            high_severity = [s for s in new_signals if s.severity >= 0.7]
            if high_severity:
                notify_parts.append(
                    f"{len(high_severity)} high-severity belief evolution signal(s):"
                )
                for s in high_severity[:5]:
                    notify_parts.append(
                        f"  - {s.signal_type}: {s.entity_a} <-> {s.entity_b} ({s.detail})"
                    )

        if escalating:
            notify_parts.append(f"{len(escalating)} escalating trend(s):")
            for key, count in sorted(escalating.items(), key=lambda x: -x[1])[:5]:
                notify_parts.append(f"  - {key} (seen {count} consecutive scans)")

        if notify_parts:
            notification = "*VaultMind Evolution Loop*\n\n" + "\n".join(notify_parts)

        return {
            "signal_ids": current_ids,
            "trend_counts": trend_counts,
            "new_signal_count": len(new_signals),
            "escalating_count": len(escalating),
            "notification": notification,
        }

    return execute
