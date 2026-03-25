"""Asyncio-native scheduler with persistent state.

Runs periodic jobs (e.g., maturation digest) as background tasks.
Persists last-run timestamps to survive bot restarts.
Supports state-aware compound loops via execute(state) -> state protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from vaultmind.bot.notifier import Notifier

logger = logging.getLogger(__name__)


def _wrap_legacy_execute(
    fn: Callable[[], Awaitable[None]],
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Wrap a legacy no-arg async callable to match the stateful execute protocol."""

    async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        await fn()
        return state

    return wrapper


def resolve_cron_expr(schedule: str, interval_days: int) -> str:
    """Resolve a cron expression from explicit schedule or interval_days fallback.

    If schedule is a non-empty string:
      - If it looks like a cron expression (contains spaces + digits), use it directly
      - Otherwise try nl_to_cron() for natural language conversion
    If schedule is empty, convert interval_days to cron.
    """
    if schedule.strip():
        from vaultmind.services.cron import CronSchedule, nl_to_cron

        # Try as raw cron first
        if CronSchedule.validate(schedule.strip()):
            return schedule.strip()
        # Try natural language
        result = nl_to_cron(schedule.strip())
        if result is not None:
            return result.expression
        logger.warning(
            "Invalid schedule '%s', falling back to interval_days=%d",
            schedule,
            interval_days,
        )

    from vaultmind.services.cron import interval_days_to_cron

    return interval_days_to_cron(interval_days)


@dataclass
class ScheduledJob:
    """A periodic job definition."""

    name: str
    interval: timedelta
    execute: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    cron_expr: str = ""  # Optional cron expression; takes priority over interval
    completion_check: Callable[[dict[str, Any]], bool] | None = field(default=None)
    trigger_event_types: list[str] | None = field(default=None)
    trigger_threshold: int = field(default=10)
    trigger_cooldown: timedelta = field(default_factory=lambda: timedelta(hours=1))

    @staticmethod
    def legacy(
        name: str,
        interval: timedelta,
        execute: Callable[[], Awaitable[None]],
    ) -> ScheduledJob:
        """Create a job from a legacy no-arg async callable.

        Note: legacy jobs do not support event-triggered early runs.
        """
        return ScheduledJob(
            name=name,
            interval=interval,
            execute=_wrap_legacy_execute(execute),
        )

    def is_overdue(self, last_run_iso: str) -> bool:
        """Check if this job should have run since last_run."""
        if self.cron_expr:
            from vaultmind.services.cron import CronSchedule

            schedule = CronSchedule(expression=self.cron_expr, description="")
            return schedule.is_overdue(last_run_iso)
        last = datetime.fromisoformat(last_run_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return datetime.now(UTC) - last > self.interval

    def should_run(self, now: datetime, last_run_iso: str | None) -> bool:
        """Check if this job should run now."""
        if last_run_iso is None:
            return True
        return self.is_overdue(last_run_iso)


class SchedulerService:
    """Asyncio-native scheduler for periodic background tasks."""

    def __init__(
        self,
        jobs: list[ScheduledJob],
        state_path: Path,
        notifier: Notifier | None = None,
    ) -> None:
        self._jobs = jobs
        self._state_path = state_path
        self._state: dict[str, Any] = self._load_state()
        self._running = False
        self._notifier = notifier
        self._event_counts: dict[str, dict[str, int]] = {}  # {job_name: {event_type: count}}
        self._last_triggered: dict[str, datetime] = {}  # {job_name: last_trigger_time}
        self._pending_triggers: set[str] = set()

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        raw: dict[str, Any] = dict(json.loads(self._state_path.read_text()))
        # Migrate old format: plain ISO string -> new dict format
        for key, val in raw.items():
            if isinstance(val, str):
                raw[key] = {
                    "last_run": val,
                    "run_count": 0,
                    "state": {},
                    "completed": False,
                }
        return raw

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, default=str))
        tmp.replace(self._state_path)

    async def run(self) -> None:
        """Start the scheduler loop. Fires missed jobs on startup."""
        self._running = True

        # Check for missed jobs on startup
        for job in self._jobs:
            job_state = self._state.get(job.name)
            if job_state is None:
                logger.info("Scheduler: first run of job '%s'", job.name)
                await self._run_job(job)
            elif not job_state.get("completed", False):
                last_run = job_state.get("last_run")
                if isinstance(last_run, str) and job.is_overdue(last_run):
                    logger.info("Scheduler: firing missed job '%s'", job.name)
                    await self._run_job(job)

        while self._running:
            now = datetime.now(UTC)
            for job in self._jobs:
                job_state = self._state.get(job.name)
                # Skip completed jobs
                if job_state is not None and job_state.get("completed", False):
                    continue

                # Check for event-triggered early runs
                if job.name in self._pending_triggers:
                    self._pending_triggers.discard(job.name)
                    await self._run_job(job)
                    continue

                last_run_str = job_state.get("last_run") if job_state else None
                if job.should_run(now, last_run_str):
                    await self._run_job(job)
            await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    def record_event(self, event_type_name: str) -> None:
        """Record a vault event that may trigger early loop runs.

        Called by the event bus subscriber. Checks if any job's threshold
        is met and the cooldown period has passed.
        """
        for job in self._jobs:
            if job.trigger_event_types is None:
                continue
            if event_type_name not in job.trigger_event_types:
                continue

            # Initialize counter
            if job.name not in self._event_counts:
                self._event_counts[job.name] = {}
            counts = self._event_counts[job.name]
            counts[event_type_name] = counts.get(event_type_name, 0) + 1

            total = sum(counts.values())
            if total >= job.trigger_threshold:
                # Check cooldown
                last = self._last_triggered.get(job.name)
                now = datetime.now(UTC)
                if last is not None and (now - last) < job.trigger_cooldown:
                    logger.debug("Scheduler: job '%s' threshold met but in cooldown", job.name)
                    continue

                logger.info(
                    "Scheduler: event trigger for job '%s' (%d events)",
                    job.name,
                    total,
                )
                self._event_counts[job.name] = {}  # Reset counters
                self._last_triggered[job.name] = now
                self._pending_triggers.add(job.name)

    async def _run_job(self, job: ScheduledJob) -> None:
        """Execute a job and update state."""
        job_state: dict[str, Any] = self._state.get(
            job.name,
            {"state": {}, "run_count": 0, "completed": False},
        )

        # Skip if already marked completed
        if job_state.get("completed", False):
            return

        inner_state: dict[str, Any] = job_state.get("state") or {}
        try:
            inner_state = await job.execute(inner_state)
        except Exception:
            logger.exception("Scheduler: job '%s' failed", job.name)

        job_state["state"] = inner_state
        job_state["run_count"] = job_state.get("run_count", 0) + 1
        job_state["last_run"] = datetime.now(UTC).isoformat()

        # Check completion after execute
        if job.completion_check is not None and job.completion_check(inner_state):
            job_state["completed"] = True
            logger.info("Scheduler: job '%s' marked completed", job.name)

        # Send notification if job produced one
        notification = inner_state.get("notification")
        if notification and self._notifier is not None:
            try:
                await self._notifier.send_if_significant(str(notification))
            except Exception:
                logger.exception("Scheduler: notification send failed for '%s'", job.name)

        self._state[job.name] = job_state
        self._save_state()

    def reset_job(self, name: str) -> None:
        """Clear completion flag, reset state and run_count for a job."""
        entry = self._state.get(name, {})
        entry["completed"] = False
        entry["state"] = {}
        entry["run_count"] = 0
        self._state[name] = entry
        self._save_state()

    def get_job_state(self, name: str) -> dict[str, Any]:
        """Return the inner state dict for a job (for external inspection)."""
        entry = self._state.get(name, {})
        return dict(entry.get("state") or {})
