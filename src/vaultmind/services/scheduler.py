"""Asyncio-native scheduler with persistent state.

Runs periodic jobs (e.g., maturation digest) as background tasks.
Persists last-run timestamps to survive bot restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """A periodic job definition."""

    name: str
    interval: timedelta
    execute: Callable[[], Awaitable[None]]

    def is_overdue(self, last_run_iso: str) -> bool:
        """Check if this job should have run since last_run."""
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

    def __init__(self, jobs: list[ScheduledJob], state_path: Path) -> None:
        self._jobs = jobs
        self._state_path = state_path
        self._state: dict[str, Any] = self._load_state()
        self._running = False

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.exists():
            return dict(json.loads(self._state_path.read_text()))
        return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, default=str))

    async def run(self) -> None:
        """Start the scheduler loop. Fires missed jobs on startup."""
        self._running = True

        # Check for missed jobs on startup
        for job in self._jobs:
            last_run = self._state.get(job.name)
            if isinstance(last_run, str) and job.is_overdue(last_run):
                logger.info("Scheduler: firing missed job '%s'", job.name)
                await self._run_job(job)
            elif last_run is None:
                logger.info("Scheduler: first run of job '%s'", job.name)
                await self._run_job(job)

        while self._running:
            now = datetime.now(UTC)
            for job in self._jobs:
                last_run = self._state.get(job.name)
                last_run_str = str(last_run) if last_run is not None else None
                if job.should_run(now, last_run_str):
                    await self._run_job(job)
            await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    async def _run_job(self, job: ScheduledJob) -> None:
        """Execute a job and update state."""
        try:
            await job.execute()
        except Exception:
            logger.exception("Scheduler: job '%s' failed", job.name)
        self._state[job.name] = datetime.now(UTC).isoformat()
        self._save_state()
