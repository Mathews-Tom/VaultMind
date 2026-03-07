"""Tests for asyncio scheduler service."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from vaultmind.services.scheduler import ScheduledJob, SchedulerService

if TYPE_CHECKING:
    from pathlib import Path


class TestScheduledJob:
    def test_is_overdue_true(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=lambda: asyncio.sleep(0),
        )
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.is_overdue(old_time) is True

    def test_is_overdue_false(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=lambda: asyncio.sleep(0),
        )
        recent_time = datetime.now(UTC).isoformat()
        assert job.is_overdue(recent_time) is False

    def test_should_run_no_last_run(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=lambda: asyncio.sleep(0),
        )
        assert job.should_run(datetime.now(UTC), None) is True

    def test_should_run_overdue(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=lambda: asyncio.sleep(0),
        )
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.should_run(datetime.now(UTC), old_time) is True

    def test_should_run_not_yet(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=lambda: asyncio.sleep(0),
        )
        recent = datetime.now(UTC).isoformat()
        assert job.should_run(datetime.now(UTC), recent) is False


class TestSchedulerService:
    async def test_state_persists(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"
        job_ran = False

        async def run_job() -> None:
            nonlocal job_ran
            job_ran = True

        job = ScheduledJob(name="test_job", interval=timedelta(hours=1), execute=run_job)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "test_job" in data
        assert job_ran is True

    async def test_fires_missed_job_on_startup(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"
        old_time = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        state_path.write_text(json.dumps({"test_job": old_time}))

        fired = False

        async def run_job() -> None:
            nonlocal fired
            fired = True

        job = ScheduledJob(name="test_job", interval=timedelta(days=7), execute=run_job)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        for j in scheduler._jobs:
            last_run = scheduler._state.get(j.name)
            if isinstance(last_run, str) and j.is_overdue(last_run):
                await scheduler._run_job(j)

        assert fired is True

    async def test_state_survives_restart(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"

        async def noop() -> None:
            pass

        job = ScheduledJob(name="persistent_job", interval=timedelta(hours=1), execute=noop)

        s1 = SchedulerService(jobs=[job], state_path=state_path)
        await s1._run_job(job)

        s2 = SchedulerService(jobs=[job], state_path=state_path)
        assert "persistent_job" in s2._state

    def test_stop(self, tmp_path: Path) -> None:
        async def noop() -> None:
            pass

        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=noop)
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")
        scheduler.stop()
        assert scheduler._running is False
