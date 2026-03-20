"""Tests for asyncio scheduler service."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from vaultmind.services.scheduler import ScheduledJob, SchedulerService

if TYPE_CHECKING:
    from pathlib import Path


async def _noop(state: dict[str, Any]) -> dict[str, Any]:
    return state


class TestScheduledJob:
    def test_is_overdue_true(self) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.is_overdue(old_time) is True

    def test_is_overdue_false(self) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        recent_time = datetime.now(UTC).isoformat()
        assert job.is_overdue(recent_time) is False

    def test_should_run_no_last_run(self) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        assert job.should_run(datetime.now(UTC), None) is True

    def test_should_run_overdue(self) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.should_run(datetime.now(UTC), old_time) is True

    def test_should_run_not_yet(self) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        recent = datetime.now(UTC).isoformat()
        assert job.should_run(datetime.now(UTC), recent) is False

    async def test_legacy_factory(self) -> None:
        called = False

        async def no_arg() -> None:
            nonlocal called
            called = True

        job = ScheduledJob.legacy(name="legacy", interval=timedelta(hours=1), execute=no_arg)
        assert job.name == "legacy"
        assert job.interval == timedelta(hours=1)
        assert job.completion_check is None
        # The wrapped execute must accept a state dict and return one
        result = await job.execute({})
        assert called is True
        assert isinstance(result, dict)


class TestSchedulerService:
    async def test_state_persists(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"
        job_ran = False

        async def run_job(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal job_ran
            job_ran = True
            return state

        job = ScheduledJob(name="test_job", interval=timedelta(hours=1), execute=run_job)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "test_job" in data
        entry = data["test_job"]
        assert "last_run" in entry
        assert "run_count" in entry
        assert "state" in entry
        assert "completed" in entry
        assert job_ran is True

    async def test_fires_missed_job_on_startup(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"
        old_time = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        # Write old-format state to ensure migration + overdue detection
        state_path.write_text(json.dumps({"test_job": old_time}))

        fired = False

        async def run_job(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal fired
            fired = True
            return state

        job = ScheduledJob(name="test_job", interval=timedelta(days=7), execute=run_job)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        # Replicate startup logic: check for missed jobs
        for j in scheduler._jobs:
            job_state = scheduler._state.get(j.name)
            if job_state is None:
                await scheduler._run_job(j)
            elif not job_state.get("completed", False):
                last_run = job_state.get("last_run")
                if isinstance(last_run, str) and j.is_overdue(last_run):
                    await scheduler._run_job(j)

        assert fired is True

    async def test_state_survives_restart(self, tmp_path: Path) -> None:
        state_path = tmp_path / "scheduler_state.json"

        job = ScheduledJob(name="persistent_job", interval=timedelta(hours=1), execute=_noop)

        s1 = SchedulerService(jobs=[job], state_path=state_path)
        await s1._run_job(job)

        s2 = SchedulerService(jobs=[job], state_path=state_path)
        assert "persistent_job" in s2._state
        entry = s2._state["persistent_job"]
        assert "last_run" in entry
        assert entry["run_count"] == 1

    def test_stop(self, tmp_path: Path) -> None:
        job = ScheduledJob(name="test", interval=timedelta(hours=1), execute=_noop)
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")
        scheduler.stop()
        assert scheduler._running is False

    async def test_stateful_job_receives_and_returns_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        async def increment(state: dict[str, Any]) -> dict[str, Any]:
            state["counter"] = state.get("counter", 0) + 1
            return state

        job = ScheduledJob(name="counter_job", interval=timedelta(hours=1), execute=increment)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)
        await scheduler._run_job(job)

        inner = scheduler.get_job_state("counter_job")
        assert inner["counter"] == 2

    async def test_completion_check_pauses_job(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        async def finish(_state: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        job = ScheduledJob(
            name="one_shot",
            interval=timedelta(hours=1),
            execute=finish,
            completion_check=lambda s: s.get("done", False),
        )
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)

        assert scheduler._state["one_shot"]["completed"] is True
        assert scheduler._state["one_shot"]["run_count"] == 1

        # Second call must be skipped because job is completed
        await scheduler._run_job(job)
        assert scheduler._state["one_shot"]["run_count"] == 1

    async def test_reset_job(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        async def finish(_state: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        job = ScheduledJob(
            name="resettable",
            interval=timedelta(hours=1),
            execute=finish,
            completion_check=lambda s: s.get("done", False),
        )
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)
        assert scheduler._state["resettable"]["completed"] is True

        scheduler.reset_job("resettable")

        entry = scheduler._state["resettable"]
        assert entry["completed"] is False
        assert entry["run_count"] == 0
        assert entry["state"] == {}

    async def test_get_job_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        async def store_data(state: dict[str, Any]) -> dict[str, Any]:
            state["value"] = 42
            return state

        job = ScheduledJob(name="data_job", interval=timedelta(hours=1), execute=store_data)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)

        result = scheduler.get_job_state("data_job")
        assert result == {"value": 42}

    async def test_legacy_job_preserves_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        async def no_arg() -> None:
            pass

        job = ScheduledJob.legacy(name="legacy_job", interval=timedelta(hours=1), execute=no_arg)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)
        first_inner = scheduler.get_job_state("legacy_job")
        assert first_inner == {}

        await scheduler._run_job(job)
        second_inner = scheduler.get_job_state("legacy_job")
        assert second_inner == {}

    def test_state_migration_from_old_format(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        old_ts = "2024-01-01T00:00:00"
        state_path.write_text(json.dumps({"job_name": old_ts}))

        job = ScheduledJob(name="job_name", interval=timedelta(hours=1), execute=_noop)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        entry = scheduler._state["job_name"]
        assert isinstance(entry, dict)
        assert entry["last_run"] == old_ts
        assert entry["run_count"] == 0
        assert entry["state"] == {}
        assert entry["completed"] is False

    async def test_atomic_write(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        job = ScheduledJob(name="atomic_job", interval=timedelta(hours=1), execute=_noop)
        scheduler = SchedulerService(jobs=[job], state_path=state_path)

        await scheduler._run_job(job)

        assert state_path.exists()
        # Temp file must not remain after atomic replace
        assert not state_path.with_suffix(".tmp").exists()
