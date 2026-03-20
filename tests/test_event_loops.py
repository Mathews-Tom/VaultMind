"""Tests for event-driven loop triggering."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from vaultmind.services.scheduler import ScheduledJob, SchedulerService

if TYPE_CHECKING:
    from pathlib import Path


async def _counter_job(state: dict[str, Any]) -> dict[str, Any]:
    state["count"] = state.get("count", 0) + 1
    return state


class TestEventTriggeredRuns:
    async def test_record_event_increments_count(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
            trigger_event_types=["NoteCreatedEvent"],
            trigger_threshold=3,
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        scheduler.record_event("NoteCreatedEvent")
        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" not in scheduler._pending_triggers

        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" in scheduler._pending_triggers

    async def test_threshold_resets_after_trigger(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
            trigger_event_types=["NoteCreatedEvent"],
            trigger_threshold=2,
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        scheduler.record_event("NoteCreatedEvent")
        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" in scheduler._pending_triggers

        # Simulate the run loop consuming the trigger
        scheduler._pending_triggers.discard("test_job")

        # Counter should have been reset — only 1 event, need 2
        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" not in scheduler._pending_triggers

    async def test_cooldown_prevents_rapid_triggers(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
            trigger_event_types=["NoteCreatedEvent"],
            trigger_threshold=1,
            trigger_cooldown=timedelta(hours=1),
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        # First trigger
        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" in scheduler._pending_triggers
        scheduler._pending_triggers.discard("test_job")

        # Second trigger should be blocked by cooldown
        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" not in scheduler._pending_triggers

    async def test_unrelated_events_ignored(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
            trigger_event_types=["NoteCreatedEvent"],
            trigger_threshold=1,
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        scheduler.record_event("NoteDeletedEvent")
        assert "test_job" not in scheduler._pending_triggers

    async def test_no_trigger_types_means_no_event_response(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" not in scheduler._pending_triggers

    async def test_pending_trigger_consumed_by_run_job(self, tmp_path: Path) -> None:
        job = ScheduledJob(
            name="test_job",
            interval=timedelta(days=7),
            execute=_counter_job,
            trigger_event_types=["NoteCreatedEvent"],
            trigger_threshold=1,
        )
        scheduler = SchedulerService(jobs=[job], state_path=tmp_path / "state.json")

        scheduler.record_event("NoteCreatedEvent")
        assert "test_job" in scheduler._pending_triggers

        # Run the job manually (as the main loop would)
        await scheduler._run_job(job)
        # State should reflect one execution
        assert scheduler.get_job_state("test_job") == {"count": 1}
