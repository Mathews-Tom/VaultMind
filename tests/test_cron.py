"""Tests for cron expression utilities and natural language schedule conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vaultmind.services.cron import CronSchedule, interval_days_to_cron, nl_to_cron
from vaultmind.services.scheduler import ScheduledJob, resolve_cron_expr

# ---------------------------------------------------------------------------
# A. CronSchedule tests
# ---------------------------------------------------------------------------


class TestCronSchedule:
    def test_validate_valid_expression_returns_true(self) -> None:
        assert CronSchedule.validate("0 0 * * *") is True

    def test_validate_invalid_expression_returns_false(self) -> None:
        assert CronSchedule.validate("not a cron") is False

    def test_is_overdue_past_next_run_returns_true(self) -> None:
        schedule = CronSchedule(expression="0 * * * *", description="hourly")
        last_run = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert schedule.is_overdue(last_run) is True

    def test_is_overdue_before_next_run_returns_false(self) -> None:
        schedule = CronSchedule(expression="0 * * * *", description="hourly")
        # Fix time to 10:05 — last run at 10:01, next run at 11:00 → not overdue
        now = datetime(2026, 1, 1, 10, 5, tzinfo=UTC)
        last_run = datetime(2026, 1, 1, 10, 1, tzinfo=UTC).isoformat()
        assert schedule.is_overdue(last_run, now=now) is False

    def test_is_overdue_handles_naive_datetime(self) -> None:
        schedule = CronSchedule(expression="0 * * * *", description="hourly")
        # Naive ISO string (no timezone info)
        last_run = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        assert schedule.is_overdue(last_run) is True


# ---------------------------------------------------------------------------
# B. nl_to_cron tests
# ---------------------------------------------------------------------------


class TestNlToCron:
    def test_daily_returns_midnight(self) -> None:
        result = nl_to_cron("daily")
        assert result is not None
        assert result.expression == "0 0 * * *"

    def test_every_day_returns_midnight(self) -> None:
        result = nl_to_cron("every day")
        assert result is not None
        assert result.expression == "0 0 * * *"

    def test_weekly_returns_sunday(self) -> None:
        result = nl_to_cron("weekly")
        assert result is not None
        assert result.expression == "0 0 * * 0"

    def test_monthly_returns_first(self) -> None:
        result = nl_to_cron("monthly")
        assert result is not None
        assert result.expression == "0 0 1 * *"

    def test_every_monday_returns_correct_dow(self) -> None:
        result = nl_to_cron("every monday")
        assert result is not None
        assert result.expression == "0 0 * * 1"

    def test_every_friday_returns_correct_dow(self) -> None:
        result = nl_to_cron("every friday")
        assert result is not None
        assert result.expression == "0 0 * * 5"

    def test_every_n_days(self) -> None:
        result = nl_to_cron("every 3 days")
        assert result is not None
        assert result.expression == "0 0 */3 * *"

    def test_every_n_hours(self) -> None:
        result = nl_to_cron("every 4 hours")
        assert result is not None
        assert result.expression == "0 */4 * * *"

    def test_weekdays(self) -> None:
        result = nl_to_cron("weekdays")
        assert result is not None
        assert result.expression == "0 0 * * 1-5"

    def test_weekends(self) -> None:
        result = nl_to_cron("weekends")
        assert result is not None
        assert result.expression == "0 0 * * 0,6"

    def test_with_time_daily(self) -> None:
        result = nl_to_cron("daily at 9:30 AM")
        assert result is not None
        assert result.expression == "30 9 * * *"

    def test_with_time_and_day(self) -> None:
        result = nl_to_cron("every monday at 3pm")
        assert result is not None
        assert result.expression == "0 15 * * 1"

    def test_unrecognized_returns_none(self) -> None:
        assert nl_to_cron("gibberish") is None

    def test_case_insensitive(self) -> None:
        upper = nl_to_cron("Every Monday")
        lower = nl_to_cron("every monday")
        assert upper is not None
        assert lower is not None
        assert upper.expression == lower.expression


# ---------------------------------------------------------------------------
# C. interval_days_to_cron tests
# ---------------------------------------------------------------------------


class TestIntervalDaysToCron:
    def test_one_day_returns_daily(self) -> None:
        assert interval_days_to_cron(1) == "0 0 * * *"

    def test_seven_days_returns_weekly(self) -> None:
        assert interval_days_to_cron(7) == "0 0 * * 0"

    def test_three_days(self) -> None:
        assert interval_days_to_cron(3) == "0 0 */3 * *"

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            interval_days_to_cron(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            interval_days_to_cron(-1)


# ---------------------------------------------------------------------------
# D. resolve_cron_expr tests
# ---------------------------------------------------------------------------


class TestResolveCronExpr:
    def test_explicit_cron_used_directly(self) -> None:
        assert resolve_cron_expr("0 9 * * 1-5", interval_days=1) == "0 9 * * 1-5"

    def test_nl_schedule_converted(self) -> None:
        assert resolve_cron_expr("every monday", interval_days=1) == "0 0 * * 1"

    def test_empty_schedule_falls_back_to_interval(self) -> None:
        assert resolve_cron_expr("", interval_days=7) == "0 0 * * 0"

    def test_invalid_schedule_falls_back_to_interval(self) -> None:
        assert resolve_cron_expr("gibberish", interval_days=3) == "0 0 */3 * *"


# ---------------------------------------------------------------------------
# E. ScheduledJob cron integration
# ---------------------------------------------------------------------------


async def _noop(state: dict[str, object]) -> dict[str, object]:
    return state


class TestScheduledJobCron:
    def test_is_overdue_with_cron_expr(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(days=999),
            execute=_noop,
            cron_expr="0 * * * *",  # hourly
        )
        last_run = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.is_overdue(last_run) is True

    def test_is_overdue_without_cron_uses_interval(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(hours=1),
            execute=_noop,
            cron_expr="",
        )
        # Last run 30 min ago — within interval, not overdue
        last_run = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        assert job.is_overdue(last_run) is False

        # Last run 2 hours ago — beyond interval, overdue
        last_run_old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert job.is_overdue(last_run_old) is True

    def test_should_run_no_last_run_always_true(self) -> None:
        job = ScheduledJob(
            name="test",
            interval=timedelta(days=365),
            execute=_noop,
            cron_expr="0 0 1 1 *",  # once a year
        )
        assert job.should_run(datetime.now(UTC), last_run_iso=None) is True
