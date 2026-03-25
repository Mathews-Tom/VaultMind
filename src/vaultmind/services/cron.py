"""Cron expression utilities and natural language schedule conversion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from croniter import croniter  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CronSchedule:
    """A validated cron expression with human-readable description."""

    expression: str
    description: str

    def is_overdue(self, last_run_iso: str, now: datetime | None = None) -> bool:
        """Check if a run is overdue given the last run timestamp."""
        if now is None:
            now = datetime.now(UTC)
        last = datetime.fromisoformat(last_run_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        cron = croniter(self.expression, last)
        next_run = cron.get_next(datetime)
        if next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=UTC)
        return bool(now >= next_run)

    @staticmethod
    def validate(expression: str) -> bool:
        """Check if a cron expression is syntactically valid."""
        return bool(croniter.is_valid(expression))


# Day name to cron day-of-week mapping (0=Sunday)
_DAY_MAP: dict[str, str] = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
    "sun": "0",
    "mon": "1",
    "tue": "2",
    "wed": "3",
    "thu": "4",
    "fri": "5",
    "sat": "6",
}


def interval_days_to_cron(days: int) -> str:
    """Convert an interval_days value to a cron expression.

    Examples:
        1 -> "0 0 * * *" (daily)
        7 -> "0 0 * * 0" (weekly, Sunday)
        N -> "0 0 */N * *" (every N days)
    """
    if days <= 0:
        msg = f"interval_days must be positive, got {days}"
        raise ValueError(msg)
    if days == 1:
        return "0 0 * * *"
    if days == 7:
        return "0 0 * * 0"
    return f"0 0 */{days} * *"


def nl_to_cron(text: str) -> CronSchedule | None:
    """Convert natural language schedule description to a CronSchedule.

    Supports common patterns via regex. Returns None if no pattern matches.
    Does NOT use LLM — pure pattern matching for deterministic, zero-cost conversion.

    Supported patterns:
        - "daily" / "every day" -> "0 0 * * *"
        - "weekly" / "every week" -> "0 0 * * 0"
        - "monthly" / "every month" -> "0 0 1 * *"
        - "every monday" / "every tuesday" etc -> "0 0 * * 1"
        - "every N days" -> "0 0 */N * *"
        - "every N hours" -> "0 */N * * *"
        - "at HH:MM" / "at H AM/PM" (combined with day patterns)
        - "weekdays" / "on weekdays" -> "0 0 * * 1-5"
        - "weekends" -> "0 0 * * 0,6"
    """
    text = text.strip().lower()

    hour = 0
    minute = 0

    # Extract time component first: "at 9:30 AM", "at 3pm", "at 14:00"
    time_match = re.search(
        r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text,
        re.IGNORECASE,
    )
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        period = (time_match.group(3) or "").lower()
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        # Remove time part from text for further pattern matching
        text = text[: time_match.start()].strip() + " " + text[time_match.end() :].strip()
        text = text.strip()

    # "daily" / "every day"
    if re.fullmatch(r"(daily|every\s*day)", text):
        return CronSchedule(f"{minute} {hour} * * *", "daily")

    # "weekly" / "every week"
    if re.fullmatch(r"(weekly|every\s*week)", text):
        return CronSchedule(f"{minute} {hour} * * 0", "weekly (Sunday)")

    # "monthly" / "every month"
    if re.fullmatch(r"(monthly|every\s*month)", text):
        return CronSchedule(f"{minute} {hour} 1 * *", "monthly (1st)")

    # "weekdays" / "on weekdays"
    if re.fullmatch(r"(on\s+)?weekdays", text):
        return CronSchedule(f"{minute} {hour} * * 1-5", "weekdays")

    # "weekends"
    if re.fullmatch(r"(on\s+)?weekends", text):
        return CronSchedule(f"{minute} {hour} * * 0,6", "weekends")

    # "every <day_name>" e.g., "every monday", "every fri"
    day_match = re.fullmatch(r"every\s+(\w+)", text)
    if day_match:
        day_name = day_match.group(1)
        if day_name in _DAY_MAP:
            day_num = _DAY_MAP[day_name]
            return CronSchedule(
                f"{minute} {hour} * * {day_num}",
                f"every {day_name}",
            )

    # "every N days"
    interval_match = re.fullmatch(r"every\s+(\d+)\s+days?", text)
    if interval_match:
        n = int(interval_match.group(1))
        if n > 0:
            expr = interval_days_to_cron(n)
            # Replace hour/minute from time extraction
            parts = expr.split()
            parts[0] = str(minute)
            parts[1] = str(hour)
            return CronSchedule(" ".join(parts), f"every {n} days")

    # "every N hours"
    hours_match = re.fullmatch(r"every\s+(\d+)\s+hours?", text)
    if hours_match:
        n = int(hours_match.group(1))
        if 0 < n < 24:
            return CronSchedule(f"0 */{n} * * *", f"every {n} hours")

    # Just a time with no day pattern — assume daily
    if time_match and not text.strip():
        return CronSchedule(f"{minute} {hour} * * *", f"daily at {hour:02d}:{minute:02d}")

    return None
