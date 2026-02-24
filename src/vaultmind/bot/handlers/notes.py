"""Notes handler — find notes by date with natural language support."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text
from vaultmind.llm.client import LLMError
from vaultmind.llm.client import Message as LLMMessage

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)

DATE_RESOLUTION_SYSTEM_PROMPT = """\
You are a date parser. Given a natural language date expression and today's date, \
return a JSON object with "start" and "end" dates in YYYY-MM-DD format.

Rules:
- "yesterday" = yesterday 00:00 to yesterday 23:59
- "last week" = Monday to Sunday of the previous week
- "over the weekend" = last Saturday and Sunday
- "last Tuesday" = the most recent Tuesday before today
- "yesterday afternoon" = yesterday (just the date)
- For single-day references, start == end
- Always return valid JSON: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
- Return ONLY the JSON, no explanation
"""


async def handle_notes(ctx: HandlerContext, message: Message, query: str) -> None:
    """Find notes by date \u2014 supports natural language and explicit dates."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(query, max_length=MAX_QUERY_LENGTH, operation="notes")
    query = san.text
    if not query:
        await message.answer("Empty query after sanitization.")
        return

    start_date, end_date = _resolve_date_range(ctx, query)
    if start_date is None or end_date is None:
        await message.answer(
            "Could not parse a date from that. "
            "Try: `2026-02-20`, `yesterday`, `last week`, `over the weekend`",
            parse_mode="Markdown",
        )
        return

    await message.answer(f"\U0001f50d Searching notes from {start_date} to {end_date}...")

    # Scan vault for notes within the date range
    notes = _find_notes_by_date(ctx, start_date, end_date)

    if not notes:
        await message.answer(f"No notes found between {start_date} and {end_date}.")
        return

    lines = [f"\U0001f4c5 **Notes from {start_date} to {end_date}** ({len(notes)} found)\n"]
    for i, (rel_path, title, created) in enumerate(notes[:20], 1):
        lines.append(f"**{i}.** {title}\n  `{rel_path}` \u2014 {created}")

    if len(notes) > 20:
        lines.append(f"\n_...and {len(notes) - 20} more_")

    for chunk in _split_message("\n".join(lines), max_len=4000):
        await message.answer(chunk, parse_mode="Markdown")


def _resolve_date_range(ctx: HandlerContext, query: str) -> tuple[str | None, str | None]:
    """Parse a date range from user input \u2014 tries formats then LLM."""
    today = datetime.now()

    # Try explicit YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            d = datetime.strptime(query.strip(), fmt)
            ds = d.strftime("%Y-%m-%d")
            return ds, ds
        except ValueError:
            continue

    # Common keywords without LLM
    lower = query.lower().strip()
    if lower == "today":
        ds = today.strftime("%Y-%m-%d")
        return ds, ds
    if lower == "yesterday":
        ds = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        return ds, ds
    if lower in ("this week", "current week"):
        mon = today - timedelta(days=today.weekday())
        return mon.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    if lower == "last week":
        mon = today - timedelta(days=today.weekday() + 7)
        sun = mon + timedelta(days=6)
        return mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")
    if lower in ("weekend", "over the weekend", "last weekend"):
        # Most recent Saturday
        days_since_sat = (today.weekday() + 2) % 7
        if days_since_sat == 0:
            days_since_sat = 7
        sat = today - timedelta(days=days_since_sat)
        sun = sat + timedelta(days=1)
        return sat.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")

    # Fall back to LLM for complex expressions
    return _llm_resolve_date(ctx, query, today)


def _llm_resolve_date(
    ctx: HandlerContext, query: str, today: datetime
) -> tuple[str | None, str | None]:
    """Use LLM to parse complex natural language date expressions."""
    model = ctx.settings.routing.chat_model or ctx.settings.llm.fast_model
    user_msg = f'Today is {today.strftime("%A, %Y-%m-%d")}. Parse this date expression: "{query}"'
    try:
        response = ctx.llm_client.complete(
            messages=[LLMMessage(role="user", content=user_msg)],
            model=model,
            max_tokens=100,
            system=DATE_RESOLUTION_SYSTEM_PROMPT,
        )
        data = json.loads(response.text.strip())
        return data.get("start"), data.get("end")
    except (LLMError, json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM date resolution failed: %s", e)
        return None, None


def _find_notes_by_date(ctx: HandlerContext, start: str, end: str) -> list[tuple[str, str, str]]:
    """Scan vault for notes created within a date range.

    Returns list of (relative_path, title, created_date) tuples.
    """
    results: list[tuple[str, str, str]] = []

    for md_file in ctx.vault_root.rglob("*.md"):
        rel = md_file.relative_to(ctx.vault_root)
        if any(part in ctx.settings.vault.excluded_folders for part in rel.parts):
            continue

        try:
            note = ctx.parser.parse_file(md_file)
            created = note.created.strftime("%Y-%m-%d")
            if start <= created <= end:
                results.append(
                    (
                        str(rel),
                        note.title,
                        note.created.strftime("%Y-%m-%d %H:%M"),
                    )
                )
        except Exception:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    return results
