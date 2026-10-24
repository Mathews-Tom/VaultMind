"""Daily handler — get or create today's daily note."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vaultmind.bot.formatter import TelegramFormatter
from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.vault.security import (
    PathResolutionError,
    PathTraversalError,
    ReservedNamespaceError,
    validate_ordinary_write_path,
)

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

DAILY_TEMPLATE = """\
---
type: daily
created: {date}
tags: [daily]
---

# {date_display}

## Captures

## Tasks

## Reflections

"""


async def handle_daily(ctx: HandlerContext, message: Message) -> None:
    """Get or create today's daily note."""
    if not _is_authorized(ctx, message):
        return

    today = datetime.now()
    filename = today.strftime("%Y-%m-%d") + ".md"
    try:
        filepath = validate_ordinary_write_path(
            str(Path(ctx.settings.vault.daily_folder) / filename), ctx.vault_root
        )
    except (PathTraversalError, PathResolutionError, ReservedNamespaceError):
        await message.answer("Path not allowed.")
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        note = ctx.parser.parse_file(filepath)
        date_str = today.strftime("%B %d, %Y")
        header_line = f"📅 <b>Daily Note — {date_str}</b>\n\n"
        formatted = TelegramFormatter.format_note(note)
        await message.answer(header_line + formatted, parse_mode="HTML")
    else:
        # Create new daily note
        content = DAILY_TEMPLATE.format(
            date=today.strftime("%Y-%m-%d"),
            date_display=today.strftime("%A, %B %d, %Y"),
        )
        filepath.write_text(content, encoding="utf-8")
        daily_folder = ctx.settings.vault.daily_folder
        await message.answer(
            f"📅 Created daily note: <code>{daily_folder}/{filename}</code>",
            parse_mode="HTML",
        )
