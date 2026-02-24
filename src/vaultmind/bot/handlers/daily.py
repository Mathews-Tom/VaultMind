"""Daily handler — get or create today's daily note."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

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
    daily_dir = ctx.vault_root / ctx.settings.vault.daily_folder
    daily_dir.mkdir(parents=True, exist_ok=True)
    filepath = daily_dir / filename

    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        # Return a summary (first 1000 chars)
        preview = content[:1000]
        if len(content) > 1000:
            preview += "\n\n_...truncated_"
        date_str = today.strftime("%B %d, %Y")
        await message.answer(
            f"\U0001f4c5 **Daily Note \u2014 {date_str}**\n\n{preview}",
            parse_mode="Markdown",
        )
    else:
        # Create new daily note
        content = DAILY_TEMPLATE.format(
            date=today.strftime("%Y-%m-%d"),
            date_display=today.strftime("%A, %B %d, %Y"),
        )
        filepath.write_text(content, encoding="utf-8")
        daily_folder = ctx.settings.vault.daily_folder
        await message.answer(
            f"\U0001f4c5 Created daily note: `{daily_folder}/{filename}`",
            parse_mode="Markdown",
        )
