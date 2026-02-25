"""Bookmark handler — save conversation exchanges as vault notes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)

INBOX_FOLDER = "00-inbox"

THINKING_BOOKMARK_TEMPLATE = """\
---
title: "{title}"
date: "{date}"
tags: [bookmark, thinking]
source: telegram-thinking
---

# {title}

*Bookmarked from thinking session on {date}*

{session_body}
"""

QA_BOOKMARK_TEMPLATE = """\
---
title: "{title}"
date: "{date}"
tags: [bookmark, chat]
source: telegram-chat
---

# {title}

*Bookmarked from chat on {date}*

**Question:** {query}

**Answer:** {response}
"""


@dataclass
class LastExchange:
    """Most recent Q&A exchange for a user."""

    query: str
    response: str
    timestamp: float  # time.monotonic()


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:60].strip("-")


def _format_session_body(history: list[dict[str, str]]) -> str:
    """Format thinking session history as readable markdown."""
    parts: list[str] = []
    for turn in history:
        user_msg = turn.get("user", "")
        assistant_msg = turn.get("assistant", "")
        parts.append(f"**User:** {user_msg}")
        parts.append("")
        parts.append(f"**Assistant:** {assistant_msg}")
        parts.append("")
    return "\n".join(parts).strip()


async def handle_bookmark(
    ctx: HandlerContext,
    message: Message,
    title: str,
    last_exchanges: dict[int, LastExchange],
) -> None:
    """Bookmark the current thinking session or last Q&A exchange as a vault note."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(title, max_length=MAX_QUERY_LENGTH, operation="bookmark")
    title = san.text
    if not title:
        await message.answer("Empty title after sanitization.")
        return

    user_id = message.from_user.id if message.from_user else 0
    now = datetime.now(tz=UTC)
    date_str = now.strftime("%Y-%m-%d")
    slug = _slugify(title)
    filename = f"{date_str}-bookmark-{slug}.md"

    if ctx.thinking.has_active_session(user_id):
        # Pull history from the in-memory session
        session = ctx.thinking._get_session(user_id)
        if not session.history:
            await message.answer(
                "Nothing to bookmark. Start a /think session or ask a question first."
            )
            return
        session_body = _format_session_body(session.history)
        note_content = THINKING_BOOKMARK_TEMPLATE.format(
            title=title,
            date=date_str,
            session_body=session_body,
        )
    elif user_id in last_exchanges:
        exchange = last_exchanges[user_id]
        note_content = QA_BOOKMARK_TEMPLATE.format(
            title=title,
            date=date_str,
            query=exchange.query,
            response=exchange.response,
        )
    else:
        await message.answer("Nothing to bookmark. Start a /think session or ask a question first.")
        return

    inbox_path = ctx.vault_root / INBOX_FOLDER
    inbox_path.mkdir(parents=True, exist_ok=True)
    filepath = inbox_path / filename
    filepath.write_text(note_content, encoding="utf-8")
    logger.info("Bookmarked note: %s", filepath)

    note = ctx.parser.parse_file(filepath)
    ctx.store.index_single_note(note, ctx.parser)

    relative_path = f"{INBOX_FOLDER}/{filename}"
    await message.answer(
        f"📌 Bookmarked: `{relative_path}`",
        parse_mode="Markdown",
    )
