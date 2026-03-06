"""Capture handler — save text as fleeting notes, with URL ingestion."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_CAPTURE_LENGTH, sanitize_text
from vaultmind.vault.ingest import IngestResult, create_vault_note, detect_url, ingest_url

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)

CAPTURE_TEMPLATE = """\
---
type: fleeting
tags: [{tags}]
created: {created}
source: telegram
status: active
---

{content}
"""


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:60]


async def handle_capture(ctx: HandlerContext, message: Message, text: str) -> None:
    """Capture text as a new fleeting note in the inbox.

    If the text contains a URL and ingestion is enabled, fetches content
    from the URL (YouTube transcript or article) and creates a source note.
    Otherwise falls through to standard fleeting note capture.
    """
    if not _is_authorized(ctx, message):
        await message.answer("\u26d4 Unauthorized")
        return

    # URL detection — ingest if enabled
    url = detect_url(text)
    if url and ctx.settings.ingest.enabled:
        try:
            result = await ingest_url(url, language=ctx.settings.ingest.youtube_language)
            if len(result.content) > ctx.settings.ingest.max_content_length:
                result = IngestResult(
                    source_type=result.source_type,
                    title=result.title,
                    content=result.content[: ctx.settings.ingest.max_content_length],
                    url=result.url,
                    metadata=result.metadata,
                )
            note_path = create_vault_note(result, ctx.vault_root, ctx.settings.ingest.inbox_folder)
            note = await asyncio.to_thread(ctx.parser.parse_file, note_path)
            await asyncio.to_thread(ctx.store.index_single_note, note, ctx.parser)
            rel = note_path.relative_to(ctx.vault_root)
            await message.answer(
                f"\U0001f517 Ingested {result.source_type} source:\n"
                f"**{result.title}**\n"
                f"Saved to: `{rel}`",
                parse_mode="Markdown",
            )
            return
        except Exception as exc:
            logger.exception("URL ingest failed: %s", url)
            await message.answer(f"URL ingest failed: {exc}\nCapturing as plain text instead.")

    san = sanitize_text(text, max_length=MAX_CAPTURE_LENGTH, operation="capture")
    text = san.text
    if not text:
        await message.answer("Empty input after sanitization.")
        return

    now = datetime.now()
    slug = now.strftime("%Y%m%d-%H%M%S")
    # Create a short title from first line or first 50 chars
    title = text.split("\n")[0][:50].strip()
    filename = f"{slug}-{_slugify(title)}.md"

    note_content = CAPTURE_TEMPLATE.format(
        tags="capture",
        created=now.strftime("%Y-%m-%d %H:%M"),
        content=text,
    )

    # Write to vault inbox
    inbox_path = ctx.vault_root / ctx.settings.vault.inbox_folder
    inbox_path.mkdir(parents=True, exist_ok=True)
    filepath = inbox_path / filename

    filepath.write_text(note_content, encoding="utf-8")
    logger.info("Captured note: %s", filepath)

    # Index immediately for instant recall (offload sync I/O to thread pool)
    try:
        note = await asyncio.to_thread(ctx.parser.parse_file, filepath)
        await asyncio.to_thread(ctx.store.index_single_note, note, ctx.parser)
    except Exception:
        logger.exception("Failed to index captured note")

    inbox = ctx.settings.vault.inbox_folder
    await message.answer(
        f"\U0001f4dd Captured \u2192 `{inbox}/{filename}`",
        parse_mode="Markdown",
    )
