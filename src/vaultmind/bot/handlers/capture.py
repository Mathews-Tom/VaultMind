"""Capture handler — save text as fleeting notes, with URL ingestion."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_CAPTURE_LENGTH, sanitize_text
from vaultmind.llm.client import ContentPart, MultimodalMessage
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


_PHOTO_NOTE_TEMPLATE = """\
---
type: fleeting
tags: [capture, photo]
created: {created}
source: telegram-photo
status: active
---

{embed}

## Description

{description}
"""

_VISION_PROMPT = (
    "You are a knowledge management assistant. Describe the content of this image "
    "concisely and precisely so it can be recalled later as a note in a personal "
    "knowledge base. Focus on key facts, text visible in the image, diagrams, or "
    "concepts depicted. Output plain text only — no markdown headings."
)


async def handle_photo_capture(ctx: HandlerContext, message: Message) -> None:
    """Download a Telegram photo, describe it via a vision model, and capture as a note."""
    if not _is_authorized(ctx, message):
        await message.answer("\u26d4 Unauthorized")
        return

    if not ctx.settings.image.enabled:
        await message.answer("Image capture is disabled.")
        return

    if not message.photo:
        await message.answer("No photo found in message.")
        return

    # Highest-resolution photo is the last entry in the array
    photo = message.photo[-1]

    # Size guard
    if photo.file_size is not None and photo.file_size > ctx.settings.image.max_image_size_bytes:
        mb = ctx.settings.image.max_image_size_bytes / 1_000_000
        await message.answer(f"Image too large (max {mb:.0f} MB).")
        return

    await message.answer("\U0001f4f8 Processing photo…")

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        bot = message.bot
        assert bot is not None, "message.bot is None"
        file = await bot.get_file(photo.file_id)
        assert file.file_path is not None, "file.file_path is None"
        await bot.download_file(file.file_path, destination=str(tmp_path))

        image_bytes = tmp_path.read_bytes()
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_data}"

        vision_model = ctx.settings.image.vision_model or ctx.settings.llm.fast_model

        mm_message = MultimodalMessage(
            role="user",
            parts=[
                ContentPart(type="text", text=_VISION_PROMPT),
                ContentPart(type="image_url", image_url=data_uri),
            ],
        )

        response = await asyncio.to_thread(
            ctx.llm_client.complete_multimodal,
            [mm_message],
            vision_model,
            1024,
        )
        description = response.text.strip()

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        inbox_path = ctx.vault_root / ctx.settings.vault.inbox_folder

        embed_ref = ""
        if ctx.settings.image.save_originals:
            images_dir = inbox_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            image_filename = f"{timestamp}.jpg"
            dest = images_dir / image_filename
            dest.write_bytes(image_bytes)
            embed_ref = f"![[images/{image_filename}]]"

        note_filename = f"{timestamp}-photo.md"
        note_content = _PHOTO_NOTE_TEMPLATE.format(
            created=now.strftime("%Y-%m-%d %H:%M"),
            embed=embed_ref,
            description=description,
        )

        inbox_path.mkdir(parents=True, exist_ok=True)
        note_path = inbox_path / note_filename
        note_path.write_text(note_content, encoding="utf-8")
        logger.info("Captured photo note: %s", note_path)

        try:
            note = await asyncio.to_thread(ctx.parser.parse_file, note_path)
            await asyncio.to_thread(ctx.store.index_single_note, note, ctx.parser)
        except Exception:
            logger.exception("Failed to index photo note")

        inbox = ctx.settings.vault.inbox_folder
        await message.answer(
            f"\U0001f4f8 Photo captured \u2192 `{inbox}/{note_filename}`\n\n_{description[:200]}_",
            parse_mode="Markdown",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
