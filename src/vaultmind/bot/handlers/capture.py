"""Telegram capture adapters: preserve first, project and index explicitly."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_CAPTURE_LENGTH, sanitize_text
from vaultmind.knowledge.intake import CaptureResult, SourceIntake
from vaultmind.llm.client import ContentPart, MultimodalMessage
from vaultmind.vault.ingest import detect_url, ingest_url
from vaultmind.vault.security import validate_ordinary_write_path
from vaultmind.wiki.workspace import WikiError, _metadata, _safe_text

if TYPE_CHECKING:
    from collections.abc import Buffer
    from pathlib import Path

    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)


class BoundedDownload(io.BytesIO):
    """Bound Telegram's actual bytes, not just its optional size declaration."""

    name = "attachment"

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: Buffer) -> int:
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise ValueError("Telegram attachment exceeds its byte limit")
        return super().write(data)


def telegram_identity(message: Message) -> dict[str, Any]:
    actor = message.from_user.id if message.from_user else None
    chat = getattr(getattr(message, "chat", None), "id", None)
    event = getattr(message, "message_id", None)
    origin = f"telegram:chat={chat}:actor={actor}"
    event_id = f"{chat}:{event}:{actor}" if type(chat) is int and type(event) is int else None
    return dict(origin=origin, event_id=event_id)


def _validate_text(original: str, logical: str) -> None:
    # Both views matter: `note:` must not hide a retention frontmatter directive.
    if len(original) > MAX_CAPTURE_LENGTH or len(logical) > MAX_CAPTURE_LENGTH:
        raise WikiError("Capture exceeds its input limit; no truncated capture was accepted")
    _metadata(_safe_text(original.encode("utf-8"), "Submitted capture"))
    _metadata(_safe_text(logical.encode("utf-8"), "Capture content").strip())
    if not logical.strip():
        raise WikiError("Empty capture refused")


def _destination(ctx: HandlerContext, folder: str) -> Path:
    return validate_ordinary_write_path(f"{folder.rstrip('/')}/capture.md", ctx.vault_root)


def _project_and_index(
    ctx: HandlerContext,
    intake: SourceIntake,
    capture: CaptureResult,
    content: str,
    *,
    folder: str,
    source: str = "telegram",
    authority: int = 5,
) -> str:
    relative = f"{folder.rstrip('/')}/{capture.source_id}.md"
    target = validate_ordinary_write_path(relative, ctx.vault_root)

    def project() -> str:
        intake.read(capture.source_id)
        destination = validate_ordinary_write_path(relative, ctx.vault_root)
        if destination.exists():
            raise WikiError("Capture projection already exists")
        body = (
            "---\ntype: fleeting\ntags: [capture]\nstatus: active\n"
            f"source: {source}\nauthority: {authority}\nsource_id: {capture.source_id}\n"
            f"created: {datetime.now().isoformat()}\n---\n\n{content}\n"
        )
        intake.workspace._atomic_bytes(destination, body.encode("utf-8"))
        return relative

    projection = capture.run_stage("projection", project)
    index_status = "not-attempted"
    if projection.status == "succeeded":

        def index() -> None:
            intake.read(capture.source_id)
            note = ctx.parser.parse_file(target)
            ctx.store.index_single_note(note, ctx.parser)

        index_status = capture.run_stage("index", index).status
    return (
        f"Preserved source `{capture.source_id}` ({capture.status}; source-only, not approved).\n"
        f"Projection: `{relative}` ({projection.status}); index: {index_status}."
    )


def _capture_sync(
    ctx: HandlerContext,
    text: str,
    original: str,
    identity: dict[str, Any],
    *,
    adapter: str,
    role: str,
    original_retained: bool,
    revision: str | None,
) -> str:
    _validate_text(original, text)
    folder = ctx.settings.vault.inbox_folder
    _destination(ctx, folder)
    url = detect_url(text) if ctx.settings.ingest.enabled else None
    if url:
        _destination(ctx, ctx.settings.ingest.inbox_folder)
    with SourceIntake(ctx.settings) as intake:
        captured = intake.preserve_text(
            original,
            original_name="telegram.txt",
            adapter=adapter,
            role=role,
            original_retained=original_retained,
            revision=revision,
            **identity,
        )
        logical = sanitize_text(text, max_length=MAX_CAPTURE_LENGTH, operation="capture").text
        summary = _project_and_index(ctx, intake, captured, logical, folder=folder)
        if not url:
            return summary

        def extract() -> Any:
            result = asyncio.run(ingest_url(url, language=ctx.settings.ingest.youtube_language))
            return result

        fetched = captured.run_stage("provider", extract)
        if fetched.status != "succeeded" or fetched.value is None:
            return (
                summary + f"\nURL fetch: {fetched.status}; original preserved; no new extraction."
            )
        result = fetched.value

        def derive() -> str:
            if not result.content or len(result.content) > ctx.settings.ingest.max_content_length:
                raise WikiError("Extracted content is empty or exceeds its input limit")
            derived = intake.preserve_text(
                result.content,
                original_name="extracted.txt",
                origin=result.url,
                adapter=f"url-extraction:{result.source_type}",
                role="extracted-text",
                event_id=captured.source_id,
                parents=(captured.source_id,),
                uri=result.url,
            )
            return _project_and_index(
                ctx,
                intake,
                derived,
                result.content,
                folder=ctx.settings.ingest.inbox_folder,
                source="url-extraction",
                authority=2,
            )

        derivative = captured.run_stage("derivative", derive)
        return (
            summary
            + f"\nURL fetch: succeeded; extraction: {derivative.status}.\n"
            + (derivative.value if isinstance(derivative.value, str) else "Original preserved.")
        )


async def handle_capture(
    ctx: HandlerContext,
    message: Message,
    text: str,
    *,
    raw_original: str | None = None,
    origin: str | None = None,
    adapter: str = "telegram-text",
    role: str = "text",
    original_retained: bool = True,
    revision: str | None = None,
) -> None:
    if not _is_authorized(ctx, message):
        await message.answer("⛔ Unauthorized")
        return
    identity = telegram_identity(message)
    if origin is not None:
        identity["origin"] = origin
    submitted = getattr(message, "text", None)
    original = (
        raw_original
        if raw_original is not None
        else (submitted if isinstance(submitted, str) else text)
    )
    try:
        summary = await asyncio.to_thread(
            _capture_sync,
            ctx,
            text,
            original,
            identity,
            adapter=adapter,
            role=role,
            original_retained=original_retained,
            revision=revision,
        )
    except Exception as exc:
        logger.warning("Capture refused or incomplete: %s", type(exc).__name__)
        await message.answer(
            "Capture refused or incomplete; no indexed-success claim. "
            "Check retention, size and destination."
        )
        return
    await message.answer(summary)


_VISION_PROMPT = (
    "Describe this image precisely for a personal knowledge note. Include visible text, "
    "facts and concepts. Output plain text only."
)


def _photo_sync(ctx: HandlerContext, image: bytes, caption: str, identity: dict[str, Any]) -> str:
    folder = ctx.settings.vault.inbox_folder
    model = ctx.settings.image.vision_model or ctx.settings.llm.fast_model
    provider = ctx.llm_client.provider_name
    _destination(ctx, folder)
    if ctx.settings.image.save_originals:
        _destination(ctx, f"{folder.rstrip('/')}/images")
    _validate_text(caption or "Telegram photo", caption or "Telegram photo")
    if len(image) > min(ctx.settings.image.max_image_size_bytes, 20 * 1024 * 1024):
        raise WikiError("Photo exceeds its byte limit")
    identity = {
        **identity,
        "revision": hashlib.sha256(
            json.dumps(
                {"image_sha256": hashlib.sha256(image).hexdigest(), "caption": caption},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    with SourceIntake(ctx.settings) as intake:
        # With originals disabled only a textual event descriptor is retained, never binary.
        if ctx.settings.image.save_originals:
            caption_source = None
            if caption:
                caption_source = intake.preserve_text(
                    caption,
                    original_name="caption.txt",
                    adapter="telegram-photo",
                    role="caption",
                    **identity,
                )
            original = intake.preserve_attachment(
                image,
                original_name="photo.jpg",
                adapter="telegram-photo",
                media_type="image/jpeg",
                parents=(caption_source.source_id,) if caption_source else (),
                **identity,
            )
        else:
            original = intake.preserve_text(
                caption or "Telegram photo: binary original not retained.",
                original_name="photo-event.txt",
                adapter="telegram-photo",
                role="nonretained-photo",
                original_retained=False,
                **identity,
            )
        image_projection = None
        if ctx.settings.image.save_originals:
            image_relative = f"{folder.rstrip('/')}/images/{original.source_id}.jpg"

            def project_image() -> str:
                retained = intake.read(original.source_id)
                target = validate_ordinary_write_path(image_relative, ctx.vault_root)
                if target.exists():
                    raise WikiError("Image projection already exists")
                intake.workspace._atomic_bytes(target, retained.content)
                return image_relative

            image_projection = original.run_stage("projection", project_image)

        def vision() -> str:
            intake.read(original.source_id)
            data = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
            response = ctx.llm_client.complete_multimodal(
                [
                    MultimodalMessage(
                        role="user",
                        parts=[
                            ContentPart(type="text", text=_VISION_PROMPT),
                            ContentPart(type="image_url", image_url=data),
                        ],
                    )
                ],
                model,
                1024,
            )
            if not response.text.strip():
                raise WikiError("Vision returned no description")
            return response.text

        generated = original.run_stage("provider", vision)
        summary = (
            f"Photo source `{original.source_id}` ({original.status}); "
            f"binary original "
            f"{'preserved' if ctx.settings.image.save_originals else 'not retained'}. "
            f"Vision: {generated.status}. "
            f"Original projection: "
            f"{image_projection.status if image_projection else 'not retained'}."
        )
        if generated.status != "succeeded" or generated.value is None:
            return summary + " No new derived capture."

        def derive() -> str:
            description = intake.preserve_text(
                generated.value,
                original_name="photo-description.txt",
                origin=identity["origin"],
                adapter=f"vision:{provider}:{model}",
                role="image-description",
                event_id=original.source_id,
                parents=(original.source_id,),
                attachments=(original.source_id,) if ctx.settings.image.save_originals else (),
                original_retained=ctx.settings.image.save_originals,
            )
            body = generated.value
            if image_projection is not None and image_projection.status == "succeeded":
                body = f"![[images/{original.source_id}.jpg]]\n\n{body}"
            return _project_and_index(
                ctx,
                intake,
                description,
                body,
                folder=folder,
                source="telegram-photo",
            )

        derivative = original.run_stage("derivative", derive)
        return (
            summary
            + f"\nDescription: {derivative.status}.\n"
            + (derivative.value if isinstance(derivative.value, str) else "No new derived capture.")
        )


async def handle_photo_capture(ctx: HandlerContext, message: Message) -> None:
    if not _is_authorized(ctx, message):
        await message.answer("⛔ Unauthorized")
        return
    if not ctx.settings.image.enabled:
        await message.answer("Image capture is disabled.")
        return
    if not message.photo:
        await message.answer("No photo found in message.")
        return
    photo = message.photo[-1]
    limit = min(ctx.settings.image.max_image_size_bytes, 20 * 1024 * 1024)
    caption = message.caption if isinstance(message.caption, str) else ""
    try:
        # Retention/destination/declared size are checked before download or provider use.
        await asyncio.to_thread(
            _validate_text, caption or "Telegram photo", caption or "Telegram photo"
        )
        await asyncio.to_thread(_destination, ctx, ctx.settings.vault.inbox_folder)
        if photo.file_size is not None and photo.file_size > limit:
            raise WikiError("Photo exceeds its byte limit")
        bot = message.bot
        if bot is None:
            raise WikiError("Telegram transport unavailable")
        file = await bot.get_file(photo.file_id)
        if not file.file_path:
            raise WikiError("Photo is unavailable")
        with BoundedDownload(limit) as stream:
            await bot.download_file(file.file_path, destination=stream)
            image = stream.getvalue()
        summary = await asyncio.to_thread(
            _photo_sync, ctx, image, caption, telegram_identity(message)
        )
    except Exception as exc:
        logger.warning("Photo capture failed or refused: %s", type(exc).__name__)
        await message.answer("Photo capture failed or refused; no derived/indexed-success claim.")
        return
    await message.answer(summary)
