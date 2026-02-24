"""Delete handler — note deletion with confirmation flow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path
from vaultmind.bot.sanitize import sanitize_path
from vaultmind.vault.security import PathTraversalError, validate_vault_path

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)


async def handle_delete(ctx: HandlerContext, message: Message, query: str) -> None:
    """Request deletion of a note \u2014 sends confirmation prompt."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_path(query)
    query = san.text
    if not query:
        await message.answer("Empty path after sanitization.")
        return

    filepath = _resolve_note_path(ctx, query)
    if filepath is None:
        await message.answer(
            f"Note not found: `{query}`",
            parse_mode="Markdown",
        )
        return

    rel_path = filepath.relative_to(ctx.vault_root)

    # Preview first 300 chars
    content = filepath.read_text(encoding="utf-8")
    preview = content[:300]
    if len(content) > 300:
        preview += "..."

    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="\U0001f5d1 Confirm Delete",
                    callback_data=f"delete_confirm:{rel_path}",
                ),
                InlineKeyboardButton(
                    text="\u274c Cancel",
                    callback_data="delete_cancel",
                ),
            ]
        ]
    )

    await message.answer(
        f"\u26a0\ufe0f **Delete note?**\n\n`{rel_path}`\n\n_{preview}_",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def handle_delete_callback(ctx: HandlerContext, callback: CallbackQuery) -> None:
    """Process delete confirmation/cancellation."""
    data = callback.data or ""

    if data == "delete_cancel":
        await callback.message.edit_text("\u274c Deletion cancelled.")  # type: ignore[union-attr]
        await callback.answer()
        return

    if data.startswith("delete_confirm:"):
        rel_path = data[len("delete_confirm:") :]
        try:
            filepath = validate_vault_path(rel_path, ctx.vault_root)
        except PathTraversalError:
            await callback.message.edit_text(  # type: ignore[union-attr]
                "Path not allowed.",
            )
            await callback.answer()
            return

        if not filepath.exists():
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"Note already removed: `{rel_path}`",
                parse_mode="Markdown",
            )
            await callback.answer()
            return

        # Remove from vector store
        ctx.store.delete_note(rel_path)

        # Delete file
        filepath.unlink()
        logger.info("Deleted note: %s", rel_path)

        await callback.message.edit_text(  # type: ignore[union-attr]
            f"\U0001f5d1 Deleted: `{rel_path}`",
            parse_mode="Markdown",
        )
        await callback.answer()
