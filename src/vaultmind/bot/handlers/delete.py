"""Delete handler — note deletion with confirmation flow.

M9: a bot-initiated delete never removes a note's text. Confirming deletion
writes a dated `> [!superseded]` callout in place of removal (the note's
full prior content stays on disk, unchanged, underneath the callout) and
drops the note from the vector store so it no longer surfaces via
`/recall`. One `LineageEdge` is recorded per confirmed deletion, queryable
via `ctx.lineage_store.get_lineage(note_path)`.

M14: the callout carries an explicit paired boundary
(`vaultmind.vault.projection`) so the note projects to no current content.
The lineage edge is recorded for the actual file mutation before search
retirement is attempted; a retirement failure is reported, never claimed as
success. Confirming again on an already-marked note performs no second
mutation, records a lineage edge only if none exists, and retries retirement.
A note whose bot supersede history is ambiguous or malformed is refused
untouched, with no lineage edge, whether the confirmation is first or repeated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path
from vaultmind.bot.sanitize import sanitize_path
from vaultmind.contradiction.marking import write_superseded_block
from vaultmind.graph.evolution import LineageStore
from vaultmind.vault.projection import (
    DELETE_MARKER,
    DELETE_RATIONALE,
    DELETE_TITLE,
    SUPERSEDED_CALLOUT,
    SUPERSEDED_KEY,
)
from vaultmind.vault.security import (
    PathResolutionError,
    PathTraversalError,
    validate_vault_path,
)

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
        f"\u26a0\ufe0f **Delete note?** _(marks superseded \u2014 content is preserved)_"
        f"\n\n`{rel_path}`\n\n_{preview}_",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def handle_delete_callback(ctx: HandlerContext, callback: CallbackQuery) -> None:
    """Process delete confirmation/cancellation."""
    if not _is_authorized(ctx, callback):
        await callback.answer("\u26d4 Unauthorized")
        return

    data = callback.data or ""

    if data == "delete_cancel":
        await callback.message.edit_text("\u274c Deletion cancelled.")  # type: ignore[union-attr]
        await callback.answer()
        return

    if data.startswith("delete_confirm:"):
        rel_path = data[len("delete_confirm:") :]
        try:
            filepath = validate_vault_path(rel_path, ctx.vault_root)
        except (PathTraversalError, PathResolutionError):
            await callback.message.edit_text(  # type: ignore[union-attr]
                "Path not allowed.",
            )
            await callback.answer()
            return

        if not filepath.exists():
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"Note not found: `{rel_path}`",
                parse_mode="Markdown",
            )
            await callback.answer()
            return

        try:
            written = write_superseded_block(
                filepath,
                callout_type=SUPERSEDED_CALLOUT,
                title=DELETE_TITLE,
                rationale=DELETE_RATIONALE,
                frontmatter_key=SUPERSEDED_KEY,
                frontmatter_value=DELETE_MARKER,
                dated=True,
                boundary="delete",
            )
        except Exception as exc:
            logger.exception("Failed to mark note as deleted: %s", rel_path)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"Delete failed for {rel_path}: {exc}\nThe delete was not applied."
            )
            await callback.answer()
            return

        # The file now (or already) carries the delete marker. Record the lineage edge for
        # the mutation first, then drop the note from search; either step may fail without
        # undoing the other, and a repeat confirmation retries both without re-marking.
        problems: list[str] = []
        try:
            _ensure_deleted_edge(ctx, rel_path)
        except Exception as exc:
            logger.exception("Failed to record delete lineage for %s", rel_path)
            problems.append(f"the lineage edge was not recorded ({exc})")
        try:
            ctx.store.delete_note(rel_path)
        except Exception as exc:
            logger.exception("Failed to retire deleted note from search: %s", rel_path)
            problems.append(f"removing it from search failed ({exc})")

        if problems:
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"\u26a0\ufe0f {rel_path} is marked as deleted on disk (original text "
                f"preserved), but {'; '.join(problems)}. Confirm the delete again to retry."
            )
            await callback.answer()
            return

        if not written:
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"Note already superseded: `{rel_path}` \u2014 removed from search again.",
                parse_mode="Markdown",
            )
            await callback.answer()
            return

        logger.info("Superseded (soft-deleted) note: %s", rel_path)

        await callback.message.edit_text(  # type: ignore[union-attr]
            f"\U0001f5d1 Superseded (not deleted): `{rel_path}`\n\n"
            "A dated callout marks this note as deleted; original text is preserved.",
            parse_mode="Markdown",
        )
        await callback.answer()


def _ensure_deleted_edge(ctx: HandlerContext, rel_path: str) -> None:
    """Record the `deleted` lineage edge unless the note already has one."""
    if not isinstance(ctx.lineage_store, LineageStore):
        return
    if any(edge.event == "deleted" for edge in ctx.lineage_store.get_lineage(rel_path)):
        return
    ctx.lineage_store.record(rel_path, "deleted", DELETE_RATIONALE)
