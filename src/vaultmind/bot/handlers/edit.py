"""Edit handler — AI-assisted note editing with confirmation flow.

M9: a bot-initiated edit never discards a note's prior text. The LLM is
shown and asked to rewrite only the note *body* (frontmatter is never at
the model's mercy); on confirmation, the new body is written above a dated
`> [!superseded]` callout, with the full prior body preserved, unchanged,
beneath it. One `LineageEdge` is recorded per confirmed edit, queryable via
`ctx.lineage_store.get_lineage(note_path)`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import frontmatter

from vaultmind.bot.handlers.utils import _is_authorized, _resolve_note_path
from vaultmind.bot.sanitize import MAX_EDIT_INSTRUCTION_LENGTH, sanitize_path, sanitize_text
from vaultmind.contradiction.marking import write_superseded_block
from vaultmind.graph.evolution import LineageStore
from vaultmind.llm.client import LLMError
from vaultmind.llm.client import Message as LLMMessage
from vaultmind.vault.security import PathTraversalError, validate_vault_path

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.handlers.context import HandlerContext

logger = logging.getLogger(__name__)

NOTE_EDIT_SYSTEM_PROMPT = """\
You are a note editor. Apply the user's requested edit to the note body below.
Return ONLY the updated body text — no frontmatter, no explanations or commentary.
"""


async def handle_edit(
    ctx: HandlerContext,
    message: Message,
    args: str,
    pending_edits: dict[int, dict[str, str]],
) -> None:
    """Edit a note via LLM \u2014 sends confirmation before applying."""
    if not _is_authorized(ctx, message):
        return

    # Parse: /edit <path> <instruction>
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            "Usage: `/edit <note path> <edit instruction>`\n"
            "Example: `/edit 00-inbox/my-note.md add a section about testing`",
            parse_mode="Markdown",
        )
        return

    note_query, instruction = parts[0], parts[1]

    san_path = sanitize_path(note_query)
    note_query = san_path.text
    san_instr = sanitize_text(
        instruction,
        max_length=MAX_EDIT_INSTRUCTION_LENGTH,
        operation="edit_instruction",
    )
    instruction = san_instr.text
    if not note_query or not instruction:
        await message.answer("Empty note path or instruction after sanitization.")
        return

    filepath = _resolve_note_path(ctx, note_query)

    if filepath is None:
        await message.answer(
            f"Note not found: `{note_query}`",
            parse_mode="Markdown",
        )
        return

    rel_path = filepath.relative_to(ctx.vault_root)
    original_body = frontmatter.load(str(filepath)).content

    await message.answer("\u270f\ufe0f Generating edit...")

    model = ctx.settings.routing.chat_model or ctx.settings.llm.fast_model
    user_msg = f"**Note body:**\n```\n{original_body}\n```\n\n**Edit instruction:** {instruction}"

    try:
        response = ctx.llm_client.complete(
            messages=[LLMMessage(role="user", content=user_msg)],
            model=model,
            max_tokens=ctx.settings.llm.max_tokens,
            system=NOTE_EDIT_SYSTEM_PROMPT,
        )
    except LLMError as e:
        await message.answer(f"Edit failed \u2014 API error ({e.provider}): {e}")
        return

    edited = response.text.strip()
    # Strip markdown code fences if the LLM wrapped the response
    if edited.startswith("```") and edited.endswith("```"):
        edited = edited.split("\n", 1)[1].rsplit("\n", 1)[0]

    # Store pending edit for confirmation
    user_id = message.from_user.id if message.from_user else 0
    pending_edits[user_id] = {
        "path": str(rel_path),
        "original": original_body,
        "edited": edited,
        "instruction": instruction,
    }

    # Show diff preview
    preview = edited[:1500]
    if len(edited) > 1500:
        preview += "\n..."

    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="\u2705 Apply Edit",
                    callback_data=f"edit_confirm:{user_id}",
                ),
                InlineKeyboardButton(
                    text="\u274c Discard",
                    callback_data=f"edit_cancel:{user_id}",
                ),
            ]
        ]
    )

    msg = (
        f"\u270f\ufe0f **Proposed edit for** `{rel_path}` _(prior text is preserved)_:"
        f"\n\n```\n{preview}\n```\n\nApply this edit?"
    )
    await message.answer(
        msg,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def handle_edit_callback(
    ctx: HandlerContext,
    callback: CallbackQuery,
    pending_edits: dict[int, dict[str, str]],
) -> None:
    """Process edit confirmation/cancellation."""
    data = callback.data or ""

    if data.startswith("edit_cancel:"):
        user_id = int(data.split(":")[1])
        pending_edits.pop(user_id, None)
        await callback.message.edit_text("\u274c Edit discarded.")  # type: ignore[union-attr]
        await callback.answer()
        return

    if data.startswith("edit_confirm:"):
        user_id = int(data.split(":")[1])
        pending = pending_edits.pop(user_id, None)

        if pending is None:
            await callback.message.edit_text(  # type: ignore[union-attr]
                "Edit expired. Run `/edit` again.",
                parse_mode="Markdown",
            )
            await callback.answer()
            return

        try:
            filepath = validate_vault_path(pending["path"], ctx.vault_root)
        except PathTraversalError:
            await callback.message.edit_text(  # type: ignore[union-attr]
                "Path not allowed.",
            )
            await callback.answer()
            return

        if not filepath.exists():
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"Note no longer exists: `{pending['path']}`",
                parse_mode="Markdown",
            )
            await callback.answer()
            return

        # Apply the new body above a dated superseded callout; the prior
        # body stays fully intact beneath it — never overwritten.
        from datetime import UTC, datetime

        ts = datetime.now(UTC)
        rationale = f"Instruction: {pending['instruction']}"
        write_superseded_block(
            filepath,
            callout_type="superseded",
            title="Edited via bot",
            rationale=rationale,
            frontmatter_key="superseded_at",
            frontmatter_value=ts.isoformat(),
            dated=True,
            timestamp=ts,
            new_content=pending["edited"],
        )

        # Re-index
        try:
            note = ctx.parser.parse_file(filepath)
            ctx.store.index_single_note(note, ctx.parser)
        except Exception:
            logger.exception("Failed to re-index edited note")

        if isinstance(ctx.lineage_store, LineageStore):
            ctx.lineage_store.record(pending["path"], "edited", rationale)

        logger.info("Edited note (superseded prior body): %s", pending["path"])
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"\u2705 Edit applied to `{pending['path']}` \u2014 prior text preserved "
            "under a dated superseded callout.",
            parse_mode="Markdown",
        )
        await callback.answer()
