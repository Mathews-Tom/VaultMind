"""Review handler — weekly review prompts + pending/failed autonomy review items.

`/review` is also the recovery surface for the review queue: pending SKIM items
(approve-all), pending BLOCK items (e.g. one whose Telegram delivery was disabled,
unavailable or failed, or whose keyboard message is gone) and FAILED items
(explicit, never automatic, retry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext

_MAX_LISTED = 5
_MAX_ITEM_SUMMARY = 500
_MAX_BUTTONS_PER_MESSAGE = 80


async def handle_review(ctx: HandlerContext, message: Message) -> None:
    """List actionable review items without exceeding Telegram transport limits."""
    if not _is_authorized(ctx, message):
        return

    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    stats = ctx.graph.stats
    bridges = ctx.graph.get_bridge_entities(3)
    orphans = ctx.graph.get_orphan_entities()[:5]
    review = [
        "\U0001f4cb Weekly Review\n",
        f"Vault: {ctx.store.count} chunks indexed",
        f"Graph: {stats['nodes']} entities, {stats['edges']} relationships\n",
        "Reflection Questions:",
        "1. What was the most important thing I learned this week?",
        "2. What project made the most progress?",
        "3. What's blocking me right now?",
        "4. What should I focus on next week?\n",
    ]
    if bridges:
        review.append("\U0001f309 Bridge Entities (connecting different knowledge areas):")
        for b in bridges:
            review.append(f"  \u2022 {b['entity']} ({b['type']})")
        review.append("")
    if orphans:
        review.append("\U0001f3dd Orphan Entities (consider connecting these):")
        for o in orphans:
            review.append(f"  \u2022 {o.get('label', o.get('id', 'unknown'))}")
        review.append("")

    sections: list[tuple[str, InlineKeyboardMarkup | None]] = [("\n".join(review), None)]
    from vaultmind.services.review_queue import Lane, ReviewQueue

    if isinstance(ctx.review_queue, ReviewQueue):
        from vaultmind.bot.handlers.autonomy import _clip, block_keyboard, retry_keyboard

        pending = ctx.review_queue.list_pending(lane=Lane.SKIM)
        if pending:
            skim = [f"\U0001f4e5 Pending Review ({len(pending)} SKIM item(s)):"]
            skim.extend(f"  \u2022 {p.summary}" for p in pending[:10])
            if len(pending) > 10:
                skim.append(f"  \u2026 and {len(pending) - 10} more")
            sections.append(
                (
                    "\n".join(skim),
                    InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text=f"Approve all {len(pending)} SKIM item(s)",
                                    callback_data="autonomy_approve_all_skim",
                                )
                            ]
                        ]
                    ),
                )
            )

        # Recover pending BLOCK items whose original delivery is unavailable.
        blocked = ctx.review_queue.list_pending(lane=Lane.BLOCK)
        if blocked:
            sections.append((f"\U0001f512 Awaiting Decision ({len(blocked)} BLOCK item(s)):", None))
            for number, p in enumerate(blocked[:_MAX_LISTED], start=1):
                sections.append(
                    (
                        f"  B{number}: {_clip(p.summary, _MAX_ITEM_SUMMARY)}",
                        block_keyboard(p, item_number=number),
                    )
                )
            if len(blocked) > _MAX_LISTED:
                sections.append((f"  \u2026 and {len(blocked) - _MAX_LISTED} more", None))

        # Listing never retries: partial side effects require explicit recovery.
        failed = ctx.review_queue.list_failed()
        if failed:
            sections.append(
                (
                    f"\u26a0\ufe0f Failed ({len(failed)} item(s)) \u2014 "
                    "operation incomplete; inspect state before retry:",
                    None,
                )
            )
            for number, p in enumerate(failed, start=1):
                sections.append(
                    (
                        f"  F{number}: {_clip(p.summary, _MAX_ITEM_SUMMARY)} "
                        f"\u2014 {p.result[:120]}",
                        retry_keyboard(p, item_number=number),
                    )
                )

    # Pack sections without detaching an item's actions from its own text.
    # Plain text also remains safe when a long summary needs a hard split.
    text = ""
    rows: list[list[InlineKeyboardButton]] = []
    for part, keyboard in sections:
        chunks = _split_message(part)
        for i, chunk in enumerate(chunks):
            candidate = f"{text}\n\n{chunk}" if text else chunk
            item_rows = (
                keyboard.inline_keyboard if keyboard is not None and i == len(chunks) - 1 else []
            )
            button_count = sum(len(row) for row in rows) + sum(len(row) for row in item_rows)
            if len(_split_message(candidate)) > 1 or button_count > _MAX_BUTTONS_PER_MESSAGE:
                await message.answer(
                    text,
                    parse_mode=None,
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if rows else None,
                )
                text, rows = chunk, []
            else:
                text = candidate
            if keyboard is not None and i == len(chunks) - 1:
                rows.extend(keyboard.inline_keyboard)
    if text:
        await message.answer(
            text,
            parse_mode=None,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if rows else None,
        )
