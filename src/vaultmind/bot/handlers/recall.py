"""Recall handler — semantic search over the vault with pagination."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext


@dataclass
class PaginatedSearch:
    query: str
    results: list[dict[str, Any]]
    page_size: int
    current_page: int = 0
    created_at: float = field(default_factory=time.monotonic)


def _render_page(session: PaginatedSearch) -> tuple[str, InlineKeyboardMarkup]:
    """Render the current page of results as text + inline keyboard."""
    results = session.results
    page_size = session.page_size
    page = session.current_page
    total_pages = max(1, (len(results) + page_size - 1) // page_size)

    start = page * page_size
    end = start + page_size
    page_results = results[start:end]

    lines = [f"\U0001f50d **Results for:** _{session.query}_\n"]
    for i, hit in enumerate(page_results, start + 1):
        meta = hit["metadata"]
        title = meta.get("note_title", "Untitled")
        note_path = meta.get("note_path", "")
        heading = meta.get("heading", "")
        distance = hit.get("distance", 0)
        relevance = max(0, round((1 - distance) * 100))

        content = hit["content"][:200].replace("\n", " ").strip()
        if len(hit["content"]) > 200:
            content += "..."

        location = f"`{note_path}`"
        if heading:
            location += f" \u2192 {heading}"

        lines.append(f"**{i}. {title}** ({relevance}% match)\n{location}\n_{content}_\n")

    text = "\n".join(lines)

    # Build navigation buttons — omit prev on page 0, omit next on last page
    buttons: list[InlineKeyboardButton] = []

    if page > 0:
        buttons.append(
            InlineKeyboardButton(
                text="\u25c0 Prev",
                callback_data=f"recall_page:__UID__:{page - 1}",
            )
        )

    buttons.append(
        InlineKeyboardButton(
            text=f"Page {page + 1}/{total_pages}",
            callback_data="noop",
        )
    )

    if page < total_pages - 1:
        buttons.append(
            InlineKeyboardButton(
                text="\u25b6 Next",
                callback_data=f"recall_page:__UID__:{page + 1}",
            )
        )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
    return text, keyboard


def _build_keyboard_for_user(
    session: PaginatedSearch, user_id: int
) -> tuple[str, InlineKeyboardMarkup]:
    """Render page and substitute __UID__ placeholder with the real user_id."""
    text, keyboard = _render_page(session)
    new_rows: list[list[InlineKeyboardButton]] = []
    for row in keyboard.inline_keyboard:
        new_row: list[InlineKeyboardButton] = []
        for btn in row:
            cd = btn.callback_data or ""
            new_row.append(
                InlineKeyboardButton(
                    text=btn.text,
                    callback_data=cd.replace("__UID__", str(user_id)),
                )
            )
        new_rows.append(new_row)
    return text, InlineKeyboardMarkup(inline_keyboard=new_rows)


async def handle_recall(
    ctx: HandlerContext,
    message: Message,
    query: str,
    search_sessions: dict[int, PaginatedSearch],
    last_exchanges: dict[int, LastExchange] | None = None,
) -> None:
    """Semantic search over the vault with paginated results."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(query, max_length=MAX_QUERY_LENGTH, operation="recall")
    query = san.text
    if not query:
        await message.answer("Empty query after sanitization.")
        return

    await message.answer("\U0001f50d Searching vault...")

    max_results = ctx.settings.search.max_results
    page_size = ctx.settings.search.page_size

    results = await asyncio.to_thread(ctx.store.search, query, n_results=max_results)

    if not results:
        await message.answer("No matching notes found.")
        return

    user_id = message.from_user.id if message.from_user else 0

    session = PaginatedSearch(
        query=query,
        results=results,
        page_size=page_size,
        current_page=0,
    )
    search_sessions[user_id] = session

    text, keyboard = _build_keyboard_for_user(session, user_id)

    if last_exchanges is not None:
        from vaultmind.bot.handlers.bookmark import LastExchange as _LastExchange

        last_exchanges[user_id] = _LastExchange(
            query=query,
            response=text,
            timestamp=time.monotonic(),
        )

    await message.answer(text, parse_mode="Markdown", reply_markup=keyboard)


async def handle_recall_page_callback(
    ctx: HandlerContext,
    callback: CallbackQuery,
    search_sessions: dict[int, PaginatedSearch],
) -> None:
    """Handle pagination button press — edits the search result message in-place."""
    data = callback.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid callback data.")
        return

    _, uid_str, page_str = parts
    try:
        uid = int(uid_str)
        page = int(page_str)
    except ValueError:
        await callback.answer("Invalid callback data.")
        return

    session = search_sessions.get(uid)
    if session is None:
        await callback.answer("Session expired, run /recall again.")
        return

    ttl = ctx.settings.search.session_ttl
    if time.monotonic() - session.created_at > ttl:
        del search_sessions[uid]
        await callback.answer("Session expired, run /recall again.")
        return

    total_pages = max(1, (len(session.results) + session.page_size - 1) // session.page_size)
    if page < 0 or page >= total_pages:
        await callback.answer("Page out of range.")
        return

    session.current_page = page
    text, keyboard = _build_keyboard_for_user(session, uid)

    if callback.message and hasattr(callback.message, "edit_text"):
        await callback.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
    await callback.answer()
