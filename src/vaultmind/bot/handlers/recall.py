"""Source-only recall with current-file snapshots and revalidated pagination."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from vaultmind.bot.formatter import TelegramFormatter
from vaultmind.bot.handlers.utils import _is_authorized
from vaultmind.bot.sanitize import MAX_QUERY_LENGTH, sanitize_text
from vaultmind.indexer.ranking import RetrievalConfidence, retrieval_confidence
from vaultmind.knowledge.sources import SourceSearchError, revalidate_source_hit, search_sources
from vaultmind.memory.gaps import GapKind, GapStore

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.handlers.bookmark import LastExchange
    from vaultmind.bot.handlers.context import HandlerContext


logger = logging.getLogger(__name__)


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

    lines = [
        f"\U0001f50d <b>Sources for:</b> <i>{escape(session.query, quote=False)}</i>\n"
        "Source-only / unapproved — not wiki answers.\n"
    ]
    for i, hit in enumerate(page_results, start + 1):
        identity = f"SHA256: <code>{escape(hit['current_digest'], quote=False)}</code>"
        if hit.get("source_id"):
            source_id = escape(hit["source_id"], quote=False)
            identity = f"Source: <code>{source_id}</code>\n{identity}"
        provenance = hit.get("provenance") or {}
        origin = [
            f"{key}: {' '.join(str(provenance[key]).split())[:120]}"
            for key in ("adapter", "origin", "role", "revision")
            if provenance.get(key)
        ]
        if origin:
            identity += "\n" + escape(" | ".join(origin)[:200], quote=False)
        lines.append(f"{TelegramFormatter.format_search_result(hit, i)}\n{identity}\n")

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
    """Discover current unapproved sources without answering or promoting them."""
    if not _is_authorized(ctx, message):
        return

    san = sanitize_text(query, max_length=MAX_QUERY_LENGTH, operation="recall")
    query = san.text
    if not query:
        await message.answer("Empty query after sanitization.")
        return

    await message.answer("\U0001f50d Searching sources...")

    max_results = ctx.settings.search.max_results
    page_size = ctx.settings.search.page_size

    try:
        results = await asyncio.to_thread(
            search_sources, ctx.settings, query, store=ctx.store, n_results=max_results
        )
    except SourceSearchError:
        await message.answer(
            "Source search unavailable. Narrow the source scope or run /recall again."
        )
        return

    user_id = message.from_user.id if message.from_user else 0

    if (
        isinstance(ctx.gap_store, GapStore)
        and retrieval_confidence(results, ctx.settings.bench.score_floor)
        is RetrievalConfidence.WEAK
    ):
        try:
            await asyncio.to_thread(
                ctx.gap_store.mint,
                query,
                GapKind.WEAK_RETRIEVAL,
                f"telegram-recall:{user_id}",
            )
        except Exception:
            logger.exception("Gap minting failed for recall query %r", query)

    if not results:
        await message.answer("No matching sources found.")
        return

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

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


async def handle_recall_page_callback(
    ctx: HandlerContext,
    callback: CallbackQuery,
    search_sessions: dict[int, PaginatedSearch],
) -> None:
    """Handle pagination button press — edits the search result message in-place."""
    if not _is_authorized(ctx, callback):
        await callback.answer("Not authorized.")
        return

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
    if callback.from_user is None or callback.from_user.id != uid:
        await callback.answer("This search belongs to another user.")
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

    start = page * session.page_size
    end = start + session.page_size
    try:
        current_hits = [
            await asyncio.to_thread(revalidate_source_hit, ctx.settings, hit)
            for hit in session.results[start:end]
        ]
    except SourceSearchError:
        del search_sessions[uid]
        notice = "Sources changed or are unavailable. Run /recall again."
        if callback.message and hasattr(callback.message, "edit_text"):
            await callback.message.edit_text(notice, reply_markup=None)
        await callback.answer(notice)
        return
    session.results[start:end] = current_hits

    session.current_page = page
    text, keyboard = _build_keyboard_for_user(session, uid)

    if callback.message and hasattr(callback.message, "edit_text"):
        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()
