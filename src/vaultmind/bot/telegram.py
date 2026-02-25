"""Telegram bot — aiogram 3.x-based PKM interface.

Provides: quick capture, semantic recall, knowledge graph queries,
daily notes, thinking partner mode, note reading, editing, and deletion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.commands import CommandHandlers

logger = logging.getLogger(__name__)

router = Router()


def create_bot(token: str) -> tuple[Bot, Dispatcher]:
    """Create and configure the Telegram bot and dispatcher."""
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    return bot, dp


def register_handlers(handlers: CommandHandlers) -> None:
    """Register all command handlers on the router."""

    @router.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "🧠 *VaultMind* — Your AI-powered second brain\n\n"
            "Send text → I'll respond using your vault context\n"
            "Prefix with `note:` to capture to inbox\n\n"
            "Commands:\n"
            "• `/recall <query>` → semantic search\n"
            "• `/think <topic>` → thinking partner mode\n"
            "• `/graph <entity>` → knowledge graph query\n"
            "• `/daily` → today's daily note\n"
            "• `/notes <date>` → notes by date (natural language)\n"
            "• `/read <note>` → read a note\n"
            "• `/edit <note> <instruction>` → edit a note\n"
            "• `/delete <note>` → delete a note\n"
            "• `/bookmark <title>` → bookmark session/last Q&A to vault\n"
            "• `/suggest <note>` → find notes to link\n"
            "• `/duplicates <note>` → find duplicate/similar notes\n"
            "• `/review` → weekly review prompts\n"
            "• `/health` → system health check\n"
            "• `/stats` → vault & graph statistics",
            parse_mode="Markdown",
        )

    @router.message(Command("bookmark"))
    async def cmd_bookmark(message: Message) -> None:
        title = message.text.replace("/bookmark", "", 1).strip() if message.text else ""
        if not title:
            await message.answer("Usage: `/bookmark <title>`", parse_mode="Markdown")
            return
        await handlers.handle_bookmark(message, title)

    @router.message(Command("recall"))
    async def cmd_recall(message: Message) -> None:
        query = message.text.replace("/recall", "", 1).strip() if message.text else ""
        if not query:
            await message.answer("Usage: `/recall <search query>`", parse_mode="Markdown")
            return
        await handlers.handle_recall(message, query)

    @router.message(Command("think"))
    async def cmd_think(message: Message) -> None:
        topic = message.text.replace("/think", "", 1).strip() if message.text else ""
        if not topic:
            await message.answer("Usage: `/think <topic or question>`", parse_mode="Markdown")
            return
        await handlers.handle_think(message, topic)

    @router.message(Command("graph"))
    async def cmd_graph(message: Message) -> None:
        entity = message.text.replace("/graph", "", 1).strip() if message.text else ""
        if not entity:
            await message.answer("Usage: `/graph <entity name>`", parse_mode="Markdown")
            return
        await handlers.handle_graph(message, entity)

    @router.message(Command("daily"))
    async def cmd_daily(message: Message) -> None:
        await handlers.handle_daily(message)

    @router.message(Command("notes"))
    async def cmd_notes(message: Message) -> None:
        query = message.text.replace("/notes", "", 1).strip() if message.text else ""
        if not query:
            await message.answer(
                "Usage: `/notes <date or expression>`\n"
                "Examples: `yesterday`, `last week`, `2026-02-20`, "
                "`over the weekend`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_notes(message, query)

    @router.message(Command("read"))
    async def cmd_read(message: Message) -> None:
        query = message.text.replace("/read", "", 1).strip() if message.text else ""
        if not query:
            await message.answer(
                "Usage: `/read <note path or search term>`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_read(message, query)

    @router.message(Command("delete"))
    async def cmd_delete(message: Message) -> None:
        query = message.text.replace("/delete", "", 1).strip() if message.text else ""
        if not query:
            await message.answer(
                "Usage: `/delete <note path or search term>`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_delete(message, query)

    @router.message(Command("edit"))
    async def cmd_edit(message: Message) -> None:
        args = message.text.replace("/edit", "", 1).strip() if message.text else ""
        if not args:
            await message.answer(
                "Usage: `/edit <note path> <edit instruction>`\n"
                "Example: `/edit my-note.md add a summary section`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_edit(message, args)

    @router.message(Command("suggest"))
    async def cmd_suggest(message: Message) -> None:
        query = message.text.replace("/suggest", "", 1).strip() if message.text else ""
        if not query:
            await message.answer(
                "Usage: `/suggest <note path or search term>`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_suggestions(message, query)

    @router.message(Command("duplicates"))
    async def cmd_duplicates(message: Message) -> None:
        query = message.text.replace("/duplicates", "", 1).strip() if message.text else ""
        if not query:
            await message.answer(
                "Usage: `/duplicates <note path or search term>`",
                parse_mode="Markdown",
            )
            return
        await handlers.handle_duplicates(message, query)

    @router.message(Command("health"))
    async def cmd_health(message: Message) -> None:
        await handlers.handle_health(message)

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(
            "📖 *VaultMind Quick Guide*\n\n"
            "*Capture*\n"
            "Prefix any message with `note:`, `save:`, `capture:`, "
            "`remember:`, `jot:`, or `log:` to save it to your inbox.\n"
            "Pasting long text (500+ chars) or multiline (3+ lines) "
            "auto-captures too.\n\n"
            "*Chat*\n"
            "Send plain text and I respond using your vault as context. "
            "Questions get vault-augmented answers; casual messages get "
            "a quick reply.\n\n"
            "*Search & Explore*\n"
            "• `/recall <query>` — semantic search across your vault\n"
            "• `/graph <entity>` — explore knowledge graph connections\n"
            "• `/notes <date>` — find notes by date\n"
            "  _Accepts: `yesterday`, `last week`, `over the weekend`, "
            "`2026-02-20`_\n\n"
            "*Notes*\n"
            "• `/read <note>` — read a note (path, filename, or search)\n"
            "• `/edit <note> <instruction>` — AI-assisted edit with "
            "confirmation\n"
            "• `/delete <note>` — delete with confirmation\n"
            "• `/suggest <note>` — find notes worth linking\n"
            "• `/duplicates <note>` — find similar/duplicate notes\n"
            "• `/daily` — get or create today's daily note\n\n"
            "*Thinking*\n"
            "• `/think <topic>` — start a thinking partner session\n"
            "  _Modes: `explore:`, `critique:`, `synthesize:`, `plan:`_\n"
            "  Follow-up messages continue the session automatically.\n"
            "• `/bookmark <title>` — save session or last Q&A to vault\n\n"
            "*System*\n"
            "• `/health` — check system status\n"
            "• `/stats` — vault & graph statistics\n"
            "• `/review` — weekly review with graph insights",
            parse_mode="Markdown",
        )

    @router.message(Command("review"))
    async def cmd_review(message: Message) -> None:
        await handlers.handle_review(message)

    @router.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        await handlers.handle_stats(message)

    @router.message(F.voice)
    async def handle_voice(message: Message) -> None:
        await handlers.handle_voice(message)

    @router.message(F.text & ~F.text.startswith("/"))
    async def handle_text(message: Message) -> None:
        """Default handler: route plain text through message classifier."""
        if message.text:
            await handlers.handle_message(message, message.text)

    # --- Callback query handlers (confirmation flows) ---

    @router.callback_query(F.data.startswith("delete_"))
    async def callback_delete(callback: CallbackQuery) -> None:
        await handlers.handle_delete_callback(callback)

    @router.callback_query(F.data.startswith("edit_"))
    async def callback_edit(callback: CallbackQuery) -> None:
        await handlers.handle_edit_callback(callback)

    @router.callback_query(F.data.startswith("recall_page:"))
    async def callback_recall_page(callback: CallbackQuery) -> None:
        await handlers.handle_recall_page_callback(callback)
