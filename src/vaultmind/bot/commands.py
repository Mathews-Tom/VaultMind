"""Command handlers — thin facade delegating to handlers/ package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.capture import handle_capture
from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.handlers.daily import handle_daily
from vaultmind.bot.handlers.delete import handle_delete, handle_delete_callback
from vaultmind.bot.handlers.duplicates import handle_duplicates
from vaultmind.bot.handlers.edit import handle_edit, handle_edit_callback
from vaultmind.bot.handlers.graph import handle_graph
from vaultmind.bot.handlers.health import handle_health
from vaultmind.bot.handlers.notes import (
    _find_notes_by_date,
    _llm_resolve_date,
    _resolve_date_range,
    handle_notes,
)
from vaultmind.bot.handlers.read import handle_read
from vaultmind.bot.handlers.recall import handle_recall
from vaultmind.bot.handlers.review import handle_review
from vaultmind.bot.handlers.routing import handle_greeting, handle_message, handle_smart_response
from vaultmind.bot.handlers.stats import handle_stats
from vaultmind.bot.handlers.suggestions import handle_suggestions
from vaultmind.bot.handlers.think import handle_think
from vaultmind.bot.handlers.voice import handle_voice

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import Settings
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.duplicate_detector import DuplicateDetector
    from vaultmind.indexer.note_suggester import NoteSuggester
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.parser import VaultParser


class CommandHandlers:
    """Implements all bot command logic, bridging Telegram to vault services."""

    def __init__(
        self,
        settings: Settings,
        store: VaultStore,
        graph: KnowledgeGraph,
        parser: VaultParser,
        thinking: ThinkingPartner,
        llm_client: LLMClient,
        duplicate_detector: DuplicateDetector | None = None,
        note_suggester: NoteSuggester | None = None,
    ) -> None:
        self._ctx = HandlerContext(
            settings=settings,
            store=store,
            graph=graph,
            parser=parser,
            thinking=thinking,
            llm_client=llm_client,
            vault_root=settings.vault.path,
        )
        self._duplicate_detector = duplicate_detector
        self._note_suggester = note_suggester
        self._pending_edits: dict[int, dict[str, str]] = {}

    # --- Delegated properties for backward compat (used by tests) ---

    @property
    def settings(self) -> Settings:
        return self._ctx.settings

    @property
    def store(self) -> VaultStore:
        return self._ctx.store

    @property
    def graph(self) -> KnowledgeGraph:
        return self._ctx.graph

    @property
    def parser(self) -> VaultParser:
        return self._ctx.parser

    @property
    def thinking(self) -> ThinkingPartner:
        return self._ctx.thinking

    @property
    def llm_client(self) -> LLMClient:
        return self._ctx.llm_client

    @property
    def vault_root(self) -> object:
        return self._ctx.vault_root

    # --- Delegated handlers ---

    async def handle_capture(self, message: Message, text: str) -> None:
        await handle_capture(self._ctx, message, text)

    async def handle_recall(self, message: Message, query: str) -> None:
        await handle_recall(self._ctx, message, query)

    async def handle_think(self, message: Message, topic: str) -> None:
        await handle_think(self._ctx, message, topic)

    async def handle_graph(self, message: Message, entity: str) -> None:
        await handle_graph(self._ctx, message, entity)

    async def handle_daily(self, message: Message) -> None:
        await handle_daily(self._ctx, message)

    async def handle_review(self, message: Message) -> None:
        await handle_review(self._ctx, message)

    async def handle_stats(self, message: Message) -> None:
        await handle_stats(self._ctx, message)

    async def handle_message(self, message: Message, text: str) -> None:
        await handle_message(
            self._ctx,
            message,
            text,
            capture_fn=handle_capture,
            think_fn=handle_think,
        )

    async def handle_greeting(self, message: Message) -> None:
        await handle_greeting(message)

    async def handle_smart_response(
        self,
        message: Message,
        text: str,
        *,
        is_question: bool,
    ) -> None:
        await handle_smart_response(self._ctx, message, text, is_question=is_question)

    async def handle_health(self, message: Message) -> None:
        await handle_health(self._ctx, message)

    async def handle_notes(self, message: Message, query: str) -> None:
        await handle_notes(self._ctx, message, query)

    async def handle_read(self, message: Message, query: str) -> None:
        await handle_read(self._ctx, message, query)

    async def handle_delete(self, message: Message, query: str) -> None:
        await handle_delete(self._ctx, message, query)

    async def handle_delete_callback(self, callback: CallbackQuery) -> None:
        await handle_delete_callback(self._ctx, callback)

    async def handle_edit(self, message: Message, args: str) -> None:
        await handle_edit(self._ctx, message, args, self._pending_edits)

    async def handle_edit_callback(self, callback: CallbackQuery) -> None:
        await handle_edit_callback(self._ctx, callback, self._pending_edits)

    async def handle_voice(self, message: Message) -> None:
        await handle_voice(self._ctx, message)

    async def handle_duplicates(self, message: Message, query: str) -> None:
        if self._duplicate_detector is None:
            await message.answer("Duplicate detection is not enabled.")
            return
        await handle_duplicates(self._ctx, message, query, self._duplicate_detector)

    async def handle_suggestions(self, message: Message, query: str) -> None:
        if self._note_suggester is None:
            await message.answer("Note suggestions are not enabled.")
            return
        await handle_suggestions(self._ctx, message, query, self._note_suggester)

    # --- Backward-compat methods (used by tests) ---

    def _is_authorized(self, message: Message) -> bool:
        from vaultmind.bot.handlers.utils import _is_authorized

        return _is_authorized(self._ctx, message)

    def _resolve_date_range(self, query: str) -> tuple[str | None, str | None]:
        return _resolve_date_range(self._ctx, query)

    def _llm_resolve_date(self, query: str, today: object) -> tuple[str | None, str | None]:
        from datetime import datetime

        assert isinstance(today, datetime)
        return _llm_resolve_date(self._ctx, query, today)

    def _find_notes_by_date(self, start: str, end: str) -> list[tuple[str, str, str]]:
        return _find_notes_by_date(self._ctx, start, end)

    @staticmethod
    def _slugify(text: str) -> str:
        from vaultmind.bot.handlers.capture import _slugify

        return _slugify(text)

    @staticmethod
    def _split_message(text: str, max_len: int = 4000) -> list[str]:
        from vaultmind.bot.handlers.utils import _split_message

        return _split_message(text, max_len)

    def _resolve_note_path(self, query: str) -> object:
        from vaultmind.bot.handlers.utils import _resolve_note_path

        return _resolve_note_path(self._ctx, query)
