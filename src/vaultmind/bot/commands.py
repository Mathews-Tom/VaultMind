"""Command handlers — implementation logic for each Telegram bot command."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from vaultmind.bot.router import Intent, MessageRouter
from vaultmind.llm.client import LLMError
from vaultmind.llm.client import Message as LLMMessage
from vaultmind.vault.security import PathTraversalError, validate_vault_path

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import CallbackQuery, Message

    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import Settings
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)

# Frontmatter template for captured notes
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

DAILY_TEMPLATE = """\
---
type: daily
created: {date}
tags: [daily]
---

# {date_display}

## Captures

## Tasks

## Reflections

"""


QUESTION_SYSTEM_PROMPT = """\
You are a vault-aware assistant with access to the user's personal knowledge base (Obsidian vault).
Answer the question using the vault context provided below. Be concise and direct.
Reference notes with [[Note Title]] format when citing vault content.
If the vault context is insufficient, say so and suggest `/think <topic>` for deeper exploration.
"""

CHAT_SYSTEM_PROMPT = """\
You are a friendly vault-aware assistant. The user's knowledge base context is provided below.
Respond naturally and conversationally. Weave in relevant vault context when it adds value.
Keep responses brief — a few sentences unless the topic demands more.
Reference notes with [[Note Title]] format when citing vault content.
"""

GREETING_RESPONSES = [
    "Hey! What's on your mind?",
    "Hi there. Ask me anything or prefix with `note:` to capture something.",
    "What's up? I've got your vault loaded.",
    "Hey. Need to look something up, think through an idea, or capture a note?",
    "Hi! Ready when you are.",
]

DATE_RESOLUTION_SYSTEM_PROMPT = """\
You are a date parser. Given a natural language date expression and today's date, \
return a JSON object with "start" and "end" dates in YYYY-MM-DD format.

Rules:
- "yesterday" = yesterday 00:00 to yesterday 23:59
- "last week" = Monday to Sunday of the previous week
- "over the weekend" = last Saturday and Sunday
- "last Tuesday" = the most recent Tuesday before today
- "yesterday afternoon" = yesterday (just the date)
- For single-day references, start == end
- Always return valid JSON: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
- Return ONLY the JSON, no explanation
"""

NOTE_EDIT_SYSTEM_PROMPT = """\
You are a note editor. Apply the user's requested edit to the note content below.
Return ONLY the updated note content (including frontmatter if present).
Do not add explanations or commentary — just the edited note text.
"""


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
    ) -> None:
        self.settings = settings
        self.store = store
        self.graph = graph
        self.parser = parser
        self.thinking = thinking
        self.llm_client = llm_client
        self.vault_root = settings.vault.path
        self._router = MessageRouter()
        self._pending_edits: dict[int, dict[str, str]] = {}

    # --- Access control ---

    def _is_authorized(self, message: Message) -> bool:
        """Check if the user is allowed to use the bot."""
        allowed = self.settings.telegram.allowed_user_ids
        if not allowed:
            return True  # No restrictions
        return message.from_user is not None and message.from_user.id in allowed

    # --- /capture (default text handler) ---

    async def handle_capture(self, message: Message, text: str) -> None:
        """Capture text as a new fleeting note in the inbox."""
        if not self._is_authorized(message):
            await message.answer("⛔ Unauthorized")
            return

        now = datetime.now()
        slug = now.strftime("%Y%m%d-%H%M%S")
        # Create a short title from first line or first 50 chars
        title = text.split("\n")[0][:50].strip()
        filename = f"{slug}-{self._slugify(title)}.md"

        note_content = CAPTURE_TEMPLATE.format(
            tags="capture",
            created=now.strftime("%Y-%m-%d %H:%M"),
            content=text,
        )

        # Write to vault inbox
        inbox_path = self.vault_root / self.settings.vault.inbox_folder
        inbox_path.mkdir(parents=True, exist_ok=True)
        filepath = inbox_path / filename

        filepath.write_text(note_content, encoding="utf-8")
        logger.info("Captured note: %s", filepath)

        # Index immediately for instant recall
        try:
            note = self.parser.parse_file(filepath)
            self.store.index_single_note(note, self.parser)
        except Exception:
            logger.exception("Failed to index captured note")

        inbox = self.settings.vault.inbox_folder
        await message.answer(
            f"📝 Captured → `{inbox}/{filename}`",
            parse_mode="Markdown",
        )

    # --- /recall ---

    async def handle_recall(self, message: Message, query: str) -> None:
        """Semantic search over the vault."""
        if not self._is_authorized(message):
            return

        await message.answer("🔍 Searching vault...")

        results = self.store.search(query, n_results=5)

        if not results:
            await message.answer("No matching notes found.")
            return

        response_parts = [f"🔍 **Results for:** _{query}_\n"]
        for i, hit in enumerate(results, 1):
            meta = hit["metadata"]
            title = meta.get("note_title", "Untitled")
            note_path = meta.get("note_path", "")
            heading = meta.get("heading", "")
            distance = hit.get("distance", 0)
            relevance = max(0, round((1 - distance) * 100))

            # Truncate content preview
            content = hit["content"][:200].replace("\n", " ").strip()
            if len(hit["content"]) > 200:
                content += "..."

            location = f"`{note_path}`"
            if heading:
                location += f" → {heading}"

            response_parts.append(
                f"**{i}. {title}** ({relevance}% match)\n{location}\n_{content}_\n"
            )

        await message.answer("\n".join(response_parts), parse_mode="Markdown")

    # --- /think ---

    async def handle_think(self, message: Message, topic: str) -> None:
        """Start or continue a thinking partner session."""
        if not self._is_authorized(message):
            return

        user_id = message.from_user.id if message.from_user else 0
        await message.answer("🧠 Thinking...")

        response = await self.thinking.think(
            user_id=user_id,
            topic=topic,
            store=self.store,
            graph=self.graph,
        )

        # Split long responses for Telegram's 4096 char limit
        for chunk in self._split_message(response, max_len=4000):
            await message.answer(chunk, parse_mode="Markdown")

    # --- /graph ---

    async def handle_graph(self, message: Message, entity: str) -> None:
        """Query the knowledge graph for an entity's connections."""
        if not self._is_authorized(message):
            return

        result = self.graph.get_neighbors(entity, depth=2)

        if result["entity"] is None:
            await message.answer(
                f"Entity `{entity}` not found in knowledge graph.",
                parse_mode="Markdown",
            )
            return

        ent = result["entity"]
        lines = [
            f"🕸 **{ent.get('label', entity)}** ({ent.get('type', 'unknown')})",
            f"Confidence: {ent.get('confidence', 0):.0%}",
            f"Source notes: {len(ent.get('source_notes', []))}",
            "",
        ]

        if result["outgoing"]:
            lines.append("**→ Outgoing:**")
            for rel in result["outgoing"][:10]:
                lines.append(f"  • {rel['relation']} → {rel['target']}")
            lines.append("")

        if result["incoming"]:
            lines.append("**← Incoming:**")
            for rel in result["incoming"][:10]:
                lines.append(f"  • {rel['source']} → {rel['relation']}")
            lines.append("")

        if result["neighbors"]:
            neighbor_labels = [n.get("label", n["id"]) for n in result["neighbors"][:15]]
            lines.append(f"**Neighborhood:** {', '.join(neighbor_labels)}")

        await message.answer("\n".join(lines), parse_mode="Markdown")

    # --- /daily ---

    async def handle_daily(self, message: Message) -> None:
        """Get or create today's daily note."""
        if not self._is_authorized(message):
            return

        today = datetime.now()
        filename = today.strftime("%Y-%m-%d") + ".md"
        daily_dir = self.vault_root / self.settings.vault.daily_folder
        daily_dir.mkdir(parents=True, exist_ok=True)
        filepath = daily_dir / filename

        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            # Return a summary (first 1000 chars)
            preview = content[:1000]
            if len(content) > 1000:
                preview += "\n\n_...truncated_"
            date_str = today.strftime("%B %d, %Y")
            await message.answer(
                f"📅 **Daily Note — {date_str}**\n\n{preview}",
                parse_mode="Markdown",
            )
        else:
            # Create new daily note
            content = DAILY_TEMPLATE.format(
                date=today.strftime("%Y-%m-%d"),
                date_display=today.strftime("%A, %B %d, %Y"),
            )
            filepath.write_text(content, encoding="utf-8")
            daily_folder = self.settings.vault.daily_folder
            await message.answer(
                f"📅 Created daily note: `{daily_folder}/{filename}`",
                parse_mode="Markdown",
            )

    # --- /review ---

    async def handle_review(self, message: Message) -> None:
        """Generate weekly review prompts with context from vault."""
        if not self._is_authorized(message):
            return

        stats = self.graph.stats
        bridges = self.graph.get_bridge_entities(3)
        orphans = self.graph.get_orphan_entities()[:5]

        review = [
            "📋 **Weekly Review**\n",
            f"**Vault:** {self.store.count} chunks indexed",
            f"**Graph:** {stats['nodes']} entities, {stats['edges']} relationships\n",
            "**Reflection Questions:**",
            "1. What was the most important thing I learned this week?",
            "2. What project made the most progress?",
            "3. What's blocking me right now?",
            "4. What should I focus on next week?\n",
        ]

        if bridges:
            review.append("**🌉 Bridge Entities** (connecting different knowledge areas):")
            for b in bridges:
                review.append(f"  • {b['entity']} ({b['type']})")
            review.append("")

        if orphans:
            review.append("**🏝 Orphan Entities** (consider connecting these):")
            for o in orphans:
                review.append(f"  • {o.get('label', o.get('id', 'unknown'))}")
            review.append("")

        await message.answer("\n".join(review), parse_mode="Markdown")

    # --- /stats ---

    async def handle_stats(self, message: Message) -> None:
        """Show vault and graph statistics."""
        if not self._is_authorized(message):
            return

        graph_stats = self.graph.stats
        chunks = self.store.count

        await message.answer(
            "📊 **VaultMind Stats**\n\n"
            f"**Vector Store:** {chunks} chunks indexed\n"
            f"**Knowledge Graph:**\n"
            f"  • Nodes: {graph_stats['nodes']}\n"
            f"  • Edges: {graph_stats['edges']}\n"
            f"  • Density: {graph_stats['density']:.3f}\n"
            f"  • Components: {graph_stats['components']}\n"
            f"  • Orphans: {graph_stats['orphans']}",
            parse_mode="Markdown",
        )

    # --- Message routing ---

    async def handle_message(self, message: Message, text: str) -> None:
        """Route a plain text message based on heuristic classification."""
        if not self._is_authorized(message):
            await message.answer("⛔ Unauthorized")
            return

        # Escape hatch: old behavior (all text → capture)
        if self.settings.routing.capture_all:
            await self.handle_capture(message, text)
            return

        # Sticky thinking sessions — continue if active
        user_id = message.from_user.id if message.from_user else 0
        if self.thinking.has_active_session(user_id):
            await self.handle_think(message, text)
            return

        # Classify and dispatch
        result = self._router.classify(text)

        if result.intent is Intent.capture:
            await self.handle_capture(message, result.content)
        elif result.intent is Intent.greeting:
            await self.handle_greeting(message)
        elif result.intent is Intent.question:
            await self.handle_smart_response(message, result.content, is_question=True)
        else:
            await self.handle_smart_response(message, result.content, is_question=False)

    async def handle_greeting(self, message: Message) -> None:
        """Respond to casual greetings with a static response."""
        await message.answer(random.choice(GREETING_RESPONSES))

    async def handle_smart_response(
        self,
        message: Message,
        text: str,
        *,
        is_question: bool,
    ) -> None:
        """Generate a vault-context-aware response using the LLM."""
        routing_cfg = self.settings.routing

        # Build vault context (reuse ThinkingPartner's method)
        vault_context = ""
        if routing_cfg.vault_context_enabled:
            vault_context = self.thinking._build_vault_context(text, self.store, self.graph)

        # Select model and system prompt
        model = routing_cfg.chat_model or self.settings.llm.fast_model
        system = QUESTION_SYSTEM_PROMPT if is_question else CHAT_SYSTEM_PROMPT

        # Build user message with vault context
        if vault_context and vault_context != "No specific vault context found for this topic.":
            user_content = f"**Context from vault:**\n{vault_context}\n\n**Message:** {text}"
        else:
            user_content = text

        messages = [LLMMessage(role="user", content=user_content)]

        try:
            response = self.llm_client.complete(
                messages=messages,
                model=model,
                max_tokens=routing_cfg.chat_max_tokens,
                system=system,
            )
            for chunk in self._split_message(response.text, max_len=4000):
                await message.answer(chunk, parse_mode="Markdown")
        except LLMError as e:
            logger.error("LLM error in smart response: %s", e)
            await message.answer(f"API error ({e.provider}): {e}")

    # --- /health ---

    async def handle_health(self, message: Message) -> None:
        """Report system health status."""
        if not self._is_authorized(message):
            return

        checks: list[str] = []

        # Vector store
        try:
            chunk_count = self.store.count
            checks.append(f"✅ **Vector Store:** {chunk_count} chunks indexed")
        except Exception as e:
            checks.append(f"❌ **Vector Store:** {e}")

        # Knowledge graph
        try:
            gs = self.graph.stats
            checks.append(f"✅ **Knowledge Graph:** {gs['nodes']} nodes, {gs['edges']} edges")
        except Exception as e:
            checks.append(f"❌ **Knowledge Graph:** {e}")

        # Vault path
        vault_path = self.vault_root
        if vault_path.exists():
            md_count = sum(1 for _ in vault_path.rglob("*.md"))
            checks.append(f"✅ **Vault:** {md_count} markdown files at `{vault_path}`")
        else:
            checks.append(f"❌ **Vault:** path not found: `{vault_path}`")

        # LLM provider
        provider = self.settings.llm.provider
        has_key = bool(self.settings.llm_api_key) or provider == "ollama"
        if has_key:
            checks.append(f"✅ **LLM:** {provider} (key configured)")
        else:
            checks.append(f"❌ **LLM:** {provider} (no API key)")

        # Graph persistence
        graph_path = self.settings.graph.persist_path
        if graph_path.exists():
            size_kb = graph_path.stat().st_size / 1024
            checks.append(f"✅ **Graph file:** {size_kb:.1f} KB")
        else:
            checks.append("⚠️ **Graph file:** not yet created")

        await message.answer(
            "🏥 **VaultMind Health**\n\n" + "\n".join(checks),
            parse_mode="Markdown",
        )

    # --- /notes (natural language date query) ---

    async def handle_notes(self, message: Message, query: str) -> None:
        """Find notes by date — supports natural language and explicit dates."""
        if not self._is_authorized(message):
            return

        start_date, end_date = self._resolve_date_range(query)
        if start_date is None or end_date is None:
            await message.answer(
                "Could not parse a date from that. "
                "Try: `2026-02-20`, `yesterday`, `last week`, `over the weekend`",
                parse_mode="Markdown",
            )
            return

        await message.answer(f"🔍 Searching notes from {start_date} to {end_date}...")

        # Scan vault for notes within the date range
        notes = self._find_notes_by_date(start_date, end_date)

        if not notes:
            await message.answer(f"No notes found between {start_date} and {end_date}.")
            return

        lines = [f"📅 **Notes from {start_date} to {end_date}** ({len(notes)} found)\n"]
        for i, (rel_path, title, created) in enumerate(notes[:20], 1):
            lines.append(f"**{i}.** {title}\n  `{rel_path}` — {created}")

        if len(notes) > 20:
            lines.append(f"\n_...and {len(notes) - 20} more_")

        for chunk in self._split_message("\n".join(lines), max_len=4000):
            await message.answer(chunk, parse_mode="Markdown")

    def _resolve_date_range(self, query: str) -> tuple[str | None, str | None]:
        """Parse a date range from user input — tries formats then LLM."""
        today = datetime.now()

        # Try explicit YYYY-MM-DD
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                d = datetime.strptime(query.strip(), fmt)
                ds = d.strftime("%Y-%m-%d")
                return ds, ds
            except ValueError:
                continue

        # Common keywords without LLM
        lower = query.lower().strip()
        if lower == "today":
            ds = today.strftime("%Y-%m-%d")
            return ds, ds
        if lower == "yesterday":
            ds = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            return ds, ds
        if lower in ("this week", "current week"):
            mon = today - timedelta(days=today.weekday())
            return mon.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        if lower == "last week":
            mon = today - timedelta(days=today.weekday() + 7)
            sun = mon + timedelta(days=6)
            return mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")
        if lower in ("weekend", "over the weekend", "last weekend"):
            # Most recent Saturday
            days_since_sat = (today.weekday() + 2) % 7
            if days_since_sat == 0:
                days_since_sat = 7
            sat = today - timedelta(days=days_since_sat)
            sun = sat + timedelta(days=1)
            return sat.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")

        # Fall back to LLM for complex expressions
        return self._llm_resolve_date(query, today)

    def _llm_resolve_date(self, query: str, today: datetime) -> tuple[str | None, str | None]:
        """Use LLM to parse complex natural language date expressions."""
        model = self.settings.routing.chat_model or self.settings.llm.fast_model
        user_msg = (
            f'Today is {today.strftime("%A, %Y-%m-%d")}. Parse this date expression: "{query}"'
        )
        try:
            response = self.llm_client.complete(
                messages=[LLMMessage(role="user", content=user_msg)],
                model=model,
                max_tokens=100,
                system=DATE_RESOLUTION_SYSTEM_PROMPT,
            )
            data = json.loads(response.text.strip())
            return data.get("start"), data.get("end")
        except (LLMError, json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM date resolution failed: %s", e)
            return None, None

    def _find_notes_by_date(self, start: str, end: str) -> list[tuple[str, str, str]]:
        """Scan vault for notes created within a date range.

        Returns list of (relative_path, title, created_date) tuples.
        """
        results: list[tuple[str, str, str]] = []

        for md_file in self.vault_root.rglob("*.md"):
            rel = md_file.relative_to(self.vault_root)
            if any(part in self.settings.vault.excluded_folders for part in rel.parts):
                continue

            try:
                note = self.parser.parse_file(md_file)
                created = note.created.strftime("%Y-%m-%d")
                if start <= created <= end:
                    results.append(
                        (
                            str(rel),
                            note.title,
                            note.created.strftime("%Y-%m-%d %H:%M"),
                        )
                    )
            except Exception:
                continue

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    # --- /read ---

    async def handle_read(self, message: Message, query: str) -> None:
        """Read a note by path or search query."""
        if not self._is_authorized(message):
            return

        filepath = self._resolve_note_path(query)
        if filepath is None:
            await message.answer(
                f"Note not found: `{query}`\n"
                "Provide a path relative to vault root, or a search term.",
                parse_mode="Markdown",
            )
            return

        content = filepath.read_text(encoding="utf-8")
        rel_path = filepath.relative_to(self.vault_root)
        header = f"📖 **{rel_path}**\n\n"

        for chunk in self._split_message(header + content, max_len=4000):
            await message.answer(chunk, parse_mode="Markdown")

    # --- /delete ---

    async def handle_delete(self, message: Message, query: str) -> None:
        """Request deletion of a note — sends confirmation prompt."""
        if not self._is_authorized(message):
            return

        filepath = self._resolve_note_path(query)
        if filepath is None:
            await message.answer(
                f"Note not found: `{query}`",
                parse_mode="Markdown",
            )
            return

        rel_path = filepath.relative_to(self.vault_root)

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
                        text="🗑 Confirm Delete",
                        callback_data=f"delete_confirm:{rel_path}",
                    ),
                    InlineKeyboardButton(
                        text="❌ Cancel",
                        callback_data="delete_cancel",
                    ),
                ]
            ]
        )

        await message.answer(
            f"⚠️ **Delete note?**\n\n`{rel_path}`\n\n_{preview}_",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    async def handle_delete_callback(self, callback: CallbackQuery) -> None:
        """Process delete confirmation/cancellation."""
        data = callback.data or ""

        if data == "delete_cancel":
            await callback.message.edit_text("❌ Deletion cancelled.")  # type: ignore[union-attr]
            await callback.answer()
            return

        if data.startswith("delete_confirm:"):
            rel_path = data[len("delete_confirm:") :]
            try:
                filepath = validate_vault_path(rel_path, self.vault_root)
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
            self.store.delete_note(rel_path)

            # Delete file
            filepath.unlink()
            logger.info("Deleted note: %s", rel_path)

            await callback.message.edit_text(  # type: ignore[union-attr]
                f"🗑 Deleted: `{rel_path}`",
                parse_mode="Markdown",
            )
            await callback.answer()

    # --- /edit ---

    async def handle_edit(self, message: Message, args: str) -> None:
        """Edit a note via LLM — sends confirmation before applying."""
        if not self._is_authorized(message):
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
        filepath = self._resolve_note_path(note_query)

        if filepath is None:
            await message.answer(
                f"Note not found: `{note_query}`",
                parse_mode="Markdown",
            )
            return

        rel_path = filepath.relative_to(self.vault_root)
        original = filepath.read_text(encoding="utf-8")

        await message.answer("✏️ Generating edit...")

        model = self.settings.routing.chat_model or self.settings.llm.fast_model
        user_msg = f"**Note content:**\n```\n{original}\n```\n\n**Edit instruction:** {instruction}"

        try:
            response = self.llm_client.complete(
                messages=[LLMMessage(role="user", content=user_msg)],
                model=model,
                max_tokens=self.settings.llm.max_tokens,
                system=NOTE_EDIT_SYSTEM_PROMPT,
            )
        except LLMError as e:
            await message.answer(f"Edit failed — API error ({e.provider}): {e}")
            return

        edited = response.text.strip()
        # Strip markdown code fences if the LLM wrapped the response
        if edited.startswith("```") and edited.endswith("```"):
            edited = edited.split("\n", 1)[1].rsplit("\n", 1)[0]

        # Store pending edit for confirmation
        user_id = message.from_user.id if message.from_user else 0
        self._pending_edits[user_id] = {
            "path": str(rel_path),
            "original": original,
            "edited": edited,
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
                        text="✅ Apply Edit",
                        callback_data=f"edit_confirm:{user_id}",
                    ),
                    InlineKeyboardButton(
                        text="❌ Discard",
                        callback_data=f"edit_cancel:{user_id}",
                    ),
                ]
            ]
        )

        await message.answer(
            f"✏️ **Proposed edit for** `{rel_path}`:\n\n```\n{preview}\n```\n\nApply this edit?",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    async def handle_edit_callback(self, callback: CallbackQuery) -> None:
        """Process edit confirmation/cancellation."""
        data = callback.data or ""

        if data.startswith("edit_cancel:"):
            user_id = int(data.split(":")[1])
            self._pending_edits.pop(user_id, None)
            await callback.message.edit_text("❌ Edit discarded.")  # type: ignore[union-attr]
            await callback.answer()
            return

        if data.startswith("edit_confirm:"):
            user_id = int(data.split(":")[1])
            pending = self._pending_edits.pop(user_id, None)

            if pending is None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    "Edit expired. Run `/edit` again.",
                    parse_mode="Markdown",
                )
                await callback.answer()
                return

            try:
                filepath = validate_vault_path(pending["path"], self.vault_root)
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

            # Write edited content
            filepath.write_text(pending["edited"], encoding="utf-8")

            # Re-index
            try:
                note = self.parser.parse_file(filepath)
                self.store.index_single_note(note, self.parser)
            except Exception:
                logger.exception("Failed to re-index edited note")

            logger.info("Edited note: %s", pending["path"])
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"✅ Edit applied to `{pending['path']}`",
                parse_mode="Markdown",
            )
            await callback.answer()

    # --- Note resolution utility ---

    def _resolve_note_path(self, query: str) -> Path | None:
        """Resolve a note query to an absolute filepath.

        Tries in order:
        1. Exact relative path match
        2. Relative path with .md extension appended
        3. Filename search across vault
        4. Semantic search (first result)
        """
        # Exact path
        candidate = validate_vault_path(query, self.vault_root)
        if candidate.is_file():
            return candidate

        # With .md
        if not query.endswith(".md"):
            candidate = validate_vault_path(query + ".md", self.vault_root)
            if candidate.is_file():
                return candidate

        # Filename search
        query_lower = query.lower().strip()
        for md_file in self.vault_root.rglob("*.md"):
            if md_file.stem.lower() == query_lower:
                return md_file
            if query_lower in md_file.stem.lower():
                return md_file

        # Semantic search fallback
        results = self.store.search(query, n_results=1)
        if results:
            note_path: str = results[0]["metadata"].get("note_path", "")
            if note_path:
                candidate = validate_vault_path(note_path, self.vault_root)
                if candidate.is_file():
                    return candidate

        return None

    # --- Voice handling ---

    async def handle_voice(self, message: Message) -> None:
        """Handle voice messages — transcribe and capture."""
        if not self._is_authorized(message):
            return
        # Voice transcription requires whisper integration
        # For now, provide a helpful message
        await message.answer(
            "🎤 Voice transcription is not yet configured.\n"
            "Enable it by installing the `whisper` extra: `uv add vaultmind[whisper]`",
            parse_mode="Markdown",
        )

    # --- Utilities ---

    @staticmethod
    def _slugify(text: str) -> str:
        """Create a filesystem-safe slug from text."""
        import re

        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug[:60]

    @staticmethod
    def _split_message(text: str, max_len: int = 4000) -> list[str]:
        """Split a long message into Telegram-compatible chunks."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # Find a good split point
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()

        return chunks
