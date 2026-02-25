"""Thinking partner — RAG + knowledge graph augmented ideation.

Supports multi-turn sessions with vault context retrieval.
Provider-agnostic via the LLMClient abstraction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from vaultmind.llm.client import LLMError, Message

if TYPE_CHECKING:
    from vaultmind.bot.session_store import SessionStore
    from vaultmind.config import LLMConfig, TelegramConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient

logger = logging.getLogger(__name__)

THINKING_SYSTEM_PROMPT = """\
You are a thinking partner with access to the user's personal knowledge base (an Obsidian vault).
You have context from their notes and knowledge graph provided below.

Your role:
1. Challenge assumptions with evidence from their own notes
2. Surface connections they might have missed
3. Play devil's advocate when appropriate
4. Synthesize across different knowledge domains in their vault
5. End with concrete next actions or open questions

When referencing their notes, use [[Note Title]] format.
Be direct, substantive, and avoid generic advice. You know this person's work — act like it.
"""

THINKING_MODES = {
    "explore": (
        "Divergent ideation: generate multiple angles, unexpected connections, and novel framings."
    ),
    "critique": "Stress-test the idea: find weaknesses, blind spots, and failure modes.",
    "synthesize": (
        "Connect dots across domains: find patterns and unifying themes in the user's knowledge."
    ),
    "plan": (
        "Break down into actionable steps: create a concrete execution plan with dependencies."
    ),
}


class ThinkingPartner:
    """Multi-turn thinking partner powered by LLM + vault context."""

    def __init__(
        self,
        llm_config: LLMConfig,
        telegram_config: TelegramConfig,
        llm_client: LLMClient,
        session_store: SessionStore | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.session_ttl = telegram_config.thinking_session_ttl
        self._client = llm_client
        self._store = session_store
        # In-memory hot cache over SQLite
        self._sessions: dict[int, _ThinkingSession] = {}

    async def think(
        self,
        user_id: int,
        topic: str,
        store: VaultStore,
        graph: KnowledgeGraph,
    ) -> str:
        """Process a thinking request — retrieves context and generates response."""

        # Detect thinking mode from prefix (e.g., "/think explore: my topic")
        mode = "explore"
        for m in THINKING_MODES:
            if topic.lower().startswith(f"{m}:"):
                mode = m
                topic = topic[len(m) + 1 :].strip()
                break

        # Get or create session
        session = self._get_session(user_id)

        # Retrieve vault context (offload sync I/O to thread pool)
        vault_context = await asyncio.to_thread(self._build_vault_context, topic, store, graph)

        # Build messages
        messages = self._build_messages(session, topic, vault_context, mode)

        try:
            response = await asyncio.to_thread(
                self._client.complete,
                messages=messages,
                model=self.llm_config.thinking_model,
                max_tokens=self.llm_config.max_tokens,
                system=THINKING_SYSTEM_PROMPT,
            )

            reply = response.text

            # Update session history
            session.add_turn(topic, reply)

            if self._store is not None:
                self._store.save(user_id, session.history)

            return reply

        except LLMError as e:
            logger.error("LLM error in thinking mode (%s): %s", e.provider, e)
            return f"API error ({e.provider}): {e}"

    def _build_vault_context(
        self,
        topic: str,
        store: VaultStore,
        graph: KnowledgeGraph,
    ) -> str:
        """Retrieve relevant context from vault and knowledge graph."""
        parts: list[str] = []

        # Semantic search for relevant notes
        results = store.search(topic, n_results=self.llm_config.max_context_notes)
        if results:
            parts.append("## Relevant Notes from Vault\n")
            for hit in results:
                meta = hit["metadata"]
                title = meta.get("note_title", "Untitled")
                path = meta.get("note_path", "")
                content = hit["content"][:500]  # Cap per-chunk context
                parts.append(f"### [[{title}]] (`{path}`)\n{content}\n")

        # Knowledge graph context — try to find entities mentioned in the topic
        words = topic.split()
        graph_context_added = False
        for word in words:
            if len(word) < 3:
                continue
            neighbors = graph.get_neighbors(word, depth=1)
            if neighbors["entity"]:
                if not graph_context_added:
                    parts.append("\n## Knowledge Graph Context\n")
                    graph_context_added = True

                ent = neighbors["entity"]
                parts.append(f"**{ent.get('label', word)}** ({ent.get('type', 'unknown')})")
                if neighbors["outgoing"]:
                    rels = [f"{r['relation']} → {r['target']}" for r in neighbors["outgoing"][:5]]
                    parts.append(f"  Connections: {', '.join(rels)}")
                parts.append("")

        # Also try multi-word entity lookup for the full topic
        neighbors = graph.get_neighbors(topic.strip(), depth=1)
        if neighbors["entity"] and not graph_context_added:
            parts.append("\n## Knowledge Graph Context\n")
            ent = neighbors["entity"]
            parts.append(f"**{ent.get('label', topic)}** ({ent.get('type', 'unknown')})")
            if neighbors["outgoing"]:
                rels = [f"{r['relation']} → {r['target']}" for r in neighbors["outgoing"][:5]]
                parts.append(f"  Connections: {', '.join(rels)}")

        return "\n".join(parts) if parts else "No specific vault context found for this topic."

    def _build_messages(
        self,
        session: _ThinkingSession,
        topic: str,
        vault_context: str,
        mode: str,
    ) -> list[Message]:
        """Build the message history."""
        messages: list[Message] = []

        # Include session history (previous turns)
        for turn in session.history[-6:]:  # Last 6 turns to stay within context
            messages.append(Message(role="user", content=turn["user"]))
            messages.append(Message(role="assistant", content=turn["assistant"]))

        # Current turn with context
        mode_instruction = THINKING_MODES.get(mode, "")
        user_message = (
            f"**Mode:** {mode}\n{mode_instruction}\n\n"
            f"**Context from vault:**\n{vault_context}\n\n"
            f"**Topic:** {topic}"
        )
        messages.append(Message(role="user", content=user_message))

        return messages

    def _get_session(self, user_id: int) -> _ThinkingSession:
        """Get or create a thinking session for a user."""
        now = time.time()

        # Clean expired sessions from cache
        expired = [
            uid for uid, s in self._sessions.items() if now - s.last_active > self.session_ttl
        ]
        for uid in expired:
            del self._sessions[uid]

        # Clean expired sessions from SQLite
        if self._store is not None:
            self._store.cleanup_expired(self.session_ttl)

        # Check in-memory cache
        if user_id in self._sessions:
            session = self._sessions[user_id]
            session.last_active = now
            return session

        # Check SQLite
        if self._store is not None:
            history = self._store.load(user_id)
            if history is not None:
                session = _ThinkingSession()
                session.history = history
                session.last_active = now
                self._sessions[user_id] = session
                return session

        # Create new session
        session = _ThinkingSession()
        self._sessions[user_id] = session
        return session

    def clear_session(self, user_id: int) -> None:
        """Clear a user's thinking session."""
        self._sessions.pop(user_id, None)
        if self._store is not None:
            self._store.delete(user_id)

    def has_active_session(self, user_id: int) -> bool:
        """Check if a user has an active (non-expired) thinking session."""
        now = time.time()
        if user_id in self._sessions:
            session = self._sessions[user_id]
            if now - session.last_active <= self.session_ttl:
                return True
            del self._sessions[user_id]
        if self._store is not None:
            return self._store.has_session(user_id)
        return False


class _ThinkingSession:
    """Tracks a multi-turn thinking conversation."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.last_active: float = time.time()

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"user": user_msg, "assistant": assistant_msg})
