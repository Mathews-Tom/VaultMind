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
    from vaultmind.graph.context import GraphContextBuilder
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
When the Knowledge Graph Context section is present, use those entity relationships
to inform your analysis — they represent explicit connections the user has established.
Be direct, substantive, and avoid generic advice. You know this person's work — act like it.

---

**OPTIONAL: Structured Extraction (for system processing only)**

If your response mentions specific entities (people, projects, tools, concepts, organizations),
you may optionally tag them using XML to help improve the knowledge graph. These tags are
automatically parsed and removed before the response is shown to the user.

Tagging guidelines:
- Only tag entities you are confident about (confidence >= 0.75)
- Entity types: person, project, concept, tool, organization, event, location
- Relationship types: related_to, part_of, depends_on, created_by, \
influences, mentioned_in, competes_with, preceded_by

Tag formats:
- <vm:entity name="..." type="..." confidence="0.X">brief description</vm:entity>
- <vm:relationship from="entity1" to="entity2" type="..." confidence="0.X" />
- <vm:episode decision="..." context="..." \
status="pending|success|failure"><lesson>...</lesson>\
<entity>...</entity></vm:episode>

Do NOT overuse tags. Focus on the most significant entities and relationships.
Incorrect or missing tags have no penalty — the user will never see them.
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
        graph_context_builder: GraphContextBuilder | None = None,
    ) -> None:
        self.llm_config = llm_config
        self._telegram_config = telegram_config
        self.session_ttl = telegram_config.thinking_session_ttl
        self._client = llm_client
        self._store = session_store
        self._graph_ctx = graph_context_builder
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

        # Build graph context via entity extraction (if available)
        graph_context_str = ""
        if self._graph_ctx is not None and self.llm_config.graph_context_enabled:
            session_id = str(user_id)
            try:
                graph_block = await self._graph_ctx.build(
                    query=topic,
                    session_id=session_id,
                    hop_depth=self.llm_config.graph_hop_depth,
                    min_confidence=self.llm_config.graph_min_confidence,
                    max_relationships=self.llm_config.graph_max_relationships,
                )
                if graph_block is not None:
                    graph_context_str = graph_block.render(
                        min_confidence=self.llm_config.graph_min_confidence
                    )
            except Exception:
                logger.warning("Graph context building failed", exc_info=True)

        full_context = vault_context
        if graph_context_str:
            full_context = f"{vault_context}\n\n{graph_context_str}"

        # Build messages (includes summaries of older turns if available)
        messages = self._build_messages(session, topic, full_context, mode, user_id)

        try:
            response = await asyncio.to_thread(
                self._client.complete,
                messages=messages,
                model=self.llm_config.thinking_model,
                max_tokens=self.llm_config.max_tokens,
                system=THINKING_SYSTEM_PROMPT,
            )

            raw_reply = response.text

            # Extract entities/relationships and strip tags from response
            reply = await self._apply_extraction(raw_reply, graph)

            # Update session history (with clean response, tags stripped)
            session.add_turn(topic, reply)

            if self._store is not None:
                self._store.save(user_id, session.history)

                # Trigger async summarization if enabled
                if self._telegram_config.thinking_summarization_enabled:
                    asyncio.ensure_future(self._summarize_if_needed(user_id))

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

    async def _apply_extraction(
        self,
        raw_response: str,
        graph: KnowledgeGraph,
    ) -> str:
        """Parse extraction tags from response, apply graph updates, return clean text."""
        if not self.llm_config.single_pass_extraction_enabled:
            return raw_response

        from vaultmind.bot.extraction_parser import parse_extraction_tags

        result = await asyncio.to_thread(parse_extraction_tags, raw_response)

        threshold = self.llm_config.extraction_confidence_threshold

        # Apply entity updates
        for entity in result.entities:
            if entity.confidence < threshold:
                continue
            try:
                graph.add_entity(
                    name=entity.name,
                    entity_type=entity.type,
                    confidence=entity.confidence,
                )
            except Exception:
                logger.debug("Failed to add extracted entity: %s", entity.name, exc_info=True)

        # Apply relationship updates
        for rel in result.relationships:
            if rel.confidence < threshold:
                continue
            try:
                graph.add_relationship(
                    source=rel.source,
                    target=rel.target,
                    relation=rel.relation_type,
                    confidence=rel.confidence,
                )
            except Exception:
                logger.debug(
                    "Failed to add extracted relationship: %s -> %s",
                    rel.source,
                    rel.target,
                    exc_info=True,
                )

        # Persist graph if anything was extracted
        if result.entities or result.relationships:
            try:
                await asyncio.to_thread(graph.save)
            except Exception:
                logger.warning("Failed to save graph after extraction", exc_info=True)

        if result.entities or result.relationships:
            logger.info(
                "Extracted %d entities, %d relationships from thinking response",
                len(result.entities),
                len(result.relationships),
            )

        return result.clean_response

    def _build_messages(
        self,
        session: _ThinkingSession,
        topic: str,
        vault_context: str,
        mode: str,
        user_id: int = 0,
    ) -> list[Message]:
        """Build the message history with optional summaries of older turns."""
        messages: list[Message] = []

        # Include summaries of older turns (if available)
        if self._store is not None and user_id:
            summaries = self._store.get_summaries(user_id)
            for summary in summaries:
                summary_msg = (
                    f"[Summary of earlier conversation — "
                    f"turns {summary['start_turn']}-{summary['end_turn']}]\n"
                    f"{summary['summary']}\n"
                    f"Key topics: {', '.join(summary['topics'])}"
                )
                if summary["questions"]:
                    summary_msg += f"\nOpen questions: {', '.join(summary['questions'])}"
                messages.append(Message(role="assistant", content=summary_msg))

        # Include recent turns in full
        recent_n = self._telegram_config.thinking_recent_turns_to_keep
        for turn in session.history[-recent_n:]:
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

    async def _summarize_if_needed(self, user_id: int, session_name: str = "default") -> None:
        """Check if summarization is needed and run it asynchronously."""
        if self._store is None:
            return

        turn_count = self._store.count_turns(user_id, session_name)
        threshold = self._telegram_config.thinking_message_count_threshold // 2
        if turn_count < threshold:
            return

        batch = self._store.get_unsummarized_batch(
            user_id,
            session_name,
            self._telegram_config.thinking_recent_turns_to_keep,
            self._telegram_config.thinking_batch_size,
        )
        if batch is None:
            return

        turns, start_idx, end_idx = batch
        try:
            summary_data = await asyncio.to_thread(self._generate_summary, turns)

            existing = self._store.get_summaries(user_id, session_name)
            next_batch = len(existing)

            self._store.save_summary(
                user_id=user_id,
                session_name=session_name,
                batch_number=next_batch,
                start_turn_index=start_idx,
                end_turn_index=end_idx,
                summary_text=str(summary_data["summary"]),
                key_topics=list(summary_data["topics"]),
                open_questions=list(summary_data["questions"]),
            )
            logger.info(
                "Summarized turns %d-%d for user %d session %s",
                start_idx,
                end_idx,
                user_id,
                session_name,
            )
        except Exception:
            logger.warning("Session summarization failed", exc_info=True)

    def _generate_summary(self, turns: list[dict[str, str]]) -> dict[str, list[str] | str]:
        """Generate an LLM summary of a batch of conversation turns."""
        conversation_text = "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in turns
        )

        prompt = (
            "Summarize this thinking partner conversation excerpt. "
            "Extract the essential insights and open threads.\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            "Format your response exactly as:\n"
            "SUMMARY: [2-3 sentence summary]\n"
            "TOPICS: [comma-separated list of main topics]\n"
            'QUESTIONS: [comma-separated list of open questions, or "none"]'
        )

        response = self._client.complete(
            messages=[Message(role="user", content=prompt)],
            model=self.llm_config.fast_model,
            max_tokens=self._telegram_config.thinking_summary_max_tokens,
            system=(
                "You are a conversation summarizer. "
                "Be concise and extract only essential information."
            ),
        )

        return self._parse_summary_response(response.text)

    @staticmethod
    def _parse_summary_response(text: str) -> dict[str, list[str] | str]:
        """Parse structured summary from LLM response."""
        summary = ""
        topics: list[str] = []
        questions: list[str] = []

        for line in text.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("SUMMARY:"):
                summary = line[len("SUMMARY:") :].strip()
            elif line.upper().startswith("TOPICS:"):
                raw = line[len("TOPICS:") :].strip()
                topics = [t.strip() for t in raw.split(",") if t.strip()]
            elif line.upper().startswith("QUESTIONS:"):
                raw = line[len("QUESTIONS:") :].strip()
                if raw.lower() != "none":
                    questions = [q.strip() for q in raw.split(",") if q.strip()]

        return {
            "summary": summary or "No summary generated.",
            "topics": topics,
            "questions": questions,
        }

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
        if self._graph_ctx is not None:
            self._graph_ctx.clear_session(str(user_id))

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
