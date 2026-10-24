"""Session intent for approved-current queries; independent idle distillation."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vaultmind.knowledge.intake import SourceIntake
from vaultmind.knowledge.query import MAX_INTENT_QUESTIONS, query_knowledge
from vaultmind.knowledge.render import render_knowledge_result
from vaultmind.pipeline.distill import (
    distill_retained_conversation,
    extract_and_store_episodes,
    mint_gap_for_unresolved,
    preserve_conversation_snapshot,
)
from vaultmind.vault.security import validate_ordinary_write_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaultmind.bot.session_store import SessionStore
    from vaultmind.config import Settings
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.memory.gaps import GapStore
    from vaultmind.vault.models import Note
    from vaultmind.vault.parser import VaultParser

THINKING_MODES = ("explore", "critique", "synthesize", "plan")


class ThinkingPartner:
    """Keep conversational intent, never conversational factual authority.

    Only complete recent user questions reach the shared query core. Assistant
    text and historical summaries remain available to explicit producer tools.
    Querying never finalizes sessions, summarizes, extracts or distills content.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm_factory: Callable[[], LLMClient],
        session_store: SessionStore | None = None,
    ) -> None:
        self._settings = settings
        self._llm_factory = llm_factory
        self._store = session_store
        self.session_ttl = settings.telegram.thinking_session_ttl
        self._sessions: dict[int, _ThinkingSession] = {}
        self._inflight: dict[int, int] = {}
        self._histories: list[tuple[int, _ThinkingSession]] = []

    async def think(
        self,
        user_id: int,
        topic: str,
        *,
        use_session: bool = True,
        evidence_only: bool = False,
    ) -> dict[str, Any]:
        """Return the core outcome after fresh approved-current selection.

        A mode prefix is a UI convention only. At most three complete previous
        user questions supply intent; the core validates every selected scalar
        without truncation and owns all completion/authority decisions.
        """
        if isinstance(topic, str):
            for mode in THINKING_MODES:
                if topic.lower().startswith(f"{mode}:"):
                    topic = topic[len(mode) + 1 :]
                    topic = topic[1:] if topic[:1].isspace() else topic
                    break
        session = self._get_session(user_id) if use_session else None
        if session is not None:
            self._inflight[user_id] = self._inflight.get(user_id, 0) + 1
        try:
            intent = (
                tuple(turn["user"] for turn in session.history[-MAX_INTENT_QUESTIONS:])
                if session is not None
                else ()
            )
            result = await asyncio.to_thread(
                query_knowledge,
                self._settings,
                topic,
                evidence_only=evidence_only,
                llm_factory=self._llm_factory,
                intent_questions=intent,
            )
            if session is not None and self._sessions.get(user_id) is session:
                session.add_turn(topic, render_knowledge_result(result))
                if self._store is not None:
                    self._store.save(user_id, session.history, session.session_name)
            return result
        finally:
            if session is not None:
                remaining = self._inflight[user_id] - 1
                if remaining:
                    self._inflight[user_id] = remaining
                else:
                    del self._inflight[user_id]

    def _get_session(self, user_id: int) -> _ThinkingSession:
        """Select active history without deleting an expired history's only copy."""
        cached = self._sessions.get(user_id)
        if cached is not None and (
            user_id in self._inflight or time.time() - cached.last_active <= self.session_ttl
        ):
            return cached
        stored = (
            self._store.load_active(user_id, self.session_ttl) if self._store is not None else None
        )
        session = _ThinkingSession()
        if stored is not None:
            session.history = stored["history"]
            session.session_name = stored["session_name"]
            session.last_active = stored["last_active"]
        elif self._store is not None and self._store.has_session(user_id):
            # Leave expired default history untouched for the non-query finalizer.
            session.session_name = f"thinking:{uuid4().hex}"
        if self._store is None:
            self._histories.append((user_id, session))
        self._sessions[user_id] = session
        return session

    def has_active_session(self, user_id: int) -> bool:
        """Check activity without reaping histories or invoking a producer."""
        session = self._sessions.get(user_id)
        if session is not None:
            return (
                user_id in self._inflight or time.time() - session.last_active <= self.session_ttl
            )
        return self._store is not None and self._store.has_active_session(user_id, self.session_ttl)

    def snapshot_for_distillation(self, user_id: int) -> dict[str, Any] | None:
        """Copy the selected available revision without running any producer."""
        if not self.has_active_session(user_id):
            return None
        session = self._get_session(user_id)
        stored = (
            self._store.snapshot(user_id, session.session_name) if self._store is not None else None
        )
        return dict(
            version=1,
            kind="selected-session",
            user_id=user_id,
            session_name=session.session_name,
            created_at=stored["created_at"] if stored is not None else session.created_at,
            last_active=session.last_active,
            history=deepcopy(session.history),
            summaries=deepcopy(stored["summaries"]) if stored is not None else [],
            summary_role="historical-generated; retained, not sent to distillation",
            participants=[dict(role="user", user_id=user_id), dict(role="assistant")],
        )

    def clear_session(self, user_id: int) -> None:
        """Explicitly clear the user's session, including retained idle records."""
        self._sessions.pop(user_id, None)
        self._histories = [(uid, s) for uid, s in self._histories if uid != user_id]
        if self._store is not None:
            self._store.delete_conversations(user_id)

    async def finalize_idle_sessions(
        self,
        *,
        store: VaultStore,
        parser: VaultParser,
        episode_store: object | None = None,
        gap_store: GapStore | None = None,
    ) -> dict[str, int]:
        """Run the enabled idle producer from the bot scheduler, not a query.

        Claim before any await/provider call. Failed or interrupted claims retain
        their original history and summaries and are not retried on every poll.
        Distillation/indexing is not atomic: a failed record may have partial
        producer effects and must not be treated as successfully finalized.
        """
        counts = {"completed": 0, "failed": 0, "skipped": 0}
        config = self._settings.distill
        if not config.enabled:
            return counts
        now = time.time()
        for uid, session in tuple(self._sessions.items()):
            if uid not in self._inflight and now - session.last_active > self.session_ttl:
                del self._sessions[uid]
        records: list[dict[str, Any]]
        if self._store is not None:
            records = self._store.claim_expired(
                self.session_ttl, excluded_user_ids=frozenset(self._inflight)
            )
        else:
            records = []
            for uid, session in self._histories:
                if (
                    session.finalization_state == "active"
                    and uid not in self._inflight
                    and now - session.last_active > self.session_ttl
                ):
                    session.finalization_state = "claimed"
                    records.append(
                        dict(
                            user_id=uid,
                            history=deepcopy(session.history),
                            last_active=session.last_active,
                            created_at=session.created_at,
                            session_name=session.session_name,
                            summaries=[],
                            generation_id=session.generation_id,
                            session=session,
                        )
                    )
        for record in records:
            state = "skipped"
            error: str | None = None
            if len(record["history"]) >= config.min_turns:
                try:
                    await asyncio.to_thread(
                        self._finalize_record, record, store, parser, episode_store, gap_store
                    )
                    state = "completed"
                except Exception as exc:
                    state = "failed"
                    error = f"{type(exc).__name__}: {exc}"
            counts[state] += 1
            if self._store is not None:
                self._store.finish_finalization(record["id"], state, error)
            else:
                record["session"].finalization_state = state
        return counts

    def _finalize_record(
        self,
        record: dict[str, Any],
        store: VaultStore,
        parser: VaultParser,
        episode_store: object | None,
        gap_store: GapStore | None,
    ) -> None:
        config = self._settings.distill
        model = config.model or self._settings.llm.thinking_model
        source_ref = f"telegram-thinking:{record['user_id']}:{record['last_active']}"
        occurred_at = datetime.fromtimestamp(record["last_active"], tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        snapshot = dict(
            version=1,
            kind="selected-idle-session",
            user_id=record["user_id"],
            session_name=record["session_name"],
            archive_id=record.get("id"),
            generation_id=record.get("generation_id"),
            created_at=record["created_at"],
            last_active=record["last_active"],
            history=record["history"],
            summaries=record["summaries"],
            summary_role="historical-generated; retained, not sent to distillation",
            participants=[
                dict(role="user", user_id=record["user_id"]),
                dict(role="assistant"),
            ],
        )
        output_dir = validate_ordinary_write_path(config.output_folder, self._settings.vault.path)
        validate_ordinary_write_path(str(output_dir / "distillation.md"), self._settings.vault.path)
        with SourceIntake(self._settings) as intake:
            captured = preserve_conversation_snapshot(
                intake,
                snapshot,
                origin=source_ref,
                adapter="idle-session",
                event_id=f"{record['user_id']}:{record.get('id', record.get('generation_id'))}",
            )
            client = self._llm_factory()
            result = distill_retained_conversation(
                intake,
                captured,
                client,
                model,
                source_ref=source_ref,
                occurred_at=occurred_at,
                output_folder=config.output_folder,
                max_tokens=config.max_tokens,
            )
            if not result.success or result.derivative is None or result.generation is None:
                raise RuntimeError(result.error)
            derivative = result.derivative
            note: Note | None = None

            def index() -> None:
                nonlocal note
                intake.read(derivative.source_id)
                note = parser.parse_file(self._settings.vault.path / result.output_path)
                store.index_single_note(note, parser)

            indexed = derivative.run_stage("index", index)
            if indexed.status != "succeeded" or note is None:
                raise RuntimeError(indexed.error or "Indexing failed; sources preserved")
            if episode_store is not None:
                extract_and_store_episodes(note, client, model, episode_store)
            mint_gap_for_unresolved(result.generation, gap_store, source_ref)


class _ThinkingSession:
    """Full exchange storage for explicit inspection/bookmark/distillation."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.last_active = time.time()
        self.created_at = self.last_active
        self.generation_id = uuid4().hex
        self.finalization_state = "active"
        self.session_name = "default"

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"user": user_msg, "assistant": assistant_msg})
        self.last_active = time.time()
