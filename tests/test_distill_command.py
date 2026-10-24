"""Tests for the manual /distill command handler."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vaultmind.bot.handlers.distill import handle_distill
from vaultmind.config import DistillConfig, Settings, VaultConfig
from vaultmind.llm.client import LLMResponse
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.distill import DISTILL_SYSTEM_PROMPT
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------


_DISTILL_JSON = {
    "question": "What storage should we use?",
    "summary": "Discussed storage options and tradeoffs.",
    "resolution": "Use ChromaDB now, migrate to pgvector past 1M chunks.",
    "systems": ["ChromaDB"],
    "participants": ["user"],
}

_EXTRACT_JSON = [
    {
        "decision": "Use ChromaDB",
        "context": "vector storage choice",
        "outcome": "Adopted",
        "outcome_status": "success",
        "lessons": [],
        "entities": ["ChromaDB"],
    }
]


def _distill_side_effect(**kwargs: object) -> LLMResponse:
    system = kwargs.get("system", "")
    if system == DISTILL_SYSTEM_PROMPT:
        return LLMResponse(text=json.dumps(_DISTILL_JSON), model="test-model", usage={})
    # extractor's system prompt (private) — detected by content, not identity
    if "decision-outcome pairs" in str(system):
        return LLMResponse(text=json.dumps(_EXTRACT_JSON), model="test-model", usage={})
    return LLMResponse(text="unused", model="test-model", usage={})


def _make_llm_client() -> MagicMock:
    client = MagicMock()
    client.complete.side_effect = _distill_side_effect
    return client


def _make_from_user(user_id: int = 42) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    return user


def _make_message(user_id: int = 42) -> AsyncMock:
    msg = AsyncMock()
    msg.from_user = _make_from_user(user_id)
    msg.answer = AsyncMock()
    msg.chat.id = 70
    msg.message_id = 8
    return msg


def _make_note_stub(path: Path, vault_root: Path) -> Note:
    rel = path.relative_to(vault_root)
    return Note(path=rel, title="qa-artifact", content="x" * 150)


def _make_thinking(*, has_session: bool, history: list[dict[str, str]] | None = None) -> MagicMock:
    thinking = MagicMock()
    selected_at = time.time()

    def snapshot(user_id: int) -> dict[str, Any] | None:
        if not has_session:
            return None
        return dict(
            version=1,
            kind="selected-session",
            user_id=user_id,
            session_name="default",
            created_at=selected_at,
            last_active=selected_at,
            history=deepcopy(history or []),
            summaries=[],
            participants=[dict(role="user", user_id=user_id), dict(role="assistant")],
        )

    thinking.snapshot_for_distillation.side_effect = snapshot
    return thinking


def _make_ctx(
    tmp_path: Path,
    thinking: MagicMock,
    store: MagicMock,
    episode_store: object | None = None,
) -> MagicMock:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()

    settings = Settings(vault=VaultConfig(path=vault_root))
    settings.telegram.allowed_user_ids = []
    settings.llm.thinking_model = "test-model"

    parser = MagicMock()
    parser.parse_file.side_effect = lambda fp: _make_note_stub(fp, vault_root)

    ctx = MagicMock()
    ctx.settings = settings
    ctx.vault_root = vault_root
    ctx.thinking = thinking
    ctx.store = store
    ctx.parser = parser
    ctx.llm_client = _make_llm_client()
    ctx.episode_store = episode_store
    ctx.gap_store = None
    return ctx


def _default_config() -> DistillConfig:
    return DistillConfig(model="", output_folder="qa-artifacts", max_tokens=800)


def _reply_text(message: AsyncMock, call_index: int = -1) -> str:
    call_args = message.answer.call_args_list[call_index]
    return call_args[0][0] if call_args[0] else call_args.kwargs.get("text", "")


# ---------------------------------------------------------------------------
# Integration tests: handle_distill
# ---------------------------------------------------------------------------


class TestDistillFromThinkingSession:
    @pytest.mark.asyncio
    async def test_creates_qa_artifact_note(self, tmp_path: Path) -> None:
        history = [
            {"user": "Which vector store?", "assistant": "ChromaDB for now."},
            {"user": "Why?", "assistant": "Zero infra."},
        ]
        thinking = _make_thinking(has_session=True, history=history)
        store = MagicMock()
        ctx = _make_ctx(tmp_path, thinking, store)
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        qa_dir = ctx.vault_root / "qa-artifacts"
        files = list(qa_dir.glob("*.md"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_note_frontmatter_is_qa_artifact(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        qa_dir = ctx.vault_root / "qa-artifacts"
        content = list(qa_dir.glob("*.md"))[0].read_text()
        assert "type: qa-artifact" in content
        assert "authority: 4" in content

    @pytest.mark.asyncio
    async def test_indexes_note_immediately(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        store = MagicMock()
        ctx = _make_ctx(tmp_path, thinking, store)
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        store.index_single_note.assert_called_once()

    @pytest.mark.asyncio
    async def test_reply_contains_output_path(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        reply = _reply_text(message)
        assert "qa-artifacts" in reply
        assert "Created qa-artifact" in reply

    @pytest.mark.asyncio
    async def test_empty_session_history_declines(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=True, history=[])
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        assert "Nothing to distill" in _reply_text(message)


class TestDistillFromLastExchange:
    @pytest.mark.asyncio
    async def test_creates_note_from_exchange(self, tmp_path: Path) -> None:
        from vaultmind.bot.handlers.bookmark import LastExchange

        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=99)
        exchanges = {
            99: LastExchange(
                query="Which vector store?", response="ChromaDB.", timestamp=time.monotonic()
            )
        }

        await handle_distill(ctx, message, _default_config(), exchanges)

        qa_dir = ctx.vault_root / "qa-artifacts"
        assert len(list(qa_dir.glob("*.md"))) == 1


class TestDistillNoContent:
    @pytest.mark.asyncio
    async def test_no_session_no_exchange_declines(self, tmp_path: Path) -> None:
        thinking = _make_thinking(has_session=False)
        ctx = _make_ctx(tmp_path, thinking, MagicMock())
        message = _make_message(user_id=1)

        await handle_distill(ctx, message, _default_config(), {})

        assert "Nothing to distill" in _reply_text(message)
        assert not (ctx.vault_root / "qa-artifacts").exists()


class TestDistillEpisodicWiring:
    @pytest.mark.asyncio
    async def test_extracts_episodes_when_configured(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        episode_store = EpisodeStore(tmp_path / "episodes.db")
        ctx = _make_ctx(tmp_path, thinking, MagicMock(), episode_store=episode_store)
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        resolved = episode_store.query_resolved()
        assert len(resolved) == 1
        assert resolved[0].decision == "Use ChromaDB"
        assert "Extracted 1 episode(s)" in _reply_text(message)
        episode_store.close()

    @pytest.mark.asyncio
    async def test_no_episode_store_skips_extraction(self, tmp_path: Path) -> None:
        history = [{"user": "q", "assistant": "a"}]
        thinking = _make_thinking(has_session=True, history=history)
        ctx = _make_ctx(tmp_path, thinking, MagicMock(), episode_store=None)
        message = _make_message()

        await handle_distill(ctx, message, _default_config(), {})

        assert "Extracted" not in _reply_text(message)


@pytest.mark.asyncio
async def test_manual_snapshot_precedes_first_await_and_retains_selection(tmp_path: Path) -> None:
    history = [dict(user="selected question", assistant="selected answer")]
    ctx = _make_ctx(tmp_path, _make_thinking(has_session=True, history=history), MagicMock())
    message = _make_message()

    async def answer(*args: object, **kwargs: object) -> None:
        history.append(dict(user="late question", assistant="late answer"))

    message.answer.side_effect = answer
    await handle_distill(ctx, message, _default_config(), {})
    model_input = ctx.llm_client.complete.call_args_list[0].kwargs["messages"][0].content
    assert "selected question" in model_input and "late question" not in model_input
    snapshots = [
        json.loads(path.read_text())
        for path in (ctx.vault_root / "_wiki/sources").glob("*.md")
        if '"kind":"selected-session"' in path.read_text()
    ]
    assert len(snapshots) == 1
    assert len(snapshots[0]["history"]) == 1
    assert snapshots[0]["selection"]["event_id"] == "70:8:42"


@pytest.mark.asyncio
async def test_duplicate_manual_delivery_does_not_repeat_producer(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        _make_thinking(has_session=True, history=[dict(user="q", assistant="a")]),
        MagicMock(),
    )
    message = _make_message()
    await handle_distill(ctx, message, _default_config(), {})
    await handle_distill(ctx, message, _default_config(), {})
    assert ctx.llm_client.complete.call_count == 1
    assert ctx.store.index_single_note.call_count == 1
    assert "preserved and unapproved" in _reply_text(message)


@pytest.mark.asyncio
async def test_manual_retention_refusal_never_calls_model(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        _make_thinking(
            has_session=True, history=[dict(user="---\ndo_not_retain: true\n---\nsecret")]
        ),
        MagicMock(),
    )
    message = _make_message()
    await handle_distill(ctx, message, _default_config(), {})
    ctx.llm_client.complete.assert_not_called()
    ctx.store.index_single_note.assert_not_called()
    assert not (ctx.vault_root / "_wiki").exists()


@pytest.mark.asyncio
async def test_index_failure_reports_preserved_unapproved_projection(tmp_path: Path) -> None:
    store = MagicMock()
    store.index_single_note.side_effect = RuntimeError("index unavailable")
    ctx = _make_ctx(
        tmp_path,
        _make_thinking(has_session=True, history=[dict(user="q", assistant="a")]),
        store,
    )
    message = _make_message()
    await handle_distill(ctx, message, _default_config(), {})
    assert "index: failed" in _reply_text(message)
    assert "unapproved" in _reply_text(message)
    assert len(list((ctx.vault_root / "qa-artifacts").glob("*.md"))) == 1
