"""Tests for the manual /distill command handler."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vaultmind.bot.handlers.distill import handle_distill
from vaultmind.config import DistillConfig
from vaultmind.llm.client import LLMResponse
from vaultmind.memory.store import EpisodeStore
from vaultmind.pipeline.distill import DISTILL_SYSTEM_PROMPT
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeVaultConfig:
    path: Path
    inbox_folder: str = "00-inbox"
    excluded_folders: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.excluded_folders is None:
            self.excluded_folders = []


@dataclass
class _FakeLLMConfig:
    thinking_model: str = "test-model"


@dataclass
class _FakeTelegramConfig:
    allowed_user_ids: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allowed_user_ids is None:
            self.allowed_user_ids = []  # empty = allow all


@dataclass
class _FakeSettings:
    vault: _FakeVaultConfig
    telegram: Any = None
    llm: Any = None

    def __post_init__(self) -> None:
        if self.telegram is None:
            self.telegram = _FakeTelegramConfig()
        if self.llm is None:
            self.llm = _FakeLLMConfig()


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
    return msg


def _make_note_stub(path: Path, vault_root: Path) -> Note:
    rel = path.relative_to(vault_root)
    return Note(path=rel, title="qa-artifact", content="x" * 150)


def _make_thinking(*, has_session: bool, history: list[dict[str, str]] | None = None) -> MagicMock:
    thinking = MagicMock()
    thinking.has_active_session.return_value = has_session
    session = MagicMock()
    session.history = history or []
    session.last_active = time.time()
    thinking._get_session.return_value = session
    return thinking


def _make_ctx(
    tmp_path: Path,
    thinking: MagicMock,
    store: MagicMock,
    episode_store: object | None = None,
) -> MagicMock:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()

    cfg = _FakeVaultConfig(path=vault_root)
    settings = _FakeSettings(vault=cfg)

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
