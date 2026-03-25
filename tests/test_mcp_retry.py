"""Tests for ToolRetryExecutor — self-healing tool execution with LLM correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from vaultmind.llm.client import LLMResponse
from vaultmind.mcp.auth import ProfileError
from vaultmind.mcp.retry import ToolRetryExecutor, ToolRetryExhaustedError
from vaultmind.vault.security import PathTraversalError

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeRetryConfig:
    enabled: bool = True
    max_retries: int = 1
    use_llm_correction: bool = True
    correction_model: str = "test-model"
    timeout_seconds: int = 30
    retryable_errors: list[str] = field(
        default_factory=lambda: [
            "ValueError",
            "LLMError",
            "TimeoutError",
            "ConnectionError",
        ]
    )


def _make_dispatch(
    results: list[dict[str, Any] | Exception],
) -> Callable[..., dict[str, Any]]:
    """Create a dispatch function that returns results in order, raising exceptions."""
    call_count = 0

    def dispatch(name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        idx = min(call_count, len(results) - 1)
        call_count += 1
        r = results[idx]
        if isinstance(r, Exception):
            raise r
        return r

    dispatch.call_count = lambda: call_count  # type: ignore[attr-defined]
    return dispatch


def _make_llm(
    response_text: str = '{"arguments": {"query": "fixed"}}',
) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(text=response_text, model="test", usage={})
    return client


# ---------------------------------------------------------------------------
# A. Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_execute_success_first_attempt_returns_result(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        executor = ToolRetryExecutor(config)
        dispatch = _make_dispatch([{"content": "ok"}])

        # Act
        result = executor.execute("vault_search", {"query": "test"}, dispatch)

        # Assert
        assert result == {"content": "ok"}
        assert dispatch.call_count() == 1

    def test_execute_disabled_passes_through(self) -> None:
        # Arrange
        config = _FakeRetryConfig(enabled=False)
        executor = ToolRetryExecutor(config)
        dispatch = _make_dispatch([{"content": "direct"}])

        # Act
        result = executor.execute("vault_search", {"query": "x"}, dispatch)

        # Assert
        assert result == {"content": "direct"}
        assert dispatch.call_count() == 1

    def test_execute_success_does_not_call_llm(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm)
        dispatch = _make_dispatch([{"content": "ok"}])

        # Act
        executor.execute("vault_search", {"query": "test"}, dispatch)

        # Assert
        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# B. Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_execute_retryable_error_triggers_retry(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("bad query"), {"content": "fixed"}])

        # Act
        result = executor.execute("vault_search", {"query": "x"}, dispatch)

        # Assert
        assert result == {"content": "fixed"}
        assert dispatch.call_count() == 2
        llm.complete.assert_called_once()

    def test_execute_non_retryable_error_raises_immediately(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm)
        dispatch = _make_dispatch([ProfileError("denied")])

        # Act / Assert
        with pytest.raises(ProfileError, match="denied"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        assert dispatch.call_count() == 1
        llm.complete.assert_not_called()

    def test_execute_path_traversal_error_not_retried(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm)
        err = PathTraversalError("../../etc/passwd", Path("/vault"))
        dispatch = _make_dispatch([err])

        # Act / Assert
        with pytest.raises(PathTraversalError):
            executor.execute("vault_search", {"path": "../../etc/passwd"}, dispatch)
        assert dispatch.call_count() == 1
        llm.complete.assert_not_called()

    def test_execute_type_error_not_retried(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm)
        dispatch = _make_dispatch([TypeError("wrong type")])

        # Act / Assert
        with pytest.raises(TypeError, match="wrong type"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        assert dispatch.call_count() == 1
        llm.complete.assert_not_called()

    def test_execute_both_attempts_fail_raises_exhausted(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("first fail"), ValueError("second fail")])

        # Act / Assert
        with pytest.raises(ToolRetryExhaustedError, match="failed on both attempts"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        assert dispatch.call_count() == 2

    def test_execute_exhausted_preserves_both_errors(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        first = ValueError("original")
        second = ValueError("retry")
        dispatch = _make_dispatch([first, second])

        # Act
        with pytest.raises(ToolRetryExhaustedError) as exc_info:
            executor.execute("vault_search", {"query": "x"}, dispatch)

        # Assert
        assert exc_info.value.original_error is first
        assert exc_info.value.retry_error is second


# ---------------------------------------------------------------------------
# C. LLM correction
# ---------------------------------------------------------------------------


class TestLLMCorrection:
    def test_correct_args_valid_json_returns_corrected(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm('{"arguments": {"query": "corrected"}}')
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("bad"), {"content": "ok"}])

        # Act
        result = executor.execute("vault_search", {"query": "x"}, dispatch)

        # Assert
        assert result == {"content": "ok"}
        # Second dispatch call should have used corrected args
        llm.complete.assert_called_once()

    def test_correct_args_invalid_json_returns_none(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm("this is not json")
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("bad")])

        # Act / Assert — original error re-raised since correction returned None
        with pytest.raises(ValueError, match="bad"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        llm.complete.assert_called_once()

    def test_correct_args_declined_returns_none(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = _make_llm('{"corrected": false, "reason": "unfixable"}')
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("bad")])

        # Act / Assert
        with pytest.raises(ValueError, match="bad"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        llm.complete.assert_called_once()

    def test_correct_args_llm_error_returns_none(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM down")
        executor = ToolRetryExecutor(config, llm_client=llm, llm_model="test-model")
        dispatch = _make_dispatch([ValueError("bad")])

        # Act / Assert — original error re-raised
        with pytest.raises(ValueError, match="bad"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        llm.complete.assert_called_once()

    def test_correct_args_disabled_returns_none(self) -> None:
        # Arrange
        config = _FakeRetryConfig(use_llm_correction=False)
        llm = _make_llm()
        executor = ToolRetryExecutor(config, llm_client=llm)
        dispatch = _make_dispatch([ValueError("bad")])

        # Act / Assert
        with pytest.raises(ValueError, match="bad"):
            executor.execute("vault_search", {"query": "x"}, dispatch)
        llm.complete.assert_not_called()

    def test_correct_args_no_llm_client_returns_none(self) -> None:
        # Arrange
        config = _FakeRetryConfig()
        executor = ToolRetryExecutor(config, llm_client=None)
        dispatch = _make_dispatch([ValueError("bad")])

        # Act / Assert
        with pytest.raises(ValueError, match="bad"):
            executor.execute("vault_search", {"query": "x"}, dispatch)


# ---------------------------------------------------------------------------
# D. Should retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def _executor(self, config: _FakeRetryConfig | None = None) -> ToolRetryExecutor:
        return ToolRetryExecutor(config or _FakeRetryConfig())

    def test_should_retry_value_error_true(self) -> None:
        executor = self._executor()
        assert executor._should_retry(ValueError("x")) is True

    def test_should_retry_timeout_error_true(self) -> None:
        executor = self._executor()
        assert executor._should_retry(TimeoutError("x")) is True

    def test_should_retry_profile_error_false(self) -> None:
        executor = self._executor()
        assert executor._should_retry(ProfileError("x")) is False

    def test_should_retry_unknown_error_false(self) -> None:
        class RandomError(Exception):
            pass

        executor = self._executor()
        assert executor._should_retry(RandomError("x")) is False
