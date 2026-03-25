"""Self-healing tool execution with LLM-based argument correction."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Errors that should never be retried (security/auth/config)
_NON_RETRYABLE_TYPES = frozenset(
    {
        "ProfileError",
        "PathTraversalError",
        "TypeError",
        "KeyError",
        "AttributeError",
    }
)


class ToolRetryExhaustedError(Exception):
    """Raised when tool retry with correction both fail."""

    def __init__(
        self,
        message: str,
        original_error: Exception,
        retry_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.retry_error = retry_error


class ToolRetryExecutor:
    """Executes MCP tools with optional LLM-based auto-correction on failure."""

    def __init__(
        self,
        config: Any,  # MCPRetryConfig
        llm_client: Any | None = None,  # LLMClient
        llm_model: str = "",
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._model = llm_model

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        dispatch_fn: Callable[..., dict[str, Any]],
        dispatch_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a tool with optional retry and LLM correction.

        Args:
            name: Tool name (e.g., "vault_search")
            args: Tool arguments dict
            dispatch_fn: The _dispatch_tool function
            dispatch_kwargs: Extra kwargs to pass to dispatch_fn (vault_path, store, etc.)

        Returns:
            Tool result dict.

        Raises:
            ToolRetryExhaustedError: Both attempts failed.
            Other exceptions: Non-retryable errors pass through immediately.
        """
        if not self._config.enabled:
            return dispatch_fn(name, args, **(dispatch_kwargs or {}))

        try:
            return dispatch_fn(name, args, **(dispatch_kwargs or {}))
        except Exception as first_error:
            if not self._should_retry(first_error):
                raise

            logger.warning(
                "Tool '%s' failed (attempt 1): %s. Attempting correction...",
                name,
                first_error,
            )

            corrected_args = self._correct_args(name, args, first_error)
            if corrected_args is None:
                raise

            try:
                result = dispatch_fn(name, corrected_args, **(dispatch_kwargs or {}))
                logger.info("Tool '%s' succeeded on retry with corrected args", name)
                return result
            except Exception as retry_error:
                raise ToolRetryExhaustedError(
                    f"Tool '{name}' failed on both attempts",
                    original_error=first_error,
                    retry_error=retry_error,
                ) from retry_error

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        error_type = type(error).__name__

        # Never retry security/auth/config errors
        if error_type in _NON_RETRYABLE_TYPES:
            return False

        # Check against configured retryable list
        return error_type in self._config.retryable_errors

    def _correct_args(
        self,
        tool_name: str,
        original_args: dict[str, Any],
        error: Exception,
    ) -> dict[str, Any] | None:
        """Use LLM to suggest corrected arguments. Returns None if correction fails."""
        if not self._config.use_llm_correction or self._llm is None:
            return None

        prompt = (
            f"An MCP tool execution failed. Suggest corrected arguments.\n\n"
            f"Tool: {tool_name}\n"
            f"Original arguments: {json.dumps(original_args, indent=2)}\n"
            f"Error: {error}\n"
            f"Error type: {type(error).__name__}\n\n"
            f"Common fixes:\n"
            f"- Query too short/long: adjust to 3-100 chars\n"
            f"- Invalid path: use relative paths like '02-projects/foo.md'\n"
            f"- Missing required fields: provide sensible defaults\n\n"
            f'Return ONLY valid JSON: {{"arguments": {{...corrected...}}}}'
            f' or {{"corrected": false, "reason": "..."}}'
        )

        try:
            from vaultmind.llm.client import Message

            response = self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                model=self._model or "default",
                max_tokens=512,
                system="You fix MCP tool arguments. Return only valid JSON.",
            )

            parsed = json.loads(response.text)
            if isinstance(parsed, dict) and "arguments" in parsed:
                corrected = parsed["arguments"]
                if isinstance(corrected, dict):
                    logger.info("LLM suggested corrected args for '%s'", tool_name)
                    return corrected

            if isinstance(parsed, dict) and parsed.get("corrected") is False:
                logger.info(
                    "LLM declined to correct '%s': %s",
                    tool_name,
                    parsed.get("reason", "unknown"),
                )
                return None

        except Exception:
            logger.debug("LLM correction failed for '%s'", tool_name, exc_info=True)

        return None
