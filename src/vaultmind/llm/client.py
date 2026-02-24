"""Unified LLM client interface and factory.

All providers implement the same Protocol: send messages, get text back.
Provider-specific details (system prompt handling, response parsing,
error types) are encapsulated in each implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

type Provider = Literal["anthropic", "openai", "gemini", "ollama"]

type MessageRole = Literal["user", "assistant", "system"]


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    model: str
    usage: dict[str, int]


class LLMError(Exception):
    """Unified error for all LLM providers."""

    def __init__(self, message: str, provider: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.original = original


class LLMClient(Protocol):
    """Protocol for LLM providers.

    Implementations must handle:
    - System prompt injection (separate param vs. prepended message)
    - Response parsing to extract text
    - Error mapping to LLMError
    """

    @property
    def provider_name(self) -> str: ...

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages and get a text response.

        Args:
            messages: Conversation history (user/assistant turns).
            model: Provider-specific model identifier.
            max_tokens: Maximum tokens in the response.
            system: Optional system prompt.

        Returns:
            LLMResponse with the generated text.

        Raises:
            LLMError: On any provider error.
        """
        ...


def create_llm_client(provider: Provider, api_key: str, base_url: str | None = None) -> LLMClient:
    """Factory: create an LLM client for the given provider.

    Args:
        provider: One of "anthropic", "openai", "gemini", "ollama".
        api_key: API key (empty string for Ollama).
        base_url: Override base URL (required for Ollama, optional for others).

    Returns:
        An LLMClient implementation.
    """
    if provider == "anthropic":
        from vaultmind.llm.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key=api_key)

    if provider == "openai":
        from vaultmind.llm.providers.openai import OpenAIClient

        return OpenAIClient(api_key=api_key, base_url=base_url)

    if provider == "gemini":
        from vaultmind.llm.providers.gemini import GeminiClient

        return GeminiClient(api_key=api_key)

    if provider == "ollama":
        from vaultmind.llm.providers.ollama import OllamaClient

        url = base_url or "http://localhost:11434"
        return OllamaClient(base_url=url)

    raise ValueError(f"Unknown LLM provider: {provider}")
