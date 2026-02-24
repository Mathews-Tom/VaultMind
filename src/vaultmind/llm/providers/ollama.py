"""Ollama provider for local models."""

from __future__ import annotations

from openai import OpenAI, OpenAIError

from vaultmind.llm.client import LLMError, LLMResponse, Message


class OllamaClient:
    """LLM client for Ollama local models.

    Uses Ollama's OpenAI-compatible API endpoint.
    No API key required.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._client = OpenAI(
            api_key="ollama",
            base_url=f"{base_url.rstrip('/')}/v1",
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Ollama supports the OpenAI chat completions format."""
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        )

        try:
            response = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=api_messages,  # type: ignore[arg-type]
            )
        except OpenAIError as e:
            raise LLMError(str(e), provider="ollama", original=e) from e

        choice = response.choices[0]
        if choice.message.content is None:
            raise LLMError("Empty response from Ollama", provider="ollama")

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens or 0,
            }
        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            usage=usage,
        )
