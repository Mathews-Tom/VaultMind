"""Anthropic Claude provider."""

from __future__ import annotations

import anthropic

from vaultmind.llm.client import LLMError, LLMResponse, Message


class AnthropicClient:
    """LLM client for Anthropic Claude models."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Anthropic uses a dedicated `system` parameter."""
        api_messages: list[anthropic.types.MessageParam] = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)  # type: ignore[arg-type,call-overload,unused-ignore]
        except anthropic.APIError as e:
            raise LLMError(str(e), provider="anthropic", original=e) from e

        block = response.content[0]
        if not isinstance(block, anthropic.types.TextBlock):
            raise LLMError(
                f"Unexpected response block: {type(block).__name__}",
                provider="anthropic",
            )

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(text=block.text, model=response.model, usage=usage)
