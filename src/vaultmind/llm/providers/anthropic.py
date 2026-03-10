"""Anthropic Claude provider."""

from __future__ import annotations

from typing import Any

import anthropic

from vaultmind.llm.client import ContentPart, LLMError, LLMResponse, Message, MultimodalMessage


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

    def complete_multimodal(
        self,
        messages: list[Message | MultimodalMessage],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Anthropic vision API: images encoded as base64 source blocks."""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, MultimodalMessage):
                content: list[dict[str, Any]] = []
                for part in msg.parts:
                    content.append(_anthropic_content_part(part))
                api_messages.append({"role": msg.role, "content": content})
            else:
                if msg.role != "system":
                    api_messages.append({"role": msg.role, "content": msg.content})

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


def _anthropic_content_part(part: ContentPart) -> dict[str, Any]:
    """Convert a ContentPart to the Anthropic API content block format."""
    if part.type == "text":
        return {"type": "text", "text": part.text}
    # image_url must be a base64 data URI: data:<media_type>;base64,<data>
    uri = part.image_url
    if uri.startswith("data:"):
        # Parse: data:image/jpeg;base64,<payload>
        header, _, payload = uri.partition(",")
        media_type = header.split(":")[1].split(";")[0]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": payload,
            },
        }
    # Plain URL — Anthropic also supports URL sources
    return {
        "type": "image",
        "source": {
            "type": "url",
            "url": uri,
        },
    }
