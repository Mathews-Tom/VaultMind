"""OpenAI provider (GPT-4.1, GPT-5, etc.)."""

from __future__ import annotations

from typing import Any

from openai import OpenAI, OpenAIError

from vaultmind.llm.client import ContentPart, LLMError, LLMResponse, Message, MultimodalMessage


class OpenAIClient:
    """LLM client for OpenAI models.

    Also works with any OpenAI-compatible API by setting base_url.
    """

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """OpenAI uses a system message prepended to the messages list."""
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        )

        try:
            response = self._client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=api_messages,  # type: ignore[arg-type]
            )
        except OpenAIError as e:
            raise LLMError(str(e), provider="openai", original=e) from e

        choice = response.choices[0]
        if choice.message.content is None:
            raise LLMError("Empty response from OpenAI", provider="openai")

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

    def complete_multimodal(
        self,
        messages: list[Message | MultimodalMessage],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """OpenAI vision API: content may be a list of text/image_url parts."""
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            if isinstance(msg, MultimodalMessage):
                parts: list[dict[str, Any]] = []
                for part in msg.parts:
                    parts.append(_openai_content_part(part))
                api_messages.append({"role": msg.role, "content": parts})
            else:
                if msg.role != "system":
                    api_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = self._client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=api_messages,  # type: ignore[arg-type]
            )
        except OpenAIError as e:
            raise LLMError(str(e), provider="openai", original=e) from e

        choice = response.choices[0]
        if choice.message.content is None:
            raise LLMError("Empty multimodal response from OpenAI", provider="openai")

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


def _openai_content_part(part: ContentPart) -> dict[str, Any]:
    """Convert a ContentPart to the OpenAI API content part format."""
    if part.type == "text":
        return {"type": "text", "text": part.text}
    # image_url — accepts base64 data URIs and plain URLs
    return {"type": "image_url", "image_url": {"url": part.image_url}}
