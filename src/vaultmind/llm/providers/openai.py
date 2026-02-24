"""OpenAI provider (GPT-4.1, GPT-5, etc.)."""

from __future__ import annotations

from openai import OpenAI, OpenAIError

from vaultmind.llm.client import LLMError, LLMResponse, Message


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
