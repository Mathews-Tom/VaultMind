"""Google Gemini provider via the OpenAI-compatible API."""

from __future__ import annotations

from openai import OpenAI, OpenAIError

from vaultmind.llm.client import LLMError, LLMResponse, Message

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient:
    """LLM client for Google Gemini models.

    Uses the OpenAI-compatible endpoint that Google provides,
    avoiding the need for the google-genai SDK as a dependency.
    """

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Gemini supports the OpenAI chat completions format."""
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
            raise LLMError(str(e), provider="gemini", original=e) from e

        choice = response.choices[0]
        if choice.message.content is None:
            raise LLMError("Empty response from Gemini", provider="gemini")

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
