"""Voice transcription — OpenAI Whisper API integration.

Uses the OpenAI SDK (already a project dependency) to transcribe voice
messages via the Whisper API. No local model needed — the API handles
all audio formats that Telegram sends (OGG/Opus).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.config import VoiceConfig

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio files using the OpenAI Whisper API."""

    def __init__(self, config: VoiceConfig, api_key: str) -> None:
        self._config = config
        self._client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (OGG, MP3, WAV, etc.).

        Returns:
            Transcribed text.

        Raises:
            RuntimeError: If transcription fails.
        """
        try:
            with open(audio_path, "rb") as f:
                kwargs: dict[str, str] = {}
                if self._config.language:
                    kwargs["language"] = self._config.language

                response = self._client.audio.transcriptions.create(
                    model=self._config.whisper_model,
                    file=f,
                    **kwargs,  # type: ignore[arg-type]
                )

            text = response.text.strip()
            logger.info("Transcribed %s: %d chars", audio_path.name, len(text))
            return text

        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path.name, e)
            raise RuntimeError(f"Transcription failed: {e}") from e
