"""Voice transcription — OpenAI Whisper API integration.

Uses the OpenAI SDK (already a project dependency) to transcribe voice
messages via the Whisper API. No local model needed — the API handles
all audio formats that Telegram sends (OGG/Opus).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from openai import OpenAI

if TYPE_CHECKING:
    from vaultmind.config import VoiceConfig

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio files using the OpenAI Whisper API."""

    def __init__(self, config: VoiceConfig, api_key: str) -> None:
        self._config = config
        self._client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: Path | BinaryIO) -> str:
        """Transcribe an audio file or transient binary stream without altering text.

        Args:
            audio_path: Audio path or an in-memory stream with a filename.

        Returns:
            Transcribed text.

        Raises:
            RuntimeError: If transcription fails.
        """
        try:
            with (
                audio_path.open("rb") if isinstance(audio_path, Path) else nullcontext(audio_path)
            ) as f:
                if self._config.language:
                    response = self._client.audio.transcriptions.create(
                        model=self._config.whisper_model,
                        file=f,
                        language=self._config.language,
                    )
                else:
                    response = self._client.audio.transcriptions.create(
                        model=self._config.whisper_model,
                        file=f,
                    )

            text: str = response.text
            logger.info("Transcribed audio: %d chars", len(text))
            return text

        except Exception as exc:
            error = type(exc).__name__
            logger.error("Transcription failed (%s)", error)
            raise RuntimeError(f"Transcription failed ({error})") from None
