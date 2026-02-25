"""Tests for voice transcription and handler routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.transcribe import Transcriber

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeVoiceConfig:
    """Minimal VoiceConfig stand-in."""

    def __init__(self) -> None:
        self.enabled = True
        self.whisper_model = "whisper-1"
        self.language = ""


class FakeAudioResponse:
    """Mimics OpenAI transcription response."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeTranscriptionsAPI:
    """Fake OpenAI audio.transcriptions API."""

    def __init__(self, text: str = "Transcribed text") -> None:
        self._text = text
        self.calls: list[dict[str, str]] = []

    def create(self, **kwargs: object) -> FakeAudioResponse:
        self.calls.append({"model": str(kwargs.get("model", ""))})
        return FakeAudioResponse(self._text)


class FakeAudioAPI:
    def __init__(self, text: str = "Transcribed text") -> None:
        self.transcriptions = FakeTranscriptionsAPI(text)


class FakeOpenAIClient:
    def __init__(self, text: str = "Transcribed text") -> None:
        self.audio = FakeAudioAPI(text)


# ---------------------------------------------------------------------------
# Tests — Transcriber
# ---------------------------------------------------------------------------


class TestTranscriber:
    def test_transcribe_returns_text(self, tmp_path: Path) -> None:
        config = FakeVoiceConfig()
        transcriber = Transcriber(config, api_key="test-key")  # type: ignore[arg-type]

        # Monkey-patch the client
        fake_client = FakeOpenAIClient("Hello world")
        transcriber._client = fake_client  # type: ignore[assignment]

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio data")

        result = transcriber.transcribe(audio_file)
        assert result == "Hello world"

    def test_transcribe_strips_whitespace(self, tmp_path: Path) -> None:
        config = FakeVoiceConfig()
        transcriber = Transcriber(config, api_key="test-key")  # type: ignore[arg-type]

        fake_client = FakeOpenAIClient("  padded text  ")
        transcriber._client = fake_client  # type: ignore[assignment]

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio data")

        result = transcriber.transcribe(audio_file)
        assert result == "padded text"

    def test_transcribe_with_language(self, tmp_path: Path) -> None:
        config = FakeVoiceConfig()
        config.language = "en"
        transcriber = Transcriber(config, api_key="test-key")  # type: ignore[arg-type]

        fake_client = FakeOpenAIClient("English text")
        transcriber._client = fake_client  # type: ignore[assignment]

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"data")

        result = transcriber.transcribe(audio_file)
        assert result == "English text"
