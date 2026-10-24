"""Voice adapter stays transient; only non-question transcripts enter intake."""

from __future__ import annotations

import asyncio
import hashlib
import io
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_capture import make_handlers, make_message, registered_callback, source_records
from vaultmind.bot.handlers import voice
from vaultmind.bot.transcribe import Transcriber
from vaultmind.config import VoiceConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def voice_message(data: bytes = b"audio-original", event: int = 20) -> MagicMock:
    message = make_message(event=event)
    message.voice = SimpleNamespace(file_id="voice-file", file_size=len(data))
    message.bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="voice.ogg"))

    async def download(path: str, destination: io.BytesIO) -> None:
        assert isinstance(destination, io.BytesIO)
        destination.write(data)

    message.bot.download_file = AsyncMock(side_effect=download)
    return message


def transcriber(text: str) -> MagicMock:
    result = MagicMock()

    def transcribe(stream: io.BytesIO) -> str:
        assert stream.name == "voice.ogg"
        assert stream.read() == b"audio-original"
        return text

    result.transcribe.side_effect = transcribe
    return result


def test_transcriber_preserves_exact_response_and_accepts_unowned_memory_stream() -> None:
    client = Transcriber(VoiceConfig(), api_key="test-key")
    fake = MagicMock()
    fake.audio.transcriptions.create.return_value = SimpleNamespace(text="  exact\r\ntext\r\n")
    client._client = fake
    with io.BytesIO(b"audio") as stream:
        stream.name = "voice.ogg"
        assert client.transcribe(stream) == "  exact\r\ntext\r\n"
        assert not stream.closed
        assert fake.audio.transcriptions.create.call_args.kwargs["file"] is stream


def test_transcriber_path_consumer_and_bodyfree_failure(tmp_path: Path) -> None:
    client = Transcriber(VoiceConfig(language="en"), api_key="test-key")
    fake = MagicMock()
    client._client = fake
    path = tmp_path / "audio.ogg"
    path.write_bytes(b"audio")
    fake.audio.transcriptions.create.return_value = SimpleNamespace(text="Exact")
    assert client.transcribe(path) == "Exact"
    assert fake.audio.transcriptions.create.call_args.kwargs["language"] == "en"
    fake.audio.transcriptions.create.side_effect = RuntimeError("private response contents")
    with pytest.raises(RuntimeError) as failure:
        client.transcribe(path)
    assert "private response contents" not in str(failure.value)


def test_voice_transcript_identity_nonretention_and_fresh_process_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    client = transcriber("  Keep exact transcript.\r\n")
    handlers._ctx.transcriber = client
    message = voice_message()
    voice._voice_events.clear()
    callback = registered_callback(handlers, "handle_voice", monkeypatch)
    asyncio.run(callback(message))
    records = source_records(handlers)
    assert len(records) == 1
    record = records[0]
    assert record.text == "  Keep exact transcript.\r\n"
    assert record.provenance["role"] == "voice-transcript"
    assert record.provenance["original_retained"] is False
    assert record.provenance["adapter"] == "transcriber:openai:whisper-1"
    assert record.provenance["event_id"] == "12:20:42"
    assert not list(tmp_path.rglob("*.ogg"))
    assert not list(tmp_path.rglob("*.bin"))
    # Simulate a fresh handler process; the persisted transcript is authoritative.
    voice._voice_events.clear()
    asyncio.run(callback(message))
    client.transcribe.assert_called_once()
    handlers._ctx.store.index_single_note.assert_called_once()
    assert len(source_records(handlers)) == 1


def test_voice_simultaneous_registered_event_has_one_transcript_and_stable_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    expected = "  First exact transcript.\r\n"
    possible_transcripts = iter((expected, "A different transcript for the same audio."))
    client = MagicMock()

    def transcribe(stream: io.BytesIO) -> str:
        assert stream.name == "voice.ogg"
        assert stream.read() == b"audio-original"
        return next(possible_transcripts)

    client.transcribe.side_effect = transcribe
    handlers._ctx.transcriber = client
    callback = registered_callback(handlers, "handle_voice", monkeypatch)
    first = voice_message(event=23)
    duplicate = voice_message(event=23)
    voice._voice_events.clear()
    lookup_results: list[object] = []

    async def exercise() -> None:
        lookup_entered = asyncio.Event()
        release_lookup = asyncio.Event()
        duplicate_decided = asyncio.Event()
        first_finished = asyncio.Event()
        real_to_thread = asyncio.to_thread
        lookup = voice._captured_transcript
        provider_starts = 0

        async def controlled_to_thread(
            func: Callable[..., object], *args: object, **kwargs: object
        ) -> object:
            nonlocal provider_starts
            if func is lookup:
                # Read real durable state before suspending: both pre-fix callers
                # must observe no transcript before either provider can run.
                result = await real_to_thread(func, *args, **kwargs)
                lookup_results.append(result)
                lookup_entered.set()
                if len(lookup_results) == 2:
                    duplicate_decided.set()
                await release_lookup.wait()
                return result
            if func is client.transcribe:
                provider_starts += 1
                if provider_starts > 1:
                    # Serialize the buggy second capture to expose two distinct
                    # sources, rather than an unrelated storage-write race.
                    await first_finished.wait()
            return await real_to_thread(func, *args, **kwargs)

        async def duplicate_answer(text: str) -> None:
            duplicate_decided.set()

        async def capture_first() -> None:
            try:
                await callback(first)
            finally:
                first_finished.set()

        monkeypatch.setattr(voice.asyncio, "to_thread", controlled_to_thread)
        duplicate.answer.side_effect = duplicate_answer
        first_task = asyncio.create_task(capture_first())
        await asyncio.wait_for(lookup_entered.wait(), timeout=5)
        duplicate_task = asyncio.create_task(callback(duplicate))
        try:
            # The duplicate either reaches the same vulnerable lookup or is
            # rejected by the reservation; neither path relies on a sleep.
            await asyncio.wait_for(duplicate_decided.wait(), timeout=5)
            client.transcribe.assert_not_called()
        finally:
            release_lookup.set()
        await asyncio.wait_for(asyncio.gather(first_task, duplicate_task), timeout=5)

        records = source_records(handlers)
        assert len(records) == 1
        retained = records[0]
        assert retained.content == expected.encode("utf-8")
        assert retained.provenance["event_id"] == "12:23:42"
        assert (
            retained.provenance["upstream_revision"]
            == hashlib.sha256(b"audio-original").hexdigest()
        )
        assert retained.provenance["original_retained"] is False
        client.transcribe.assert_called_once()
        handlers._ctx.store.index_single_note.assert_called_once()
        assert "already handled" in duplicate.answer.call_args.args[0]

        # A new process must reuse this same complete transcript, not mint a
        # second source or fail an ambiguous authenticated-event lookup.
        voice._voice_events.clear()
        await callback(voice_message(event=23))
        replayed = source_records(handlers)
        assert len(replayed) == 1
        assert replayed[0].source_id == retained.source_id
        assert replayed[0].content == retained.content
        assert lookup_results == [None, expected]
        client.transcribe.assert_called_once()
        handlers._ctx.store.index_single_note.assert_called_once()
        assert not list(tmp_path.rglob("*.ogg"))
        assert not list(tmp_path.rglob("*.bin"))

    asyncio.run(exercise())


def test_voice_question_is_not_a_source_and_no_audio_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    client = transcriber("What is this question?")
    handlers._ctx.transcriber = client
    question = AsyncMock()
    monkeypatch.setattr("vaultmind.bot.handlers.routing.handle_smart_response", question)
    message = voice_message(event=21)
    voice._voice_events.clear()
    callback = registered_callback(handlers, "handle_voice", monkeypatch)
    asyncio.run(callback(message))
    question.assert_awaited_once_with(
        handlers._ctx, message, "What is this question?", is_question=True
    )
    assert not (tmp_path / "_wiki").exists()
    assert not (tmp_path / "00-inbox").exists()
    assert not list(tmp_path.rglob("*.ogg"))
    asyncio.run(callback(message))
    client.transcribe.assert_called_once()
    handlers._ctx.store.index_single_note.assert_not_called()


def test_voice_actual_size_is_bounded_before_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = make_handlers(tmp_path)
    client = transcriber("Must not be reached")
    handlers._ctx.transcriber = client
    message = voice_message(b"oversized")
    message.voice.file_size = 1
    monkeypatch.setattr(voice, "_MAX_VOICE_BYTES", 4)
    asyncio.run(handlers.handle_voice(message))
    client.transcribe.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_failed_transcription_is_not_claimed_capture_and_not_retried(tmp_path: Path) -> None:
    handlers = make_handlers(tmp_path)
    client = MagicMock()
    client.transcribe.side_effect = RuntimeError("private speech")
    handlers._ctx.transcriber = client
    message = voice_message(event=22)
    voice._voice_events.clear()
    asyncio.run(handlers.handle_voice(message))
    assert "nothing captured" in message.answer.call_args.args[0]
    assert "private speech" not in message.answer.call_args.args[0]
    assert not (tmp_path / "_wiki").exists()
    asyncio.run(handlers.handle_voice(message))
    client.transcribe.assert_called_once()
