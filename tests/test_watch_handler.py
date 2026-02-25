"""Tests for IncrementalWatchHandler — debounce, hash stability, event bus."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.vault.events import (
    NoteCreatedEvent,
    NoteDeletedEvent,
    NoteModifiedEvent,
    VaultEventBus,
)
from vaultmind.vault.watch_handler import IncrementalWatchHandler, _file_content_hash

if TYPE_CHECKING:
    from vaultmind.vault.models import Note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeWatchConfig:
    """Minimal WatchConfig stand-in for tests."""

    def __init__(
        self,
        debounce_ms: int = 50,
        hash_stability_check: bool = False,
        reextract_graph: bool = False,
        batch_graph_interval_seconds: int = 300,
    ) -> None:
        self.debounce_ms = debounce_ms
        self.hash_stability_check = hash_stability_check
        self.reextract_graph = reextract_graph
        self.batch_graph_interval_seconds = batch_graph_interval_seconds


class FakeParser:
    """Minimal VaultParser stand-in."""

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root

    def parse_file(self, path: Path) -> Note:
        from vaultmind.vault.models import Note

        rel = path.relative_to(self.vault_root)
        return Note(
            path=rel,
            title=path.stem,
            content=path.read_text(),
        )

    def chunk_note(self, note: Note) -> list[object]:
        return [
            MagicMock(
                chunk_id=f"{note.path}::0",
                content=note.content,
                to_chroma_metadata=lambda: {},
            )
        ]


class FakeStore:
    """Minimal VaultStore stand-in tracking calls."""

    def __init__(self) -> None:
        self.indexed: list[str] = []
        self.deleted: list[str] = []

    def index_single_note(self, note: Note, parser: object) -> int:
        self.indexed.append(str(note.path))
        return 1

    def delete_note(self, note_path: str) -> None:
        self.deleted.append(note_path)


def _write_md(vault: Path, name: str, content: str) -> Path:
    p = vault / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Tests — _file_content_hash
# ---------------------------------------------------------------------------


class TestFileContentHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("hello world")
        assert _file_content_hash(f) == _file_content_hash(f)

    def test_different_content(self, tmp_path: Path) -> None:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("alpha")
        b.write_text("beta")
        assert _file_content_hash(a) != _file_content_hash(b)

    def test_length_16(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("content")
        assert len(_file_content_hash(f)) == 16


# ---------------------------------------------------------------------------
# Tests — VaultEventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self) -> None:
        bus = VaultEventBus()
        received: list[NoteCreatedEvent] = []

        async def on_created(event: NoteCreatedEvent) -> None:
            received.append(event)

        bus.subscribe(NoteCreatedEvent, on_created)  # type: ignore[arg-type]

        event = NoteCreatedEvent(path=Path("test.md"), chunks_indexed=3)
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].chunks_indexed == 3

    @pytest.mark.asyncio
    async def test_type_isolation(self) -> None:
        bus = VaultEventBus()
        created_count = 0
        deleted_count = 0

        async def on_created(event: NoteCreatedEvent) -> None:
            nonlocal created_count
            created_count += 1

        async def on_deleted(event: NoteDeletedEvent) -> None:
            nonlocal deleted_count
            deleted_count += 1

        bus.subscribe(NoteCreatedEvent, on_created)  # type: ignore[arg-type]
        bus.subscribe(NoteDeletedEvent, on_deleted)  # type: ignore[arg-type]

        await bus.publish(NoteCreatedEvent(path=Path("a.md")))
        await bus.publish(NoteDeletedEvent(path=Path("b.md")))
        await bus.publish(NoteCreatedEvent(path=Path("c.md")))

        assert created_count == 2
        assert deleted_count == 1

    @pytest.mark.asyncio
    async def test_subscriber_error_isolated(self) -> None:
        bus = VaultEventBus()
        good_received: list[NoteCreatedEvent] = []

        async def bad_subscriber(event: NoteCreatedEvent) -> None:
            raise RuntimeError("boom")

        async def good_subscriber(event: NoteCreatedEvent) -> None:
            good_received.append(event)

        bus.subscribe(NoteCreatedEvent, bad_subscriber)  # type: ignore[arg-type]
        bus.subscribe(NoteCreatedEvent, good_subscriber)  # type: ignore[arg-type]

        await bus.publish(NoteCreatedEvent(path=Path("test.md")))
        assert len(good_received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus = VaultEventBus()
        count = 0

        async def handler(event: NoteCreatedEvent) -> None:
            nonlocal count
            count += 1

        bus.subscribe(NoteCreatedEvent, handler)  # type: ignore[arg-type]
        await bus.publish(NoteCreatedEvent(path=Path("a.md")))
        assert count == 1

        bus.unsubscribe(NoteCreatedEvent, handler)  # type: ignore[arg-type]
        await bus.publish(NoteCreatedEvent(path=Path("b.md")))
        assert count == 1

    def test_subscriber_count(self) -> None:
        bus = VaultEventBus()

        async def noop(event: NoteCreatedEvent) -> None:
            pass

        assert bus.subscriber_count == 0
        bus.subscribe(NoteCreatedEvent, noop)  # type: ignore[arg-type]
        bus.subscribe(NoteDeletedEvent, noop)  # type: ignore[arg-type]
        assert bus.subscriber_count == 2


# ---------------------------------------------------------------------------
# Tests — IncrementalWatchHandler
# ---------------------------------------------------------------------------


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def handler_deps(vault: Path) -> tuple[FakeParser, FakeStore, VaultEventBus]:
    return FakeParser(vault), FakeStore(), VaultEventBus()


class TestWatchHandlerBasic:
    @pytest.mark.asyncio
    async def test_created_indexes_note(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "# Hello\nWorld")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        assert len(store.indexed) == 1
        assert "note.md" in store.indexed[0]

    @pytest.mark.asyncio
    async def test_modified_reindexes(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "version 1")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        md.write_text("version 2")
        handler.handle_change(md, "modified")
        await asyncio.sleep(0.05)

        assert len(store.indexed) == 2

    @pytest.mark.asyncio
    async def test_deleted_removes_from_store(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "content")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        md.unlink()
        handler.handle_change(md, "deleted")
        await asyncio.sleep(0.05)

        assert len(store.deleted) == 1

    @pytest.mark.asyncio
    async def test_unchanged_content_skipped(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "static content")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        # Same content, should skip
        handler.handle_change(md, "modified")
        await asyncio.sleep(0.05)

        assert len(store.indexed) == 1


class TestDebounce:
    @pytest.mark.asyncio
    async def test_rapid_changes_coalesced(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=50, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "v1")

        # Fire 5 rapid changes — only the last should trigger indexing
        for i in range(5):
            md.write_text(f"version {i}")
            handler.handle_change(md, "modified")
            await asyncio.sleep(0.01)

        # Wait for debounce to fire
        await asyncio.sleep(0.15)

        # Should have indexed only once (debounce coalesced the 5 events)
        assert len(store.indexed) == 1


class TestHashStability:
    @pytest.mark.asyncio
    async def test_stable_hash_indexes(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=True)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "stable content")
        handler.handle_change(md, "created")

        # With stability check: debounce (10ms) + stability sleep (10ms) + processing
        await asyncio.sleep(0.15)

        assert len(store.indexed) == 1

    @pytest.mark.asyncio
    async def test_unstable_hash_retries(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=20, hash_stability_check=True)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        md = _write_md(vault, "note.md", "initial")
        handler.handle_change(md, "created")

        # Change content during the stability window
        await asyncio.sleep(0.025)
        md.write_text("changed during stability")

        # Wait for full retry cycle
        await asyncio.sleep(0.2)

        # Should still index eventually (content stabilized)
        assert len(store.indexed) >= 1


class TestEventPublishing:
    @pytest.mark.asyncio
    async def test_created_event_published(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        events: list[NoteCreatedEvent] = []

        async def capture(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, capture)  # type: ignore[arg-type]

        md = _write_md(vault, "test.md", "# Test")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        assert len(events) == 1
        assert events[0].chunks_indexed == 1

    @pytest.mark.asyncio
    async def test_modified_event_published(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        events: list[NoteModifiedEvent] = []

        async def capture(event: NoteModifiedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteModifiedEvent, capture)  # type: ignore[arg-type]

        md = _write_md(vault, "test.md", "v1")
        handler.handle_change(md, "created")
        await asyncio.sleep(0.05)

        md.write_text("v2")
        handler.handle_change(md, "modified")
        await asyncio.sleep(0.05)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_deleted_event_published(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        events: list[NoteDeletedEvent] = []

        async def capture(event: NoteDeletedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteDeletedEvent, capture)  # type: ignore[arg-type]

        md = _write_md(vault, "test.md", "content")
        md.unlink()
        handler.handle_change(md, "deleted")
        await asyncio.sleep(0.05)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_vanished_file_no_event(self, vault: Path, handler_deps: tuple) -> None:
        parser, store, bus = handler_deps
        config = FakeWatchConfig(debounce_ms=10, hash_stability_check=False)
        handler = IncrementalWatchHandler(
            config=config,
            parser=parser,
            store=store,
            event_bus=bus,  # type: ignore[arg-type]
        )

        events: list[NoteCreatedEvent] = []

        async def capture(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, capture)  # type: ignore[arg-type]

        md = _write_md(vault, "ghost.md", "boo")
        handler.handle_change(md, "created")
        md.unlink()  # vanish before debounce fires
        await asyncio.sleep(0.05)

        assert len(events) == 0
        assert len(store.indexed) == 0
