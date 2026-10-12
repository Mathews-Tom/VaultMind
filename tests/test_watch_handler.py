"""Tests for IncrementalWatchHandler — debounce, hash stability, event bus.

The first half drives the handler with lightweight stand-ins to pin debounce and
event semantics. The ``TestReal*`` classes drive the real chain — files
on disk, the watchdog observer, ``VaultWatcher``, the handler, the parser and real
Chroma + SQLite BM25 indexes — with only the embedding provider synthetic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vaultmind.config import VaultConfig, WatchConfig
from vaultmind.contradiction.marking import write_superseded_block
from vaultmind.vault.events import (
    NoteCreatedEvent,
    NoteDeletedEvent,
    NoteModifiedEvent,
    VaultEventBus,
)
from vaultmind.vault.projection import (
    DELETE_MARKER,
    DELETE_RATIONALE,
    DELETE_TITLE,
    EDIT_RATIONALE_PREFIX,
    EDIT_TITLE,
    SUPERSEDED_CALLOUT,
    SUPERSEDED_KEY,
)
from vaultmind.vault.watch_handler import IncrementalWatchHandler, _file_content_hash
from vaultmind.vault.watcher import VaultWatcher

from .index_support import ALL_SURFACES, IndexEnv, make_env

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

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


# ---------------------------------------------------------------------------
# Real chain: filesystem -> watchdog observer -> VaultWatcher -> handler -> indexes
# ---------------------------------------------------------------------------

THREE_SECTIONS = (
    "## Tides\nobserved tidewrack along the north pier\n\n"
    "## Birds\nemberquill gulls nest on the second jetty\n\n"
    "## Lamps\ncobaltmoor lamp replaced at the lighthouse"
)
REVISED_THREE = (
    "## Tides\nobserved ironweft along the south pier\n\n"
    "## Birds\nsaltreed terns nest on the first jetty\n\n"
    "## Lamps\nbrasswick lamp replaced at the harbour"
)
REVISED_ONE = "## Tides\nobserved ironweft along the south pier"
OLD_TOKENS = ("tidewrack", "emberquill", "cobaltmoor")
HANDLER_LOGGER = "vaultmind.vault.watch_handler"


@pytest.fixture
def env(tmp_path: Path) -> Iterator[IndexEnv]:
    environment = make_env(tmp_path.resolve())
    try:
        yield environment
    finally:
        environment.close()


async def eventually(predicate: Callable[[], object], what: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


def index_failures(caplog: pytest.LogCaptureFixture, path: Path) -> list[logging.LogRecord]:
    """Error records with a traceback that the handler logged about *path*."""
    return [
        record
        for record in caplog.records
        if record.name == HANDLER_LOGGER
        and record.levelno >= logging.ERROR
        and record.exc_info is not None
        and str(path) in record.getMessage()
    ]


class WatchedVault:
    """The real watch chain over an isolated vault, recording the events it publishes."""

    def __init__(self, env: IndexEnv, *, hash_stability_check: bool) -> None:
        self.env = env
        self.bus = VaultEventBus()
        self.handler = IncrementalWatchHandler(
            WatchConfig(debounce_ms=40, hash_stability_check=hash_stability_check),
            env.parser,
            env.store,
            self.bus,
        )
        self.watcher = VaultWatcher(VaultConfig(path=env.vault), self.handler.handle_change)
        self.indexed: list[NoteCreatedEvent | NoteModifiedEvent] = []
        self.deleted: list[NoteDeletedEvent] = []
        self.bus.subscribe(NoteCreatedEvent, self._on_indexed)  # type: ignore[arg-type]
        self.bus.subscribe(NoteModifiedEvent, self._on_indexed)  # type: ignore[arg-type]
        self.bus.subscribe(NoteDeletedEvent, self._on_deleted)  # type: ignore[arg-type]

    async def _on_indexed(self, event: NoteCreatedEvent | NoteModifiedEvent) -> None:
        self.indexed.append(event)

    async def _on_deleted(self, event: NoteDeletedEvent) -> None:
        self.deleted.append(event)

    def indexed_for(self, path: Path) -> list[NoteCreatedEvent | NoteModifiedEvent]:
        return [event for event in self.indexed if event.path == path]

    async def indexed_count(self, path: Path, count: int) -> None:
        await eventually(
            lambda: len(self.indexed_for(path)) >= count, f"{count} index event(s) for {path.name}"
        )
        # A quiet period: a duplicate or premature event would show up as an extra one.
        await asyncio.sleep(0.4)
        assert len(self.indexed_for(path)) == count

    async def wait_until_live(self) -> None:
        """Prove real filesystem events reach the handler before the scenario starts."""
        marker = self.env.vault / "watcher-ready.md"
        deadline = time.monotonic() + 20.0
        while marker not in self.handler.indexed_hashes:
            if time.monotonic() > deadline:
                raise AssertionError("filesystem events never reached the watch handler")
            self.env.write(marker.name, "## Ready\nreadiness marker")
            await asyncio.sleep(0.5)


@asynccontextmanager
async def watching(
    env: IndexEnv, *, hash_stability_check: bool = True
) -> AsyncIterator[WatchedVault]:
    watched = WatchedVault(env, hash_stability_check=hash_stability_check)
    watched.watcher.start()
    try:
        await watched.wait_until_live()
        yield watched
    finally:
        watched.watcher.stop()


class TestRealFilesystemEvents:
    @pytest.mark.asyncio
    async def test_note_lifecycle_reaches_every_retrieval_path(self, env: IndexEnv) -> None:
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)
            assert watched.indexed_for(note)[0].chunks_indexed == 3
            assert watched.handler.indexed_hashes[note] == _file_content_hash(note)
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES

            env.write("log.md", REVISED_ONE)
            await watched.indexed_count(note, 2)
            assert watched.handler.indexed_hashes[note] == _file_content_hash(note)
            assert env.surfaces("ironweft") == ALL_SURFACES
            for token in OLD_TOKENS:
                assert env.surfaces(token) == set()

            note.unlink()
            await eventually(lambda: any(e.path == note for e in watched.deleted), "delete event")
            assert note not in watched.handler.indexed_hashes
            assert env.surfaces("ironweft") == set()
            # Only the readiness marker remains indexed.
            assert (env.store.count, env.bm25.count) == (1, 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stability", [True, False])
    async def test_embedding_failure_keeps_last_good_state_until_identical_retry(
        self, env: IndexEnv, caplog: pytest.LogCaptureFixture, stability: bool
    ) -> None:
        caplog.set_level(logging.ERROR, logger=HANDLER_LOGGER)
        async with watching(env, hash_stability_check=stability) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)
            last_good = _file_content_hash(note)

            env.embedder.failure = RuntimeError("embedding provider unavailable")
            env.write("log.md", REVISED_THREE)
            await eventually(lambda: index_failures(caplog, note), "the logged index failure")

            assert watched.handler.indexed_hashes[note] == last_good
            assert len(watched.indexed_for(note)) == 1
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES
            assert env.surfaces("ironweft") == set()

            # The next event carrying the identical failed bytes is retried.
            env.embedder.failure = None
            env.write("log.md", REVISED_THREE)
            await watched.indexed_count(note, 2)
            assert watched.handler.indexed_hashes[note] == _file_content_hash(note)
            assert _file_content_hash(note) != last_good
            assert env.surfaces("ironweft") == ALL_SURFACES
            for token in OLD_TOKENS:
                assert env.surfaces(token) == set()

    @pytest.mark.asyncio
    async def test_partial_index_failure_is_retried_even_when_bytes_revert_to_last_good(
        self, env: IndexEnv, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.ERROR, logger=HANDLER_LOGGER)
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)
            last_good = _file_content_hash(note)

            # The vector index accepts the shorter revision, the keyword index fails.
            env.bm25.fail_on = {"replace_note"}
            env.write("log.md", REVISED_ONE)
            await eventually(lambda: index_failures(caplog, note), "the logged index failure")
            assert watched.handler.indexed_hashes[note] == last_good
            assert len(watched.indexed_for(note)) == 1

            # Reverting to the last-good bytes must still repair the half-applied index.
            env.bm25.fail_on = set()
            env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 2)
            assert _file_content_hash(note) == last_good
            assert env.store.count == env.bm25.count == 4
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES
            assert env.surfaces("ironweft") == set()

    @pytest.mark.asyncio
    async def test_unparseable_revision_keeps_last_good_index_and_hash(
        self, env: IndexEnv, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.ERROR, logger=HANDLER_LOGGER)
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)
            last_good = _file_content_hash(note)

            note.write_text("---\ntitle: [unclosed\n---\n\nbroken", encoding="utf-8")
            await eventually(lambda: index_failures(caplog, note), "the logged parse failure")

            assert watched.handler.indexed_hashes[note] == last_good
            assert len(watched.indexed_for(note)) == 1
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES

            env.write("log.md", REVISED_THREE)
            await watched.indexed_count(note, 2)
            assert env.surfaces("ironweft") == ALL_SURFACES

    @pytest.mark.asyncio
    async def test_failed_retirement_is_logged_without_a_deleted_event(
        self, env: IndexEnv, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.ERROR, logger=HANDLER_LOGGER)
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)

            env.bm25.fail_on = {"delete_note"}
            note.unlink()
            await eventually(lambda: index_failures(caplog, note), "the logged retirement failure")

            assert not any(event.path == note for event in watched.deleted)
            assert note not in watched.handler.indexed_hashes

            # Recreating the identical file must index it afresh, not be skipped as unchanged.
            env.bm25.fail_on = set()
            env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 2)
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES
            assert env.store.count == env.bm25.count == 4

    @pytest.mark.asyncio
    async def test_bot_delete_on_disk_retires_the_note_from_every_index(
        self, env: IndexEnv
    ) -> None:
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)
            for token in OLD_TOKENS:
                assert env.surfaces(token) == ALL_SURFACES

            write_superseded_block(
                note,
                callout_type=SUPERSEDED_CALLOUT,
                title=DELETE_TITLE,
                rationale=DELETE_RATIONALE,
                frontmatter_key=SUPERSEDED_KEY,
                frontmatter_value=DELETE_MARKER,
                dated=True,
                boundary="delete",
            )
            await watched.indexed_count(note, 2)

            assert watched.indexed_for(note)[1].chunks_indexed == 0
            for token in OLD_TOKENS:
                assert env.surfaces(token) == set()
            assert (env.store.count, env.bm25.count) == (1, 1)  # the readiness marker only
            # Canonical content stays on disk.
            assert all(token in note.read_text(encoding="utf-8") for token in OLD_TOKENS)

    @pytest.mark.asyncio
    async def test_two_bot_edits_on_disk_index_only_the_latest_claim(self, env: IndexEnv) -> None:
        async with watching(env) as watched:
            note = env.write("log.md", THREE_SECTIONS)
            await watched.indexed_count(note, 1)

            claims = [
                "## Tides\nobserved ironweft along the south pier",
                "## Tides\nobserved saltreed along the east pier",
            ]
            started = datetime.now(UTC)
            for position, claim in enumerate(claims):
                stamp = started + timedelta(seconds=position)
                write_superseded_block(
                    note,
                    callout_type=SUPERSEDED_CALLOUT,
                    title=EDIT_TITLE,
                    rationale=f"{EDIT_RATIONALE_PREFIX}revise the tide entry",
                    frontmatter_key=SUPERSEDED_KEY,
                    frontmatter_value=stamp.isoformat(),
                    dated=True,
                    timestamp=stamp,
                    new_content=claim,
                    boundary="edit",
                )
                await watched.indexed_count(note, 2 + position)

            assert env.surfaces("saltreed") == ALL_SURFACES
            for token in ("ironweft", *OLD_TOKENS):
                assert env.surfaces(token) == set()
            # Every earlier revision is still on disk.
            text = note.read_text(encoding="utf-8")
            assert all(token in text for token in ("tidewrack", "ironweft", "saltreed"))


class TestRealHandlerEntryPoint:
    @pytest.mark.asyncio
    async def test_change_event_for_a_vanished_file_retires_the_note(self, env: IndexEnv) -> None:
        bus = VaultEventBus()
        deleted: list[NoteDeletedEvent] = []

        async def on_deleted(event: NoteDeletedEvent) -> None:
            deleted.append(event)

        bus.subscribe(NoteDeletedEvent, on_deleted)  # type: ignore[arg-type]
        handler = IncrementalWatchHandler(
            WatchConfig(debounce_ms=10, hash_stability_check=False), env.parser, env.store, bus
        )
        note = env.write("log.md", THREE_SECTIONS)
        env.index("log.md")
        note.unlink()

        handler.handle_change(note, "modified")
        await eventually(lambda: deleted, "the deleted event")

        assert deleted[0].path == note
        assert (env.store.count, env.bm25.count) == (0, 0)

    @pytest.mark.asyncio
    async def test_overlapping_events_for_one_save_index_it_once(self, env: IndexEnv) -> None:
        bus = VaultEventBus()
        events: list[NoteCreatedEvent] = []

        async def on_created(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, on_created)  # type: ignore[arg-type]
        handler = IncrementalWatchHandler(
            WatchConfig(debounce_ms=20, hash_stability_check=False), env.parser, env.store, bus
        )
        env.embedder.delay_seconds = 0.3
        note = env.write("log.md", THREE_SECTIONS)

        handler.handle_change(note, "created")
        await asyncio.sleep(0.1)  # the first pass is now inside the slow embedding call
        handler.handle_change(note, "created")
        await eventually(lambda: events, "the created event")
        await asyncio.sleep(0.8)

        assert len(events) == 1
        assert env.embedder.calls == 1
        assert handler.indexed_hashes[note] == _file_content_hash(note)
