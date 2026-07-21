"""Tests for sources/pipeline.py — ingestion, review-queue routing, and
event-bus finalization (no live LLM/network calls)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import frontmatter
import pytest

from vaultmind.llm.client import LLMResponse
from vaultmind.services.review_queue import (
    Impact,
    Lane,
    ProposalKind,
    ProposalStatus,
    ReviewProposal,
    ReviewQueue,
)
from vaultmind.sources.models import ConnectorDefinition, FetchResult, SourceInstance, SourceItem
from vaultmind.sources.pipeline import (
    AUTO_INGESTED_AUTHORITY,
    finalize_ingested_note,
    ingest_item,
    make_applier,
    propose_ingestion,
    run_connector_once,
)
from vaultmind.sources.registry import REGISTRY, register_connector
from vaultmind.sources.store import SourceStore
from vaultmind.vault.events import NoteCreatedEvent, VaultEventBus

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from vaultmind.vault.models import Note

_GOOD_RESPONSE = {
    "question": "What's new in the fixture feed?",
    "summary": "A short summary of the fetched item.",
    "resolution": "",
    "systems": ["VaultMind"],
    "participants": [],
}


def _make_llm(response_obj: object | None = None, *, raise_error: bool = False) -> MagicMock:
    client = MagicMock()
    if raise_error:
        client.complete.side_effect = RuntimeError("LLM unavailable")
    else:
        client.complete.return_value = LLMResponse(
            text=json.dumps(response_obj or _GOOD_RESPONSE),
            model="fake-model",
            usage={"total_tokens": 40},
        )
    return client


def _item(item_id: str = "item-1", title: str = "Fixture Item") -> SourceItem:
    return SourceItem(
        item_id=item_id,
        title=title,
        content="Some fetched content about the fixture feed.",
        url="https://example.com/fixture-item",
        published_at="2026-06-05T09:00:00Z",
    )


def _instance() -> SourceInstance:
    return SourceInstance(
        name="rss-fixture", kind="rss", target="ignored", output_folder="00-inbox/sources"
    )


class FakeParser:
    """Minimal VaultParser stand-in, mirroring test_watch_handler.py's convention."""

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root

    def parse_file(self, path: Path) -> Note:
        from vaultmind.vault.models import Note

        rel = path.relative_to(self.vault_root)
        post = frontmatter.loads(path.read_text())
        return Note(path=rel, title=str(post.get("title", path.stem)), content=path.read_text())


class FakeStore:
    """Minimal VaultStore stand-in tracking calls, mirroring test_watch_handler.py's convention."""

    def __init__(self) -> None:
        self.indexed: list[str] = []

    def index_single_note(self, note: Note, parser: object) -> int:
        self.indexed.append(str(note.path))
        return 3


class TestIngestItem:
    def test_successful_distillation_stamps_auto_ingested_authority(self, tmp_path: Path) -> None:
        llm = _make_llm()
        rel_path = ingest_item(
            _item(), _instance(), vault_root=tmp_path, llm_client=llm, model="fake-model"
        )
        post = frontmatter.loads((tmp_path / rel_path).read_text())
        assert post.metadata["authority"] == AUTO_INGESTED_AUTHORITY == 1
        assert post.metadata["type"] == "qa-artifact"
        assert post.metadata["question"] == _GOOD_RESPONSE["question"]

    def test_distillation_failure_falls_back_to_plain_note(self, tmp_path: Path) -> None:
        llm = _make_llm(raise_error=True)
        rel_path = ingest_item(
            _item(title="Failed Item"),
            _instance(),
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
        )
        note_path = tmp_path / rel_path
        assert note_path.exists()
        post = frontmatter.loads(note_path.read_text())
        assert post.metadata["authority"] == AUTO_INGESTED_AUTHORITY == 1
        assert "type" not in post.metadata or post.metadata.get("type") != "qa-artifact"
        assert "Some fetched content" in note_path.read_text()

    def test_plain_note_fallback_never_overwrites(self, tmp_path: Path) -> None:
        llm = _make_llm(raise_error=True)
        instance = _instance()
        first = ingest_item(
            _item(item_id="a", title="Dup Title"),
            instance,
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
        )
        second = ingest_item(
            _item(item_id="b", title="Dup Title"),
            instance,
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
        )
        assert first != second
        assert (tmp_path / first).exists()
        assert (tmp_path / second).exists()


class TestMakeApplier:
    def test_applier_writes_note_and_returns_path(self, tmp_path: Path) -> None:
        llm = _make_llm()
        applier = make_applier(vault_root=tmp_path, llm_client=llm, model="fake-model")
        payload = {
            "item_id": "item-1",
            "title": "Fixture Item",
            "content": "content here",
            "url": "https://example.com/x",
            "published_at": "",
            "instance_name": "rss-fixture",
            "instance_kind": "rss",
            "output_folder": "00-inbox/sources",
        }
        result_path = applier(payload)
        assert (tmp_path / result_path).exists()


class TestProposeIngestion:
    def test_routes_auto_by_default_thresholds(self, tmp_path: Path) -> None:
        llm = _make_llm()
        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={
                ProposalKind.SOURCE_INGESTION: make_applier(
                    vault_root=tmp_path, llm_client=llm, model="fake-model"
                )
            },
        )
        proposal = propose_ingestion(queue, _item(), _instance())
        assert proposal.kind is ProposalKind.SOURCE_INGESTION
        assert proposal.lane is Lane.AUTO
        assert proposal.confidence == 1.0
        assert proposal.impact is Impact.LOW

    def test_routes_skim_under_stricter_thresholds(self, tmp_path: Path) -> None:
        from vaultmind.services.review_queue import AutonomyThresholds

        queue = ReviewQueue(
            tmp_path / "queue.db", AutonomyThresholds(block_below=0.4, skim_below=1.0)
        )
        proposal = propose_ingestion(queue, _item(), _instance())
        assert proposal.lane is Lane.SKIM

    def test_never_bypasses_queue_even_without_applier(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")  # no appliers registered
        proposal = propose_ingestion(queue, _item(), _instance())
        # capped at SKIM (never AUTO) since no applier is registered for the kind
        assert proposal.lane is Lane.SKIM
        assert proposal.status is ProposalStatus.PENDING


class TestFinalizeIngestedNote:
    async def test_publishes_note_created_event_for_applied_ingestion(self, tmp_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text('---\ntitle: "X"\n---\n\nbody\n')
        proposal = ReviewProposal(
            proposal_id="p1",
            kind=ProposalKind.SOURCE_INGESTION,
            lane=Lane.AUTO,
            lane_reason="test",
            confidence=1.0,
            impact=Impact.LOW,
            summary="test",
            payload={},
            status=ProposalStatus.APPLIED,
            result="note.md",
            created=datetime.now(),
            last_seen=datetime.now(),
            resolved=None,
            occurrence_count=1,
        )
        bus = VaultEventBus()
        received: list[NoteCreatedEvent] = []

        async def _on_created(event: NoteCreatedEvent) -> None:
            received.append(event)

        bus.subscribe(NoteCreatedEvent, _on_created)  # type: ignore[arg-type]
        store = FakeStore()
        await finalize_ingested_note(
            proposal,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )
        assert len(received) == 1
        assert store.indexed == ["note.md"]

    async def test_noop_for_non_source_ingestion_kind(self, tmp_path: Path) -> None:
        proposal = ReviewProposal(
            proposal_id="p2",
            kind=ProposalKind.TAG_APPLICATION,
            lane=Lane.AUTO,
            lane_reason="test",
            confidence=1.0,
            impact=Impact.LOW,
            summary="test",
            payload={},
            status=ProposalStatus.APPLIED,
            result="note.md",
            created=datetime.now(),
            last_seen=datetime.now(),
            resolved=None,
            occurrence_count=1,
        )
        bus = VaultEventBus()
        received: list[NoteCreatedEvent] = []
        bus.subscribe(NoteCreatedEvent, lambda e: received.append(e))  # type: ignore[arg-type,misc]
        store = FakeStore()
        await finalize_ingested_note(
            proposal,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )
        assert store.indexed == []

    async def test_noop_for_pending_status(self, tmp_path: Path) -> None:
        proposal = ReviewProposal(
            proposal_id="p3",
            kind=ProposalKind.SOURCE_INGESTION,
            lane=Lane.SKIM,
            lane_reason="test",
            confidence=1.0,
            impact=Impact.LOW,
            summary="test",
            payload={},
            status=ProposalStatus.PENDING,
            result="",
            created=datetime.now(),
            last_seen=datetime.now(),
            resolved=None,
            occurrence_count=1,
        )
        bus = VaultEventBus()
        store = FakeStore()
        await finalize_ingested_note(
            proposal,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )
        assert store.indexed == []


class TestRunConnectorOnce:
    @pytest.fixture(autouse=True)
    def _register_fixture_connector(self) -> Iterator[None]:
        REGISTRY.pop("rss", None)

        async def _fetch(instance: SourceInstance, state: object) -> FetchResult:
            return FetchResult(
                items=[_item("a", "First"), _item("b", "Second")], next_cursor_id="a"
            )

        register_connector(ConnectorDefinition(kind="rss", fetch=_fetch, description="fixture"))
        yield
        REGISTRY.pop("rss", None)

    async def test_full_run_ingests_and_advances_cursor(self, tmp_path: Path) -> None:
        llm = _make_llm()
        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={
                ProposalKind.SOURCE_INGESTION: make_applier(
                    vault_root=tmp_path, llm_client=llm, model="fake-model"
                )
            },
        )
        source_store = SourceStore(tmp_path / "sources.db")
        bus = VaultEventBus()
        store = FakeStore()

        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )

        assert result.items_fetched == 2
        assert result.items_ingested == 2
        assert result.error == ""

        state = source_store.get_state("rss-fixture")
        assert state.last_seen_id == "a"
        assert state.run_count == 1

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert runs[0].items_ingested == 2

    async def test_connector_fetch_failure_records_error_without_crashing(
        self, tmp_path: Path
    ) -> None:
        REGISTRY.pop("rss", None)

        async def _failing_fetch(instance: SourceInstance, state: object) -> FetchResult:
            raise RuntimeError("feed unreachable")

        register_connector(
            ConnectorDefinition(kind="rss", fetch=_failing_fetch, description="fixture-fail")
        )

        queue = ReviewQueue(tmp_path / "queue.db")
        source_store = SourceStore(tmp_path / "sources.db")
        bus = VaultEventBus()
        store = FakeStore()

        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )
        assert result.items_fetched == 0
        assert result.items_ingested == 0
        assert "feed unreachable" in result.error

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert "feed unreachable" in runs[0].error

    async def test_one_item_failure_does_not_block_sibling_items(self, tmp_path: Path) -> None:
        """Mirrors a real embedding/index failure (e.g. a bad API key)
        blocking only the item that hit it, not the whole batch."""

        class _FailOnceStore:
            def __init__(self) -> None:
                self.indexed: list[str] = []
                self._calls = 0

            def index_single_note(self, note: object, parser: object) -> int:
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("embedding provider 401")
                self.indexed.append(str(note.path))  # type: ignore[attr-defined]
                return 1

        llm = _make_llm()
        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={
                ProposalKind.SOURCE_INGESTION: make_applier(
                    vault_root=tmp_path, llm_client=llm, model="fake-model"
                )
            },
        )
        source_store = SourceStore(tmp_path / "sources.db")
        bus = VaultEventBus()
        store = _FailOnceStore()

        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=FakeParser(tmp_path),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )

        assert result.items_fetched == 2
        assert result.items_ingested == 1  # only the item whose finalize step succeeded counts
        assert "embedding provider 401" in result.error
        assert len(store.indexed) == 1  # only the second item's index call succeeded

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert "embedding provider 401" in runs[0].error
