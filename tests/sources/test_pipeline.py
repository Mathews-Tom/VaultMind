"""Tests for sources/pipeline.py — ingestion, review-queue routing, and
event-bus finalization (no live LLM/network calls)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import frontmatter
import pytest

from vaultmind.config import Settings
from vaultmind.knowledge.intake import SourceIntake
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
from vaultmind.vault.parser import VaultParser
from vaultmind.wiki.workspace import WikiError, proposal_digest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from vaultmind.knowledge.intake import SourceRecord
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


@pytest.fixture
def intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SourceIntake]:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    settings = Settings(vault={"path": tmp_path})
    with SourceIntake(settings) as selected:
        yield selected


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
    def test_successful_distillation_stamps_auto_ingested_authority(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
        llm = _make_llm()
        rel_path = ingest_item(
            _item(),
            _instance(),
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
            intake=intake,
        )
        post = frontmatter.loads((tmp_path / rel_path).read_text())
        assert post.metadata["authority"] == AUTO_INGESTED_AUTHORITY == 1
        assert post.metadata["type"] == "qa-artifact"
        assert post.metadata["question"] == _GOOD_RESPONSE["question"]

    def test_distillation_failure_falls_back_to_plain_note(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
        llm = _make_llm(raise_error=True)
        rel_path = ingest_item(
            _item(title="Failed Item"),
            _instance(),
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
            intake=intake,
        )
        note_path = tmp_path / rel_path
        assert note_path.exists()
        post = frontmatter.loads(note_path.read_text())
        assert post.metadata["authority"] == AUTO_INGESTED_AUTHORITY == 1
        assert "type" not in post.metadata or post.metadata.get("type") != "qa-artifact"
        assert "Some fetched content" in note_path.read_text()

    def test_plain_note_fallback_never_overwrites(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
        llm = _make_llm(raise_error=True)
        instance = _instance()
        first = ingest_item(
            _item(item_id="a", title="Dup Title"),
            instance,
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
            intake=intake,
        )
        second = ingest_item(
            _item(item_id="b", title="Dup Title"),
            instance,
            vault_root=tmp_path,
            llm_client=llm,
            model="fake-model",
            intake=intake,
        )
        assert first != second
        assert (tmp_path / first).exists()
        assert (tmp_path / second).exists()


class TestMakeApplier:
    def test_applier_writes_note_and_returns_path(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
        llm = _make_llm()
        applier = make_applier(
            vault_root=tmp_path, llm_client=llm, model="fake-model", intake=intake
        )
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
    def test_routes_auto_by_default_thresholds(self, tmp_path: Path, intake: SourceIntake) -> None:
        llm = _make_llm()
        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={
                ProposalKind.SOURCE_INGESTION: make_applier(
                    vault_root=tmp_path, llm_client=llm, model="fake-model", intake=intake
                )
            },
        )
        proposal = propose_ingestion(queue, _item(), _instance(), vault_root=tmp_path)
        assert proposal.kind is ProposalKind.SOURCE_INGESTION
        assert proposal.lane is Lane.AUTO
        assert proposal.confidence == 1.0
        assert proposal.impact is Impact.LOW

    def test_routes_skim_under_stricter_thresholds(self, tmp_path: Path) -> None:
        from vaultmind.services.review_queue import AutonomyThresholds

        queue = ReviewQueue(
            tmp_path / "queue.db", AutonomyThresholds(block_below=0.4, skim_below=1.0)
        )
        proposal = propose_ingestion(queue, _item(), _instance(), vault_root=tmp_path)
        assert proposal.lane is Lane.SKIM

    def test_never_bypasses_queue_even_without_applier(self, tmp_path: Path) -> None:
        queue = ReviewQueue(tmp_path / "queue.db")  # no appliers registered
        proposal = propose_ingestion(queue, _item(), _instance(), vault_root=tmp_path)
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

    @pytest.mark.parametrize("root_alias", [False, True], ids=["canonical", "logical-alias"])
    async def test_full_run_ingests_and_advances_cursor(
        self, tmp_path: Path, intake: SourceIntake, root_alias: bool
    ) -> None:
        llm = _make_llm()
        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={
                ProposalKind.SOURCE_INGESTION: make_applier(
                    vault_root=tmp_path, llm_client=llm, model="fake-model", intake=intake
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
            parser=VaultParser(Settings(vault={"path": tmp_path}).vault),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path / ".." / tmp_path.name if root_alias else tmp_path,
            event_bus=bus,
            intake=intake,
        )

        assert result.items_fetched == 2
        assert result.items_ingested == 2
        assert result.error == ""
        assert store.indexed == [outcome.path for outcome in result.outcomes]
        assert len(result.outcomes) == 2
        assert all(outcome.source_ids for outcome in result.outcomes)
        assert all(outcome.generation_status == "succeeded" for outcome in result.outcomes)
        assert all(
            outcome.indexed is None and outcome.event_published is None
            for outcome in result.outcomes
        )
        assert all(
            outcome.finalization_status == "returned-no-durable-receipt"
            for outcome in result.outcomes
        )

        state = source_store.get_state("rss-fixture")
        assert state.last_seen_id == "a"
        assert state.run_count == 1

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert runs[0].items_ingested == 2

    async def test_changed_projection_reports_mismatch_without_skipping_raw_finalization(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
        applier = make_applier(
            vault_root=tmp_path, llm_client=_make_llm(), model="fake-model", intake=intake
        )

        def edit_after_capture(proposal: ReviewProposal) -> str:
            relative_path = applier(proposal)
            path = tmp_path / relative_path
            projected = frontmatter.loads(path.read_text())
            projected.content += "\n\nUser-edited ordinary source."
            path.write_text(str(frontmatter.dumps(projected)))
            return relative_path

        class ObservedStore(FakeStore):
            def __init__(self) -> None:
                super().__init__()
                self.contents: list[str] = []

            def index_single_note(self, note: Note, parser: object) -> int:
                self.contents.append(note.content)
                return super().index_single_note(note, parser)

        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={ProposalKind.SOURCE_INGESTION: edit_after_capture},
        )
        source_store = SourceStore(tmp_path / "sources.db")
        bus = VaultEventBus()
        events: list[NoteCreatedEvent] = []

        async def collect(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, collect)  # type: ignore[arg-type]
        store = ObservedStore()
        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=VaultParser(Settings(vault={"path": tmp_path}).vault),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            intake=intake,
            event_bus=bus,
        )

        assert result.items_ingested == 2
        assert "projection differs from its preserved source" in result.error.lower()
        assert store.indexed == [outcome.path for outcome in result.outcomes]
        assert all(content.endswith("User-edited ordinary source.") for content in store.contents)
        assert [str(event.path) for event in events] == store.indexed
        assert all(event.note.content.endswith("User-edited ordinary source.") for event in events)
        for outcome in result.outcomes:
            assert outcome.status == "applied"
            assert outcome.projection_status == "unverified"
            assert outcome.finalization_status == "returned-no-durable-receipt"
            assert outcome.indexed is None
            assert outcome.event_published is None
            retained = intake.read((tmp_path / outcome.path).stem)
            assert "User-edited ordinary source." not in retained.content.decode()
        assert source_store.get_state("rss-fixture").last_seen_id == "a"
        assert source_store.list_runs("rss-fixture")[0].error == result.error

    @pytest.mark.parametrize("rejected_field", ["title", "item_id"])
    async def test_rejected_original_stays_out_of_logs_and_run_history(
        self,
        tmp_path: Path,
        intake: SourceIntake,
        caplog: pytest.LogCaptureFixture,
        rejected_field: str,
    ) -> None:
        marker = "NO-RETAIN-LOG-MARKER"
        fields = {
            "item_id": "rejected",
            "title": "Rejected item",
            "content": "Permitted content",
            "url": "https://example.invalid/rejected",
        }
        fields[rejected_field] = f"---\ndo_not_retain: true\n---\n{marker}"
        rejected = SourceItem(**fields)

        async def fetch(instance: SourceInstance, state: object) -> FetchResult:
            return FetchResult(items=[rejected], next_cursor_id="fetched-cursor")

        REGISTRY.pop("rss", None)
        register_connector(ConnectorDefinition(kind="rss", fetch=fetch, description="fixture"))
        queue = ReviewQueue(tmp_path / "queue.db")
        source_store = SourceStore(tmp_path / "sources.db")
        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=VaultParser(Settings(vault={"path": tmp_path}).vault),
            store=FakeStore(),  # type: ignore[arg-type]
            vault_root=tmp_path,
            intake=intake,
            event_bus=VaultEventBus(),
        )

        assert result.items_fetched == 1
        assert result.items_ingested == 0
        assert result.outcomes[0].proposal_id == ""
        assert result.outcomes[0].source_ids == []
        assert marker not in caplog.text
        assert marker not in result.error
        assert marker not in result.outcomes[0].item_id
        assert marker not in source_store.list_runs("rss-fixture")[0].error
        assert "do-not-retain" in result.error.lower()
        assert not (tmp_path / "_wiki" / "sources").exists()

    @pytest.mark.parametrize("target_kind", ["reserved", "outside"])
    async def test_applied_projection_alias_is_denied_before_raw_finalization(
        self, tmp_path: Path, intake: SourceIntake, target_kind: str
    ) -> None:
        target = (
            tmp_path / "wiki" / "protected.md"
            if target_kind == "reserved"
            else tmp_path.parent / f"{tmp_path.name}-outside.md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# Never index this alias\n\nProtected target bytes.\n")
        before = target.read_bytes()
        applier = make_applier(
            vault_root=tmp_path, llm_client=_make_llm(), model="fake-model", intake=intake
        )

        def alias_after_capture(proposal: ReviewProposal) -> str:
            relative_path = applier(proposal)
            path = tmp_path / relative_path
            path.unlink()
            path.symlink_to(target)
            return relative_path

        queue = ReviewQueue(
            tmp_path / "queue.db",
            appliers={ProposalKind.SOURCE_INGESTION: alias_after_capture},
        )
        source_store = SourceStore(tmp_path / "sources.db")
        store = FakeStore()
        bus = VaultEventBus()
        events: list[NoteCreatedEvent] = []

        async def collect(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, collect)  # type: ignore[arg-type]
        result = await run_connector_once(
            _instance(),
            source_store=source_store,
            review_queue=queue,
            parser=VaultParser(Settings(vault={"path": tmp_path}).vault),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            intake=intake,
            event_bus=bus,
        )

        assert result.items_ingested == 0
        assert result.error
        assert store.indexed == []
        assert events == []
        assert target.read_bytes() == before
        assert all(outcome.status == "applied" for outcome in result.outcomes)
        assert all(outcome.finalization_status == "not-attempted" for outcome in result.outcomes)

    async def test_connector_fetch_failure_records_error_without_crashing(
        self, tmp_path: Path, intake: SourceIntake
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
            parser=FakeParser(tmp_path),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
            intake=intake,
        )
        assert result.items_fetched == 0
        assert result.items_ingested == 0
        assert "feed unreachable" in result.error

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert "feed unreachable" in runs[0].error

    async def test_one_item_failure_does_not_block_sibling_items(
        self, tmp_path: Path, intake: SourceIntake
    ) -> None:
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
                    vault_root=tmp_path, llm_client=llm, model="fake-model", intake=intake
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
            parser=FakeParser(tmp_path),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
            intake=intake,
        )

        assert result.items_fetched == 2
        assert result.items_ingested == 1  # only the item whose finalize step succeeded counts
        assert "embedding provider 401" in result.error
        assert len(store.indexed) == 1  # only the second item's index call succeeded
        assert result.outcomes[0].status == "applied"
        assert result.outcomes[0].source_ids
        assert result.outcomes[0].finalization_status == "failed"
        assert result.outcomes[0].indexed is None
        assert result.outcomes[0].event_published is None

        runs = source_store.list_runs("rss-fixture")
        assert len(runs) == 1
        assert "embedding provider 401" in runs[0].error


def _retained_by_role(intake: SourceIntake) -> dict[str, SourceRecord]:
    records = {}
    for path in (intake.workspace.root / "_wiki" / "sources").glob("*.intake.json"):
        record = intake.read(path.name.removesuffix(".intake.json"))
        assert record.provenance is not None
        records[record.provenance["role"]] = record
    return records


def test_connector_preserves_exact_original_and_request_before_provider(
    tmp_path: Path, intake: SourceIntake
) -> None:
    item = SourceItem(
        item_id="exact",
        title='A "quoted": title\ncontinued',
        content="Full original\r\n\r\nPreserve every character: café.\r\n",
        url="https://example.invalid/exact",
        published_at="2026-10-24T00:00:00Z",
    )
    llm = _make_llm()
    response = llm.complete.return_value

    def complete(**kwargs: object) -> LLMResponse:
        records = _retained_by_role(intake)
        assert records["item-content"].content == item.content.encode("utf-8")
        assert json.loads(records["item"].text or "") == {
            "item_id": item.item_id,
            "title": item.title,
            "content": item.content,
            "url": item.url,
            "published_at": item.published_at,
        }
        retained_request = json.loads(records["model-request"].text or "")
        assert retained_request["messages"][0]["content"] == records["distillation-input"].text
        messages = kwargs["messages"]
        assert messages[0].content == retained_request["messages"][0]["content"]  # type: ignore[index]
        for key in ("model", "max_tokens", "system"):
            assert kwargs[key] == retained_request[key]
        assert not list((tmp_path / _instance().output_folder).glob("*.md"))
        return response

    llm.complete.side_effect = complete
    relative = ingest_item(
        item,
        _instance(),
        vault_root=tmp_path,
        intake=intake,
        llm_client=llm,
        model="scripted",
    )
    derivative = intake.read((tmp_path / relative).stem)
    assert derivative.provenance is not None
    assert derivative.provenance["original_retained"] is False
    assert derivative.provenance["parents"]
    assert (tmp_path / relative).read_bytes() == derivative.content
    assert frontmatter.loads(derivative.text or "")["approved"] is False
    assert not (tmp_path / "wiki").exists()
    llm.complete.assert_called_once()


def test_failed_connector_generation_is_plain_original_not_synthesis_or_replay(
    tmp_path: Path, intake: SourceIntake
) -> None:
    llm = _make_llm(raise_error=True)
    relative = ingest_item(
        _item(),
        _instance(),
        vault_root=tmp_path,
        intake=intake,
        llm_client=llm,
        model="scripted",
    )
    post = frontmatter.loads((tmp_path / relative).read_text())
    assert post["approved"] is False
    assert post["generation_status"] == "failed"
    assert post.get("type") != "qa-artifact"
    assert _item().content in post.content
    records = _retained_by_role(intake)
    assert "distillation" not in records
    processing = (
        tmp_path
        / "_wiki"
        / "sources"
        / (records["distillation-input"].source_id + ".processing.json")
    )
    assert json.loads(processing.read_text())["provider"] == "failed"
    before = {
        str(path): path.read_bytes() for path in (tmp_path / "_wiki").rglob("*") if path.is_file()
    }
    with pytest.raises(WikiError, match="no automatic replay"):
        ingest_item(
            _item(),
            _instance(),
            vault_root=tmp_path,
            intake=intake,
            llm_client=llm,
            model="scripted",
        )
    assert {
        str(path): path.read_bytes() for path in (tmp_path / "_wiki").rglob("*") if path.is_file()
    } == before
    llm.complete.assert_called_once()


@pytest.mark.parametrize(
    "content",
    [
        "---\ndo_not_retain: true\n---\nPrivate original",
        "---\nretention: do-not-retain\n---\nPrivate original",
        "---\nunterminated: true",
        "unsupported\x00control",
        "x" * (256 * 1024 + 1),
        "x" * (256 * 1024),
        '"' * (128 * 1024),
    ],
    ids=[
        "do-not-retain",
        "retention-policy",
        "malformed-frontmatter",
        "null-byte",
        "oversized",
        "oversized-snapshot",
        "escaped-snapshot",
    ],
)
def test_connector_original_policy_cannot_be_escaped_in_json(
    content: str, tmp_path: Path, intake: SourceIntake
) -> None:
    llm = _make_llm()
    item = SourceItem("policy", "Policy", content, "https://example.invalid/policy")
    queue = MagicMock(spec=ReviewQueue)
    with pytest.raises(WikiError):
        propose_ingestion(queue, item, _instance(), vault_root=tmp_path)
    queue.propose.assert_not_called()
    with pytest.raises(WikiError):
        ingest_item(
            item,
            _instance(),
            vault_root=tmp_path,
            intake=intake,
            llm_client=llm,
            model="scripted",
        )
    llm.complete.assert_not_called()
    assert not (tmp_path / "_wiki" / "sources").exists()


async def test_old_eight_field_pending_payload_survives_queue_restart(
    tmp_path: Path, intake: SourceIntake
) -> None:
    from vaultmind.services.review_queue import Lane

    payload = {
        "item_id": "old",
        "title": "Old queued item",
        "content": "Exact old content.",
        "url": "https://example.invalid/old",
        "published_at": "",
        "instance_name": "old-feed",
        "instance_kind": "rss",
        "output_folder": "00-inbox/sources",
    }
    queue_path = tmp_path / "delayed.db"
    queue = ReviewQueue(queue_path)
    proposal = queue.propose(
        ProposalKind.SOURCE_INGESTION,
        1.0,
        Impact.LOW,
        "Old operation",
        payload,
        lane_override=Lane.SKIM,
    )
    queue.close()
    llm = _make_llm()
    queue = ReviewQueue(
        queue_path,
        appliers={
            ProposalKind.SOURCE_INGESTION: make_applier(
                vault_root=tmp_path,
                intake=intake,
                llm_client=llm,
                model="scripted",
            )
        },
    )
    try:
        approved = queue.approve(proposal.proposal_id)
        assert approved is not None
        assert approved.payload == payload
        assert approved.status is ProposalStatus.APPLIED
        assert isinstance(approved.result, str)
        assert (tmp_path / approved.result).is_file()
        store = FakeStore()
        bus = VaultEventBus()
        events = []

        async def collect(event: NoteCreatedEvent) -> None:
            events.append(event)

        bus.subscribe(NoteCreatedEvent, collect)  # type: ignore[arg-type]
        await finalize_ingested_note(
            approved,
            parser=FakeParser(tmp_path),
            store=store,  # type: ignore[arg-type]
            vault_root=tmp_path,
            event_bus=bus,
        )
        assert store.indexed == [approved.result]
        assert len(events) == 1
        assert not (tmp_path / "wiki").exists()
    finally:
        queue.close()


def test_connector_derivative_uses_existing_exact_wiki_review_and_parent_checks(
    tmp_path: Path, intake: SourceIntake
) -> None:
    relative = ingest_item(
        _item(),
        _instance(),
        vault_root=tmp_path,
        intake=intake,
        llm_client=_make_llm(),
        model="scripted",
    )
    derivative_id = (tmp_path / relative).stem
    draft = tmp_path / "candidate.md"
    draft.write_text(
        "# Evidence\n\n## Current understanding\n\n"
        f"An explicitly reviewed claim. [[_wiki/sources/{derivative_id}]]\n"
    )
    workspace = intake.workspace
    proposal = workspace.propose("wiki/evidence.md", [derivative_id], draft_file=draft)
    review = workspace.review(proposal.proposal_id)
    assert "Exact before/after unified diff" in review
    with pytest.raises(WikiError, match="digest"):
        workspace.apply(proposal.proposal_id, "0" * 64)
    records = _retained_by_role(intake)
    (tmp_path / records["item-content"].path).unlink()
    with pytest.raises(WikiError):
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    assert not (tmp_path / "wiki" / "evidence.md").exists()


def test_connector_projection_failure_does_not_become_plain_fallback_or_retry(
    tmp_path: Path, intake: SourceIntake, monkeypatch: pytest.MonkeyPatch
) -> None:
    atomic_write = intake.workspace._atomic_bytes
    output_folder = tmp_path / _instance().output_folder

    def fail_projection(path: Path, content: bytes) -> None:
        if path.is_relative_to(output_folder):
            raise OSError("scripted ordinary-projection failure")
        atomic_write(path, content)

    monkeypatch.setattr(intake.workspace, "_atomic_bytes", fail_projection)
    llm = _make_llm()
    with pytest.raises(WikiError, match="no automatic replay"):
        ingest_item(
            _item(),
            _instance(),
            vault_root=tmp_path,
            intake=intake,
            llm_client=llm,
            model="scripted",
        )
    roles = _retained_by_role(intake)
    assert "distillation" in roles
    assert "original-projection" not in roles
    assert not output_folder.exists()
    with pytest.raises(WikiError, match="no automatic replay"):
        ingest_item(
            _item(),
            _instance(),
            vault_root=tmp_path,
            intake=intake,
            llm_client=llm,
            model="scripted",
        )
    llm.complete.assert_called_once()


def test_connector_absolute_ordinary_folder_still_returns_relative_path(
    tmp_path: Path, intake: SourceIntake
) -> None:
    instance = SourceInstance(
        name="absolute",
        kind="rss",
        target="unused",
        output_folder=str(tmp_path / "00-inbox" / "absolute"),
    )
    relative = ingest_item(
        _item(),
        instance,
        vault_root=tmp_path,
        intake=intake,
        llm_client=_make_llm(),
        model="scripted",
    )
    assert relative.startswith("00-inbox/absolute/")
    assert (tmp_path / relative).is_file()


def test_connector_rejects_wrong_vault_capability_before_provider(
    tmp_path: Path, intake: SourceIntake
) -> None:
    llm = _make_llm()
    with pytest.raises(WikiError, match="different vault"):
        ingest_item(
            _item(),
            _instance(),
            vault_root=tmp_path / "other",
            intake=intake,
            llm_client=llm,
            model="scripted",
        )
    llm.complete.assert_not_called()
    assert not (tmp_path / "_wiki" / "sources").exists()


def test_connector_source_id_can_be_explicitly_curated_without_reingestion(
    tmp_path: Path, intake: SourceIntake
) -> None:
    llm = _make_llm()
    relative = ingest_item(
        _item(),
        _instance(),
        vault_root=tmp_path,
        intake=intake,
        llm_client=llm,
        model="scripted",
    )
    source_id = (tmp_path / relative).stem
    draft = tmp_path / "reviewed-source.md"
    draft.write_text(
        "# Source evidence\n\n## Current understanding\n\n"
        f"A manually reviewed statement. [[_wiki/sources/{source_id}]]\n"
    )
    workspace = intake.workspace
    proposal = workspace.propose("wiki/source-evidence.md", [source_id], draft_file=draft)
    review = workspace.review(proposal.proposal_id)
    digest = proposal_digest(proposal.payload)
    assert digest in review
    assert not (tmp_path / "wiki" / "source-evidence.md").exists()
    workspace.apply(proposal.proposal_id, digest)
    page = frontmatter.loads((tmp_path / "wiki" / "source-evidence.md").read_text())
    assert source_id in page["sources"]
    assert page["source_revisions"][source_id] == intake.read(source_id).snapshot["sha256"]
    assert intake.read(source_id).provenance["original_retained"] is False
    llm.complete.assert_called_once()
