"""Ordinary producers reject reserved output aliases before side effects."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaultmind.config import MaturationConfig, Settings
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.pipeline.distill import distill_retained_conversation
from vaultmind.pipeline.maturation import MaturationPipeline
from vaultmind.research.pipeline import ResearchConfig, run_research_pipeline
from vaultmind.services.review_queue import Impact, Lane, ProposalKind, ProposalStatus, ReviewQueue
from vaultmind.sources.models import SourceInstance, SourceItem
from vaultmind.sources.pipeline import (
    ingest_item,
    make_applier,
    propose_ingestion,
)
from vaultmind.vault.security import ReservedNamespaceError


@pytest.fixture(params=["wiki", "_wiki"])
def reserved_alias(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, Path]:
    root = tmp_path / "vault"
    reserved = root / str(request.param)
    reserved.mkdir(parents=True)
    page = reserved / "page.md"
    page.write_text("approved page", encoding="utf-8")
    (root / "configured-output").symlink_to(reserved, target_is_directory=True)
    return root, page


@pytest.fixture
def intake(reserved_alias: tuple[Path, Path]) -> Iterator[SourceIntake]:
    root, _ = reserved_alias
    with SourceIntake(Settings(vault={"path": root})) as capability:
        yield capability


def _item() -> SourceItem:
    return SourceItem(item_id="one", title="page", content="raw source", url="https://example.com")


def _instance() -> SourceInstance:
    return SourceInstance(
        name="feed", kind="rss", target="https://example.com", output_folder="configured-output"
    )


def test_distill_alias_rejected_before_provider(
    reserved_alias: tuple[Path, Path], intake: SourceIntake
) -> None:
    root, page = reserved_alias
    llm = MagicMock()
    capture = intake.preserve_text(
        "User: source body",
        original_name="selected-conversation.md",
        origin="selected synthetic session",
        adapter="synthetic-distill",
    )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ReservedNamespaceError):
        distill_retained_conversation(
            intake,
            capture,
            llm,
            "fake",
            source_ref="session",
            occurred_at="2026-10-24T00:00:00+00:00",
            output_folder="configured-output",
        )
    llm.complete.assert_not_called()
    assert page.read_text() == "approved page"
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_synthesis_alias_rejected_before_source_read(
    reserved_alias: tuple[Path, Path], intake: SourceIntake
) -> None:
    root, page = reserved_alias
    llm = MagicMock()
    cluster = MagicMock()
    cluster.top_entity = "entity"
    cluster.note_paths = [str(page.relative_to(root))]
    with pytest.raises(ReservedNamespaceError):
        pipeline = MaturationPipeline(
            MaturationConfig(inbox_folder="configured-output"),
            MagicMock(),
            MagicMock(),
            llm,
            intake,
        )
        pipeline.synthesize(cluster)
    llm.complete.assert_not_called()
    assert page.read_text() == "approved page"
    assert list(page.parent.iterdir()) == [page]


@pytest.mark.asyncio
async def test_research_alias_rejected_before_search(
    reserved_alias: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, intake: SourceIntake
) -> None:
    root, page = reserved_alias
    search = AsyncMock()
    monkeypatch.setattr("vaultmind.research.pipeline.search_youtube", search)
    llm = MagicMock()
    with pytest.raises(ReservedNamespaceError):
        await run_research_pipeline(
            "topic",
            root,
            llm,
            MagicMock(),
            MagicMock(),
            ResearchConfig(output_folder="configured-output"),
            intake=intake,
        )
    search.assert_not_called()
    llm.complete.assert_not_called()
    assert page.read_text() == "approved page"
    assert list(page.parent.iterdir()) == [page]


def test_connector_alias_rejected_before_queue_and_fallback(
    reserved_alias: tuple[Path, Path], intake: SourceIntake
) -> None:
    root, page = reserved_alias
    queue = MagicMock()
    llm = MagicMock()
    with pytest.raises(ReservedNamespaceError):
        propose_ingestion(queue, _item(), _instance(), vault_root=root)
    queue.propose.assert_not_called()
    with pytest.raises(ReservedNamespaceError):
        ingest_item(
            _item(), _instance(), vault_root=root, llm_client=llm, model="fake", intake=intake
        )
    llm.complete.assert_not_called()
    assert page.read_text() == "approved page"
    assert list(page.parent.iterdir()) == [page]


def test_queued_connector_rechecks_alias_and_stays_failed(
    reserved_alias: tuple[Path, Path],
    tmp_path: Path,
    intake: SourceIntake,
) -> None:
    root, page = reserved_alias
    destination = root / "later-output"
    destination.mkdir()
    llm = MagicMock()
    queue = ReviewQueue(
        tmp_path / "review.db",
        appliers={
            ProposalKind.SOURCE_INGESTION: make_applier(
                vault_root=root, llm_client=llm, model="fake", intake=intake
            )
        },
    )
    item = _item()
    proposal = queue.propose(
        ProposalKind.SOURCE_INGESTION,
        confidence=1.0,
        impact=Impact.LOW,
        summary="new source",
        lane_override=Lane.SKIM,
        payload={
            "item_id": item.item_id,
            "title": item.title,
            "content": item.content,
            "url": item.url,
            "published_at": "",
            "instance_name": "feed",
            "instance_kind": "rss",
            "output_folder": "later-output",
        },
    )
    destination.rmdir()
    destination.symlink_to(page.parent, target_is_directory=True)
    failed = queue.approve(proposal.proposal_id)
    assert failed is not None
    assert failed.status is ProposalStatus.FAILED
    assert queue.approve(proposal.proposal_id) is None
    assert queue.approve_all() == []
    llm.complete.assert_not_called()
    assert page.read_text() == "approved page"
    assert list(page.parent.iterdir()) == [page]
