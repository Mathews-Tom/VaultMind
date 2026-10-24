"""Connector operations preserve originals before generating discovery material.

SOURCE_INGESTION authorizes ingestion only. Its old payload and relative-path
result survive; asynchronous indexing/events and cursor/run history retain
their existing non-atomic, non-recovery semantics. Wiki publication remains the
separate selected-page, exact-review, digest-apply workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.knowledge.intake import CaptureResult, SourceIntake
from vaultmind.pipeline.distill import (
    distill_retained_conversation,
    mint_gap_for_unresolved,
    render_conversation,
)
from vaultmind.services.review_delivery import deliver_block
from vaultmind.services.review_queue import Impact, ProposalKind, ProposalStatus
from vaultmind.sources import (
    connectors as _connectors,  # noqa: F401 - registers connectors as import side effect
)
from vaultmind.sources.models import FetchResult, RunSummary, SourceInstance, SourceItem
from vaultmind.sources.registry import get_connector
from vaultmind.sources.store import advance_cursor
from vaultmind.vault.events import NoteCreatedEvent
from vaultmind.vault.security import validate_ordinary_write_path
from vaultmind.wiki.workspace import WikiError, _metadata, _safe_text

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.indexer.store import VaultStore
    from vaultmind.knowledge.intake import SourceRecord
    from vaultmind.llm.client import LLMClient
    from vaultmind.services.review_delivery import BlockNotifier
    from vaultmind.services.review_queue import Applier, ReviewProposal, ReviewQueue
    from vaultmind.sources.store import SourceStore
    from vaultmind.vault.events import VaultEventBus
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)

# M2's provenance table (authored=5, distilled=4, research=3, web clip=2,
# auto-ingested=1): connector items are unattended background content, the
# table's lowest-trust tier — distinct from `distill.DISTILLED_AUTHORITY`
# (=4), which stays the default for a user's own conversation-distillation
# session (bot/thinking.py, bot/handlers/distill.py).
AUTO_INGESTED_AUTHORITY = 1


def _item_snapshot(item: SourceItem) -> dict[str, str]:
    """The actual available connector object, never credentials/configuration."""
    return {
        "item_id": item.item_id,
        "title": item.title,
        "content": item.content,
        "url": item.url,
        "published_at": item.published_at,
    }


def _validate_item(item: SourceItem) -> None:
    """Reject forbidden originals before queue persistence as well as capture."""
    for value in _item_snapshot(item).values():
        _metadata(_safe_text(value.encode("utf-8"), "Connector item"))
    _safe_text(_json_text(_item_snapshot(item)).encode("utf-8"), "Connector item snapshot")


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _item_identity(item: SourceItem, instance: SourceInstance) -> tuple[str, str, str]:
    serialized = _json_text(_item_snapshot(item))
    revision = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    event_id = hashlib.sha256(
        _json_text([instance.kind, instance.name, item.item_id]).encode("utf-8")
    ).hexdigest()
    return serialized, revision, event_id


def _render_plain_note(
    item: SourceItem, instance: SourceInstance, original: CaptureResult, error: str, *, created: str
) -> str:
    """Render the existing original-source fallback, explicitly not synthesis."""
    return str(
        frontmatter.dumps(
            frontmatter.Post(
                f"# {item.title}\n\n**Source:** {item.url}\n\n{item.content}",
                title=item.title,
                source=item.url,
                source_type=instance.kind,
                source_instance=instance.name,
                tags=["source-ingested"],
                authority=AUTO_INGESTED_AUTHORITY,
                created=created,
                status="active",
                source_id=original.source_id,
                source_path=original.path,
                approved=False,
                generation_status="failed",
                generation_error=error,
            )
        )
    )


def ingest_item(
    item: SourceItem,
    instance: SourceInstance,
    *,
    vault_root: Path,
    intake: SourceIntake,
    llm_client: LLMClient,
    model: str,
    gap_store: object = None,
) -> str:
    """Preserve originals, generate an unapproved derivative, project it.

    The SOURCE_INGESTION operation still returns only a relative ordinary-note
    path. Its async caller owns indexing/events; neither this return nor queue
    approval approves any wiki page. Failed/unchanged stages are not replayed.
    """
    vault_root = vault_root.resolve()
    output_folder = validate_ordinary_write_path(instance.output_folder, vault_root)
    folder = output_folder.relative_to(vault_root).as_posix()
    if intake.workspace.root.resolve() != vault_root:
        raise WikiError("Connector intake belongs to a different vault")
    _validate_item(item)
    serialized, revision, event_id = _item_identity(item, instance)
    adapter = "connector"
    origin = f"{instance.kind}:{instance.name}"
    content = intake.preserve_text(
        item.content,
        original_name="connector-content.md",
        origin=origin,
        adapter=adapter,
        role="item-content",
        event_id=event_id,
        revision=revision,
        uri=item.url or None,
    )
    original = intake.preserve_text(
        serialized,
        original_name="connector-item.json",
        origin=origin,
        adapter=adapter,
        role="item",
        event_id=event_id,
        revision=revision,
        uri=item.url or None,
        parents=[content.source_id],
    )
    input_capture = intake.preserve_text(
        render_conversation([{"user": f"{item.title}\n\n{item.content}".strip(), "assistant": ""}]),
        original_name="connector-input.md",
        origin=origin,
        adapter=adapter,
        role="distillation-input",
        event_id=event_id,
        revision=revision,
        uri=item.url or None,
        parents=[original.source_id],
        original_retained=False,
    )
    occurred_at = item.published_at or str(
        intake.read(original.source_id).snapshot["metadata"]["captured_at"]
    )
    result = distill_retained_conversation(
        intake,
        input_capture,
        llm_client,
        model,
        source_ref=item.url,
        occurred_at=occurred_at,
        output_folder=folder,
        authority=AUTO_INGESTED_AUTHORITY,
    )
    if result.success:
        if result.generation is not None:
            mint_gap_for_unresolved(result.generation, gap_store, item.url)
        return result.output_path
    if (
        result.generation is None
        or result.generation.success
        or result.generation_disposition is None
        or result.generation_disposition.status != "failed"
    ):
        raise WikiError(
            f"{result.error}; original retained: {original.source_id}; "
            f"input retained: {input_capture.source_id}; no automatic replay"
        )

    # Preserve the old provider-outage UX only for an actual failed generation,
    # not for failed/skipped projection or incomplete prior processing.
    def preserve_plain() -> CaptureResult:
        return intake.preserve_text(
            _render_plain_note(
                item,
                instance,
                original,
                result.error,
                created=str(intake.read(original.source_id).snapshot["metadata"]["captured_at"]),
            ),
            original_name="connector-original-projection.md",
            origin=origin,
            adapter=adapter,
            role="original-projection",
            parents=[
                result.request.source_id if result.request is not None else input_capture.source_id
            ],
            original_retained=False,
        )

    fallback = original.run_stage("derivative", preserve_plain)
    if fallback.status == "succeeded" and isinstance(fallback.value, CaptureResult):
        projection = intake.project_text(fallback.value, folder)
        if projection.status == "succeeded" and isinstance(projection.value, str):
            return projection.value
    raise WikiError(
        f"Original-source projection not completed; original retained: {original.source_id}"
    )


def make_applier(
    *,
    vault_root: Path,
    intake: SourceIntake,
    llm_client: LLMClient,
    model: str,
    gap_store: object = None,
) -> Applier:
    """Build the sync `Applier` registered for `ProposalKind.SOURCE_INGESTION`.

    Bound via closure to the collaborators `ingest_item` needs, matching
    `cli.py::bot`'s existing `_maturation_applier`/
    `ContradictionDetector._resolution_applier` closure-over-scope convention.
    """
    if intake.workspace.root.resolve() != vault_root.resolve():
        raise WikiError("Connector intake belongs to a different vault")

    def _applier(payload: dict[str, Any]) -> str:
        item = SourceItem(
            item_id=str(payload["item_id"]),
            title=str(payload["title"]),
            content=str(payload["content"]),
            url=str(payload["url"]),
            published_at=str(payload.get("published_at", "")),
        )
        instance = SourceInstance(
            name=str(payload["instance_name"]),
            kind=str(payload["instance_kind"]),
            target="",  # not needed for note-writing
            output_folder=str(payload["output_folder"]),
        )
        return ingest_item(
            item,
            instance,
            vault_root=vault_root,
            intake=intake,
            llm_client=llm_client,
            model=model,
            gap_store=gap_store,
        )

    return _applier


def propose_ingestion(
    queue: ReviewQueue, item: SourceItem, instance: SourceInstance, *, vault_root: Path
) -> ReviewProposal:
    """Route one fetched item into the review queue as a `SOURCE_INGESTION`
    proposal.

    `confidence=1.0`/`impact=Impact.LOW`, no `lane_override`: "this item
    exists and is newer than the stored cursor" has no LLM judgment call to
    be uncertain about (unlike tag-suggestion confidence), so under
    `DEFAULT_THRESHOLDS` this routes AUTO — the literal referent of the
    milestone's own Acceptance wording, "a connector proposal that would
    auto-apply." A stricter configured `AutonomyThresholds` still degrades
    it to SKIM/BLOCK correctly. Every item goes through `propose()`, never
    bypassing the queue, regardless of which lane it lands in.
    """
    validate_ordinary_write_path(instance.output_folder, vault_root)
    _validate_item(item)
    payload = {
        "item_id": item.item_id,
        "title": item.title,
        "content": item.content,
        "url": item.url,
        "published_at": item.published_at,
        "instance_name": instance.name,
        "instance_kind": instance.kind,
        "output_folder": instance.output_folder,
    }
    return queue.propose(
        ProposalKind.SOURCE_INGESTION,
        confidence=1.0,
        impact=Impact.LOW,
        summary=f"New {instance.kind} item from '{instance.name}': {item.title}",
        payload=payload,
    )


async def finalize_ingested_note(
    proposal: ReviewProposal,
    *,
    parser: VaultParser,
    store: VaultStore,
    vault_root: Path,
    event_bus: VaultEventBus,
) -> None:
    """After a `SOURCE_INGESTION` proposal is applied — AUTO-immediate (the
    scheduler job / `source run` CLI awaits this right after `propose()`),
    or later approved (`bot/handlers/autonomy.py`'s approve-later path) —
    parse the written note and publish `NoteCreatedEvent` on `event_bus`.

    Triggers the already-wired dedup (`_duplicate_review_subscriber`) and
    contradiction (`ContradictionDetector.on_note_changed`) subscribers
    exactly as `IncrementalWatchHandler` does for a file-watch-detected
    note — this is the "evaluated by the existing dedup/contradiction
    event-bus path" the milestone's Acceptance requires.

    No-op for any other kind/status — safe to call unconditionally after
    `propose()`/`approve()` without checking the kind first.
    """
    if proposal.kind is not ProposalKind.SOURCE_INGESTION:
        return
    if proposal.status is not ProposalStatus.APPLIED or not proposal.result:
        return

    note_path = vault_root / proposal.result
    if not note_path.exists():
        logger.warning(
            "Applied SOURCE_INGESTION proposal %s has no note at %s — skipping event publish",
            proposal.proposal_id,
            note_path,
        )
        return

    note = parser.parse_file(note_path)
    chunks = store.index_single_note(note, parser)
    await event_bus.publish(NoteCreatedEvent(path=note.path, note=note, chunks_indexed=chunks))


@dataclass(slots=True)
class ItemOutcome:
    """Observed run outcomes, not durable receipts or a cursor-recovery claim."""

    item_id: str
    proposal_id: str = ""
    status: str = "failed"
    path: str = ""
    source_ids: list[str] = field(default_factory=list)
    generation_status: str = "unknown"
    projection_status: str = "not-attempted"
    finalization_status: str = "not-attempted"
    indexed: bool | None = False
    event_published: bool | None = False
    error: str = ""


def _append_source_ids(outcome: ItemOutcome, record: SourceRecord) -> None:
    seen = set(outcome.source_ids)

    def visit(source: SourceRecord) -> None:
        if source.source_id in seen:
            return
        seen.add(source.source_id)
        outcome.source_ids.append(source.source_id)
        for dependency in source.dependencies:
            visit(dependency)

    visit(record)


def _describe_failed_capture(
    outcome: ItemOutcome, item: SourceItem, instance: SourceInstance, intake: SourceIntake
) -> None:
    """Read existing identities only; never repair or retry a failed operation."""
    _, revision, event_id = _item_identity(item, instance)
    try:
        record = intake.find_event("connector", event_id, revision, "item")
        if record is None:
            record = intake.find_event("connector", event_id, revision, "item-content")
        if record is not None:
            _append_source_ids(outcome, record)
    except WikiError:
        outcome.error += "; retained identities could not be verified"


def _describe_projection(outcome: ItemOutcome, intake: SourceIntake, path: Path) -> None:
    """Describe retained provenance after ordinary-destination authorization."""
    # Old APPLIED rows keep their old meaning. No migration or retroactive
    # source preservation is inferred from an ordinary legacy note.
    if re.fullmatch(r"i-[0-9a-f]{60}", path.stem) is None:
        outcome.projection_status = "legacy-unverified"
        return
    record = intake.read(path.stem)
    _append_source_ids(outcome, record)
    outcome.projection_status = "unverified"
    if not path.exists() or intake.workspace._read(path, "Connector projection") != record.content:
        raise WikiError("Connector projection differs from its preserved source")
    metadata, _ = _metadata(record.text or "")
    outcome.generation_status = (
        "failed" if metadata.get("generation_status") == "failed" else "succeeded"
    )
    outcome.projection_status = "verified"
    if outcome.generation_status == "failed":
        outcome.error = str(metadata.get("generation_error", "Generation failed"))


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one `run_connector_once()` call."""

    instance_name: str
    items_fetched: int
    items_ingested: int
    error: str = ""
    outcomes: list[ItemOutcome] = field(default_factory=list)


async def run_connector_once(
    instance: SourceInstance,
    *,
    source_store: SourceStore,
    review_queue: ReviewQueue,
    parser: VaultParser,
    store: VaultStore,
    vault_root: Path,
    intake: SourceIntake,
    event_bus: VaultEventBus,
    on_block: BlockNotifier | None = None,
) -> RunResult:
    """Fetch, propose, and finalize every new item for one connector
    instance, then advance its durable cursor and record a run summary.

    The full per-instance run, shared by the scheduler's per-instance
    `ScheduledJob` (`cli.py::bot`) and the `vaultmind source run <name>`
    CLI command — neither duplicates this orchestration.
    """
    vault_root = vault_root.resolve()
    validate_ordinary_write_path(instance.output_folder, vault_root)
    if intake.workspace.root.resolve() != vault_root:
        raise WikiError("Connector intake belongs to a different vault")
    started = datetime.now(UTC)
    connector = get_connector(instance.kind)
    state = source_store.get_state(instance.name)
    items_ingested = 0
    error = ""
    fetch_result = FetchResult()
    outcomes: list[ItemOutcome] = []

    try:
        fetch_result = await connector.fetch(instance, state)
    except Exception as exc:  # noqa: BLE001 - a connector's own fetch failure must not crash the run loop
        logger.exception("Connector fetch failed for instance %r", instance.name)
        error = str(exc)
        fetch_result = FetchResult()
    else:
        item_errors: list[str] = []
        for item in fetch_result.items:
            outcome = ItemOutcome(item_id=item.item_id)
            outcomes.append(outcome)
            try:
                proposal = propose_ingestion(review_queue, item, instance, vault_root=vault_root)
                outcome.proposal_id = proposal.proposal_id
                outcome.status = proposal.status.value
                if proposal.status is ProposalStatus.APPLIED:
                    outcome.path = proposal.result
                    if outcome.path:
                        path = validate_ordinary_write_path(outcome.path, vault_root)
                        try:
                            _describe_projection(outcome, intake, path)
                        except WikiError as exc:
                            outcome.projection_status = "unverified"
                            outcome.error = f"Projection verification failed: {exc}"
                            item_errors.append(f"{item.item_id}: {outcome.error}")
                elif proposal.status is ProposalStatus.FAILED:
                    outcome.error = proposal.result
                    _describe_failed_capture(outcome, item, instance, intake)
                await deliver_block(proposal, on_block)
                if proposal.status is ProposalStatus.APPLIED:
                    outcome.finalization_status = "attempted"
                await finalize_ingested_note(
                    proposal,
                    parser=parser,
                    store=store,
                    vault_root=vault_root,
                    event_bus=event_bus,
                )
                if proposal.status is ProposalStatus.APPLIED:
                    outcome.finalization_status = "returned-no-durable-receipt"
                    outcome.indexed = None
                    outcome.event_published = None
            except Exception as exc:  # noqa: BLE001 - one item's failure must not block the rest of the batch
                if not outcome.proposal_id and isinstance(exc, WikiError):
                    # A rejected original can put forbidden text in its own ID.
                    # Keep that text out of diagnostics and durable run history.
                    outcome.item_id = f"rejected-item-{len(outcomes)}"
                logger.exception(
                    "Ingestion failed for %s item %r from instance %r",
                    instance.kind,
                    outcome.item_id,
                    instance.name,
                )
                item_errors.append(f"{outcome.item_id}: {exc}")
                outcome.error = str(exc)
                if outcome.finalization_status == "attempted":
                    outcome.finalization_status = "failed"
                    # Finalization may already have indexed or published before
                    # raising. This layer has no durable receipt to prove either.
                    outcome.indexed = None
                    outcome.event_published = None
                continue
            if proposal.status is ProposalStatus.APPLIED:
                items_ingested += 1
        if item_errors:
            error = "; ".join(item_errors)

    advance_cursor(
        source_store,
        instance.name,
        next_cursor_id=fetch_result.next_cursor_id,
        next_cursor_at=fetch_result.next_cursor_at,
        next_etag=fetch_result.next_etag,
    )

    finished = datetime.now(UTC)
    source_store.record_run(
        RunSummary(
            instance_name=instance.name,
            started=started,
            finished=finished,
            items_fetched=len(fetch_result.items),
            items_ingested=items_ingested,
            error=error,
        )
    )
    return RunResult(
        instance_name=instance.name,
        items_fetched=len(fetch_result.items),
        items_ingested=items_ingested,
        outcomes=outcomes,
        error=error,
    )


__all__ = [
    "AUTO_INGESTED_AUTHORITY",
    "RunResult",
    "ItemOutcome",
    "finalize_ingested_note",
    "ingest_item",
    "make_applier",
    "propose_ingestion",
    "run_connector_once",
]
