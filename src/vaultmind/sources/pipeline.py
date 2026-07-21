"""Connector-agnostic ingestion pipeline (DEVELOPMENT_PLAN.md M8).

Every fetched `SourceItem` is routed through M4-style distillation (stamped
with M2's `AUTO_INGESTED_AUTHORITY`), M7's review queue (`propose_ingestion`
— never bypassed, even for a proposal that ends up AUTO-lane), and — once
applied — the existing dedup/contradiction event-bus path
(`finalize_ingested_note`). `run_connector_once` is the full per-instance
orchestration, shared by the scheduler job wiring (`cli.py::bot`) and the
`vaultmind source run <name>` CLI command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vaultmind.pipeline.distill import distill_conversation, mint_gap_for_unresolved
from vaultmind.services.review_queue import Impact, ProposalKind, ProposalStatus
from vaultmind.sources import (
    connectors as _connectors,  # noqa: F401 - registers connectors as import side effect
)
from vaultmind.sources.models import FetchResult, RunSummary, SourceInstance, SourceItem
from vaultmind.sources.registry import get_connector
from vaultmind.sources.store import advance_cursor
from vaultmind.vault.events import NoteCreatedEvent
from vaultmind.vault.ingest import _sanitize_filename
from vaultmind.vault.security import validate_vault_path

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
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


def _write_plain_note(item: SourceItem, instance: SourceInstance, vault_root: Path) -> str:
    """Fallback note-write when distillation fails (no LLM key, LLM error,
    or an empty extracted question) — a plain frontmatter'd note mirroring
    `vault/ingest.py::create_vault_note`'s non-LLM shape. An ingested item
    is never silently dropped by an LLM outage."""
    folder = vault_root / instance.output_folder
    folder.mkdir(parents=True, exist_ok=True)

    base_name = _sanitize_filename(item.title) or "untitled"
    note_path = folder / f"{base_name}.md"
    counter = 1
    while note_path.exists():
        note_path = folder / f"{base_name}-{counter}.md"
        counter += 1
    validate_vault_path(str(note_path), vault_root)

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = f"""\
---
title: "{item.title}"
source: "{item.url}"
source_type: {instance.kind}
source_instance: {instance.name}
tags: [source-ingested]
created: {now}
authority: {AUTO_INGESTED_AUTHORITY}
status: active
---

# {item.title}

**Source:** {item.url}
**Connector:** {instance.kind} ({instance.name})

---

{item.content}
"""
    note_path.write_text(content, encoding="utf-8")
    logger.info("Wrote fallback plain note to %s (distillation unavailable)", note_path)
    return str(note_path.relative_to(vault_root))


def ingest_item(
    item: SourceItem,
    instance: SourceInstance,
    *,
    vault_root: Path,
    llm_client: LLMClient,
    model: str,
    gap_store: object = None,
) -> str:
    """Write one connector-fetched item into the vault as a note.

    Distills via `distill_conversation()` (M4-style: the item is framed as
    a single synthetic conversation turn), stamped at `AUTO_INGESTED_AUTHORITY`
    — falls back to `_write_plain_note` on any distillation failure. This is
    the sole responsibility of the sync `Applier` registered for
    `ProposalKind.SOURCE_INGESTION` (`make_applier` below): write-only,
    matching `ReviewQueue`'s strictly-synchronous `Applier` contract.
    Event-bus publication happens in the async caller (`finalize_ingested_note`).

    Returns the created note's vault-relative path.
    """
    turns = [{"user": f"{item.title}\n\n{item.content}".strip(), "assistant": ""}]
    result = distill_conversation(
        turns=turns,
        llm_client=llm_client,
        model=model,
        vault_root=vault_root,
        output_folder=instance.output_folder,
        source_ref=item.url,
        occurred_at=item.published_at or None,
        authority=AUTO_INGESTED_AUTHORITY,
    )
    if result.success:
        mint_gap_for_unresolved(result, gap_store, item.url)
        return result.output_path

    logger.warning(
        "Distillation failed for %s item %r from %r (%s) — writing a plain note",
        instance.kind,
        item.title,
        instance.name,
        result.error,
    )
    return _write_plain_note(item, instance, vault_root)


def make_applier(
    *,
    vault_root: Path,
    llm_client: LLMClient,
    model: str,
    gap_store: object = None,
) -> Applier:
    """Build the sync `Applier` registered for `ProposalKind.SOURCE_INGESTION`.

    Bound via closure to the collaborators `ingest_item` needs, matching
    `cli.py::bot`'s existing `_maturation_applier`/
    `ContradictionDetector._resolution_applier` closure-over-scope convention.
    """

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
            llm_client=llm_client,
            model=model,
            gap_store=gap_store,
        )

    return _applier


def propose_ingestion(
    queue: ReviewQueue, item: SourceItem, instance: SourceInstance
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


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one `run_connector_once()` call."""

    instance_name: str
    items_fetched: int
    items_ingested: int
    error: str = ""


async def run_connector_once(
    instance: SourceInstance,
    *,
    source_store: SourceStore,
    review_queue: ReviewQueue,
    parser: VaultParser,
    store: VaultStore,
    vault_root: Path,
    event_bus: VaultEventBus,
) -> RunResult:
    """Fetch, propose, and finalize every new item for one connector
    instance, then advance its durable cursor and record a run summary.

    The full per-instance run, shared by the scheduler's per-instance
    `ScheduledJob` (`cli.py::bot`) and the `vaultmind source run <name>`
    CLI command — neither duplicates this orchestration.
    """
    started = datetime.now(UTC)
    connector = get_connector(instance.kind)
    state = source_store.get_state(instance.name)
    items_ingested = 0
    error = ""
    fetch_result = FetchResult()

    try:
        fetch_result = await connector.fetch(instance, state)
    except Exception as exc:  # noqa: BLE001 - a connector's own fetch failure must not crash the run loop
        logger.exception("Connector fetch failed for instance %r", instance.name)
        error = str(exc)
        fetch_result = FetchResult()
    else:
        item_errors: list[str] = []
        for item in fetch_result.items:
            try:
                proposal = propose_ingestion(review_queue, item, instance)
                await finalize_ingested_note(
                    proposal,
                    parser=parser,
                    store=store,
                    vault_root=vault_root,
                    event_bus=event_bus,
                )
            except Exception as exc:  # noqa: BLE001 - one item's failure must not block the rest of the batch
                logger.exception(
                    "Ingestion failed for %s item %r from instance %r",
                    instance.kind,
                    item.title,
                    instance.name,
                )
                item_errors.append(f"{item.item_id}: {exc}")
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
        error=error,
    )


__all__ = [
    "AUTO_INGESTED_AUTHORITY",
    "RunResult",
    "finalize_ingested_note",
    "ingest_item",
    "make_applier",
    "propose_ingestion",
    "run_connector_once",
]
