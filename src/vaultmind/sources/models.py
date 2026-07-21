"""Data models for the source connector framework (M8).

`SourceInstance` is one configured connector (from `config/sources.toml`).
`ConnectorState` is the durable per-instance cursor persisted in `sources.db`.
`SourceItem`/`FetchResult` are the connector-agnostic shape every connector's
`fetch()` callable returns, consumed by `sources/pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One fetched item from a connector, prior to distillation.

    `item_id` is the connector-specific stable identifier used for cursor
    advancement (RSS guid, YouTube video ID, GitHub commit SHA/PR number).
    `published_at` is ISO-8601; connectors without a reliable per-item
    timestamp (e.g. some RSS feeds) may leave it empty — cursor advancement
    then relies solely on `item_id` ordering the connector itself guarantees.
    """

    item_id: str
    title: str
    content: str
    url: str
    published_at: str = ""


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of one connector fetch call.

    `items` are every item newer than the cursor passed in (the connector is
    responsible for filtering against `ConnectorState.last_seen_id`/`etag` —
    the pipeline never re-filters). `next_cursor_id`/`next_etag`, if set,
    replace the stored cursor after a successful run; `None` leaves the
    stored cursor unchanged (e.g. a fetch that found zero new items).
    """

    items: list[SourceItem] = field(default_factory=list)
    next_cursor_id: str | None = None
    next_etag: str | None = None


# A connector's fetch callable: (instance, prior cursor state) -> FetchResult.
type ConnectorFetch = Callable[["SourceInstance", "ConnectorState"], Awaitable[FetchResult]]


@dataclass(frozen=True, slots=True)
class ConnectorDefinition:
    """One entry in the closed connector registry (`registry.REGISTRY`)."""

    kind: str
    fetch: Callable[[SourceInstance, ConnectorState], Awaitable[FetchResult]]
    description: str


@dataclass(frozen=True, slots=True)
class SourceInstance:
    """One configured connector instance, loaded from `config/sources.toml`."""

    name: str
    kind: str
    target: str
    enabled: bool = False
    interval_hours: int = 24
    output_folder: str = "00-inbox/sources"
    options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One bounded run-history entry for a connector instance."""

    instance_name: str
    started: datetime
    finished: datetime
    items_fetched: int
    items_ingested: int
    error: str = ""


@dataclass(frozen=True, slots=True)
class ConnectorState:
    """Durable per-instance cursor state, persisted in `sources.db`."""

    instance_name: str
    last_seen_id: str = ""
    etag: str = ""
    last_run: datetime | None = None
    run_count: int = 0


def _row_to_run_summary(row: Any) -> RunSummary:
    return RunSummary(
        instance_name=row["instance_name"],
        started=datetime.fromisoformat(row["started"]),
        finished=datetime.fromisoformat(row["finished"]),
        items_fetched=row["items_fetched"],
        items_ingested=row["items_ingested"],
        error=row["error"] or "",
    )


__all__ = [
    "ConnectorDefinition",
    "ConnectorFetch",
    "ConnectorState",
    "FetchResult",
    "RunSummary",
    "SourceInstance",
    "SourceItem",
    "_row_to_run_summary",
]
