"""Episodic memory data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003 — used in dataclass field defaults
from enum import StrEnum


class OutcomeStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class MemoryHorizon(StrEnum):
    """Temporal classification for episodic memory retrieval."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class Episode:
    episode_id: str
    decision: str
    context: str
    outcome: str
    outcome_status: OutcomeStatus
    lessons: list[str]
    entities: list[str]  # linked graph entities
    source_notes: list[str]  # vault note paths
    created: datetime
    resolved: datetime | None = None
    tags: list[str] = field(default_factory=list)
    memory_horizon: MemoryHorizon = MemoryHorizon.SHORT_TERM
