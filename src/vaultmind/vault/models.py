"""Data models for vault notes and chunks."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — Pydantic needs Path at runtime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class NoteType(StrEnum):
    """Note types following Zettelkasten conventions."""

    FLEETING = "fleeting"
    LITERATURE = "literature"
    PERMANENT = "permanent"
    DAILY = "daily"
    PROJECT = "project"
    AREA = "area"
    PERSON = "person"
    CONCEPT = "concept"
    TEMPLATE = "template"


class Note(BaseModel):
    """Represents a parsed Obsidian markdown note."""

    path: Path
    title: str
    content: str
    note_type: NoteType = NoteType.FLEETING
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    status: str = "active"
    source: str = "manual"
    created: datetime = Field(default_factory=datetime.now)
    modified: datetime = Field(default_factory=datetime.now)
    frontmatter: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def wikilinks(self) -> list[str]:
        """Extract [[wikilinks]] from content."""
        import re

        return re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", self.content)

    @property
    def relative_path(self) -> str:
        """Path relative to vault root (for use as IDs)."""
        return str(self.path)

    def body_without_frontmatter(self) -> str:
        """Return content body without YAML frontmatter."""
        if self.content.startswith("---"):
            parts = self.content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return self.content


class NoteChunk(BaseModel):
    """A semantically meaningful chunk of a note, ready for embedding."""

    note_path: str
    note_title: str
    chunk_idx: int
    heading: str = ""
    content: str
    note_type: NoteType = NoteType.FLEETING
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    created: str = ""
    modified: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.note_path}::{self.chunk_idx}"

    def to_chroma_metadata(self) -> dict[str, str | int]:
        """Convert to ChromaDB-compatible metadata dict."""
        return {
            "note_path": self.note_path,
            "note_title": self.note_title,
            "note_type": self.note_type.value,
            "heading": self.heading,
            "tags": ",".join(self.tags),
            "entities": ",".join(self.entities),
            "created": self.created,
            "modified": self.modified,
            "chunk_idx": self.chunk_idx,
        }
