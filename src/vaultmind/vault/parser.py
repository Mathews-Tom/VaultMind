"""Vault parser — reads markdown files, extracts frontmatter, and produces heading-aware chunks."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.vault.models import Note, NoteChunk, NoteType

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.config import VaultConfig

logger = logging.getLogger(__name__)

# Heading pattern: matches ## Heading but not code blocks
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]*)", re.MULTILINE)


class VaultParser:
    """Parses an Obsidian vault into structured Note objects."""

    def __init__(self, config: VaultConfig) -> None:
        self.config = config
        self.vault_root = config.path

    def iter_notes(self) -> list[Note]:
        """Iterate all markdown files in the vault, yielding parsed Notes."""
        notes: list[Note] = []
        for md_file in self.vault_root.rglob("*.md"):
            # Skip excluded folders
            rel = md_file.relative_to(self.vault_root)
            if any(part in self.config.excluded_folders for part in rel.parts):
                continue
            try:
                note = self.parse_file(md_file)
                notes.append(note)
            except Exception:
                logger.exception("Failed to parse %s", md_file)
        logger.info("Parsed %d notes from vault", len(notes))
        return notes

    def parse_file(self, filepath: Path) -> Note:
        """Parse a single markdown file into a Note."""
        with open(filepath, encoding="utf-8") as f:
            post = frontmatter.load(f)

        meta = post.metadata or {}
        rel_path = filepath.relative_to(self.vault_root)

        # Infer note type from frontmatter or folder
        note_type = self._infer_type(meta, rel_path)

        # Extract inline tags from content (merge with frontmatter tags)
        inline_tags = TAG_PATTERN.findall(post.content)
        fm_tags = meta.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        all_tags = list(set(fm_tags + inline_tags))

        return Note(
            path=rel_path,
            title=meta.get("title", filepath.stem.replace("-", " ").replace("_", " ")),
            content=post.content,
            note_type=note_type,
            tags=all_tags,
            entities=meta.get("entities", []),
            related=meta.get("related", []),
            status=meta.get("status", "active"),
            source=meta.get("source", "manual"),
            created=self._parse_date(meta.get("created")),
            modified=self._parse_date(meta.get("modified", datetime.now())),
            frontmatter=meta,
        )

    def chunk_note(self, note: Note, max_tokens: int = 500) -> list[NoteChunk]:
        """Split a note into heading-aware chunks.

        Strategy:
        1. Split by headings (## level sections)
        2. If a section exceeds max_tokens, split by paragraphs
        3. Each chunk preserves the heading context
        """
        body = note.body_without_frontmatter()
        if not body.strip():
            return []

        sections = self._split_by_headings(body)
        chunks: list[NoteChunk] = []

        for heading, content in sections:
            content = content.strip()
            if not content:
                continue

            # Rough token estimate: ~4 chars per token
            estimated_tokens = len(content) // 4

            if estimated_tokens <= max_tokens:
                chunks.append(
                    NoteChunk(
                        note_path=str(note.path),
                        note_title=note.title,
                        chunk_idx=len(chunks),
                        heading=heading,
                        content=f"{heading}\n\n{content}" if heading else content,
                        note_type=note.note_type,
                        tags=note.tags,
                        entities=note.entities,
                        created=note.created.isoformat(),
                        modified=note.modified.isoformat(),
                    )
                )
            else:
                # Split large sections by paragraphs
                paragraphs = content.split("\n\n")
                current_chunk = ""

                for para in paragraphs:
                    if len((current_chunk + para).encode()) // 4 > max_tokens and current_chunk:
                        chunks.append(
                            NoteChunk(
                                note_path=str(note.path),
                                note_title=note.title,
                                chunk_idx=len(chunks),
                                heading=heading,
                                content=(
                                    f"{heading}\n\n{current_chunk}" if heading else current_chunk
                                ),
                                note_type=note.note_type,
                                tags=note.tags,
                                entities=note.entities,
                                created=note.created.isoformat(),
                                modified=note.modified.isoformat(),
                            )
                        )
                        current_chunk = para
                    else:
                        current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

                if current_chunk.strip():
                    chunks.append(
                        NoteChunk(
                            note_path=str(note.path),
                            note_title=note.title,
                            chunk_idx=len(chunks),
                            heading=heading,
                            content=f"{heading}\n\n{current_chunk}" if heading else current_chunk,
                            note_type=note.note_type,
                            tags=note.tags,
                            entities=note.entities,
                            created=note.created.isoformat(),
                            modified=note.modified.isoformat(),
                        )
                    )

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split text into (heading, content) pairs."""
        matches = list(HEADING_PATTERN.finditer(text))

        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []

        # Content before the first heading
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((heading, content))

        return sections

    def _infer_type(self, meta: dict[str, Any], rel_path: Path) -> NoteType:
        """Infer note type from frontmatter or folder structure."""
        if "type" in meta:
            try:
                return NoteType(meta["type"])
            except ValueError:
                pass

        # Infer from folder
        folder_map = {
            "00-inbox": NoteType.FLEETING,
            "01-daily": NoteType.DAILY,
            "02-projects": NoteType.PROJECT,
            "04-resources": NoteType.LITERATURE,
            "06-templates": NoteType.TEMPLATE,
        }
        first_folder = rel_path.parts[0] if rel_path.parts else ""
        return folder_map.get(first_folder, NoteType.FLEETING)

    def _parse_date(self, value: str | datetime | None) -> datetime:
        """Parse a date from frontmatter (string or datetime)."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return datetime.now()
