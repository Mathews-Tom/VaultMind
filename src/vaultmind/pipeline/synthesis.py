"""LLM synthesis for Zettelkasten maturation.

Given a cluster of source notes, synthesizes a permanent Zettelkasten note
with validated frontmatter and wikilinks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import frontmatter

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.llm.client import LLMClient
    from vaultmind.pipeline.clustering import NoteCluster

logger = logging.getLogger(__name__)

WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

SYNTHESIS_PROMPT = """\
You are synthesizing a permanent Zettelkasten note from a cluster of source notes.

Source notes:
{source_notes_content}

Shared entities detected: {entity_list}

Instructions:
1. Write a single atomic permanent note (300-500 words) capturing the core insight
2. Use [[wikilink]] syntax to reference source notes by filename (without .md extension)
3. Output YAML frontmatter: type: permanent, tags: [...], entities: [...], created: {today}
4. Do not copy sentences verbatim — synthesize
5. End with a ## Sources section listing the source note paths

Output the complete note content including frontmatter fenced by ---."""


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Result of a synthesis attempt."""

    success: bool
    content: str
    output_path: str
    error: str


def synthesize_cluster(
    cluster: NoteCluster,
    vault_root: Path,
    inbox_folder: str,
    llm: LLMClient,
    model: str,
    max_tokens: int = 1500,
) -> SynthesisResult:
    """Synthesize a permanent note from a cluster of source notes.

    Reads source note content, calls LLM, validates output, writes to inbox.

    Returns:
        SynthesisResult with success status and output path or error.
    """
    from vaultmind.llm.client import Message

    # Read source notes
    source_contents: list[str] = []
    for note_path in cluster.note_paths:
        full_path = vault_root / note_path
        if full_path.exists():
            text = full_path.read_text(encoding="utf-8")
            source_contents.append(f"### {note_path}\n{text}")

    if not source_contents:
        return SynthesisResult(
            success=False,
            content="",
            output_path="",
            error="No source notes found on disk",
        )

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    prompt = SYNTHESIS_PROMPT.format(
        source_notes_content="\n\n---\n\n".join(source_contents),
        entity_list=cluster.top_entity,
        today=today,
    )

    response = llm.complete(
        messages=[Message(role="user", content=prompt)],
        model=model,
        max_tokens=max_tokens,
    )

    content = response.text.strip()

    # Validate
    validation_error = _validate_synthesis(content, vault_root)
    if validation_error:
        logger.error("Synthesis validation failed: %s", validation_error)
        return SynthesisResult(
            success=False,
            content=content,
            output_path="",
            error=validation_error,
        )

    # Write to inbox
    inbox_path = vault_root / inbox_folder
    inbox_path.mkdir(parents=True, exist_ok=True)

    slug = cluster.top_entity.lower().replace(" ", "-")[:50]
    filename = f"synthesis-{slug}-{today}.md"
    output_path = inbox_path / filename

    # Avoid overwriting
    counter = 1
    while output_path.exists():
        filename = f"synthesis-{slug}-{today}-{counter}.md"
        output_path = inbox_path / filename
        counter += 1

    output_path.write_text(content, encoding="utf-8")
    rel_path = str(output_path.relative_to(vault_root))
    logger.info("Wrote synthesized note to %s", rel_path)

    return SynthesisResult(
        success=True,
        content=content,
        output_path=rel_path,
        error="",
    )


def _validate_synthesis(content: str, vault_root: Path) -> str:
    """Validate synthesized content. Returns error string or empty on success."""
    # Check frontmatter parses
    try:
        post = frontmatter.loads(content)
        if not post.metadata:
            return "No YAML frontmatter found"
    except Exception as e:
        return f"YAML frontmatter parse error: {e}"

    # Check wikilinks resolve
    wikilinks = WIKILINK_PATTERN.findall(content)
    unresolved: list[str] = []
    for link in wikilinks:
        # Search vault for matching file
        matches = list(vault_root.rglob(f"{link}.md"))
        if not matches:
            unresolved.append(link)

    if unresolved:
        return f"Unresolved wikilinks: {', '.join(unresolved)}"

    return ""
