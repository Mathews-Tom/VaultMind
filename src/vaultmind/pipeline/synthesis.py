"""Filesystem-free generation for retained Zettelkasten source snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import frontmatter

from vaultmind.llm.client import Message
from vaultmind.wiki.workspace import _metadata, _safe_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vaultmind.llm.client import LLMClient

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
class SynthesisGeneration:
    """Generated content, not a persistence or approval result."""

    success: bool
    content: str = ""
    error: str = ""


def render_synthesis_prompt(
    source_notes: Sequence[tuple[str, str]], entity: str, today: str
) -> str:
    """Render the complete selected source texts without filesystem access."""
    return SYNTHESIS_PROMPT.format(
        source_notes_content="\n\n---\n\n".join(
            f"### {path}\n{text}" for path, text in source_notes
        ),
        entity_list=entity,
        today=today,
    )


def generate_synthesis(
    prompt_text: str,
    llm: LLMClient,
    model: str,
    max_tokens: int = 1500,
) -> SynthesisGeneration:
    """Generate and validate a permanent-note-shaped artifact only in memory."""
    if not prompt_text.strip():
        return SynthesisGeneration(False, error="No source notes supplied")
    try:
        response = llm.complete(
            messages=[Message(role="user", content=prompt_text)],
            model=model,
            max_tokens=max_tokens,
        )
    except Exception:
        return SynthesisGeneration(False, error="LLM call failed")
    content = response.text.strip()
    try:
        _metadata(_safe_text(content.encode("utf-8"), "Generated synthesis"))
        post = frontmatter.loads(content)
        if not post.metadata:
            return SynthesisGeneration(False, content, "No YAML frontmatter found")
        post.metadata["type"] = "permanent"
        post.metadata["status"] = "unapproved"
        post.metadata["approved"] = False
        post.metadata["source"] = "synthesized"
        rendered = str(frontmatter.dumps(post))
    except ValueError as exc:
        return SynthesisGeneration(False, content, f"Invalid synthesis: {exc}")
    return SynthesisGeneration(True, rendered)
