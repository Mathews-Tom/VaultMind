"""Research pipeline — search, fetch, analyze, and create vault notes."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vaultmind.research.analyzer import analyze_sources
from vaultmind.research.searcher import search_youtube
from vaultmind.vault.security import validate_vault_path

if TYPE_CHECKING:
    from vaultmind.indexer import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.parser import VaultParser

_DEFAULT_MODEL = "gpt-4.1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Pipeline configuration."""

    max_results: int = 5
    output_folder: str = "research"
    youtube_language: str = "en"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    query: str
    sources_created: list[Path] = field(default_factory=list)
    summary_path: Path = field(default_factory=lambda: Path())
    analysis_summary: str = ""


def _sanitize_dirname(name: str) -> str:
    """Create a filesystem-safe directory name."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name.lower())
    name = re.sub(r"\s+", "-", name)
    return name[:80].strip("-_")


def _create_source_note(
    title: str,
    content: str,
    url: str,
    query: str,
    output_dir: Path,
    vault_root: Path,
) -> Path:
    """Create an individual source note."""
    filename = re.sub(r'[<>:"/\\|?*]', "_", title)[:80] + ".md"
    note_path = output_dir / filename
    validate_vault_path(str(note_path), vault_root)

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    note_content = f"""\
---
title: "{title}"
source: "{url}"
source_type: youtube
tags: [research, {query}]
created: {now}
---

# {title}

**Source:** {url}

---

{content}
"""
    note_path.write_text(note_content, encoding="utf-8")
    return note_path


async def run_research_pipeline(
    query: str,
    vault_root: Path,
    llm_client: LLMClient,
    store: VaultStore,
    parser: VaultParser,
    config: ResearchConfig | None = None,
    model: str = _DEFAULT_MODEL,
) -> PipelineResult:
    """Execute the full research pipeline.

    1. Search YouTube
    2. Fetch transcripts
    3. Create source notes
    4. LLM analysis
    5. Create summary note
    6. Index everything
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    if config is None:
        config = ResearchConfig()

    # 1. Search
    logger.info("Searching YouTube for: %s", query)
    results = await search_youtube(query, max_results=config.max_results)
    if not results:
        return PipelineResult(query=query, analysis_summary="No results found.")

    # 2. Fetch transcripts
    sources: list[dict[str, str]] = []
    for result in results:
        try:
            segments = await asyncio.to_thread(
                YouTubeTranscriptApi.get_transcript,
                result.video_id,
                languages=[config.youtube_language],
            )
            transcript = " ".join(s["text"] for s in segments)
            sources.append(
                {
                    "title": result.title,
                    "content": transcript,
                    "url": result.url,
                    "video_id": result.video_id,
                }
            )
            logger.info("Fetched transcript: %s", result.title)
        except Exception:
            logger.warning("Could not fetch transcript for: %s", result.title)

    if not sources:
        return PipelineResult(query=query, analysis_summary="No transcripts available.")

    # 3. Create source notes
    dir_name = _sanitize_dirname(query)
    output_dir = vault_root / config.output_folder / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths: list[Path] = []
    for src in sources:
        path = _create_source_note(
            title=src["title"],
            content=src["content"],
            url=src["url"],
            query=query,
            output_dir=output_dir,
            vault_root=vault_root,
        )
        source_paths.append(path)

    # 4. LLM analysis
    logger.info("Running LLM analysis across %d sources", len(sources))
    analysis = await analyze_sources(
        sources=sources,
        query=query,
        llm_client=llm_client,
        model=model,
    )

    # 5. Create summary note
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_links = "\n".join(f"- [[{s['title']}]]" for s in sources)
    themes_list = "\n".join(f"- {t}" for t in analysis.key_themes) if analysis.key_themes else "N/A"

    summary_content = f"""\
---
title: "Research: {query}"
tags: [research-summary]
sources: [{", ".join(s["title"] for s in sources)}]
created: {now}
---

# Research: {query}

## Sources Analyzed

{source_links}

## Summary

{analysis.summary}

## Key Themes

{themes_list}

## Comparative Insights

{analysis.comparative_insights}

## Gaps & Opportunities

{analysis.gaps}

## Recommendations

{analysis.recommendations}
"""
    summary_path = output_dir / "summary.md"
    validate_vault_path(str(summary_path), vault_root)
    summary_path.write_text(summary_content, encoding="utf-8")

    # 6. Index all notes
    all_paths = [*source_paths, summary_path]
    for path in all_paths:
        try:
            note = await asyncio.to_thread(parser.parse_file, path)
            await asyncio.to_thread(store.index_single_note, note, parser)
        except Exception:
            logger.warning("Failed to index: %s", path)

    logger.info(
        "Research pipeline complete: %d sources, summary at %s",
        len(sources),
        summary_path,
    )

    return PipelineResult(
        query=query,
        sources_created=source_paths,
        summary_path=summary_path,
        analysis_summary=analysis.summary,
    )
