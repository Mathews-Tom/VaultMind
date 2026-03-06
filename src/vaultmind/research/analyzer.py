"""LLM-powered comparative analysis of research sources."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from vaultmind.llm.client import LLMClient, Message

logger = logging.getLogger(__name__)

_ANALYSIS_SYSTEM_PROMPT = """\
You are a research analyst. Analyze the provided sources and produce a comparative analysis.

Output valid JSON with these keys:
- "summary": 2-3 sentence overview of all sources
- "key_themes": list of strings identifying major themes across sources
- "comparative_insights": string comparing/contrasting perspectives across sources
- "gaps": string identifying gaps in coverage or missing perspectives
- "recommendations": string with actionable recommendations

When referencing specific sources, use [[Source Title]] wikilink syntax.
Focus on cross-source patterns, not per-source summaries.
"""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    summary: str
    key_themes: list[str] = field(default_factory=list)
    comparative_insights: str = ""
    gaps: str = ""
    recommendations: str = ""


async def analyze_sources(
    sources: list[dict[str, str]],
    query: str,
    llm_client: LLMClient,
    model: str,
) -> AnalysisResult:
    """Send all source content to LLM for comparative analysis."""
    # Build source context (truncate long transcripts)
    source_blocks: list[str] = []
    for src in sources:
        title = src["title"]
        content = src["content"][:8000]  # Cap per-source to fit context
        source_blocks.append(f"## [[{title}]]\n\n{content}")

    combined = "\n\n---\n\n".join(source_blocks)

    user_msg = (
        f"Research query: {query}\n\n"
        f"Sources ({len(sources)} total):\n\n{combined}\n\n"
        "Produce a comparative analysis as JSON."
    )

    messages = [Message(role="user", content=user_msg)]

    response = await asyncio.to_thread(
        llm_client.complete,
        messages=messages,
        model=model,
        max_tokens=4096,
        system=_ANALYSIS_SYSTEM_PROMPT,
    )

    return _parse_analysis(response.text)


def _parse_analysis(text: str) -> AnalysisResult:
    """Parse LLM response into AnalysisResult."""
    # Try JSON extraction
    try:
        # Handle markdown code blocks
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

        data = json.loads(cleaned)
        return AnalysisResult(
            summary=data.get("summary", ""),
            key_themes=data.get("key_themes", []),
            comparative_insights=data.get("comparative_insights", ""),
            gaps=data.get("gaps", ""),
            recommendations=data.get("recommendations", ""),
        )
    except (json.JSONDecodeError, IndexError):
        logger.warning("Could not parse LLM analysis as JSON, using raw text")
        return AnalysisResult(summary=text[:2000])
