"""LLM-powered comparative analysis of research sources."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from vaultmind.llm.client import LLMClient, Message

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
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
ANALYSIS_MAX_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    summary: str
    key_themes: list[str] = field(default_factory=list)
    comparative_insights: str = ""
    gaps: str = ""
    recommendations: str = ""


def render_analysis_request(sources: list[dict[str, str]], query: str) -> str:
    """Render the exact model input; full originals are retained separately."""
    source_blocks: list[str] = []
    for src in sources:
        title = src["title"]
        content = src["content"][:8000]  # Cap per-source to fit context
        source_blocks.append(f"## [[{title}]]\n\n{content}")

    combined = "\n\n---\n\n".join(source_blocks)

    return (
        f"Research query: {query}\n\n"
        f"Sources ({len(sources)} total):\n\n{combined}\n\n"
        "Produce a comparative analysis as JSON."
    )


def analyze_sources(
    request_text: str,
    llm_client: LLMClient,
    model: str,
    *,
    system: str = ANALYSIS_SYSTEM_PROMPT,
    max_tokens: int = ANALYSIS_MAX_TOKENS,
) -> AnalysisResult:
    """Analyze an already retained request without reading or writing files."""
    response = llm_client.complete(
        messages=[Message(role="user", content=request_text)],
        model=model,
        max_tokens=max_tokens,
        system=system,
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
        if not isinstance(data, dict):
            raise ValueError("Analysis must be an object")
        text_keys = ("summary", "comparative_insights", "gaps", "recommendations")
        if any(not isinstance(data.get(key, ""), str) for key in text_keys):
            raise ValueError("Analysis fields must be text")
        themes = data.get("key_themes", [])
        if not isinstance(themes, list) or any(not isinstance(theme, str) for theme in themes):
            raise ValueError("Analysis themes must be a text list")
        return AnalysisResult(
            summary=data.get("summary", ""),
            key_themes=themes,
            comparative_insights=data.get("comparative_insights", ""),
            gaps=data.get("gaps", ""),
            recommendations=data.get("recommendations", ""),
        )
    except (ValueError, IndexError):
        logger.warning("Could not parse LLM analysis as JSON, using raw text")
        return AnalysisResult(summary=text[:2000])
