"""Research sources and analysis retained separately from discovery projections."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.knowledge.intake import CaptureResult, SourceIntake, StageDisposition
from vaultmind.pipeline.distill import validate_derivative_parent
from vaultmind.research.analyzer import (
    ANALYSIS_MAX_TOKENS,
    ANALYSIS_SYSTEM_PROMPT,
    AnalysisResult,
    analyze_sources,
    render_analysis_request,
)
from vaultmind.research.searcher import search_youtube
from vaultmind.vault.security import validate_ordinary_write_path
from vaultmind.wiki.workspace import WikiError, _metadata, _safe_text

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.indexer import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.parser import VaultParser

_DEFAULT_MODEL = "gpt-4.1"


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    max_results: int = 5
    output_folder: str = "research"
    youtube_language: str = "en"


@dataclass(slots=True)
class PipelineResult:
    query: str
    sources_created: list[Path] = field(default_factory=list)
    summary_path: Path | None = None
    analysis_summary: str = ""
    captures: list[CaptureResult] = field(default_factory=list)
    original_source_ids: list[str] = field(default_factory=list)
    request_source_id: str | None = None
    derivative_source_id: str | None = None
    dispositions: list[StageDisposition] = field(default_factory=list)
    error: str = ""


def _sanitize_dirname(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name.lower())
    name = re.sub(r"\s+", "-", name)
    return name[:80].strip("-_") or "research"


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _render_source_note(
    title: str, content: str, url: str, query: str, capture: CaptureResult, *, created: str
) -> str:
    """Render a discovery view, not a source writer or an approval."""
    return str(
        frontmatter.dumps(
            frontmatter.Post(
                f"# {title}\n\n**Source:** {url}\n\n{content}",
                title=title,
                source=url,
                source_type="youtube",
                tags=["research", query],
                authority=1,
                created=created,
                source_id=capture.source_id,
                source_path=capture.path,
                approved=False,
            )
        )
    )


def _render_summary(
    query: str, analysis: AnalysisResult, source_ids: list[str], *, created: str
) -> str:
    themes = "\n".join(f"- {theme}" for theme in analysis.key_themes) or "N/A"
    links = "\n".join(f"- [[_wiki/sources/{source_id}]]" for source_id in source_ids)
    body = (
        f"# Research: {query}\n\n## Sources Analyzed\n\n{links}\n\n"
        f"## Summary\n\n{analysis.summary}\n\n## Key Themes\n\n{themes}\n\n"
        f"## Comparative Insights\n\n{analysis.comparative_insights}\n\n"
        f"## Gaps & Opportunities\n\n{analysis.gaps}\n\n"
        f"## Recommendations\n\n{analysis.recommendations}\n"
    )
    return str(
        frontmatter.dumps(
            frontmatter.Post(
                body,
                title=f"Research: {query}",
                tags=["research-summary"],
                authority=3,
                created=created,
                source_ids=source_ids,
                approved=False,
            )
        )
    )


def _project_source(
    intake: SourceIntake, capture: CaptureResult, content: str, folder: str
) -> StageDisposition:
    relative = f"{folder}/{capture.source_id}.md"
    validate_ordinary_write_path(relative, intake.workspace.root)

    def project() -> str:
        intake.read(capture.source_id)
        path = validate_ordinary_write_path(relative, intake.workspace.root)
        if path.exists():
            raise WikiError("Research projection destination already exists")
        intake.workspace._atomic_bytes(path, content.encode("utf-8"))
        return relative

    return capture.run_stage("projection", project)


async def run_research_pipeline(
    query: str,
    vault_root: Path,
    llm_client: LLMClient,
    store: VaultStore,
    parser: VaultParser,
    config: ResearchConfig | None = None,
    model: str = _DEFAULT_MODEL,
    *,
    intake: SourceIntake,
) -> PipelineResult:
    """Retain fetched representations and exact requests before invoking a model.

    Ordinary notes remain unapproved discovery views. Index failures do not erase
    retained sources or claim index/event/wiki-approval success. No events are
    published here. Each explicit research invocation is a new request, not a
    repair or replay of a previously failed provider stage.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    config = config or ResearchConfig()
    vault_root = vault_root.resolve()
    output_folder = validate_ordinary_write_path(config.output_folder, vault_root)
    output_dir = validate_ordinary_write_path(
        str(output_folder / _sanitize_dirname(query)), vault_root
    )
    if intake.workspace.root.resolve() != vault_root:
        raise WikiError("Research intake belongs to a different vault")
    folder = output_dir.relative_to(vault_root).as_posix()
    result = PipelineResult(query=query)
    projected: list[tuple[CaptureResult, Path]] = []

    async def index(capture: CaptureResult, path: Path) -> None:
        def operation() -> int:
            return store.index_single_note(parser.parse_file(path), parser)

        try:
            disposition = await asyncio.to_thread(capture.run_stage, "index", operation)
        except WikiError as exc:
            disposition = StageDisposition("index", "failed", error=str(exc))
        result.dispositions.append(disposition)

    try:
        query_capture = intake.preserve_text(
            query,
            original_name="research-query.md",
            origin="research",
            adapter="research",
            role="query",
        )
        result.captures.append(query_capture)
        results = await search_youtube(query, max_results=config.max_results)
        if not results:
            result.analysis_summary = "No results found."
            return result

        sources: list[dict[str, str]] = []
        transcript_source_ids: list[str] = []
        for found in results:
            try:
                segments: list[dict[str, Any]] = await asyncio.to_thread(
                    YouTubeTranscriptApi.get_transcript,
                    found.video_id,
                    languages=[config.youtube_language],
                )
            except Exception:
                result.dispositions.append(
                    StageDisposition(
                        "fetch", "failed", error=f"Transcript unavailable: {found.video_id}"
                    )
                )
                continue
            # Validate the actual original text before a JSON envelope could hide
            # retention frontmatter. No normalization or silent truncation here.
            search_snapshot = asdict(found)
            for value in search_snapshot.values():
                if isinstance(value, str):
                    _metadata(_safe_text(value.encode("utf-8"), "Search result"))
            for segment in segments:
                _metadata(_safe_text(segment["text"].encode("utf-8"), "Transcript segment"))
            transcript = " ".join(segment["text"] for segment in segments)
            _metadata(_safe_text(transcript.encode("utf-8"), "Transcript"))
            original = _json_text(
                dict(search=search_snapshot, language=config.youtube_language, segments=segments)
            )
            revision = hashlib.sha256(original.encode("utf-8")).hexdigest()
            event = _json_text([found.video_id, config.youtube_language]).strip()
            raw = intake.preserve_text(
                original,
                original_name="youtube-transcript.json",
                origin="youtube",
                adapter="research",
                role="transcript-segments",
                event_id=event,
                revision=revision,
                uri=found.url,
            )
            result.captures.append(raw)
            result.original_source_ids.append(raw.source_id)
            text = intake.preserve_text(
                transcript,
                original_name="youtube-transcript.md",
                origin="youtube",
                adapter="research",
                role="transcript-text",
                event_id=event,
                revision=revision,
                uri=found.url,
                parents=[raw.source_id],
                original_retained=False,
            )
            result.captures.append(text)
            transcript_source_ids.append(text.source_id)
            sources.append(dict(title=found.title, content=transcript, url=found.url))
            projection = _project_source(
                intake,
                text,
                _render_source_note(
                    found.title,
                    transcript,
                    found.url,
                    query,
                    text,
                    created=str(intake.read(text.source_id).snapshot["metadata"]["captured_at"]),
                ),
                folder,
            )
            result.dispositions.append(projection)
            if projection.status == "succeeded" and isinstance(projection.value, str):
                path = vault_root / projection.value
                result.sources_created.append(path)
                projected.append((text, path))

        if not sources:
            result.analysis_summary = "No transcripts available."
            return result

        prompt = intake.preserve_text(
            render_analysis_request(sources, query),
            original_name="research-input.md",
            origin="research",
            adapter="research",
            role="analysis-input",
            parents=[query_capture.source_id, *transcript_source_ids],
            original_retained=False,
        )
        result.captures.append(prompt)
        request = intake.preserve_text(
            _json_text(
                dict(
                    input_source_id=prompt.source_id,
                    model=model,
                    messages=[dict(role="user", content=intake.read(prompt.source_id).text)],
                    system=ANALYSIS_SYSTEM_PROMPT,
                    max_tokens=ANALYSIS_MAX_TOKENS,
                )
            ),
            original_name="research-request.json",
            origin="research",
            adapter="research",
            role="analysis-request",
            parents=[prompt.source_id],
            original_retained=False,
        )
        result.captures.append(request)
        result.request_source_id = request.source_id
        validate_derivative_parent(intake.read(request.source_id))

        def generate() -> AnalysisResult:
            selected = json.loads(intake.read(request.source_id).text or "")
            return analyze_sources(
                selected["messages"][0]["content"],
                llm_client,
                selected["model"],
                system=selected["system"],
                max_tokens=selected["max_tokens"],
            )

        provider = await asyncio.to_thread(request.run_stage, "provider", generate)
        result.dispositions.append(provider)
        if provider.status != "succeeded" or not isinstance(provider.value, AnalysisResult):
            result.error = provider.error or "Analysis was not generated; originals preserved"
            return result
        analysis = provider.value
        result.analysis_summary = analysis.summary

        def preserve_analysis() -> CaptureResult:
            return intake.preserve_text(
                _render_summary(
                    query,
                    analysis,
                    result.original_source_ids,
                    created=str(intake.read(request.source_id).snapshot["metadata"]["captured_at"]),
                ),
                original_name="research-analysis.md",
                origin="research",
                adapter="research",
                role="analysis",
                parents=[request.source_id],
                original_retained=False,
            )

        derivative = request.run_stage("derivative", preserve_analysis)
        result.dispositions.append(derivative)
        if derivative.status != "succeeded" or not isinstance(derivative.value, CaptureResult):
            result.error = derivative.error or "Analysis not retained; originals preserved"
            return result
        captured = derivative.value
        result.captures.append(captured)
        result.derivative_source_id = captured.source_id
        projection = intake.project_text(captured, folder)
        result.dispositions.append(projection)
        if projection.status == "succeeded" and isinstance(projection.value, str):
            result.summary_path = vault_root / projection.value
            projected.append((captured, result.summary_path))
        else:
            result.error = projection.error or "Analysis projection not created"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.dispositions.append(StageDisposition("research", "failed", error=result.error))
    finally:
        for capture, path in projected:
            await index(capture, path)
    return result
