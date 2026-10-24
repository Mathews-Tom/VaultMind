"""Filesystem-free conversation generation and retained-source orchestration.

Ordinary qa-artifacts are unapproved discovery projections, never wiki approval.
Indexing, event delivery and independent memory production belong to callers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.knowledge.intake import (
    MAX_CLOSURE_BYTES,
    MAX_CLOSURE_DEPTH,
    MAX_DEPENDENCIES,
    CaptureResult,
    SourceIntake,
    SourceRecord,
    StageDisposition,
)
from vaultmind.llm.client import Message
from vaultmind.vault.security import validate_ordinary_write_path
from vaultmind.wiki.workspace import MAX_INPUT_BYTES, WikiError, _canonical, _metadata, _safe_text

if TYPE_CHECKING:
    from vaultmind.llm.client import LLMClient
    from vaultmind.memory.store import EpisodeStore
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)

DISTILL_SYSTEM_PROMPT = """\
You distill a finished conversation into a structured Q&A artifact for a \
personal knowledge base.

Analyze the exchange below and return a JSON object with these fields:
- question: the core question or topic explored (concise, one sentence)
- summary: a synthesis of what was discussed (2-4 sentences)
- resolution: the concrete conclusion or decision reached, or "" if unresolved
- systems: array of systems/projects/tools mentioned (strings, may be empty)
- participants: array of participant names or roles mentioned (strings, may be empty)

Return ONLY valid JSON, no other text."""

# Provenance-default authority for LLM-distilled artifacts (M2's default table:
# authored=5, distilled=4, research=3, web clip=2, auto-ingested=1).
DISTILLED_AUTHORITY = 4


@dataclass(frozen=True, slots=True)
class DistillationGeneration:
    """Parsed/rendered model output; success says nothing about persistence."""

    success: bool
    content: str = ""
    frontmatter: dict[str, object] = field(default_factory=dict)
    error: str = ""


@dataclass
class PreservedDistillation:
    """Actual producer dispositions, not index/event/approval guarantees."""

    input_capture: CaptureResult
    request: CaptureResult | None = None
    generation: DistillationGeneration | None = None
    generation_disposition: StageDisposition | None = None
    derivative: CaptureResult | None = None
    derivative_disposition: StageDisposition | None = None
    projection: StageDisposition | None = None

    @property
    def output_path(self) -> str:
        if (
            self.projection is not None
            and self.projection.status == "succeeded"
            and isinstance(self.projection.value, str)
        ):
            return self.projection.value
        return ""

    @property
    def success(self) -> bool:
        return bool(self.output_path)

    @property
    def error(self) -> str:
        if self.generation is not None and not self.generation.success:
            return self.generation.error
        for disposition in (
            self.generation_disposition,
            self.derivative_disposition,
            self.projection,
        ):
            if disposition is not None and disposition.status != "succeeded":
                return disposition.error or f"{disposition.stage}: {disposition.status}"
        return "" if self.success else "No generated projection was completed"


def render_conversation(turns: list[dict[str, str]]) -> str:
    """Format complete turns without hiding an original retention directive."""
    parts: list[str] = []
    for i, turn in enumerate(turns, start=1):
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        for text in (user, assistant):
            _metadata(_safe_text(text.encode("utf-8"), "Conversation turn"))
        parts.append(f"Turn {i}\nUser: {user}\nAssistant: {assistant}")
    return "\n\n".join(parts)


def preserve_conversation_snapshot(
    intake: SourceIntake,
    snapshot: dict[str, Any],
    *,
    origin: str,
    adapter: str,
    event_id: str | None,
) -> CaptureResult:
    """Retain a selected revision and its exact formatted model-input text."""
    conversation = render_conversation(snapshot["history"])

    def validate(value: Any) -> None:
        if isinstance(value, str):
            _metadata(_safe_text(value.encode("utf-8"), "Selected conversation"))
        elif isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)

    validate(snapshot)
    original = intake.preserve_text(
        _canonical(snapshot),
        original_name="selected-conversation.json",
        origin=origin,
        adapter=adapter,
        role="selected-conversation",
        event_id=event_id,
    )
    return intake.preserve_text(
        conversation,
        original_name="conversation.txt",
        origin=origin,
        adapter=adapter,
        role="conversation-input",
        event_id=event_id,
        revision=original.source_id,
        parents=[original.source_id],
        original_retained=False,
    )


def validate_derivative_parent(record: SourceRecord) -> None:
    """Check that one generated child can fit the already-validated closure."""
    seen: set[str] = set()
    total = 0

    def visit(source: SourceRecord, depth: int) -> None:
        nonlocal total
        if depth >= MAX_CLOSURE_DEPTH:
            raise WikiError("Generated source would exceed retained dependency depth")
        if source.source_id in seen:
            return
        seen.add(source.source_id)
        total += len(source.content)
        for dependency in source.dependencies:
            visit(dependency, depth + 1)

    visit(record, 0)
    if len(seen) >= MAX_DEPENDENCIES:
        raise WikiError("Generated source would exceed retained dependency count")
    if total + MAX_INPUT_BYTES > MAX_CLOSURE_BYTES:
        raise WikiError("Generated source would exceed retained closure byte limit")


def _render_note(
    fm: dict[str, object],
    question: str,
    summary: str,
    resolution: str,
    systems: list[str],
    participants: list[str],
) -> str:
    body_parts = [f"# {question}", ""]
    if summary:
        body_parts += ["## Summary", "", summary, ""]
    body_parts += ["## Resolution", "", resolution or "_Unresolved._"]
    if systems:
        links = ", ".join(f"[[{s}]]" for s in systems)
        body_parts += ["", "## Systems", "", links]
    if participants:
        links = ", ".join(f"[[{p}]]" for p in participants)
        body_parts += ["", "## Participants", "", links]
    body = "\n".join(body_parts)
    post = frontmatter.Post(body, **fm)
    return str(frontmatter.dumps(post))


def generate_distillation(
    conversation_text: str,
    llm_client: LLMClient,
    model: str,
    *,
    source_ref: str,
    occurred_at: str,
    max_tokens: int = 800,
    authority: int = DISTILLED_AUTHORITY,
) -> DistillationGeneration:
    """Generate a qa-artifact in memory, without reading or writing files."""
    if not conversation_text.strip():
        return DistillationGeneration(success=False, error="No conversation turns to distill")
    try:
        response = llm_client.complete(
            messages=[Message(role="user", content=conversation_text)],
            model=model,
            max_tokens=max_tokens,
            system=DISTILL_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("LLM call failed during conversation distillation")
        return DistillationGeneration(success=False, error="LLM call failed")

    try:
        parsed = json.loads(response.text.strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        return DistillationGeneration(success=False, error="Invalid JSON from LLM")
    if not isinstance(parsed, dict):
        return DistillationGeneration(success=False, error="Expected a JSON object from the LLM")

    question = str(parsed.get("question", "")).strip()
    if not question:
        return DistillationGeneration(success=False, error="Distillation produced no question")

    summary = str(parsed.get("summary", "")).strip()
    resolution = str(parsed.get("resolution", "")).strip()
    raw_systems = parsed.get("systems", [])
    raw_participants = parsed.get("participants", [])
    systems = (
        [str(s).strip() for s in raw_systems if str(s).strip()]
        if isinstance(raw_systems, list)
        else []
    )
    participants = (
        [str(p).strip() for p in raw_participants if str(p).strip()]
        if isinstance(raw_participants, list)
        else []
    )

    fm: dict[str, object] = {
        "type": "qa-artifact",
        "title": question,
        "question": question,
        "summary": summary,
        "resolution": resolution,
        "systems": systems,
        "participants": participants,
        "source_ref": source_ref,
        "occurred_at": occurred_at,
        "authority": authority,
        "tags": ["qa-artifact"],
        "status": "unapproved",
        "approved": False,
        "source": "distilled",
        "created": occurred_at,
    }

    content = _render_note(fm, question, summary, resolution, systems, participants)
    return DistillationGeneration(success=True, content=content, frontmatter=fm)


def distill_retained_conversation(
    intake: SourceIntake,
    input_capture: CaptureResult,
    llm_client: LLMClient,
    model: str,
    *,
    source_ref: str,
    occurred_at: str,
    output_folder: str,
    max_tokens: int = 800,
    authority: int = DISTILLED_AUTHORITY,
) -> PreservedDistillation:
    """Generate once from retained input, then preserve and project its derivative."""
    root = intake.workspace.root.resolve()
    output_dir = validate_ordinary_write_path(output_folder, root)
    validate_ordinary_write_path(str(output_dir / "distillation.md"), root)
    folder = output_dir.relative_to(root).as_posix()
    source = intake.read(input_capture.source_id)
    if source.text is None or source.provenance is None or not source.provenance["parents"]:
        raise WikiError("Distillation requires retained text with its original snapshot parents")
    request_values = dict(
        messages=[dict(role="user", content=source.text)],
        system=DISTILL_SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        source_ref=source_ref,
        occurred_at=occurred_at,
        authority=authority,
    )
    result = PreservedDistillation(input_capture)
    request = intake.preserve_text(
        _canonical(request_values),
        original_name="distillation-request.json",
        origin=source_ref,
        adapter="conversation-distillation",
        role="model-request",
        event_id=input_capture.source_id,
        parents=[input_capture.source_id],
        original_retained=False,
    )
    result.request = request
    retained_request = intake.read(request.source_id)
    validate_derivative_parent(retained_request)
    selected = json.loads(retained_request.text or "")

    def generate() -> DistillationGeneration:
        # The retained values are the actual values passed to the completion.
        generated = generate_distillation(
            selected["messages"][0]["content"],
            llm_client,
            selected["model"],
            source_ref=selected["source_ref"],
            occurred_at=selected["occurred_at"],
            max_tokens=selected["max_tokens"],
            authority=selected["authority"],
        )
        result.generation = generated
        if not generated.success:
            raise WikiError(generated.error)
        return generated

    result.generation_disposition = input_capture.run_stage("provider", generate)
    if result.generation_disposition.status != "succeeded" or result.generation is None:
        return result
    generation = result.generation
    post = frontmatter.loads(generation.content)
    post.metadata["parent_source_ids"] = [request.source_id]
    derivative_content = str(frontmatter.dumps(post))

    def preserve() -> CaptureResult:
        return intake.preserve_text(
            derivative_content,
            original_name="distillation.md",
            origin=source_ref,
            adapter="conversation-distillation",
            role="distillation",
            event_id=input_capture.source_id,
            revision=request.source_id,
            parents=[request.source_id],
            original_retained=False,
        )

    result.derivative_disposition = request.run_stage("derivative", preserve)
    if result.derivative_disposition.status != "succeeded":
        return result
    if not isinstance(result.derivative_disposition.value, CaptureResult):
        return result
    result.derivative = result.derivative_disposition.value
    result.projection = intake.project_text(result.derivative, folder)
    return result


def extract_and_store_episodes(
    note: Note,
    llm_client: LLMClient,
    model: str,
    episode_store: object,
) -> int:
    """Run existing episodic-memory extraction over a distilled note and persist results.

    Calls `memory.extractor.extract_episodes` unmodified — the `qa-artifact`
    note type flows through it like any other note (the extractor has no
    note-type filtering). No-op if `episode_store` is not an `EpisodeStore`.

    Returns the number of episodes extracted.
    """
    from vaultmind.memory.extractor import extract_episodes
    from vaultmind.memory.models import OutcomeStatus
    from vaultmind.memory.store import EpisodeStore as _EpisodeStore

    if not isinstance(episode_store, _EpisodeStore):
        return 0

    store: EpisodeStore = episode_store
    episodes = extract_episodes(note, llm_client, model)
    for ep in episodes:
        entities_raw = ep.get("entities", [])
        entities = [str(e) for e in entities_raw] if isinstance(entities_raw, list) else []
        episode = store.create(
            decision=str(ep.get("decision", "")),
            context=str(ep.get("context", "")),
            entities=entities,
            source_notes=[str(note.path)],
        )
        outcome = str(ep.get("outcome", ""))
        if outcome:
            status_str = str(ep.get("outcome_status", "pending"))
            try:
                status = OutcomeStatus(status_str)
            except ValueError:
                status = OutcomeStatus.UNKNOWN
            lessons_raw = ep.get("lessons", [])
            lessons = [str(item) for item in lessons_raw] if isinstance(lessons_raw, list) else []
            store.resolve(episode.episode_id, outcome, status, lessons)
    return len(episodes)


def mint_gap_for_unresolved(
    result: DistillationGeneration,
    gap_store: object,
    source_ref: str,
) -> None:
    """Mint an `unanswered_question` gap for a qa-artifact with an empty `resolution`.

    Shared by both the auto-trigger (`bot/thinking.py`) and manual `/distill`
    (`bot/handlers/distill.py`) call sites — mirrors `extract_and_store_episodes`'s
    "one shared pipeline helper, wired from two call sites" convention.
    No-op if `gap_store` is not a `GapStore`, distillation failed, or the
    artifact's `resolution` is non-empty.
    """
    from vaultmind.memory.gaps import GapKind
    from vaultmind.memory.gaps import GapStore as _GapStore

    if not isinstance(gap_store, _GapStore) or not result.success:
        return
    if result.frontmatter.get("resolution"):
        return
    question = str(result.frontmatter.get("question", ""))
    if not question:
        return
    try:
        gap_store.mint(question, GapKind.UNANSWERED_QUESTION, evidence_ref=source_ref)
    except Exception:
        logger.exception("Gap minting failed for qa-artifact question %r", question)
