"""Conversation distillation — turns a finished thinking/chat exchange into a
structured `qa-artifact` vault note.

Mirrors `pipeline/synthesis.py`'s convention: a pure pipeline function that
calls an LLM, validates the structured output, and writes the note to disk.
Parsing and indexing the written note is the caller's responsibility (same
division of concerns `vault/ingest.py` and `bot/handlers/capture.py` use).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import frontmatter

from vaultmind.llm.client import Message

if TYPE_CHECKING:
    from pathlib import Path

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

_SLUG_RE = re.compile(r"[^a-z0-9\s-]")
_WS_RE = re.compile(r"[\s-]+")


@dataclass(frozen=True, slots=True)
class DistillResult:
    """Result of distilling a conversation into a qa-artifact note."""

    success: bool
    output_path: str = ""
    frontmatter: dict[str, object] = field(default_factory=dict)
    error: str = ""


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("", text.lower().strip())
    slug = _WS_RE.sub("-", slug).strip("-")
    return slug[:60] or "untitled"


def _format_conversation(turns: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, turn in enumerate(turns, start=1):
        parts.append(
            f"Turn {i}\nUser: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
        )
    return "\n\n".join(parts)


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


def distill_conversation(
    turns: list[dict[str, str]],
    llm_client: LLMClient,
    model: str,
    vault_root: Path,
    output_folder: str,
    source_ref: str,
    occurred_at: str | None = None,
    max_tokens: int = 800,
    authority: int = DISTILLED_AUTHORITY,
) -> DistillResult:
    """Distill a finished conversation into a `qa-artifact` note.

    Calls the LLM to extract structured fields (question, summary, resolution,
    systems, participants), validates the response, and writes a `qa-artifact`
    note under `{vault_root}/{output_folder}/`. Never overwrites an existing
    file. The caller is responsible for parsing and indexing the written note.
    `authority` (M2's provenance table) defaults to `DISTILLED_AUTHORITY`
    (=4, "distilled") for the original conversation-distillation call sites
    (`bot/thinking.py`, `bot/handlers/distill.py`); `sources/pipeline.py`
    (M8) passes `authority=1` ("auto-ingested") explicitly, since connector
    items are unattended background content, not a user's own session.

    Returns a `DistillResult`; `success=False` means no file was written.
    """
    if not turns:
        return DistillResult(success=False, error="No conversation turns to distill")

    conversation = _format_conversation(turns)
    occurred = occurred_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        response = llm_client.complete(
            messages=[Message(role="user", content=conversation)],
            model=model,
            max_tokens=max_tokens,
            system=DISTILL_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("LLM call failed during conversation distillation for %s", source_ref)
        return DistillResult(success=False, error="LLM call failed")

    try:
        parsed = json.loads(response.text.strip())
    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON from LLM during distillation for %s: %r",
            source_ref,
            response.text[:200],
        )
        return DistillResult(success=False, error="Invalid JSON from LLM")

    if not isinstance(parsed, dict):
        return DistillResult(success=False, error="Expected a JSON object from the LLM")

    question = str(parsed.get("question", "")).strip()
    if not question:
        return DistillResult(success=False, error="Distillation produced no question")

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
        "occurred_at": occurred,
        "authority": authority,
        "tags": ["qa-artifact"],
        "status": "active",
        "source": "distilled",
        "created": occurred,
    }

    output_dir = vault_root / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = occurred[:10]
    slug = _slugify(question)
    filename = f"{date_str}-{slug}.md"
    output_path = output_dir / filename
    counter = 1
    while output_path.exists():
        filename = f"{date_str}-{slug}-{counter}.md"
        output_path = output_dir / filename
        counter += 1

    content = _render_note(fm, question, summary, resolution, systems, participants)
    output_path.write_text(content, encoding="utf-8")
    rel_path = str(output_path.relative_to(vault_root))
    logger.info("Wrote qa-artifact note to %s", rel_path)

    return DistillResult(success=True, output_path=rel_path, frontmatter=fm)


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
    result: DistillResult,
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
