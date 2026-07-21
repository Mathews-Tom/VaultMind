"""Auto-tagging — LLM-based tag classification against the vault vocabulary.

Classifies notes using the existing vault tag vocabulary (plus any
previously-approved novel tags the caller passes in). Returns known vs
novel tags for the caller to route through `services.review_queue.ReviewQueue`
(M7) as `TAG_APPLICATION`/`TAG_VOCABULARY` proposals — this module has no
persistence or approval state of its own.

Design choices:
- Uses the fast LLM model (not thinking model) to minimize cost
- Max 2 new tags per note to prevent over-tagging
- Existing vault tags are passed as vocabulary context
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import frontmatter

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.config import AutoTagConfig
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)

TAGGING_PROMPT = """\
You are a knowledge management assistant. Classify the note below by assigning \
up to {max_tags} tags from the existing vocabulary. If none of the existing tags \
fit well, you may suggest up to 1 new tag.

Existing tag vocabulary:
{vocabulary}

Rules:
- Prefer existing tags over new ones
- Tags should be lowercase, hyphen-separated (e.g., "machine-learning")
- Only assign tags that genuinely describe the note's content
- Do not assign generic tags like "note", "misc", "other"

Respond with ONLY valid JSON:
{{"tags": ["tag1", "tag2"], "new_tags": ["optional-new-tag"]}}

Note title: {title}
Note type: {note_type}
Current tags: {current_tags}

Content (first 2000 chars):
{content}
"""


@dataclass(frozen=True, slots=True)
class TagSuggestion:
    """Tag suggestion for a single note."""

    note_path: str
    note_title: str
    existing_tags: list[str]
    suggested_tags: list[str]
    new_tags: list[str]


class AutoTagger:
    """LLM-based auto-tagger — pure suggestion generation, no approval state."""

    def __init__(
        self,
        config: AutoTagConfig,
        llm_client: LLMClient,
        model: str,
    ) -> None:
        self._config = config
        self._client = llm_client
        self._model = model

    def suggest_tags(
        self,
        note: Note,
        vault_tags: set[str],
    ) -> TagSuggestion | None:
        """Generate tag suggestions for a single note using the LLM.

        Args:
            note: The note to classify.
            vault_tags: Set of tags currently considered "known" vocabulary
                (existing vault tags plus any caller-supplied approved
                novel tags).

        Returns:
            TagSuggestion or None if the note is too short.
        """
        from vaultmind.llm.client import LLMError, Message

        body = note.body_without_frontmatter().strip()
        if len(body) < self._config.min_content_length:
            return None

        prompt = TAGGING_PROMPT.format(
            max_tags=self._config.max_tags_per_note,
            vocabulary=", ".join(sorted(vault_tags)) if vault_tags else "(no existing tags)",
            title=note.title,
            note_type=note.note_type.value,
            current_tags=", ".join(note.tags) if note.tags else "none",
            content=body[:2000],
        )

        try:
            response = self._client.complete(
                messages=[Message(role="user", content=prompt)],
                model=self._model,
                max_tokens=256,
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            result: dict[str, Any] = json.loads(raw)
            suggested = result.get("tags", [])[: self._config.max_tags_per_note]
            new_tags = result.get("new_tags", [])[:1]

            # Separate known vs novel tags — novel tags are the caller's
            # responsibility to route through the review queue as
            # TAG_VOCABULARY proposals; they are never auto-included here.
            known = [t for t in suggested if t in vault_tags]
            novel = [t for t in suggested if t not in vault_tags] + new_tags

            return TagSuggestion(
                note_path=str(note.path),
                note_title=note.title,
                existing_tags=list(note.tags),
                suggested_tags=known,
                new_tags=[t for t in novel if t],
            )

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning("Failed to parse tagging result for %s: %s", note.title, e)
            return None
        except LLMError as e:
            logger.error("LLM error during tagging for %s: %s", note.title, e)
            return None

    def suggest_batch(
        self,
        notes: list[Note],
        vault_tags: set[str],
    ) -> list[TagSuggestion]:
        """Generate tag suggestions for multiple notes."""
        suggestions: list[TagSuggestion] = []
        for note in notes:
            result = self.suggest_tags(note, vault_tags)
            if result and (result.suggested_tags or result.new_tags):
                suggestions.append(result)
        return suggestions

    def apply_tags(self, note_path: Path, tags: list[str]) -> None:
        """Write tags to a note's frontmatter.

        Merges with existing tags (no duplicates). Uses python-frontmatter
        for safe round-trip editing.
        """
        with open(note_path, encoding="utf-8") as f:
            post = frontmatter.load(f)

        existing = post.metadata.get("tags", [])
        if isinstance(existing, str):
            existing = [existing]

        merged = list(dict.fromkeys(existing + tags))
        post.metadata["tags"] = merged

        with open(note_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        logger.info("Applied tags %s to %s", tags, note_path)

    def collect_vault_tags(self, notes: list[Note]) -> set[str]:
        """Collect all unique tags from a list of notes."""
        tags: set[str] = set()
        for note in notes:
            tags.update(note.tags)
        return tags
