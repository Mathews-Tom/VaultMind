"""Auto-tagging — LLM-based tag classification with quarantine for new tags.

Classifies notes using the existing vault tag vocabulary. New tags suggested
by the LLM go into a quarantine list that must be approved before entering
the canonical vocabulary. Tags are written to note frontmatter only when
explicitly applied (``--apply`` flag).

Design choices:
- Uses the fast LLM model (not thinking model) to minimize cost
- Max 2 new tags per note to prevent over-tagging
- Existing vault tags are passed as vocabulary context
- New (unseen) tags go to quarantine, not directly to notes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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


@dataclass
class QuarantineState:
    """Tracks new tags pending approval and approved vocabulary."""

    approved_tags: set[str] = field(default_factory=set)
    quarantined_tags: set[str] = field(default_factory=set)

    def approve(self, tag: str) -> None:
        self.quarantined_tags.discard(tag)
        self.approved_tags.add(tag)

    def reject(self, tag: str) -> None:
        self.quarantined_tags.discard(tag)

    def approve_all(self) -> None:
        self.approved_tags |= self.quarantined_tags
        self.quarantined_tags.clear()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "approved": sorted(self.approved_tags),
            "quarantined": sorted(self.quarantined_tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> QuarantineState:
        return cls(
            approved_tags=set(data.get("approved", [])),
            quarantined_tags=set(data.get("quarantined", [])),
        )


class AutoTagger:
    """LLM-based auto-tagger with quarantine for new tag vocabulary."""

    def __init__(
        self,
        config: AutoTagConfig,
        llm_client: LLMClient,
        model: str,
        quarantine_path: Path,
    ) -> None:
        self._config = config
        self._client = llm_client
        self._model = model
        self._quarantine_path = quarantine_path
        self._quarantine = self._load_quarantine()

    @property
    def quarantine(self) -> QuarantineState:
        return self._quarantine

    def suggest_tags(
        self,
        note: Note,
        vault_tags: set[str],
    ) -> TagSuggestion | None:
        """Generate tag suggestions for a single note using the LLM.

        Args:
            note: The note to classify.
            vault_tags: Set of all tags currently used in the vault.

        Returns:
            TagSuggestion or None if the note is too short.
        """
        from vaultmind.llm.client import LLMError, Message

        body = note.body_without_frontmatter().strip()
        if len(body) < self._config.min_content_length:
            return None

        # Include approved quarantine tags in vocabulary
        full_vocab = vault_tags | self._quarantine.approved_tags

        prompt = TAGGING_PROMPT.format(
            max_tags=self._config.max_tags_per_note,
            vocabulary=", ".join(sorted(full_vocab)) if full_vocab else "(no existing tags)",
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

            # Separate known vs new tags
            known = [t for t in suggested if t in full_vocab]
            novel = [t for t in suggested if t not in full_vocab] + new_tags

            # Quarantine novel tags
            for tag in novel:
                if tag and tag not in self._quarantine.approved_tags:
                    self._quarantine.quarantined_tags.add(tag)

            all_suggested = known + [t for t in novel if t in self._quarantine.approved_tags]

            return TagSuggestion(
                note_path=str(note.path),
                note_title=note.title,
                existing_tags=list(note.tags),
                suggested_tags=all_suggested,
                new_tags=novel,
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

    def save_quarantine(self) -> None:
        """Persist quarantine state to disk."""
        self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._quarantine_path, "w") as f:
            json.dump(self._quarantine.to_dict(), f, indent=2)

    def _load_quarantine(self) -> QuarantineState:
        """Load quarantine state from disk, or create fresh."""
        if self._quarantine_path.exists():
            with open(self._quarantine_path) as f:
                return QuarantineState.from_dict(json.load(f))
        return QuarantineState()

    def collect_vault_tags(self, notes: list[Note]) -> set[str]:
        """Collect all unique tags from a list of notes."""
        tags: set[str] = set()
        for note in notes:
            tags.update(note.tags)
        return tags
