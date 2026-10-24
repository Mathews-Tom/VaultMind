"""Real frontmatter and approved-vocabulary operations shared by CLI and bot."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.services.review_queue import ProposalKind
from vaultmind.vault.security import (
    validate_ordinary_output_path,
    validate_ordinary_write_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.services.review_queue import Applier

logger = logging.getLogger(__name__)


def apply_tags(note_path: Path, tags: list[str], *, vault_root: Path) -> None:
    """Merge tags on an ordinary note, preserving its content and metadata."""
    note_path = validate_ordinary_write_path(str(note_path), vault_root)
    with note_path.open(encoding="utf-8") as stream:
        post = frontmatter.load(stream)
    existing = post.metadata.get("tags", [])
    if isinstance(existing, str):
        existing = [existing]
    if not isinstance(existing, list) or not all(isinstance(tag, str) for tag in existing):
        raise TypeError("frontmatter tags must be a string or list of strings")
    post.metadata["tags"] = list(dict.fromkeys(existing + tags))
    with note_path.open("w", encoding="utf-8") as stream:
        stream.write(frontmatter.dumps(post))
    logger.info("Applied tags %s to %s", tags, note_path)


def load_tag_vocabulary(path: Path) -> set[str]:
    """Load the existing durable approved-tag vocabulary."""
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("approved", []))


def save_tag_vocabulary(path: Path, tags: set[str], *, vault_root: Path) -> None:
    """Persist approved tags only outside the selected vault's reserved trees."""
    path = validate_ordinary_output_path(path, vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"approved": sorted(tags)}, indent=2))


def tag_appliers(vault_root: Path, vocabulary_path: Path) -> dict[ProposalKind, Applier]:
    """Bind the same real queued tag operations for every queue assembly."""
    # Reject fixed/configured destinations before the caller opens its queue;
    # retain the request so execution also checks aliases changed while pending.
    validate_ordinary_output_path(vocabulary_path, vault_root)

    def apply_application(payload: dict[str, Any]) -> str:
        note_path = validate_ordinary_write_path(str(payload["note_path"]), vault_root)
        tags = list(payload["tags"])
        apply_tags(note_path, tags, vault_root=vault_root)
        return f"applied {len(tags)} tag(s) to {payload['note_path']}"

    def apply_vocabulary(payload: dict[str, Any]) -> str:
        path = validate_ordinary_output_path(vocabulary_path, vault_root)
        tag = str(payload["tag"])
        approved = load_tag_vocabulary(path)
        approved.add(tag)
        save_tag_vocabulary(path, approved, vault_root=vault_root)
        return f"'{tag}' added to vocabulary"

    return {
        ProposalKind.TAG_APPLICATION: apply_application,
        ProposalKind.TAG_VOCABULARY: apply_vocabulary,
    }
