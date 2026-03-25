"""Content-based importance scoring computed at indexing time."""

from __future__ import annotations

import re


def compute_importance(
    content: str,
    tags: list[str] | None = None,
    entities: list[str] | None = None,
) -> float:
    """Compute content-based importance score [0.0, 1.0].

    Four factors with equal weight (0.25 each):
    - Entity density: min(entity_count / 5, 1.0)
    - Link density: min(wikilink_count / 10, 1.0)
    - Content length: min(word_count / 500, 1.0)
    - Tag count: min(tag_count / 5, 1.0)
    """
    tags = tags or []
    entities = entities or []

    entity_score = min(len(entities) / 5, 1.0) if entities else 0.0

    wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content)
    link_score = min(len(wikilinks) / 10, 1.0)

    word_count = len(content.split())
    length_score = min(word_count / 500, 1.0)

    tag_score = min(len(tags) / 5, 1.0)

    return (entity_score + link_score + length_score + tag_score) / 4.0
