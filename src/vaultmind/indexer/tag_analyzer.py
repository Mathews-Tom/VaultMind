"""Tag synonym and merge detection — zero-LLM-cost analysis.

Uses string similarity (difflib) and co-occurrence ratios to surface
tag pairs that are likely synonyms or candidates for merging.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaultmind.vault.models import Note


@dataclass(frozen=True, slots=True)
class TagSynonym:
    """A pair of tags that may be synonyms or merge candidates."""

    tag_a: str
    tag_b: str
    similarity: float
    co_occurrence_ratio: float
    suggested_canonical: str


def compute_tag_stats(
    notes: list[Note],
) -> tuple[dict[str, int], dict[frozenset[str], int]]:
    """Compute per-tag counts and pairwise co-occurrence counts.

    Args:
        notes: Parsed vault notes.

    Returns:
        A 2-tuple of:
        - tag_counts: mapping tag -> total occurrence count across notes
        - co_occurrences: mapping frozenset({tag_a, tag_b}) -> notes where both appear
    """
    tag_counts: dict[str, int] = {}
    co_occurrences: dict[frozenset[str], int] = {}

    for note in notes:
        tags = list(note.tags)
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # All unique pairs within this note
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair: frozenset[str] = frozenset({tags[i], tags[j]})
                co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

    return tag_counts, co_occurrences


def find_synonyms(
    tag_counts: dict[str, int],
    co_occurrences: dict[frozenset[str], int],
    min_similarity: float = 0.75,
    min_co_occurrence: float = 0.5,
) -> list[TagSynonym]:
    """Detect likely tag synonyms using string similarity and co-occurrence.

    A pair is flagged if EITHER:
    - String similarity ratio >= min_similarity (catches plural/singular, hyphens)
    - Co-occurrence ratio >= min_co_occurrence (tags that always appear together)

    Co-occurrence ratio = co_occurrences[pair] / min(count_a, count_b).
    The canonical tag is the one with the higher usage count. Ties go to
    alphabetical order.

    Args:
        tag_counts: Per-tag document frequency (from compute_tag_stats).
        co_occurrences: Pairwise note co-occurrence counts (from compute_tag_stats).
        min_similarity: Minimum SequenceMatcher ratio to flag a pair.
        min_co_occurrence: Minimum co-occurrence ratio to flag a pair.

    Returns:
        Sorted list of TagSynonym (highest similarity first).
    """
    tags = sorted(tag_counts)
    results: list[TagSynonym] = []

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            tag_a = tags[i]
            tag_b = tags[j]

            similarity = difflib.SequenceMatcher(None, tag_a, tag_b).ratio()

            pair: frozenset[str] = frozenset({tag_a, tag_b})
            co_count = co_occurrences.get(pair, 0)
            min_count = min(tag_counts[tag_a], tag_counts[tag_b])
            co_ratio = co_count / min_count if min_count > 0 else 0.0

            if similarity >= min_similarity or co_ratio >= min_co_occurrence:
                count_a = tag_counts[tag_a]
                count_b = tag_counts[tag_b]
                if count_a > count_b:
                    canonical = tag_a
                elif count_b > count_a:
                    canonical = tag_b
                else:
                    canonical = min(tag_a, tag_b)  # alphabetical tiebreak

                results.append(
                    TagSynonym(
                        tag_a=tag_a,
                        tag_b=tag_b,
                        similarity=similarity,
                        co_occurrence_ratio=co_ratio,
                        suggested_canonical=canonical,
                    )
                )

    results.sort(key=lambda s: s.similarity, reverse=True)
    return results
