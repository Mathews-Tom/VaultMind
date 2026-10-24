"""Deterministic lexical selection, never an authority or truth score."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

from vaultmind.knowledge.sections import _page

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaultmind.wiki.workspace import WikiWorkspace

MAX_QUERY_PAGES = 8
LEXICAL_LIMITATION = (
    "Lexical discovery can miss synonyms; selection is not an exhaustive wiki conclusion."
)
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    )
)
_STRUCTURAL_HEADINGS = frozenset({"Current understanding", "Conflicting information", "Resolution"})


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(re.findall(r"[^\W_]+", normalized)) - _STOPWORDS


def discover_pages(
    workspace: WikiWorkspace,
    question: str,
    candidates: list[str],
    allows_path: Callable[[str], bool] | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Validate every permitted candidate before ranking complete current bodies.

    The caller holds the workspace's read boundary and has authorized the entire
    ``_wiki`` provenance tree under the adapter's effective first-folder policy.
    No excluded page is opened or included in counts or diagnostics.
    """
    payloads = [
        workspace.approved_page(path)
        for path in candidates
        if allows_path is None or allows_path(path)
    ]
    if not payloads:
        return [], False, "no_eligible_pages"
    question_terms = _terms(question)
    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for payload in payloads:
        title, sections, _ = _page(payload)
        heading_terms = _terms(
            "\n".join(
                [
                    title,
                    *(heading for heading, _ in sections if heading not in _STRUCTURAL_HEADINGS),
                ]
            )
        )
        body_terms = _terms(
            "".join(
                text.partition("\n")[2] if heading in _STRUCTURAL_HEADINGS else text
                for heading, text in sections
            )
        )
        heading_matches = len(question_terms & heading_terms)
        body_matches = len(question_terms & body_terms)
        if heading_matches or body_matches:
            # The question denominator is identical for every candidate, so
            # integer coverage counts preserve normalized coverage ordering.
            ranked.append((-heading_matches, -body_matches, payload["page"], payload))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    if not ranked:
        return [], False, "no_lexical_support"
    return [row[3] for row in ranked[:MAX_QUERY_PAGES]], len(ranked) > MAX_QUERY_PAGES, None
