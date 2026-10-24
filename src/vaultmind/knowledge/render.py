"""Lossless plain-text presentation shared by Telegram answer routes."""

from __future__ import annotations

import json
from typing import Any


def render_knowledge_result(result: dict[str, Any]) -> str:
    """Present the accepted outcome and its complete provenance/qualifications.

    JSON-encoded metadata keeps repository-controlled names unambiguous. No
    Markdown parsing, shortening or model-generated citation rewriting occurs.
    """
    mapping = [
        {key: row[key] for key in ("id", "page", "page_sha256", "heading", "sources")}
        for row in result["evidence"]
    ]
    details = dict(
        citations=result["citations"],
        section_mapping=mapping,
        selected_pages=result["selected_pages"],
        sources=result["sources"],
        unresolved_conflicts=result["unresolved_conflicts"],
        selection=result["selection"],
        reason=result["reason"],
    )
    if result["status"] == "evidence":
        details["evidence"] = result["evidence"]
    return (
        f"Status: {result['status']}\n\n{result['answer']}\n\n"
        "Evidence and selection qualifications:\n"
        + json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False)
    )


def split_knowledge_text(text: str, max_len: int = 4000) -> list[str]:
    """Split without losing even boundary whitespace or citation characters."""
    return [text[start : start + max_len] for start in range(0, len(text), max_len)]
