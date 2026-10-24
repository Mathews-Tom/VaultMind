"""Complete Markdown sections and conflict interpretation for approved pages."""

from __future__ import annotations

import re
from typing import Any

from vaultmind.wiki.workspace import _metadata


def _sections(body: str) -> list[tuple[str, str]]:
    """Keep every body line, splitting at Markdown headings outside fenced code."""
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    fence = ""
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            token, rest = marker.groups()
            if not fence and not (token[0] == "`" and "`" in rest):
                fence = token
                lines.append(line)
                continue
            if fence and token[0] == fence[0] and len(token) >= len(fence) and not rest.strip():
                fence = ""
                lines.append(line)
                continue
        match = re.match(r"^#{1,6} ([^\r\n]+)", line) if not fence else None
        if match:
            if lines:
                sections.append((heading, "".join(lines)))
            heading = match.group(1).strip()
            lines = []
        lines.append(line)
    if lines:
        sections.append((heading, "".join(lines)))
    return sections


def _page(payload: dict[str, Any]) -> tuple[str, list[tuple[str, str]], bool]:
    metadata, body = _metadata(payload["after"])
    sections = _sections(body)
    blocks: dict[str, list[str]] = {}
    active = ""
    for heading, text in sections:
        boundary = re.match(r"^#{1,2} ", text)
        if boundary:
            active = heading if text.startswith("## ") else ""
        if active:
            blocks.setdefault(active, []).append(text.partition("\n")[2] if boundary else text)
    conflict = bool("".join(blocks.get("Conflicting information", [])).strip())
    resolution = bool("".join(blocks.get("Resolution", [])).strip())
    return str(metadata["title"]), sections, conflict and not resolution
