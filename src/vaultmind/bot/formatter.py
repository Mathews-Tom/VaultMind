"""Telegram HTML formatter for vault notes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Compiled patterns — module level, never re-compiled per call
# ---------------------------------------------------------------------------

# Fenced code blocks (``` ... ```) — multi-line, captured before anything else
_RE_CODE_BLOCK = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)

# Obsidian callout: > [!type] optional title  (raw markdown line)
_RE_CALLOUT_LINE = re.compile(r"^>\s*\[!(\w+)\][ \t]*(.*)$", re.IGNORECASE)

# Blockquote prefix (raw markdown line) — matched after callout check
_RE_BLOCKQUOTE_PREFIX = re.compile(r"^>[ \t]?")

# Headings — all levels → bold  (raw markdown line)
_RE_HEADING_LINE = re.compile(r"^#{1,6}\s+(.+)$")

# Horizontal rules (raw markdown line)
_RE_HRULE_LINE = re.compile(r"^[ \t]*([-*_]){3,}[ \t]*$")

# Task list items (raw markdown line)
_RE_TASK_OPEN_LINE = re.compile(r"^(\s*)[-*]\s+\[ \]\s+(.*)$")
_RE_TASK_DONE_LINE = re.compile(r"^(\s*)[-*]\s+\[x\]\s+(.*)$", re.IGNORECASE)

# Unordered list items (raw markdown line, after tasks)
_RE_LIST_ITEM_LINE = re.compile(r"^(\s*)[-*]\s+(.*)$")

# Inline patterns (applied to already-escaped text segments)
_RE_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD_UNDER = re.compile(r"__(.+?)__")
_RE_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_ITALIC_UNDER = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")

# Images (raw markdown — checked before escape)
_RE_IMG_WIKI = re.compile(r"!\[\[([^\]]+)\]\]")
_RE_IMG_MD = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")

# Wikilinks (raw markdown — checked before escape)
_RE_WIKILINK_ALIAS = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
_RE_WIKILINK_PLAIN = re.compile(r"\[\[([^\]]+)\]\]")

# HTML tags — strip them from raw text before processing
_RE_HTML_TAG = re.compile(r"<[^>]+>")

# Callout emoji map
_CALLOUT_EMOJI: dict[str, str] = {
    "note": "📝",
    "tip": "💡",
    "warning": "⚠️",
    "danger": "🔴",
    "info": "ℹ️",
}
_CALLOUT_DEFAULT_EMOJI = "📌"

# Placeholder sentinel — used to protect extracted code blocks
_PLACEHOLDER_PREFIX = "\x00CB"
_PLACEHOLDER_SUFFIX = "\x00"


class TelegramFormatter:
    """Convert VaultNote content to Telegram-compatible HTML."""

    @classmethod
    def format_note(cls, note: Note, max_length: int = 4000) -> str:
        """Format a full note into a Telegram HTML string.

        Renders a header block followed by the converted body.
        Truncates at the last complete line before max_length.
        """
        header = cls._build_header(note)
        body_raw = note.body_without_frontmatter()
        body = cls._convert_body(body_raw)

        if header and body:
            full = header + "\n\n" + body
        elif header:
            full = header
        else:
            full = body

        if len(full) <= max_length:
            return full

        total_chars = len(full)
        notice = f"\n\n<i>…truncated ({total_chars} chars total)</i>"
        cutoff = max_length - len(notice)
        split_at = full.rfind("\n", 0, cutoff)
        if split_at == -1:
            split_at = cutoff
        return full[:split_at] + notice

    @classmethod
    def format_search_result(cls, hit: dict[str, Any], index: int) -> str:
        """Format a single search result hit as an HTML snippet."""
        meta = hit.get("metadata", {})
        title = _escape(meta.get("note_title", "Untitled"))
        note_path = _escape(meta.get("note_path", ""))
        heading = _escape(meta.get("heading", ""))
        distance = hit.get("distance", 0.0)
        relevance = max(0, round((1 - distance) * 100))

        raw_content = hit.get("content", "")
        content_raw = raw_content[:200].replace("\n", " ").strip()
        if len(raw_content) > 200:
            content_raw += "..."
        content = _escape(content_raw)

        location = f"<code>{note_path}</code>"
        if heading:
            location += f" → {heading}"

        return f"<b>{index}. {title}</b> ({relevance}% match)\n{location}\n<i>{content}</i>"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_header(cls, note: Note) -> str:
        """Build the structured header block for a note."""
        parts: list[str] = []

        title = note.title or note.path.stem
        if title:
            parts.append(f"<b>{_escape(title)}</b>")

        rel = str(note.path)
        if rel:
            parts.append(f"<code>{_escape(rel)}</code>")

        if note.tags:
            tag_str = " ".join(f"#{_escape(t)}" for t in note.tags)
            parts.append(f"<i>{tag_str}</i>")

        fm = note.frontmatter

        date_val = fm.get("date") or fm.get("created") or note.created
        if date_val:
            date_str = str(date_val)
            if "T" in date_str:
                date_str = date_str.split("T")[0]
            elif " " in date_str and len(date_str) > 10:
                date_str = date_str[:10]
            parts.append(_escape(date_str))

        note_type_val = fm.get("type") or fm.get("note_type") or note.note_type.value
        if note_type_val and note_type_val != "fleeting":
            parts.append(f"<i>{_escape(str(note_type_val))}</i>")

        return "\n".join(parts)

    @classmethod
    def _convert_body(cls, raw: str) -> str:
        """Convert raw markdown body to Telegram HTML.

        Processing order is critical:
        1. Extract fenced code blocks → placeholders (protect their content)
        2. Strip raw HTML tags
        3. Process each line structurally (block-level identification)
           — escape HTML entities in the *text portion* of each line
           — apply block-level markup (headings, quotes, lists, rules)
        4. Apply inline markdown (bold, italic, code, wikilinks, images)
           to the already-structured, already-escaped text
        5. Restore code block placeholders
        """
        if not raw.strip():
            return ""

        # --- Step 1: extract code blocks ---
        placeholders: dict[str, str] = {}
        counter = 0

        def _extract(m: re.Match[str]) -> str:
            nonlocal counter
            lang = (m.group(1) or "").lower()
            body = m.group(2)
            if lang in ("dataview", "mermaid"):
                html = f"[{lang.capitalize()} block]"
            else:
                html = f"<pre>{_escape(body)}</pre>"
            key = f"{_PLACEHOLDER_PREFIX}{counter}{_PLACEHOLDER_SUFFIX}"
            placeholders[key] = html
            counter += 1
            return key

        text = _RE_CODE_BLOCK.sub(_extract, raw)

        # --- Step 2: strip HTML tags from non-placeholder content ---
        # Do this line-by-line to avoid touching placeholders
        lines_raw = text.split("\n")
        clean_lines: list[str] = []
        for line in lines_raw:
            if _PLACEHOLDER_PREFIX in line:
                clean_lines.append(line)
            else:
                clean_lines.append(_RE_HTML_TAG.sub("", line))

        # --- Step 3: per-line block-level processing ---
        # Each line is classified, its text portion is HTML-escaped,
        # and it is wrapped in the appropriate block markup.
        result_lines: list[str] = []
        for line in clean_lines:
            # Preserve placeholder lines verbatim
            if _PLACEHOLDER_PREFIX in line:
                result_lines.append(line)
                continue

            # Horizontal rule
            if _RE_HRULE_LINE.match(line):
                result_lines.append("─────")
                continue

            # Obsidian callout: > [!type] title
            m_callout = _RE_CALLOUT_LINE.match(line)
            if m_callout:
                ctype = m_callout.group(1).lower()
                ctitle = (
                    m_callout.group(2).strip() if m_callout.group(2).strip() else ctype.capitalize()
                )
                emoji = _CALLOUT_EMOJI.get(ctype, _CALLOUT_DEFAULT_EMOJI)
                result_lines.append(f"{emoji} <b>{_escape(ctitle)}</b>")
                continue

            # Plain blockquote
            if _RE_BLOCKQUOTE_PREFIX.match(line):
                inner = _RE_BLOCKQUOTE_PREFIX.sub("", line)
                result_lines.append(f"┃ <i>{_escape(inner)}</i>")
                continue

            # Heading
            m_heading = _RE_HEADING_LINE.match(line)
            if m_heading:
                result_lines.append(f"<b>{_escape(m_heading.group(1))}</b>")
                continue

            # Task — done
            m_task_done = _RE_TASK_DONE_LINE.match(line)
            if m_task_done:
                indent = m_task_done.group(1)
                rest = _escape(m_task_done.group(2))
                result_lines.append(f"{indent}☑ {rest}")
                continue

            # Task — open
            m_task_open = _RE_TASK_OPEN_LINE.match(line)
            if m_task_open:
                indent = m_task_open.group(1)
                rest = _escape(m_task_open.group(2))
                result_lines.append(f"{indent}☐ {rest}")
                continue

            # Unordered list item
            m_list = _RE_LIST_ITEM_LINE.match(line)
            if m_list:
                indent = m_list.group(1)
                rest = _escape(m_list.group(2))
                result_lines.append(f"{indent}• {rest}")
                continue

            # Plain line — escape and pass through
            result_lines.append(_escape(line))

        text = "\n".join(result_lines)

        # --- Step 4: inline transformations (on escaped text) ---

        # Images (still in raw form — they were not escaped because
        # _escape only touches &, <, > and the image patterns use [ ] ( ))
        text = _RE_IMG_WIKI.sub(lambda m: f"[🖼 {m.group(1).split('.')[0]}]", text)
        text = _RE_IMG_MD.sub(lambda m: f"[🖼 {m.group(1) or 'Image'}]", text)

        # Wikilinks
        text = _RE_WIKILINK_ALIAS.sub(lambda m: f"<b>{m.group(2)}</b>", text)
        text = _RE_WIKILINK_PLAIN.sub(lambda m: f"<b>{m.group(1)}</b>", text)

        # Bold (before italic)
        text = _RE_BOLD_STAR.sub(lambda m: f"<b>{m.group(1)}</b>", text)
        text = _RE_BOLD_UNDER.sub(lambda m: f"<b>{m.group(1)}</b>", text)

        # Italic
        text = _RE_ITALIC_STAR.sub(lambda m: f"<i>{m.group(1)}</i>", text)
        text = _RE_ITALIC_UNDER.sub(lambda m: f"<i>{m.group(1)}</i>", text)

        # Inline code
        text = _RE_INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", text)

        # --- Step 5: restore code block placeholders ---
        for key, value in placeholders.items():
            text = text.replace(key, value)

        return text.strip()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
