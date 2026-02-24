"""Input sanitization for Telegram bot handlers.

Strips null bytes, trims whitespace, truncates to length limits,
and logs potential injection patterns (detection only, never blocks).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Length limits per operation type
MAX_CAPTURE_LENGTH = 10_000
MAX_QUERY_LENGTH = 500
MAX_LLM_INPUT_LENGTH = 8_000
MAX_EDIT_INSTRUCTION_LENGTH = 2_000
MAX_PATH_LENGTH = 500

# Injection detection patterns (log-only, never block)
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|(?:im_start|im_end|system|endoftext)\|>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
]


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitizing user input."""

    text: str
    was_modified: bool
    flags: list[str] = field(default_factory=list)


def sanitize_text(
    text: str,
    *,
    max_length: int,
    operation: str = "",
) -> SanitizationResult:
    """Sanitize user input text.

    - Strips null bytes
    - Strips leading/trailing whitespace
    - Truncates to max_length
    - Logs injection pattern matches (never blocks)
    """
    flags: list[str] = []
    original = text

    # Strip null bytes
    if "\x00" in text:
        text = text.replace("\x00", "")
        flags.append("null_bytes_stripped")

    # Strip whitespace
    text = text.strip()

    # Truncate
    if len(text) > max_length:
        text = text[:max_length]
        flags.append("length_truncated")
        logger.warning(
            "Input truncated from %d to %d chars for %s",
            len(original),
            max_length,
            operation,
        )

    # Log injection attempts (detection only, never block)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Potential injection detected in %s input: %s",
                operation,
                pattern.pattern,
            )
            flags.append("injection_pattern_detected")
            break  # Log once per input, not per pattern

    return SanitizationResult(
        text=text,
        was_modified=text != original,
        flags=flags,
    )


def sanitize_path(path: str) -> SanitizationResult:
    """Sanitize a file path input."""
    return sanitize_text(path, max_length=MAX_PATH_LENGTH, operation="path")
