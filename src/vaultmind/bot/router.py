"""Message router — heuristic-based intent classification for plain text messages.

Classifies user messages into intents (capture, greeting, question, conversational)
using pure regex/string checks. Zero LLM cost for routing decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class Intent(Enum):
    """Classified intent for an incoming text message."""

    capture = auto()
    greeting = auto()
    question = auto()
    conversational = auto()


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Routing decision with cleaned content."""

    intent: Intent
    content: str  # Original text or prefix-stripped for capture


# Prefixes that signal explicit capture intent
CAPTURE_PREFIXES = ("note:", "save:", "capture:", "remember:", "jot:", "log:")

# Short casual phrases — matched case-insensitively against full text
_GREETING_PATTERNS = re.compile(
    r"^("
    r"h(i|ey|ello|owdy|ola)"
    r"|yo"
    r"|sup"
    r"|thanks?( you)?|thx|ty"
    r"|ok(ay)?"
    r"|cool"
    r"|nice"
    r"|great"
    r"|good (morning|afternoon|evening|night|day)"
    r"|gm|gn"
    r"|bye|cya|later|see ya"
    r"|lol|lmao|haha"
    r"|yes|no|nah|yep|nope|yea|yeah|ya"
    r"|sure"
    r"|got it"
    r"|what'?s up"
    r"|np"
    r")!*\.?$",
    re.IGNORECASE,
)

# Interrogative starters for question detection
_INTERROGATIVE_STARTERS = re.compile(
    r"^(who|what|where|when|why|how|which|is|are|was|were|do|does|did|can|could|should|would|will)\b",
    re.IGNORECASE,
)

# Thresholds
_LONG_TEXT_CHARS = 500
_MULTILINE_THRESHOLD = 3
_GREETING_MAX_CHARS = 60
_MIN_QUESTION_CHARS = 10


class MessageRouter:
    """Classifies plain text messages into routing intents."""

    def classify(self, text: str) -> RoutingResult:
        """Classify text into an intent using heuristics only.

        Order of checks:
        1. Explicit capture prefix → capture
        2. Long text or multiline → capture
        3. Short casual phrase → greeting
        4. Question pattern → question
        5. Fallback → conversational
        """
        stripped = text.strip()

        # 1. Explicit capture prefix
        lower = stripped.lower()
        for prefix in CAPTURE_PREFIXES:
            if lower.startswith(prefix):
                content = stripped[len(prefix) :].strip()
                if content:
                    return RoutingResult(intent=Intent.capture, content=content)
                # Empty after prefix — treat as conversational
                return RoutingResult(intent=Intent.conversational, content=stripped)

        # 2. Long text or multiline (pasted content = intentional capture)
        newline_count = stripped.count("\n")
        if len(stripped) > _LONG_TEXT_CHARS or newline_count >= _MULTILINE_THRESHOLD:
            return RoutingResult(intent=Intent.capture, content=stripped)

        # 3. Short casual greeting
        if len(stripped) <= _GREETING_MAX_CHARS and _GREETING_PATTERNS.match(stripped):
            return RoutingResult(intent=Intent.greeting, content=stripped)

        # 4. Question detection
        if stripped.endswith("?") and len(stripped) > _MIN_QUESTION_CHARS:
            return RoutingResult(intent=Intent.question, content=stripped)
        if _INTERROGATIVE_STARTERS.match(stripped) and len(stripped) > _MIN_QUESTION_CHARS:
            return RoutingResult(intent=Intent.question, content=stripped)

        # 5. Fallback
        return RoutingResult(intent=Intent.conversational, content=stripped)
