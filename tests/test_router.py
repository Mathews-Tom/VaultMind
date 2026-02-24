"""Tests for MessageRouter — heuristic intent classification."""

from __future__ import annotations

import pytest

from vaultmind.bot.router import Intent, MessageRouter


@pytest.fixture
def router() -> MessageRouter:
    return MessageRouter()


class TestCaptureIntent:
    """Messages that should route to capture."""

    @pytest.mark.parametrize(
        "text",
        [
            "note: buy groceries",
            "save: interesting article about LLMs",
            "capture: meeting notes from standup",
            "remember: dentist appointment thursday",
            "jot: idea for blog post",
            "log: completed migration to v2",
            "Note: case insensitive prefix",
            "SAVE: uppercase prefix",
        ],
    )
    def test_explicit_capture_prefix(self, router: MessageRouter, text: str) -> None:
        result = router.classify(text)
        assert result.intent is Intent.capture

    def test_capture_prefix_strips_prefix(self, router: MessageRouter) -> None:
        result = router.classify("note: buy groceries")
        assert result.content == "buy groceries"

    def test_capture_prefix_strips_with_extra_spaces(self, router: MessageRouter) -> None:
        result = router.classify("save:   important thing  ")
        assert result.content == "important thing"

    def test_empty_after_prefix_is_conversational(self, router: MessageRouter) -> None:
        """Empty content after prefix falls through to conversational."""
        result = router.classify("note:")
        assert result.intent is Intent.conversational

    def test_long_text_is_capture(self, router: MessageRouter) -> None:
        """Text over 500 chars routes to capture."""
        long_text = "a" * 501
        result = router.classify(long_text)
        assert result.intent is Intent.capture

    def test_multiline_is_capture(self, router: MessageRouter) -> None:
        """Three or more newlines routes to capture."""
        multiline = "line 1\nline 2\nline 3\nline 4"
        result = router.classify(multiline)
        assert result.intent is Intent.capture

    def test_two_newlines_not_capture(self, router: MessageRouter) -> None:
        """Two newlines is not enough for auto-capture."""
        text = "line 1\nline 2\nline 3"
        result = router.classify(text)
        assert result.intent is not Intent.capture


class TestGreetingIntent:
    """Short casual phrases that should route to greeting."""

    @pytest.mark.parametrize(
        "text",
        [
            "hi",
            "hey",
            "hello",
            "Hi!",
            "Hey!",
            "howdy",
            "yo",
            "sup",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "ok",
            "okay",
            "cool",
            "nice",
            "good morning",
            "good night",
            "gm",
            "gn",
            "bye",
            "lol",
            "yes",
            "no",
            "nah",
            "yep",
            "nope",
            "sure",
            "got it",
            "what's up",
            "whats up",
        ],
    )
    def test_greeting_detection(self, router: MessageRouter, text: str) -> None:
        result = router.classify(text)
        assert result.intent is Intent.greeting, (
            f"Expected greeting for '{text}', got {result.intent}"
        )

    def test_long_greeting_is_not_greeting(self, router: MessageRouter) -> None:
        """Greeting-like text over 60 chars falls through."""
        text = "hello there my friend how are you doing on this fine day today"
        assert len(text) > 60
        result = router.classify(text)
        assert result.intent is not Intent.greeting


class TestQuestionIntent:
    """Messages that should route to question handling."""

    @pytest.mark.parametrize(
        "text",
        [
            "What did I write about distributed systems?",
            "How does the indexer pipeline work?",
            "Who mentioned machine learning in my notes?",
            "Where are my notes about Python async?",
            "When did I start the VaultMind project?",
            "Why did I choose ChromaDB over Pinecone?",
            "Which notes reference knowledge graphs?",
            "Can you find my notes on embeddings?",
            "Is there anything about Rust in my vault?",
            "Does my vault have notes on Docker?",
        ],
    )
    def test_question_detection(self, router: MessageRouter, text: str) -> None:
        result = router.classify(text)
        assert result.intent is Intent.question, (
            f"Expected question for '{text}', got {result.intent}"
        )

    def test_question_mark_detection(self, router: MessageRouter) -> None:
        result = router.classify("anything about RAG pipelines?")
        assert result.intent is Intent.question

    def test_very_short_question_mark_is_not_question(self, router: MessageRouter) -> None:
        """Very short question-like text (<=10 chars) is not classified as question."""
        result = router.classify("huh?")
        assert result.intent is not Intent.question


class TestConversationalIntent:
    """Messages that should fall through to conversational."""

    @pytest.mark.parametrize(
        "text",
        [
            "Tell me about my recent projects",
            "Summarize my vault notes on AI",
            "I've been thinking about knowledge management",
            "Compare my notes on React and Vue",
            "Interesting that I wrote so much about embeddings",
        ],
    )
    def test_conversational_fallback(self, router: MessageRouter, text: str) -> None:
        result = router.classify(text)
        assert result.intent is Intent.conversational, (
            f"Expected conversational for '{text}', got {result.intent}"
        )

    def test_whitespace_handling(self, router: MessageRouter) -> None:
        """Whitespace is stripped before classification."""
        result = router.classify("  Tell me about my projects  ")
        assert result.intent is Intent.conversational
        assert result.content == "Tell me about my projects"


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_exactly_500_chars_not_capture(self, router: MessageRouter) -> None:
        text = "a" * 500
        result = router.classify(text)
        assert result.intent is not Intent.capture

    def test_501_chars_is_capture(self, router: MessageRouter) -> None:
        text = "a" * 501
        result = router.classify(text)
        assert result.intent is Intent.capture

    def test_exactly_3_newlines_is_capture(self, router: MessageRouter) -> None:
        text = "a\nb\nc\nd"
        result = router.classify(text)
        assert result.intent is Intent.capture

    def test_capture_prefix_priority_over_question(self, router: MessageRouter) -> None:
        """Capture prefix takes priority over question patterns."""
        result = router.classify("note: what is the meaning of life?")
        assert result.intent is Intent.capture
        assert result.content == "what is the meaning of life?"

    def test_empty_string(self, router: MessageRouter) -> None:
        result = router.classify("")
        assert result.intent is Intent.conversational

    def test_whitespace_only(self, router: MessageRouter) -> None:
        result = router.classify("   ")
        assert result.intent is Intent.conversational
