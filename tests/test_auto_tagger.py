"""Tests for AutoTagger — LLM tag suggestions and frontmatter application.

Quarantine/approval-state tests moved to `tests/test_review_queue.py`
(M7) — `AutoTagger` is now pure suggestion generation with no persistence
or approval state of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vaultmind.indexer.auto_tagger import AutoTagger
from vaultmind.llm.client import LLMResponse, Message
from vaultmind.vault.models import Note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAutoTagConfig:
    """Minimal AutoTagConfig stand-in."""

    def __init__(
        self,
        max_tags_per_note: int = 2,
        min_content_length: int = 50,
    ) -> None:
        self.enabled = True
        self.max_tags_per_note = max_tags_per_note
        self.min_content_length = min_content_length
        self.tagging_model = "test-model"


class FakeLLMClient:
    """Fake LLM client returning canned JSON responses."""

    def __init__(self, tags: list[str] | None = None, new_tags: list[str] | None = None) -> None:
        self._tags = tags or []
        self._new_tags = new_tags or []
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages})
        result = json.dumps({"tags": self._tags, "new_tags": self._new_tags})
        return LLMResponse(text=result, model=model, usage={"total_tokens": 100})


def _make_note(
    path: str = "test.md",
    title: str = "Test Note",
    content: str = "This is a test note with enough content to pass length checks. " * 5,
    tags: list[str] | None = None,
) -> Note:
    return Note(
        path=Path(path),
        title=title,
        content=content,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# Tests — AutoTagger.suggest_tags
# ---------------------------------------------------------------------------


class TestSuggestTags:
    def test_returns_known_tags(self) -> None:
        client = FakeLLMClient(tags=["python", "automation"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"python", "automation", "ml"})

        assert result is not None
        assert "python" in result.suggested_tags
        assert "automation" in result.suggested_tags

    def test_separates_novel_tags_into_new_tags(self) -> None:
        client = FakeLLMClient(tags=["python", "novel-concept"], new_tags=["brand-new"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"python"})

        assert result is not None
        # Novel tags are the caller's responsibility (TAG_VOCABULARY proposals
        # in the review queue) — never auto-included in suggested_tags.
        assert "novel-concept" in result.new_tags
        assert "brand-new" in result.new_tags
        assert "python" in result.suggested_tags
        assert "novel-concept" not in result.suggested_tags

    def test_short_note_returns_none(self) -> None:
        client = FakeLLMClient(tags=["python"])
        config = FakeAutoTagConfig(min_content_length=100)
        tagger = AutoTagger(config, client, "test-model")

        note = _make_note(content="short")
        result = tagger.suggest_tags(note, vault_tags={"python"})
        assert result is None

    def test_caller_supplied_approved_vocabulary_is_treated_as_known(self) -> None:
        """The caller (CLI) merges any previously-approved novel tags into
        `vault_tags` before calling — AutoTagger has no vocabulary state of
        its own to consult."""
        client = FakeLLMClient(tags=["custom-tag"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"custom-tag"})

        assert result is not None
        assert "custom-tag" in result.suggested_tags
        assert result.new_tags == []

    def test_respects_max_tags(self) -> None:
        client = FakeLLMClient(tags=["a", "b", "c", "d"])
        config = FakeAutoTagConfig(max_tags_per_note=2)
        tagger = AutoTagger(config, client, "test-model")

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"a", "b", "c", "d"})

        assert result is not None
        assert len(result.suggested_tags) <= 2


class TestApplyTags:
    def test_writes_tags_to_frontmatter(self, tmp_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("---\ntitle: Test\n---\nBody content.\n")

        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        tagger.apply_tags(note_path, ["python", "automation"])

        import frontmatter

        post = frontmatter.load(note_path)
        assert post.metadata["tags"] == ["python", "automation"]

    def test_merges_with_existing_tags_no_duplicates(self, tmp_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("---\ntitle: Test\ntags: [python]\n---\nBody content.\n")

        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        tagger.apply_tags(note_path, ["python", "new-tag"])

        import frontmatter

        post = frontmatter.load(note_path)
        assert post.metadata["tags"] == ["python", "new-tag"]
        assert post.metadata["tags"].count("python") == 1


class TestCollectVaultTags:
    def test_collects_from_multiple_notes(self) -> None:
        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model")

        notes = [
            _make_note(path="a.md", tags=["python", "ml"]),
            _make_note(path="b.md", tags=["automation"]),
            _make_note(path="c.md", tags=["python"]),
        ]
        tags = tagger.collect_vault_tags(notes)
        assert tags == {"python", "ml", "automation"}
