"""Tests for AutoTagger — tag suggestions, quarantine, and frontmatter application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vaultmind.indexer.auto_tagger import AutoTagger, QuarantineState
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
# Tests — QuarantineState
# ---------------------------------------------------------------------------


class TestQuarantineState:
    def test_approve(self) -> None:
        q = QuarantineState(quarantined_tags={"new-tag"})
        q.approve("new-tag")
        assert "new-tag" in q.approved_tags
        assert "new-tag" not in q.quarantined_tags

    def test_reject(self) -> None:
        q = QuarantineState(quarantined_tags={"bad-tag"})
        q.reject("bad-tag")
        assert "bad-tag" not in q.quarantined_tags
        assert "bad-tag" not in q.approved_tags

    def test_approve_all(self) -> None:
        q = QuarantineState(quarantined_tags={"a", "b", "c"})
        q.approve_all()
        assert q.quarantined_tags == set()
        assert q.approved_tags == {"a", "b", "c"}

    def test_roundtrip(self) -> None:
        q = QuarantineState(approved_tags={"x"}, quarantined_tags={"y"})
        data = q.to_dict()
        q2 = QuarantineState.from_dict(data)
        assert q2.approved_tags == {"x"}
        assert q2.quarantined_tags == {"y"}


# ---------------------------------------------------------------------------
# Tests — AutoTagger
# ---------------------------------------------------------------------------


@pytest.fixture
def quarantine_path(tmp_path: Path) -> Path:
    return tmp_path / "quarantine.json"


class TestSuggestTags:
    def test_returns_known_tags(self, quarantine_path: Path) -> None:
        client = FakeLLMClient(tags=["python", "automation"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"python", "automation", "ml"})

        assert result is not None
        assert "python" in result.suggested_tags
        assert "automation" in result.suggested_tags

    def test_quarantines_new_tags(self, quarantine_path: Path) -> None:
        client = FakeLLMClient(tags=["python", "novel-concept"], new_tags=["brand-new"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"python"})

        assert result is not None
        # novel-concept and brand-new should be in quarantine
        assert "novel-concept" in tagger.quarantine.quarantined_tags
        assert "brand-new" in tagger.quarantine.quarantined_tags
        # Only known tags should be in suggested_tags
        assert "python" in result.suggested_tags
        assert "novel-concept" not in result.suggested_tags

    def test_short_note_returns_none(self, quarantine_path: Path) -> None:
        client = FakeLLMClient(tags=["python"])
        config = FakeAutoTagConfig(min_content_length=100)
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note(content="short")
        result = tagger.suggest_tags(note, vault_tags={"python"})
        assert result is None

    def test_approved_quarantine_tags_used(self, quarantine_path: Path) -> None:
        # Pre-approve some tags
        q = QuarantineState(approved_tags={"custom-tag"})
        quarantine_path.write_text(json.dumps(q.to_dict()))

        client = FakeLLMClient(tags=["custom-tag"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags=set())

        assert result is not None
        assert "custom-tag" in result.suggested_tags

    def test_respects_max_tags(self, quarantine_path: Path) -> None:
        client = FakeLLMClient(tags=["a", "b", "c", "d"])
        config = FakeAutoTagConfig(max_tags_per_note=2)
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note()
        result = tagger.suggest_tags(note, vault_tags={"a", "b", "c", "d"})

        assert result is not None
        assert len(result.suggested_tags) <= 2


class TestApplyTags:
    def test_writes_tags_to_frontmatter(self, tmp_path: Path, quarantine_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("---\ntitle: Test\ntags: [existing]\n---\n\nBody content.\n")

        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        tagger.apply_tags(note_path, ["new-tag", "another"])

        import frontmatter

        with open(note_path) as f:
            post = frontmatter.load(f)

        assert "existing" in post.metadata["tags"]
        assert "new-tag" in post.metadata["tags"]
        assert "another" in post.metadata["tags"]

    def test_no_duplicates_on_reapply(self, tmp_path: Path, quarantine_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("---\ntitle: Test\ntags: [python]\n---\n\nBody content.\n")

        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        tagger.apply_tags(note_path, ["python", "new"])

        import frontmatter

        with open(note_path) as f:
            post = frontmatter.load(f)

        assert post.metadata["tags"].count("python") == 1


class TestQuarantinePersistence:
    def test_save_and_load(self, quarantine_path: Path) -> None:
        client = FakeLLMClient(tags=["unknown-tag"])
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        note = _make_note()
        tagger.suggest_tags(note, vault_tags=set())
        tagger.save_quarantine()

        assert quarantine_path.exists()
        data = json.loads(quarantine_path.read_text())
        assert "unknown-tag" in data["quarantined"]


class TestCollectVaultTags:
    def test_collects_from_multiple_notes(self, quarantine_path: Path) -> None:
        client = FakeLLMClient()
        config = FakeAutoTagConfig()
        tagger = AutoTagger(config, client, "test-model", quarantine_path)  # type: ignore[arg-type]

        notes = [
            _make_note(tags=["python", "ml"]),
            _make_note(tags=["automation", "python"]),
        ]
        tags = tagger.collect_vault_tags(notes)
        assert tags == {"python", "ml", "automation"}
