"""Tests for AutoTagger — LLM tag suggestions and frontmatter application.

Quarantine/approval-state tests moved to `tests/test_review_queue.py`
(M7) — `AutoTagger` is now pure suggestion generation with no persistence
or approval state of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vaultmind.indexer.auto_tagger import AutoTagger
from vaultmind.indexer.tags import apply_tags, tag_appliers
from vaultmind.llm.client import LLMResponse, Message
from vaultmind.services.review_queue import Impact, Lane, ProposalKind, ProposalStatus, ReviewQueue
from vaultmind.vault.models import Note
from vaultmind.vault.security import PathTraversalError

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

        apply_tags(note_path, ["python", "automation"])

        import frontmatter

        post = frontmatter.load(note_path)
        assert post.metadata["tags"] == ["python", "automation"]

    def test_merges_with_existing_tags_no_duplicates(self, tmp_path: Path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("---\ntitle: Test\ntags: [python]\n---\nBody content.\n")

        apply_tags(note_path, ["python", "new-tag"])

        import frontmatter

        post = frontmatter.load(note_path)
        assert post.metadata["tags"] == ["python", "new-tag"]
        assert post.metadata["tags"].count("python") == 1


class TestQueuedTagOperations:
    def test_missing_note_fails_and_explicit_retry_merges_without_losing_body(
        self, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        queue = ReviewQueue(
            tmp_path / "queue.db", appliers=tag_appliers(vault, tmp_path / "tag_vocabulary.json")
        )
        proposal = queue.propose(
            ProposalKind.TAG_APPLICATION,
            0.6,
            Impact.LOW,
            "apply",
            {"note_path": "note.md", "tags": ["python", "new-tag"]},
        )
        failed = queue.approve(proposal.proposal_id)
        assert failed.status is ProposalStatus.FAILED
        note = vault / "note.md"
        note.write_text("---\ntitle: Kept\ntags: [python]\n---\nOriginal body.\n")
        retried = queue.retry(proposal.proposal_id)
        assert retried.status is ProposalStatus.APPLIED
        import frontmatter

        post = frontmatter.load(note)
        assert post.content == "Original body."
        assert post.metadata == {"title": "Kept", "tags": ["python", "new-tag"]}
        before = note.stat().st_mtime_ns
        assert queue.approve(proposal.proposal_id) is None
        assert queue.retry(proposal.proposal_id) is None
        assert note.stat().st_mtime_ns == before
        queue.close()

    def test_queued_targets_cannot_escape_vault(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("Preserve outside.")
        (vault / "escape.md").symlink_to(outside)
        operation = tag_appliers(vault, tmp_path / "vocab.json")[ProposalKind.TAG_APPLICATION]
        for target in ("../outside.md", "escape.md"):
            with pytest.raises(PathTraversalError):
                operation({"note_path": target, "tags": ["no"]})
        assert outside.read_text() == "Preserve outside."


def test_cli_approve_skim_preserves_unrelated_pending_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frontmatter
    from click.testing import CliRunner

    import vaultmind.cli as app
    import vaultmind.config as config

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("---\ntags: [original]\n---\nPreserved original body.\n")
    home = tmp_path / "home"
    monkeypatch.setattr(config, "VAULTMIND_HOME", home)
    monkeypatch.setattr(app, "_create_llm_client", lambda settings: FakeLLMClient())
    database = tmp_path / "review.db"
    toml = tmp_path / "fixture.toml"
    toml.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        '[llm]\nprovider = "ollama"\n'
        f"[autonomy]\ndb_path = {json.dumps(str(database))}\n"
    )
    queue = ReviewQueue(database)
    tag = queue.propose(
        ProposalKind.TAG_APPLICATION,
        0.6,
        Impact.LOW,
        "Apply reviewed tag",
        {"note_path": "note.md", "tags": ["reviewed"]},
        lane_override=Lane.SKIM,
    )
    unrelated = queue.propose(
        ProposalKind.MATURATION_SYNTHESIS,
        0.6,
        Impact.LOW,
        "Unrelated synthesis",
        {"fingerprint": "other-cluster"},
        lane_override=Lane.SKIM,
    )
    queue.close()

    result = CliRunner().invoke(app.cli, ["--config", str(toml), "auto-tag", "--approve-skim"])

    reopened = ReviewQueue(database)
    try:
        assert reopened.get(unrelated.proposal_id) == unrelated
        assert reopened.get(tag.proposal_id).status is ProposalStatus.APPLIED
        assert frontmatter.load(note).metadata["tags"] == ["original", "reviewed"]
        assert frontmatter.load(note).content == "Preserved original body."
        assert result.exit_code == 0, result.output
    finally:
        reopened.close()


def test_cli_apply_reports_deduplicated_pending_and_prior_human_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    import vaultmind.cli as app
    import vaultmind.config as config

    vault = tmp_path / "vault"
    vault.mkdir()
    seed = vault / "seed.md"
    seed.write_text("---\ntags: [known]\n---\n" + "Preserved vocabulary source. " * 8)
    note = vault / "note.md"
    note.write_text("Preserved untagged note. " * 8)
    before = {path: path.read_bytes() for path in (seed, note)}
    monkeypatch.setattr(config, "VAULTMIND_HOME", tmp_path / "home")
    client = FakeLLMClient(tags=["unreviewed"], new_tags=["novel"])
    monkeypatch.setattr(app, "_create_llm_client", lambda settings: client)
    database = tmp_path / "review.db"
    toml = tmp_path / "fixture.toml"
    toml.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        '[llm]\nprovider = "ollama"\n'
        f"[autonomy]\ndb_path = {json.dumps(str(database))}\n"
    )
    queue = ReviewQueue(database)
    rejected = queue.propose(
        ProposalKind.TAG_VOCABULARY,
        0.6,
        Impact.LOW,
        "Previously rejected tag",
        {"tag": "novel"},
        lane_override=Lane.SKIM,
    )
    queue.reject(rejected.proposal_id, reason="Preserve this human decision.")
    queue.close()

    result = CliRunner().invoke(app.cli, ["--config", str(toml), "auto-tag", "--apply"])

    reopened = ReviewQueue(database)
    try:
        assert result.exit_code == 0, result.output
        assert {path: path.read_bytes() for path in (seed, note)} == before
        assert reopened.get(rejected.proposal_id).status is ProposalStatus.REJECTED
        pending = reopened.list_pending()
        assert len(pending) == 1
        assert pending[0].kind is ProposalKind.TAG_VOCABULARY
        assert pending[0].payload == {"tag": "unreviewed"}
        assert pending[0].occurrence_count == 2
        # These are reported operation counts, not a label/format snapshot.
        assert "2 unique proposal(s)" in result.output
        assert "0 newly applied" in result.output
        assert "1 previously resolved" in result.output
        assert "1 pending review" in result.output
    finally:
        reopened.close()


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
