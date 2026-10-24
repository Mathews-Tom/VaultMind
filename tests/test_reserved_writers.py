"""Ordinary tag/mark mutations cannot cross the manual wiki namespace boundary."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import frontmatter
import pytest

from vaultmind.bot.handlers.context import HandlerContext
from vaultmind.bot.handlers.delete import handle_delete_callback
from vaultmind.bot.handlers.edit import handle_edit_callback
from vaultmind.config import AutoTagConfig, ContradictionConfig, load_settings
from vaultmind.contradiction.detector import ContradictionDetector
from vaultmind.contradiction.marking import mark_contradicted, write_superseded_block
from vaultmind.graph.evolution import LineageStore
from vaultmind.indexer.auto_tagger import AutoTagger
from vaultmind.indexer.duplicate_detector import DuplicateMatch, MatchType
from vaultmind.indexer.tags import apply_tags, save_tag_vocabulary, tag_appliers
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.services.review_queue import Impact, Lane, ProposalKind, ProposalStatus, ReviewQueue
from vaultmind.vault.events import NoteCreatedEvent
from vaultmind.vault.models import Note
from vaultmind.vault.security import ReservedNamespaceError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class ProtectedVault:
    root: Path
    intake: SourceIntake
    source_id: str
    source_path: str

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for namespace in ("wiki", "_wiki")
            for path in (self.root / namespace).rglob("*")
            if path.is_file()
        }

    def target(self, namespace: str, access: str) -> Path:
        relative = (
            Path("curated.md")
            if namespace == "wiki"
            else Path(self.source_path).relative_to("_wiki")
        )
        if access == "alias":
            alias = self.root / "ordinary-link"
            alias.symlink_to(self.root / namespace, target_is_directory=True)
            return alias / relative
        if access == "case":
            return self.root / namespace.upper() / relative
        return self.root / namespace / relative


@pytest.fixture
def protected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ProtectedVault]:
    for name in tuple(os.environ):
        if name.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "vault"
    root.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(root))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n',
        encoding="utf-8",
    )
    with SourceIntake(load_settings(config)) as intake:
        captured = intake.preserve_text(
            "# Retained source\n\nOriginal evidence must survive.\n",
            original_name="original.md",
            origin="synthetic:test",
            adapter="test",
        )
        (root / "wiki").mkdir()
        (root / "wiki" / "curated.md").write_text(
            "---\ntitle: Curated\ntags: [approved]\n---\n\nReviewed knowledge.\n",
            encoding="utf-8",
        )
        yield ProtectedVault(root, intake, captured.source_id, captured.path)


def detector(root: Path, queue: ReviewQueue | None = None) -> ContradictionDetector:
    return ContradictionDetector(
        ContradictionConfig(enabled=True, auto_resolve=True),
        MagicMock(),
        MagicMock(),
        "scripted-test",
        root,
        MagicMock(),
        gap_store=MagicMock(),
        review_queue=queue,
    )


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("access", ["direct", "alias", "case"])
@pytest.mark.parametrize("operation", ["tags", "contradiction", "supersede"])
def test_low_level_mutations_reject_before_reading(
    protected: ProtectedVault, namespace: str, access: str, operation: str
) -> None:
    target = protected.target(namespace, access)
    before = protected.snapshot()
    source_before = protected.intake.read(protected.source_id).snapshot
    with patch.object(frontmatter, "load") as load:
        with pytest.raises(ReservedNamespaceError) as error:
            if operation == "tags":
                apply_tags(target, ["unreviewed"], vault_root=protected.root)
            elif operation == "contradiction":
                mark_contradicted(
                    target, "ordinary.md", "Ordinary", "not approved", vault_root=protected.root
                )
            else:
                write_superseded_block(
                    target,
                    vault_root=protected.root,
                    callout_type="superseded",
                    title="Unreviewed edit",
                    rationale="not approved",
                    frontmatter_key="superseded_at",
                    frontmatter_value="bot:delete",
                    new_content="Unreviewed replacement.",
                )
        load.assert_not_called()
    assert str(target) not in str(error.value)
    assert protected.snapshot() == before
    assert protected.intake.read(protected.source_id).snapshot == source_before


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("access", ["direct", "alias"])
def test_vocabulary_destination_rejects_before_mkdir_or_queue_creation(
    protected: ProtectedVault, tmp_path: Path, namespace: str, access: str
) -> None:
    destination = protected.target(namespace, access).parent / "not-created" / "approved.json"
    before = protected.snapshot()
    with pytest.raises(ReservedNamespaceError):
        save_tag_vocabulary(destination, {"unreviewed"}, vault_root=protected.root)
    database = tmp_path / "not-created" / "queue.db"
    with pytest.raises(ReservedNamespaceError):
        ReviewQueue(database, appliers=tag_appliers(protected.root, destination))
    assert not destination.parent.exists()
    assert not database.parent.exists()
    assert protected.snapshot() == before


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
def test_lexical_reserved_alias_cannot_launder_an_ordinary_note(
    tmp_path: Path, namespace: str
) -> None:
    root = tmp_path / "vault"
    ordinary = root / "ordinary"
    ordinary.mkdir(parents=True)
    target = ordinary / "note.md"
    target.write_text("---\ntitle: Ordinary\n---\n\nOriginal.\n", encoding="utf-8")
    (root / namespace).symlink_to(ordinary, target_is_directory=True)
    before = target.read_bytes()
    with pytest.raises(ReservedNamespaceError):
        apply_tags(root / namespace / "note.md", ["unreviewed"], vault_root=root)
    assert target.read_bytes() == before


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("access", ["direct", "alias"])
@pytest.mark.parametrize(
    "kind", [ProposalKind.TAG_APPLICATION, ProposalKind.CONTRADICTION_RESOLUTION]
)
def test_queued_mutation_rechecks_reserved_destination_at_execution(
    protected: ProtectedVault,
    tmp_path: Path,
    namespace: str,
    access: str,
    kind: ProposalKind,
) -> None:
    target = protected.target(namespace, access)
    queue = ReviewQueue(
        tmp_path / "queue.db",
        appliers=tag_appliers(protected.root, tmp_path / "approved.json"),
    )
    detector(protected.root, queue)
    payload = (
        {"note_path": str(target), "tags": ["unreviewed"]}
        if kind is ProposalKind.TAG_APPLICATION
        else {
            "loser_path": str(target),
            "winner_path": "ordinary.md",
            "winner_title": "Ordinary",
            "rationale": "not approved",
        }
    )
    try:
        proposal = queue.propose(
            kind, 0.5, Impact.HIGH, "pending", payload, lane_override=Lane.BLOCK
        )
        before = protected.snapshot()
        result = queue.approve(proposal.proposal_id)
        assert result is not None
        assert result.status is ProposalStatus.FAILED
        assert result.result.startswith("ReservedNamespaceError:")
        assert str(target) not in result.result
        assert protected.snapshot() == before
        assert queue.approve(proposal.proposal_id) is None
        assert queue.list_applied(kind) == []
    finally:
        queue.close()


@pytest.mark.parametrize(
    "kind", [ProposalKind.TAG_APPLICATION, ProposalKind.CONTRADICTION_RESOLUTION]
)
def test_pending_note_alias_is_revalidated_after_retargeting(
    protected: ProtectedVault, tmp_path: Path, kind: ProposalKind
) -> None:
    ordinary = protected.root / "ordinary.md"
    ordinary.write_text("---\ntitle: Ordinary\n---\n\nOriginal.\n", encoding="utf-8")
    alias = protected.root / "pending.md"
    alias.symlink_to(ordinary)
    queue = ReviewQueue(
        tmp_path / "queue.db", appliers=tag_appliers(protected.root, tmp_path / "approved.json")
    )
    detector(protected.root, queue)
    payload = (
        {"note_path": "pending.md", "tags": ["unreviewed"]}
        if kind is ProposalKind.TAG_APPLICATION
        else {
            "loser_path": "pending.md",
            "winner_path": "ordinary.md",
            "winner_title": "Ordinary",
            "rationale": "not approved",
        }
    )
    try:
        proposal = queue.propose(
            kind, 0.5, Impact.HIGH, "pending", payload, lane_override=Lane.BLOCK
        )
        alias.unlink()
        alias.symlink_to(protected.root / protected.source_path)
        before = protected.snapshot()
        ordinary_before = ordinary.read_bytes()
        result = queue.approve(proposal.proposal_id)
        assert result is not None
        assert result.status is ProposalStatus.FAILED
        assert result.result.startswith("ReservedNamespaceError:")
        assert protected.snapshot() == before
        assert ordinary.read_bytes() == ordinary_before
    finally:
        queue.close()


def test_pending_vocabulary_alias_is_revalidated_before_reading(
    protected: ProtectedVault, tmp_path: Path
) -> None:
    ordinary = tmp_path / "ordinary-vocabulary"
    ordinary.mkdir()
    alias = tmp_path / "vocabulary-link"
    alias.symlink_to(ordinary, target_is_directory=True)
    queue = ReviewQueue(
        tmp_path / "queue.db", appliers=tag_appliers(protected.root, alias / "approved.json")
    )
    try:
        proposal = queue.propose(
            ProposalKind.TAG_VOCABULARY,
            0.5,
            Impact.HIGH,
            "pending",
            {"tag": "unreviewed"},
            lane_override=Lane.BLOCK,
        )
        alias.unlink()
        alias.symlink_to(protected.root / "_wiki", target_is_directory=True)
        before = protected.snapshot()
        with patch("vaultmind.indexer.tags.load_tag_vocabulary") as load:
            result = queue.approve(proposal.proposal_id)
            load.assert_not_called()
        assert result is not None
        assert result.status is ProposalStatus.FAILED
        assert result.result.startswith("ReservedNamespaceError:")
        assert protected.snapshot() == before
        assert not (ordinary / "approved.json").exists()
    finally:
        queue.close()


def test_external_vocabulary_output_is_still_supported(
    protected: ProtectedVault, tmp_path: Path
) -> None:
    destination = tmp_path / "state" / "approved.json"
    before = protected.snapshot()
    save_tag_vocabulary(destination, {"z", "a"}, vault_root=protected.root)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"approved": ["a", "z"]}
    assert protected.snapshot() == before


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("batch", [False, True])
def test_auto_tag_preflights_before_any_provider_call(
    protected: ProtectedVault, namespace: str, batch: bool
) -> None:
    client = MagicMock()
    tagger = AutoTagger(AutoTagConfig(), client, "scripted-test", vault_root=protected.root)
    reserved = Note(
        path=protected.target(namespace, "alias").relative_to(protected.root),
        title="Protected",
        content="Evidence must not enter an ordinary tagging prompt. " * 10,
    )
    before = protected.snapshot()
    with pytest.raises(ReservedNamespaceError):
        if batch:
            ordinary = Note(path="ordinary.md", title="Ordinary", content="Ordinary text. " * 20)
            tagger.suggest_batch([ordinary, reserved], {"approved"})
        else:
            tagger.suggest_tags(reserved, {"approved"})
    client.complete.assert_not_called()
    assert protected.snapshot() == before


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("reserved_event", [False, True])
def test_contradiction_preflights_before_parser_provider_or_proposal(
    protected: ProtectedVault, tmp_path: Path, namespace: str, reserved_event: bool
) -> None:
    target = str(protected.target(namespace, "alias").relative_to(protected.root))
    queue = ReviewQueue(tmp_path / "queue.db")
    service = detector(protected.root, queue)
    service._duplicate_detector.find_duplicates.return_value = [
        DuplicateMatch(
            source_path="ordinary.md",
            source_title="Ordinary",
            match_path=target,
            match_title="Protected",
            similarity=0.85,
            match_type=MatchType.MERGE,
        )
    ]
    note = Note(path=target if reserved_event else "ordinary.md", title="Event", content="Text.")
    before = protected.snapshot()
    try:
        with pytest.raises(ReservedNamespaceError):
            asyncio.run(service.on_note_changed(NoteCreatedEvent(note=note, path=str(note.path))))
        if reserved_event:
            service._duplicate_detector.find_duplicates.assert_not_called()
        service._parser.parse_file.assert_not_called()
        service._llm_client.complete.assert_not_called()
        service._gap_store.mint.assert_not_called()
        assert queue.list_pending() == []
        assert queue.list_failed() == []
        assert protected.snapshot() == before
    finally:
        queue.close()


@pytest.mark.parametrize("namespace", ["wiki", "_wiki"])
@pytest.mark.parametrize("access", ["direct", "alias"])
@pytest.mark.parametrize("operation", ["delete", "edit"])
def test_reserved_callbacks_preserve_source_index_and_lineage(
    protected: ProtectedVault, tmp_path: Path, namespace: str, access: str, operation: str
) -> None:
    relative = str(protected.target(namespace, access).relative_to(protected.root))
    lineage_path = tmp_path / "lineage.json"
    lineage = LineageStore(store_path=lineage_path)
    lineage.record("ordinary.md", "edited", "existing audit history")
    context: Any = MagicMock(spec=HandlerContext)
    context.vault_root = protected.root
    context.settings = MagicMock()
    context.settings.telegram.allowed_user_ids = []
    context.store = MagicMock()
    context.parser = MagicMock()
    context.llm_client = MagicMock()
    context.lineage_store = lineage
    callback = MagicMock()
    callback.from_user = MagicMock(id=7)
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()
    before = protected.snapshot()
    source_before = protected.intake.read(protected.source_id).snapshot
    lineage_before = lineage_path.read_bytes()
    if operation == "delete":
        callback.data = f"delete_confirm:{relative}"
        asyncio.run(handle_delete_callback(context, callback))
    else:
        callback.data = "edit_confirm:7"
        pending = {
            7: {
                "path": relative,
                "original": "Original.",
                "edited": "Unreviewed replacement.",
                "instruction": "replace",
            }
        }
        asyncio.run(handle_edit_callback(context, callback, pending))
    context.store.delete_note.assert_not_called()
    context.store.index_single_note.assert_not_called()
    context.parser.parse_file.assert_not_called()
    context.llm_client.complete.assert_not_called()
    assert lineage.get_lineage(relative) == []
    assert lineage_path.read_bytes() == lineage_before
    assert protected.snapshot() == before
    assert protected.intake.read(protected.source_id).snapshot == source_before


def test_explicit_selected_root_allows_nonreserved_nested_names(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    target = root / "ordinary" / "wiki" / "note.md"
    target.parent.mkdir(parents=True)
    target.write_text("---\ntitle: Ordinary\n---\n\nOriginal body.\n", encoding="utf-8")
    apply_tags(target, ["known"], vault_root=root)
    post = frontmatter.load(str(target))
    assert post.metadata == {"title": "Ordinary", "tags": ["known"]}
    assert post.content == "Original body."
