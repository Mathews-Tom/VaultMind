"""Synthetic exact-byte capture, provenance closure and downstream dispositions."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from vaultmind.config import Settings, load_settings
from vaultmind.knowledge.intake import MAX_ATTACHMENT_BYTES, SourceIntake
from vaultmind.knowledge.query import query_knowledge
from vaultmind.wiki.workspace import MAX_INPUT_BYTES, WikiError, WikiWorkspace, proposal_digest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f'[vault]\npath = {json.dumps(str(vault))}\nexcluded_folders = ["wiki", "_wiki"]\n',
        encoding="utf-8",
    )
    return load_settings(config)


def capture(intake: SourceIntake, content: str | bytes = "Exact source\n", **kwargs: Any) -> Any:
    return intake.preserve_text(
        content,
        original_name="message.md",
        origin="telegram:chat:19",
        adapter="telegram",
        **kwargs,
    )


def files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_exact_text_binary_and_legacy_raw_metadata_digest(
    settings: Settings, tmp_path: Path
) -> None:
    body = b"# Exact\r\n\r\nEmoji: \xf0\x9f\x8c\x8d\r\n"
    binary = b"\x00\xff\xfe\r\n" * 100
    with SourceIntake(settings) as intake:
        saved = capture(intake, body, event_id="19:1")
        attachment = intake.preserve_attachment(
            binary,
            original_name="photo.jpg",
            origin="telegram:file:7",
            adapter="telegram",
            event_id="19:2",
            media_type="image/jpeg",
        )
        assert (settings.vault.path / saved.path).read_bytes() == body
        assert intake.read(saved.source_id).content == body
        assert intake.read(attachment.source_id).content == binary
        assert intake.read(attachment.source_id).text is None
        assert saved.preserved and not saved.indexed and not saved.approved
        original = tmp_path / "legacy.md"
        original.write_bytes(body)
        intake.workspace.add_source(original, "legacy", uri="urn:legacy")
        metadata = settings.vault.path / "_wiki/sources/legacy.json"
        raw_metadata = metadata.read_bytes()
        snapshot = intake.read("legacy").snapshot
        assert set(snapshot) == {"id", "sha256", "metadata_sha256", "metadata"}
        assert set(snapshot["metadata"]) == {
            "id",
            "original_name",
            "origin",
            "uri",
            "captured_at",
            "sha256",
        }
        assert snapshot["metadata"]["origin"] == str(original.absolute())
        assert snapshot["metadata_sha256"] == hashlib.sha256(raw_metadata).hexdigest()
        intake.workspace.add_source(original, "legacy")
        assert metadata.read_bytes() == raw_metadata
        assert set(intake.read("legacy").identity()) == {
            "id",
            "path",
            "sha256",
            "metadata_sha256",
            "uri",
        }


def test_event_revision_role_and_origin_identity(settings: Settings) -> None:
    with SourceIntake(settings) as intake:
        first = capture(intake, event_id="19:10", revision="server-rev")
        before = files(settings.vault.path)
        replay = capture(intake, event_id="19:10", revision="server-rev")
        assert replay.source_id == first.source_id and replay.status == "unchanged"
        assert files(settings.vault.path) == before
        changed = capture(intake, "Changed bytes\n", event_id="19:10", revision="server-rev")
        revised = capture(intake, event_id="19:10", revision="server-rev-2")
        distinct = capture(intake, event_id="19:11", revision="server-rev")
        derivative = capture(
            intake,
            event_id="19:10",
            revision="server-rev",
            role="transcript",
            original_retained=False,
        )
        fresh = capture(intake)
        another = capture(intake)
        assert (
            len(
                {
                    first.source_id,
                    changed.source_id,
                    revised.source_id,
                    distinct.source_id,
                    derivative.source_id,
                    fresh.source_id,
                    another.source_id,
                }
            )
            == 7
        )
        assert intake.read(first.source_id).provenance["origin"] == "telegram:chat:19"
        assert (
            intake.read(derivative.source_id).identity()["retention"]["original_retained"] is False
        )
        with pytest.raises(WikiError, match="different provenance"):
            intake.preserve_text(
                "Exact source\n",
                original_name="message.md",
                origin="telegram:other-origin",
                adapter="telegram",
                event_id="19:10",
                revision="server-rev",
            )


@pytest.mark.parametrize("corruption", ["missing", "changed", "copied", "downgraded"])
def test_manifest_completion_binding_is_fail_closed(settings: Settings, corruption: str) -> None:
    with SourceIntake(settings) as intake:
        first = capture(intake, event_id="19:20")
        second = capture(intake, event_id="19:21")
        record = settings.vault.path / f"_wiki/sources/{first.source_id}.intake.json"
        if corruption == "missing":
            record.unlink()
        elif corruption == "changed":
            record.write_bytes(record.read_bytes() + b" ")
        elif corruption == "copied":
            record.write_bytes(
                (settings.vault.path / f"_wiki/sources/{second.source_id}.intake.json").read_bytes()
            )
        else:
            metadata_path = settings.vault.path / f"_wiki/sources/{first.source_id}.json"
            metadata = json.loads(metadata_path.read_bytes())
            metadata["origin"] = "/fake/legacy.md"
            metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(WikiError):
            intake.read(first.source_id)


@pytest.mark.parametrize("dependency", ["parent", "attachment"])
@pytest.mark.parametrize("mutation", ["missing", "bytes", "metadata"])
def test_dependency_closure_rejects_tampering(
    settings: Settings, dependency: str, mutation: str
) -> None:
    with SourceIntake(settings) as intake:
        if dependency == "parent":
            parent = capture(intake, event_id="19:30")
        else:
            parent = intake.preserve_attachment(
                b"\xff\x00image",
                original_name="image.bin",
                origin="telegram:file:1",
                adapter="telegram",
                event_id="19:30",
            )
        child = capture(
            intake,
            "Derived observation\n",
            event_id="19:31",
            role="vision-description",
            **{("parents" if dependency == "parent" else "attachments"): [parent.source_id]},
        )
        assert (
            intake.read(child.source_id).identity()["retained_dependencies"][0]["id"]
            == parent.source_id
        )
        path = settings.vault.path / parent.path
        if mutation == "missing":
            path.unlink()
        elif mutation == "bytes":
            path.write_bytes(b"tampered")
        else:
            metadata = settings.vault.path / f"_wiki/sources/{parent.source_id}.json"
            metadata.write_bytes(metadata.read_bytes() + b" ")
        with pytest.raises(WikiError):
            intake.read(child.source_id)


@pytest.mark.parametrize(
    "content",
    [b"---\nretention: do-not-retain\n---\nSECRET", b"\xff", b"x" * (MAX_INPUT_BYTES + 1)],
)
def test_retention_and_bounds_refuse_before_any_persistence(
    settings: Settings, content: bytes
) -> None:
    with SourceIntake(settings) as intake, pytest.raises(WikiError):
        capture(intake, content, event_id="19:40")
    assert files(settings.vault.path) == {}
    assert not (settings.vault.path / "_wiki").exists()


def test_binary_limit_and_permissions_precede_lock_writes(settings: Settings) -> None:
    with SourceIntake(settings, allows_path=lambda _: False) as intake, pytest.raises(WikiError):
        capture(intake)
    assert files(settings.vault.path) == {}
    with SourceIntake(settings) as intake, pytest.raises(WikiError):
        intake.preserve_attachment(
            b"x" * (MAX_ATTACHMENT_BYTES + 1),
            original_name="large.bin",
            origin="file:1",
            adapter="test",
        )
    assert not (settings.vault.path / "_wiki").exists()


def test_parent_authorization_precedes_read_and_no_new_capture(settings: Settings) -> None:
    with SourceIntake(settings) as intake:
        parent = capture(intake, event_id="19:41")
        child = capture(intake, event_id="19:42", parents=[parent.source_id])
    before = files(settings.vault.path)
    denied = f"_wiki/sources/{parent.source_id}.json"
    with SourceIntake(settings, allows_path=lambda path: path != denied) as intake:
        with pytest.raises(WikiError, match="authorized"):
            intake.read(child.source_id)
        with pytest.raises(WikiError, match="authorized"):
            capture(intake, event_id="19:43", parents=[parent.source_id])
    assert files(settings.vault.path) == before


def test_processing_failure_replay_and_incomplete_claim_never_repeat(settings: Settings) -> None:
    calls: list[str] = []
    with SourceIntake(settings) as intake:
        saved = capture(intake, event_id="19:50")

        def fail() -> None:
            calls.append("provider")
            raise RuntimeError("SECRET SOURCE BODY")

        disposition = saved.run_stage("provider", fail)
        assert disposition.status == "failed"
        assert "SECRET" not in str(disposition.as_dict())
        assert saved.preserved and not saved.approved and not saved.indexed
        saved.run_stage("provider", fail)
        state = settings.vault.path / f"_wiki/sources/{saved.source_id}.processing.json"
        value = json.loads(state.read_bytes())
        value["index"] = "attempted"
        state.write_text(json.dumps(value))
    with SourceIntake(settings) as intake:
        replay = capture(intake, event_id="19:50")
        assert replay.run_stage("provider", fail).status == "failed"
        assert replay.run_stage("index", lambda: calls.append("index")).status == "incomplete"
        assert intake.read(replay.source_id).content == b"Exact source\n"
    assert calls == ["provider"]


def test_projection_is_deterministic_guarded_and_replay_safe(settings: Settings) -> None:
    with SourceIntake(settings) as intake:
        saved = capture(intake, event_id="19:51")
        with pytest.raises(ValueError):
            intake.project_text(saved, "wiki")
        assert not (settings.vault.path / "wiki").exists()
        projected = intake.project_text(saved, "inbox")
        assert projected.status == "succeeded"
        assert (settings.vault.path / projected.value).read_bytes() == b"Exact source\n"
    before = files(settings.vault.path)
    with SourceIntake(settings) as intake:
        replay = capture(intake, event_id="19:51")
        intake.project_text(replay, "inbox")
    assert files(settings.vault.path) == before


def test_partial_preservation_never_completes_on_replay(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SourceIntake(settings) as intake:
        original_json = intake.workspace._json

        def fail_metadata(path: Path, value: dict[str, Any]) -> None:
            raise OSError("disk failure")

        monkeypatch.setattr(intake.workspace, "_json", fail_metadata)
        with pytest.raises(WikiError):
            capture(intake, event_id="19:52")
        monkeypatch.setattr(intake.workspace, "_json", original_json)
        before = files(settings.vault.path)
        with pytest.raises(WikiError):
            capture(intake, event_id="19:52")
        assert files(settings.vault.path) == before
        manifests = list((settings.vault.path / "_wiki/sources").glob("*.intake.json"))
        assert len(manifests) == 1
        assert (
            not manifests[0].with_name(manifests[0].name.replace(".intake.json", ".json")).exists()
        )


def test_approved_queries_validate_retained_attachment_after_generation(
    settings: Settings, tmp_path: Path
) -> None:
    from vaultmind.llm.client import LLMResponse

    with WikiWorkspace(settings) as workspace:
        intake = SourceIntake.from_workspace(workspace)
        attachment = intake.preserve_attachment(
            b"image",
            original_name="photo.bin",
            origin="telegram:file:2",
            adapter="telegram",
            event_id="19:60",
        )
        derivative = capture(
            intake,
            "Photo shows a blue door.\n",
            role="vision-description",
            event_id="19:60",
            attachments=[attachment.source_id],
        )
        draft = tmp_path / "draft.md"
        draft.write_text(
            "# Door\n\n## Current understanding\n\n"
            f"A blue door. [[_wiki/sources/{derivative.source_id}]]\n"
        )
        proposal = workspace.propose("wiki/door.md", [derivative.source_id], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    evidence = query_knowledge(settings, "door", ["wiki/door.md"], evidence_only=True)
    assert evidence["sources"][0]["retained_dependencies"][0]["id"] == attachment.source_id
    citation = evidence["evidence"][0]["id"]

    class MutatingProvider:
        provider_name = "synthetic"
        calls = 0

        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            self.calls += 1
            (settings.vault.path / attachment.path).write_bytes(b"changed")
            return LLMResponse(
                text=json.dumps(dict(status="supported", answer="Blue door", citations=[citation])),
                model="scripted",
                usage={},
            )

        def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
            raise AssertionError("Unexpected multimodal query")

    provider = MutatingProvider()
    with pytest.raises(WikiError):
        query_knowledge(settings, "door", ["wiki/door.md"], llm_factory=lambda: provider)
    assert provider.calls == 1
    with pytest.raises(WikiError):
        query_knowledge(settings, "door", ["wiki/door.md"], evidence_only=True)


def test_preflight_refusal_and_symlink_never_persist_source(
    settings: Settings, tmp_path: Path
) -> None:
    with SourceIntake(settings) as intake:

        def deny_projection(source_id: str) -> None:
            assert source_id.startswith("i-")
            raise WikiError("Projection permission refused")

        with pytest.raises(WikiError):
            capture(intake, event_id="19:70", preflight=deny_projection)
        assert not (settings.vault.path / "_wiki").exists()
        outside = tmp_path / "outside"
        outside.mkdir()
        (settings.vault.path / "_wiki").symlink_to(outside, target_is_directory=True)
        with pytest.raises(WikiError):
            capture(intake, event_id="19:71")
        assert files(outside) == {}


def test_dependency_depth_is_bounded_before_write(settings: Settings) -> None:
    from vaultmind.knowledge.intake import MAX_CLOSURE_DEPTH

    with SourceIntake(settings) as intake:
        previous = capture(intake, event_id="depth:0")
        for depth in range(1, MAX_CLOSURE_DEPTH + 1):
            previous = capture(intake, event_id=f"depth:{depth}", parents=[previous.source_id])
        before = files(settings.vault.path)
        with pytest.raises(WikiError, match="depth"):
            capture(intake, event_id="depth:too-deep", parents=[previous.source_id])
        assert files(settings.vault.path) == before


def test_authenticated_event_lookup_is_read_only_complete_and_unambiguous(
    settings: Settings,
) -> None:
    with SourceIntake(settings) as intake:
        assert intake.find_event("telegram", "voice:1", "audio-sha", "voice-transcript") is None
        assert not (settings.vault.path / "_wiki").exists()
        saved = capture(
            intake,
            "First transcript\\n",
            event_id="voice:1",
            revision="audio-sha",
            role="voice-transcript",
            original_retained=False,
        )
        before = files(settings.vault.path)
        found = intake.find_event("telegram", "voice:1", "audio-sha", "voice-transcript")
        assert found is not None and found.source_id == saved.source_id
        assert found.text == "First transcript\\n"
        assert (
            intake.find_event("telegram", "voice:1", "different-audio", "voice-transcript") is None
        )
        assert files(settings.vault.path) == before
        capture(
            intake,
            "Different transcript\\n",
            event_id="voice:1",
            revision="audio-sha",
            role="voice-transcript",
            original_retained=False,
        )
        with pytest.raises(WikiError, match="ambiguous"):
            intake.find_event("telegram", "voice:1", "audio-sha", "voice-transcript")


@pytest.mark.parametrize("operation", ["review", "apply"])
def test_pending_proposal_revalidates_dependency_before_acceptance(
    settings: Settings, tmp_path: Path, operation: str
) -> None:
    with WikiWorkspace(settings) as workspace:
        intake = SourceIntake.from_workspace(workspace)
        parent = capture(intake, event_id="proposal:parent")
        child = capture(intake, "Derived\n", event_id="proposal:child", parents=[parent.source_id])
        draft = tmp_path / "proposal.md"
        draft.write_text(
            "# Evidence\n\n## Current understanding\n\n"
            f"Derived fact. [[_wiki/sources/{child.source_id}]]\n"
        )
        proposal = workspace.propose("wiki/evidence.md", [child.source_id], draft_file=draft)
        (settings.vault.path / parent.path).unlink()
        with pytest.raises(WikiError):
            if operation == "review":
                workspace.review(proposal.proposal_id)
            else:
                workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert not (settings.vault.path / "wiki/evidence.md").exists()


def test_incomplete_matching_record_refuses_event_replay(settings: Settings) -> None:
    with SourceIntake(settings) as intake:
        saved = capture(intake, event_id="incomplete:1", revision="revision-1")
        (settings.vault.path / f"_wiki/sources/{saved.source_id}.json").unlink()
        before = files(settings.vault.path)
        with pytest.raises(WikiError):
            intake.find_event("telegram", "incomplete:1", "revision-1", "text")
        assert files(settings.vault.path) == before


def test_event_lookup_counts_only_adjacent_provenance_candidates(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SourceIntake(settings) as intake:
        capture(intake, event_id="other:1")
        saved = capture(intake, event_id="target:1")
        directory = settings.vault.path / "_wiki/sources"
        for index in range(5):
            (directory / f"unrelated-{index}.md").write_text("Not intake provenance\n")
        monkeypatch.setattr("vaultmind.knowledge.intake.MAX_EVENT_SCAN_FILES", 2)
        found = intake.find_event("telegram", "target:1", None, "text")
        assert found is not None and found.source_id == saved.source_id
        capture(intake, event_id="other:2")
        with pytest.raises(WikiError, match="source scan limit"):
            intake.find_event("telegram", "target:1", None, "text")


def test_event_lookup_does_not_read_unrelated_retained_closures(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SourceIntake(settings) as intake:
        unrelated = [capture(intake, "Other bytes\n", event_id=f"other:{i}") for i in range(3)]
        saved = capture(intake, "Target bytes\n", event_id="target:1")
        # Every source fits individually; unrelated retained bytes together do not.
        monkeypatch.setattr("vaultmind.knowledge.intake.MAX_CLOSURE_BYTES", len("Target bytes\n"))
        found = intake.find_event("telegram", "target:1", None, "text")
        assert found is not None and found.source_id == saved.source_id
        # An unrelated incomplete closure is not a reason to read/return its body.
        for source in unrelated:
            (settings.vault.path / source.path).unlink()
        before = files(settings.vault.path)
        found = intake.find_event("telegram", "target:1", None, "text")
        assert found is not None and found.content == b"Target bytes\n"
        assert intake.find_event("telegram", "absent:1", None, "text") is None
        assert files(settings.vault.path) == before


def test_event_lookup_matching_closures_have_independent_byte_limits(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SourceIntake(settings) as intake:
        capture(intake, "First bytes\n", event_id="target:1")
        capture(intake, "Other bytes\n", event_id="target:1")
        monkeypatch.setattr("vaultmind.knowledge.intake.MAX_CLOSURE_BYTES", len("First bytes\n"))
        with pytest.raises(WikiError, match="ambiguous"):
            intake.find_event("telegram", "target:1", None, "text")


@pytest.mark.parametrize(
    "failure",
    ["missing", "corrupt", "dependency", "unauthorized", "unauthorized-provenance", "limit"],
)
def test_event_lookup_matching_invalid_closure_refuses_replay(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    with SourceIntake(settings) as intake:
        parent = capture(intake, "Parent bytes\n", event_id="parent:1")
        saved = capture(intake, "Target bytes\n", event_id="target:1", parents=[parent.source_id])
        if failure == "missing":
            (settings.vault.path / saved.path).unlink()
        elif failure == "corrupt":
            (settings.vault.path / saved.path).write_text("Changed bytes\n")
        elif failure == "dependency":
            (settings.vault.path / parent.path).unlink()
        elif failure == "unauthorized":
            intake.allows_path = lambda path: path != parent.path
        elif failure == "unauthorized-provenance":
            record_path = f"_wiki/sources/{saved.source_id}.intake.json"
            intake.allows_path = lambda path: path != record_path
        else:
            monkeypatch.setattr(
                "vaultmind.knowledge.intake.MAX_CLOSURE_BYTES", len("Target bytes\n")
            )
        before = files(settings.vault.path)
        with pytest.raises(WikiError):
            intake.find_event("telegram", "target:1", None, "text")
        assert files(settings.vault.path) == before


@pytest.mark.parametrize("matching", [False, True])
def test_event_lookup_unvalidated_provenance_never_establishes_miss_or_exposes_body(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, matching: bool
) -> None:
    with SourceIntake(settings) as intake:
        saved = capture(intake, event_id="target:1")
        path = settings.vault.path / f"_wiki/sources/{saved.source_id}.intake.json"
        provenance = json.loads(path.read_bytes())
        provenance["event_id"] = "target:1" if matching else "other:1"
        provenance["origin"] = "Corrupt provenance"
        path.write_text(json.dumps(provenance))

        def refuse_body(relative: str, limit: int) -> bytes:
            pytest.fail(f"Unvalidated provenance exposed source body: {relative}")

        monkeypatch.setattr(intake, "_binary", refuse_body)
        before = files(settings.vault.path)
        with pytest.raises(WikiError, match="provenance hash"):
            intake.find_event("telegram", "target:1", None, "text")
        assert files(settings.vault.path) == before
