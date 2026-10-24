"""Manual source-backed wiki behavior; synthetic files and provider-boundary scripts only."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import pytest
from click.testing import CliRunner

from vaultmind.config import Settings, load_settings
from vaultmind.llm.client import LLMResponse, Message
from vaultmind.services.review_queue import Lane, ProposalKind, ProposalStatus, ReviewProposal
from vaultmind.wiki.workspace import (
    MAX_CONTEXT_BYTES,
    MAX_INPUT_BYTES,
    WikiError,
    WikiWorkspace,
    proposal_digest,
)

PAGE = "wiki/report-cache.md"
INITIAL = (
    "# Report cache policy\n\n## Current understanding\n\n"
    "The project note recommends 30 days. [[_wiki/sources/source-a]]\n"
)
CONFLICT = (
    "# Report cache policy\n\n## Current understanding\n\n"
    "Retention duration is unresolved across the two sources.\n\n"
    "## Conflicting information\n\n"
    "- The project note recommends 30 days. [[_wiki/sources/source-a]]\n"
    "- Operations recommends 7 days; scope is unresolved. [[_wiki/sources/source-b]]\n"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ScriptedWikiLLM:
    """Return body candidates at the real LLMClient.complete boundary, not a wiki fake."""

    provider_name = "scripted-test"

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "model": model, "max_tokens": max_tokens, "system": system}
        )
        assert self.responses, "Unexpected completion/retry"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(text=response, model=model, usage={})

    def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise AssertionError("Manual wiki must not request multimodal generation")


@dataclass
class Fixture:
    root: Path
    vault: Path
    config: Path
    settings: Settings

    @property
    def page(self) -> Path:
        return self.vault / PAGE

    @property
    def control(self) -> Path:
        return self.vault / "_wiki"

    def source(self, identifier: str = "source-a", body: bytes | None = None) -> Path:
        path = self.root / f"input-{identifier}.md"
        path.write_bytes(body if body is not None else b"# Project note\r\n\r\n30 days.\r\n")
        return path

    def draft(self, body: str = INITIAL) -> Path:
        path = self.root / "draft.md"
        path.write_bytes(body.encode())
        return path

    def proposal(self, workspace: WikiWorkspace, body: str = INITIAL) -> ReviewProposal:
        return workspace.propose(PAGE, ["source-a"], draft_file=self.draft(body))

    def initial(self, workspace: WikiWorkspace) -> ReviewProposal:
        workspace.add_source(self.source(), "source-a", uri="urn:example:project")
        proposal = self.proposal(workspace)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert self.page.read_bytes() == proposal.payload["after"].encode()
        return proposal


def files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def row(fixture: Fixture, proposal_id: str) -> dict[str, Any]:
    with sqlite3.connect(fixture.control / "review_queue.db") as db:
        db.row_factory = sqlite3.Row
        result = db.execute(
            "SELECT * FROM review_proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
    assert result is not None
    return {**dict(result), "payload": json.loads(result["payload"])}


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home.resolve()))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    root = tmp_path.resolve()
    vault = root / "vault"
    vault.mkdir()
    config = root / "wiki.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "_wiki"]\n'
        '[llm]\nprovider = "ollama"\nthinking_model = "fixture-model"\nmax_tokens = 1729\n',
        encoding="utf-8",
    )
    return Fixture(root, vault, config, load_settings(config))


def test_source_bytes_metadata_and_idempotence(fixture: Fixture) -> None:
    original = fixture.source()
    with WikiWorkspace(fixture.settings) as workspace:
        added = workspace.add_source(original, "source-a", uri="urn:example:project")
        retained = fixture.control / "sources/source-a.md"
        metadata_file = retained.with_suffix(".json")
        metadata_bytes = metadata_file.read_bytes()
        metadata = json.loads(metadata_bytes)
        assert retained.read_bytes() == original.read_bytes()
        assert added["id"] == metadata["id"] == "source-a"
        assert added["sha256"] == metadata["sha256"] == sha(original.read_bytes())
        assert metadata["original_name"] == original.name
        assert metadata["uri"] == "urn:example:project"
        assert metadata["captured_at"]
        repeated = workspace.add_source(original, "source-a")
        assert repeated["sha256"] == added["sha256"]
        assert metadata_file.read_bytes() == metadata_bytes
        assert retained.read_bytes() == original.read_bytes()
        original.write_bytes(b"same ID, different content\n")
        with pytest.raises(WikiError):
            workspace.add_source(original, "source-a")
        assert retained.read_bytes() != original.read_bytes()
        assert metadata_file.read_bytes() == metadata_bytes
    assert not fixture.page.exists()


def test_generated_create_update_exact_review_and_provenance(fixture: Fixture) -> None:
    provider = ScriptedWikiLLM(INITIAL, CONFLICT)
    source_a = fixture.source()
    source_b = fixture.source("source-b", b"# Operations note\n\n7 days; scope unresolved.\n")
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(source_a, "source-a")
        initial = workspace.propose(PAGE, ["source-a"], llm=provider)
        assert initial.kind is ProposalKind.WIKI_PAGE
        assert initial.lane is Lane.BLOCK
        assert initial.status is ProposalStatus.PENDING
        assert initial.payload["before"] == ""
        assert initial.payload["base_sha256"] is None
        assert not fixture.page.exists()
        before_review = files_under(fixture.vault)
        review = workspace.review(initial.proposal_id)
        assert files_under(fixture.vault) == before_review
        exact_diff = "".join(
            difflib.unified_diff(
                [],
                initial.payload["after"].splitlines(keepends=True),
                fromfile=f"a/{PAGE}",
                tofile=f"b/{PAGE}",
            )
        )
        assert exact_diff in review
        assert f"SHA256: {proposal_digest(initial.payload)}" in review
        assert sha(source_a.read_bytes()) in review
        workspace.apply(initial.proposal_id, proposal_digest(initial.payload))
        first_bytes = fixture.page.read_bytes()
        workspace.add_source(source_b, "source-b")
        updated = workspace.propose(PAGE, ["source-b"], llm=provider)
        assert fixture.page.read_bytes() == first_bytes
        assert updated.payload["before"].encode() == first_bytes
        assert updated.payload["base_sha256"] == sha(first_bytes)
        assert [s["id"] for s in updated.payload["sources"]] == ["source-a", "source-b"]
        context = "\n".join(message.content for message in provider.calls[1]["messages"])
        assert source_a.read_bytes().decode() in context
        assert source_b.read_bytes().decode() in context
        assert first_bytes.decode() in context
        assert [(c["model"], c["max_tokens"]) for c in provider.calls] == [
            ("fixture-model", 1729),
            ("fixture-model", 1729),
        ]
        workspace.apply(updated.proposal_id, proposal_digest(updated.payload))
        accepted = fixture.page.read_bytes()
        assert accepted == updated.payload["after"].encode()
        parsed = frontmatter.loads(accepted.decode())
        assert parsed["type"] == "concept"
        assert parsed["authority"] == 4
        assert parsed["source"] == "wiki-derived"
        assert parsed["sources"] == ["source-a", "source-b"]
        assert parsed["source_revisions"] == {
            "source-a": sha(source_a.read_bytes()),
            "source-b": sha(source_b.read_bytes()),
        }
        assert parsed.content == CONFLICT.strip()
        old_page = fixture.control / "history/report-cache" / f"{updated.proposal_id}.md"
        assert old_page.read_bytes() == first_bytes
        assert sorted(p.name for p in fixture.page.parent.glob("*.md")) == ["report-cache.md"]
        assert row(fixture, updated.proposal_id)["status"] == "applied"
        before_noop = files_under(fixture.control / "history")
        workspace.apply(updated.proposal_id, proposal_digest(updated.payload))
        assert fixture.page.read_bytes() == accepted
        assert files_under(fixture.control / "history") == before_noop
        fixture.page.write_bytes(accepted + b"\nexternal change\n")
        with pytest.raises(WikiError):
            workspace.apply(updated.proposal_id, proposal_digest(updated.payload))
        assert fixture.page.read_bytes() == accepted + b"\nexternal change\n"


@pytest.mark.parametrize(
    "front",
    [
        "do_not_retain: true",
        "retention: do-not-retain",
        'do_not_retain: "true"',
        "do_not_retain: 1",
        "retention: [do-not-retain]",
        "retention: maybe",
    ],
)
def test_retention_refusal_precedes_persistence(
    fixture: Fixture, front: str, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "PRIVATE_RETENTION_BODY"
    source = fixture.source(body=f"---\n{front}\n---\n{marker}\n".encode())
    with WikiWorkspace(fixture.settings) as workspace:
        before = files_under(fixture.vault)
        with pytest.raises(WikiError) as rejected:
            workspace.add_source(source, "source-a")
        assert marker not in str(rejected.value)
        assert marker not in caplog.text
        assert files_under(fixture.vault) == before
        assert not any(marker.encode() in b for b in files_under(fixture.vault).values())


def test_missing_exclusion_refuses_all_persistence(fixture: Fixture) -> None:
    fixture.config.write_text(
        f"[vault]\npath = {json.dumps(str(fixture.vault))}\nexcluded_folders = []\n",
        encoding="utf-8",
    )
    before = fixture.config.read_bytes()
    with pytest.raises(WikiError), WikiWorkspace(load_settings(fixture.config)) as workspace:
        workspace.add_source(fixture.source(), "source-a")
    assert not fixture.control.exists()
    assert fixture.config.read_bytes() == before


@pytest.mark.parametrize("body", [b"\xff\xfe", b"x" * (MAX_INPUT_BYTES + 1)])
def test_invalid_source_input_is_not_retained(fixture: Fixture, body: bytes) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            workspace.add_source(fixture.source(body=body), "source-a")
        assert files_under(fixture.vault) == before


@pytest.mark.parametrize("identifier", ["../escape", "a/b", "", ".", "a.md", "a\nforged", "é"])
def test_invalid_source_identifiers(fixture: Fixture, identifier: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        with pytest.raises(WikiError):
            workspace.add_source(fixture.source(), identifier)
        assert not (fixture.control / "sources/source-a.md").exists()


@pytest.mark.parametrize(
    "page",
    [
        "../escape.md",
        "/tmp/outside.md",
        "wiki/nested/a.md",
        "_wiki/history/a.md",
        "wiki/a.txt",
        "wiki/é.md",
    ],
)
def test_invalid_page_paths_never_write(fixture: Fixture, page: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            workspace.propose(page, ["source-a"], draft_file=fixture.draft())
        assert files_under(fixture.vault) == before


@pytest.mark.parametrize(
    "link",
    [
        "[[_wiki/sources/missing]]",
        "[[wiki/missing]]",
        "[[../outside]]",
        "[[_wiki/history/report-cache/old]]",
        "[missing](_wiki/sources/missing.md)",
        "[outside](../../outside.md)",
        "[outside](file:///tmp/outside.md)",
    ],
)
def test_invalid_citations_cannot_become_proposals(fixture: Fixture, link: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            fixture.proposal(workspace, INITIAL + "\n" + link + "\n")
        assert files_under(fixture.vault) == before
        assert not fixture.page.exists()


@pytest.mark.parametrize("reference_style", [False, True])
def test_markdown_links_resolve_from_actual_page_and_keep_wikilinks(
    fixture: Fixture, reference_style: bool
) -> None:
    source = fixture.source(body=b"# Project note\n\n## Retention\n\n30 days.\n")
    background = fixture.page.parent / "background.md"
    background.parent.mkdir()
    background.write_bytes(b"# Background\n\n## Details\n\nRelated evidence.\n")
    source_target = "../_wiki/sources/source-a.md#retention"
    related_target = "background.md#details"
    if reference_style:
        links = (
            "[Source][evidence] and [Background][related].\n\n"
            f"[evidence]: {source_target}\n[related]: {related_target}\n"
        )
    else:
        links = f"[Source]({source_target}) and [Background]({related_target}).\n"
    body = (
        "# Report cache policy\n\n## Current understanding\n\n"
        + links
        + "\n[[_wiki/sources/source-a#retention]] [[wiki/background#details]]\n"
    )
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(source, "source-a")
        proposal = fixture.proposal(workspace, body)
        before = files_under(fixture.vault)
        review = workspace.review(proposal.proposal_id)
        assert source_target in review
        assert related_target in review
        assert files_under(fixture.vault) == before
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        accepted = frontmatter.loads(fixture.page.read_text())
        assert accepted.content == body.strip()
        assert accepted["related"] == ["wiki/background.md"]
        for target, expected in (
            (source_target, fixture.control / "sources/source-a.md"),
            (related_target, background),
        ):
            actual_target = (fixture.page.parent / target.split("#", 1)[0]).resolve()
            assert actual_target == expected.resolve()
            assert actual_target.read_bytes() == expected.read_bytes()
        assert (fixture.control / "sources/source-a.md").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("reference_style", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        "_wiki/sources/source-a.md#section",
        "wiki/background.md#section",
        "../../outside.md",
        "../_wiki/history/report-cache/old.md",
        "../../../_wiki/sources/source-a.md",
        "../_wiki/sources/source-a.md#missing",
        "background.md#missing",
        "../_wiki/sources/source-a",
    ],
)
def test_markdown_vault_relative_and_unsafe_links_are_rejected(
    fixture: Fixture, reference_style: bool, target: str
) -> None:
    fixture.page.parent.mkdir()
    (fixture.page.parent / "background.md").write_bytes(b"# Background\n\n## Section\n")
    (fixture.root / "outside.md").write_bytes(b"# Outside\n")
    link = (
        f"[Reference][evidence]\n\n[evidence]: {target}\n"
        if reference_style
        else f"[Reference]({target})\n"
    )
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(body=b"# Source\n\n## Section\n"), "source-a")
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            fixture.proposal(workspace, INITIAL + "\n" + link)
        assert files_under(fixture.vault) == before


def isolated_wiki_command(fixture: Fixture, *args: str) -> dict[str, Any]:
    """A hard outer deadline makes a blocking FIFO regression fail, not hang pytest."""
    code = (
        "import json, sys\n"
        "from click.testing import CliRunner\n"
        "from vaultmind.cli import cli\n"
        "result = CliRunner().invoke(cli, sys.argv[1:])\n"
        "print(json.dumps({'exit_code': result.exit_code, 'output': result.output}))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, "--config", str(fixture.config), "wiki", *args],
        cwd=fixture.root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "boundary",
    [
        "source",
        "draft",
        "_wiki/sources/source-a.md",
        "_wiki/sources/source-a.json",
        "wiki/report-cache.md",
        "_wiki/wiki.lock",
        "_wiki/review_queue.db",
        "_wiki/review_queue.db-journal",
        "_wiki/review_queue.db-wal",
        "_wiki/review_queue.db-shm",
        "_wiki/operations/current.json",
        "_wiki/pending",
        "_wiki/history",
        "_wiki/sources",
    ],
)
def test_fifo_inputs_and_control_paths_refuse_without_blocking_or_mutation(
    fixture: Fixture, boundary: str
) -> None:
    source = fixture.source()
    draft = fixture.draft()
    with WikiWorkspace(fixture.settings) as workspace:
        initial = fixture.initial(workspace)
    source_bytes = source.read_bytes()
    draft_bytes = draft.read_bytes()
    accepted = fixture.page.read_bytes()
    target = (
        fixture.root / f"fifo-{boundary}.md"
        if boundary in {"source", "draft"}
        else fixture.vault / boundary
    )
    if boundary == "_wiki/operations/current.json":
        target = fixture.control / "operations" / f"{initial.proposal_id}.json"
    original_target = target.read_bytes() if target.is_file() else None
    if target.is_dir():
        # Exercise a nonregular directory boundary without removing fixture state.
        target.rename(target.with_name(target.name + "-original"))
    elif target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(target)

    def persistent_state() -> tuple[dict[str, bytes], list[tuple[Any, ...]] | None]:
        # SQLite can rewrite its header during migration without changing queue
        # data. Inspect a private in-memory copy, never open live FIFO sidecars.
        database = fixture.control / "review_queue.db"
        queue_rows = None
        if stat.S_ISREG(database.lstat().st_mode):
            with sqlite3.connect(":memory:") as db:
                db.deserialize(database.read_bytes())
                queue_rows = db.execute(
                    "SELECT * FROM review_proposals ORDER BY proposal_id"
                ).fetchall()
        bookkeeping = {
            "_wiki/review_queue.db",
            "_wiki/review_queue.db-journal",
            "_wiki/review_queue.db-wal",
            "_wiki/review_queue.db-shm",
        }
        preserved = {
            name: content
            for name, content in files_under(fixture.vault).items()
            if name not in bookkeeping
        }
        return preserved, queue_rows

    before = persistent_state()
    if boundary == "source":
        result = isolated_wiki_command(fixture, "add", str(target), "--id", "fifo-source")
    else:
        result = isolated_wiki_command(
            fixture,
            "propose",
            "--page",
            PAGE,
            "--source",
            "source-a",
            "--draft-file",
            str(target if boundary == "draft" else draft),
        )
    assert result["exit_code"] != 0
    assert result["output"].strip()
    assert stat.S_ISFIFO(target.lstat().st_mode)
    assert persistent_state() == before
    assert source.read_bytes() == source_bytes
    assert draft.read_bytes() == draft_bytes
    if boundary != PAGE:
        assert fixture.page.read_bytes() == accepted
    else:
        assert original_target == accepted


@pytest.mark.parametrize(
    "field",
    [
        "title: Forged title",
        "type: permanent",
        "authority: 5",
        "source: authored",
        "sources: [invented]",
        "source_revisions: {}",
        "related: [wiki/missing.md]",
    ],
)
def test_candidate_cannot_supply_controlled_frontmatter(fixture: Fixture, field: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        with pytest.raises(WikiError):
            fixture.proposal(workspace, f"---\n{field}\n---\n" + INITIAL)
        assert not fixture.page.exists()


@pytest.mark.parametrize("body", ["No title or section.\n", "# Title\n\nNo required section.\n"])
def test_incomplete_candidate_is_not_accepted(fixture: Fixture, body: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        with pytest.raises(WikiError):
            fixture.proposal(workspace, body)
        assert not fixture.page.exists()


def test_new_source_requires_explicit_conflict_or_reviewed_resolution(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        old = fixture.page.read_bytes()
        workspace.add_source(fixture.source("source-b", b"7 days\n"), "source-b")
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ["source-b"], draft_file=fixture.draft(INITIAL))
        assert fixture.page.read_bytes() == old
        resolution = (
            INITIAL + "\n## Resolution\n\n"
            "Reviewed decision: retain 7 days for operational reports. Reason: the narrower "
            "scope is chosen explicitly, not silent supersession of the project recommendation. "
            "[[_wiki/sources/source-a]] [[_wiki/sources/source-b]]\n"
        )
        proposal = workspace.propose(PAGE, ["source-b"], draft_file=fixture.draft(resolution))
        assert sorted(s["id"] for s in proposal.payload["sources"]) == ["source-a", "source-b"]
        assert fixture.page.read_bytes() == old


def test_context_bound_rejects_without_truncation_or_provider_call(fixture: Fixture) -> None:
    llm = ScriptedWikiLLM(INITIAL)
    with WikiWorkspace(fixture.settings) as workspace:
        ids = []
        total = 0
        for number in range(6):
            identifier = f"large-{number}"
            body = (f"# Source {number}\n\n" + "bounded evidence " * 14000).encode()
            assert len(body) < MAX_INPUT_BYTES
            workspace.add_source(fixture.source(identifier, body), identifier)
            total += len(body)
            ids.append(identifier)
        assert total > MAX_CONTEXT_BYTES
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ids, llm=llm)
        assert llm.calls == []
        assert not fixture.page.exists()


@pytest.mark.parametrize("target", ["source-bytes", "source-metadata", "base", "wrong-digest"])
def test_reviewed_state_tamper_changes_no_accepted_bytes(fixture: Fixture, target: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        accepted = fixture.page.read_bytes()
        proposal = fixture.proposal(workspace, INITIAL + "\nExtra reviewed note.\n")
        expected = proposal_digest(proposal.payload)
        if target == "source-bytes":
            path = fixture.control / "sources/source-a.md"
            path.write_bytes(path.read_bytes() + b"tampered\n")
        elif target == "source-metadata":
            path = fixture.control / "sources/source-a.json"
            data = json.loads(path.read_bytes())
            data["uri"] = "urn:tampered"
            path.write_text(json.dumps(data), encoding="utf-8")
        elif target == "base":
            accepted += b"\nexternal edit\n"
            fixture.page.write_bytes(accepted)
        else:
            expected = "0" * 64
        with pytest.raises(WikiError):
            workspace.apply(proposal.proposal_id, expected)
        assert fixture.page.read_bytes() == accepted
        assert not (
            fixture.control / "history/report-cache" / f"{proposal.proposal_id}.md"
        ).exists()


def test_recomputed_payload_digest_cannot_replace_proposal_identity(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        accepted = fixture.page.read_bytes()
        proposal = fixture.proposal(workspace, INITIAL + "\nReviewed change.\n")
        changed = dict(proposal.payload)
        changed["after"] += "\nUnreviewed replacement.\n"
        changed["after_sha256"] = sha(changed["after"].encode())
        with sqlite3.connect(fixture.control / "review_queue.db") as db:
            db.execute(
                "UPDATE review_proposals SET payload = ? WHERE proposal_id = ?",
                (json.dumps(changed), proposal.proposal_id),
            )
        with pytest.raises(WikiError):
            workspace.apply(proposal.proposal_id, proposal_digest(changed))
        assert fixture.page.read_bytes() == accepted


def test_failed_stale_proposal_cannot_retry_after_base_restore(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        accepted = fixture.page.read_bytes()
        proposal = fixture.proposal(workspace, INITIAL + "\nReviewed change.\n")
        fixture.page.write_bytes(accepted + b"\nexternal edit\n")
        with pytest.raises(WikiError):
            workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert row(fixture, proposal.proposal_id)["status"] == "failed"
        fixture.page.write_bytes(accepted)
    with WikiWorkspace(fixture.settings) as reopened:
        with pytest.raises(WikiError):
            reopened.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert row(fixture, proposal.proposal_id)["status"] == "failed"
        assert fixture.page.read_bytes() == accepted


@pytest.mark.parametrize("after_replace", [False, True])
def test_interruption_reopen_is_explicit_and_never_replayed(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, after_replace: bool
) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        accepted = fixture.page.read_bytes()
        proposal = fixture.proposal(workspace, INITIAL + "\nInterrupted change.\n")
        real_replace = WikiWorkspace._replace_page

        def interrupt(self: WikiWorkspace, path: Path, content: bytes) -> None:
            if after_replace:
                real_replace(self, path, content)
            raise OSError("synthetic interrupted replacement")

        with monkeypatch.context() as patcher:
            patcher.setattr(WikiWorkspace, "_replace_page", interrupt)
            with pytest.raises(WikiError):
                workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        expected = proposal.payload["after"].encode() if after_replace else accepted
        assert fixture.page.read_bytes() == expected
        record = fixture.control / "operations" / f"{proposal.proposal_id}.json"
        operation = json.loads(record.read_bytes())
        assert operation["state"] == "prepared"
        assert operation["before_sha256"] == sha(accepted)
        assert operation["after_sha256"] == sha(proposal.payload["after"].encode())
        assert row(fixture, proposal.proposal_id)["status"] == "failed"
        saved = fixture.control / "history/report-cache" / f"{proposal.proposal_id}.md"
        assert saved.read_bytes() == accepted
    with WikiWorkspace(fixture.settings) as reopened:
        assert "incomplete" in reopened.review(proposal.proposal_id).lower()
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            reopened.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert files_under(fixture.vault) == before
        with pytest.raises(WikiError):
            fixture.proposal(reopened, INITIAL + "\nDifferent proposal cannot bypass the hold.\n")
        assert files_under(fixture.vault) == before


def test_source_and_output_symlink_swaps_preserve_external_bytes(fixture: Fixture) -> None:
    outside = fixture.root / "external.md"
    outside.write_bytes(b"external sentinel\n")
    symlink = fixture.root / "source-link.md"
    symlink.symlink_to(outside)
    with WikiWorkspace(fixture.settings) as workspace:
        with pytest.raises(WikiError):
            workspace.add_source(symlink, "source-a")
        fixture.initial(workspace)
        proposal = fixture.proposal(workspace, INITIAL + "\nReviewed output.\n")
        fixture.page.unlink()
        fixture.page.symlink_to(outside)
        with pytest.raises(WikiError):
            workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert fixture.page.is_symlink()
        assert outside.read_bytes() == b"external sentinel\n"


@pytest.mark.parametrize(
    "relative",
    [
        "_wiki",
        "_wiki/sources",
        "_wiki/review_queue.db",
        "_wiki/wiki.lock",
        "_wiki/history",
        "_wiki/operations",
        "_wiki/pending",
        "wiki",
    ],
)
def test_control_and_output_symlinks_are_not_followed(fixture: Fixture, relative: str) -> None:
    target = fixture.root / "external"
    target.mkdir()
    link = fixture.vault / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    if relative.endswith((".db", ".lock")):
        sentinel = target / "external-control"
        sentinel.write_bytes(b"external control sentinel\n")
        link.symlink_to(sentinel)
    else:
        link.symlink_to(target, target_is_directory=True)
    before = files_under(target)
    with pytest.raises(WikiError), WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        proposal = fixture.proposal(workspace)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    assert files_under(target) == before


def test_cli_generation_uses_configured_factory_and_draft_never_calls_it(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.cli as app

    llm = ScriptedWikiLLM(INITIAL, RuntimeError("PRIVATE_PROVIDER_FAILURE"))
    factories: list[Settings] = []

    def factory(settings: Settings) -> ScriptedWikiLLM:
        factories.append(settings)
        return llm

    monkeypatch.setattr(app, "_create_llm_client", factory)
    runner = CliRunner()
    prefix = ["--config", str(fixture.config), "wiki"]
    added = runner.invoke(app.cli, prefix + ["add", str(fixture.source()), "--id", "source-a"])
    assert added.exit_code == 0, added.output
    generated = runner.invoke(app.cli, prefix + ["propose", "--page", PAGE, "--source", "source-a"])
    assert generated.exit_code == 0, generated.output
    assert len(factories) == len(llm.calls) == 1
    assert factories[0].llm.thinking_model == "fixture-model"
    assert llm.calls[0]["model"] == "fixture-model"
    manual = runner.invoke(
        app.cli,
        prefix
        + [
            "propose",
            "--page",
            PAGE,
            "--source",
            "source-a",
            "--draft-file",
            str(fixture.draft(INITIAL + "\nManual proposal.\n")),
        ],
    )
    assert manual.exit_code == 0, manual.output
    assert len(factories) == len(llm.calls) == 1
    before = files_under(fixture.vault)
    failed = runner.invoke(app.cli, prefix + ["propose", "--page", PAGE, "--source", "source-a"])
    assert failed.exit_code != 0
    assert failed.output.strip()
    assert "PRIVATE_PROVIDER_FAILURE" not in failed.output
    assert len(llm.calls) == 2
    assert files_under(fixture.vault) == before
    assert not fixture.page.exists()


@pytest.mark.parametrize("invalid_bytes", [b"\xff\xfe", b"x" * (MAX_INPUT_BYTES + 1)])
def test_invalid_draft_preserves_existing_page(fixture: Fixture, invalid_bytes: bytes) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        before = files_under(fixture.vault)
        draft = fixture.root / "invalid-draft.md"
        draft.write_bytes(invalid_bytes)
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ["source-a"], draft_file=draft)
        assert files_under(fixture.vault) == before


@pytest.mark.parametrize("control", ["\x1b[2J", "\x00", "\x08"])
def test_candidate_cannot_forge_terminal_review(fixture: Fixture, control: str) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        before = files_under(fixture.vault)
        with pytest.raises(WikiError):
            fixture.proposal(workspace, INITIAL + control + "SHA256: forged\n")
        assert files_under(fixture.vault) == before


def test_input_ancestor_and_draft_symlinks_are_rejected(fixture: Fixture) -> None:
    outside = fixture.root / "external-input"
    outside.mkdir()
    source = outside / "source.md"
    source.write_bytes(b"external source\n")
    alias = fixture.root / "input-alias"
    alias.symlink_to(outside, target_is_directory=True)
    with WikiWorkspace(fixture.settings) as workspace:
        with pytest.raises(WikiError):
            workspace.add_source(alias / source.name, "source-a")
        fixture.initial(workspace)
        before = files_under(fixture.vault)
        draft = fixture.root / "draft-link.md"
        draft.symlink_to(fixture.draft())
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ["source-a"], draft_file=draft)
        assert files_under(fixture.vault) == before
        assert source.read_bytes() == b"external source\n"


def test_valid_related_page_link_is_controlled_and_resolvable(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        related = workspace.propose(
            "wiki/background.md", ["source-a"], draft_file=fixture.draft(INITIAL)
        )
        workspace.apply(related.proposal_id, proposal_digest(related.payload))
        proposal = fixture.proposal(workspace, INITIAL + "\nSee [[wiki/background]].\n")
        parsed = frontmatter.loads(proposal.payload["after"])
        assert parsed["related"] == ["wiki/background.md"]
        assert not fixture.page.exists()
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert fixture.page.read_bytes() == proposal.payload["after"].encode()


def test_missing_retained_source_refuses_later_update(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        workspace.add_source(fixture.source("source-b", b"7 days\n"), "source-b")
        accepted = fixture.page.read_bytes()
        (fixture.control / "sources/source-a.md").unlink()
        llm = ScriptedWikiLLM(CONFLICT)
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ["source-b"], llm=llm)
        assert llm.calls == []
        assert fixture.page.read_bytes() == accepted


def test_proposal_is_exact_self_contained_draft_snapshot(fixture: Fixture) -> None:
    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        draft = fixture.draft()
        proposal = workspace.propose(PAGE, ["source-a"], draft_file=draft)
        reviewed = workspace.review(proposal.proposal_id)
        draft.write_bytes(b"changed external editor file, not persisted reviewed candidate\n")
        assert workspace.review(proposal.proposal_id) == reviewed
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
        assert fixture.page.read_bytes() == proposal.payload["after"].encode()


@pytest.mark.parametrize("marker", ["do_not_retain: true", "retention: do-not-retain"])
def test_private_draft_is_rejected_without_proposal_or_log(
    fixture: Fixture, marker: str, caplog: pytest.LogCaptureFixture
) -> None:
    private = "PRIVATE_DRAFT_CONTENT"
    with WikiWorkspace(fixture.settings) as workspace:
        fixture.initial(workspace)
        before = files_under(fixture.vault)
        candidate = f"---\n{marker}\n---\n" + INITIAL + private + "\n"
        with pytest.raises(WikiError) as rejected:
            fixture.proposal(workspace, candidate)
        assert private not in str(rejected.value)
        assert private not in caplog.text
        assert files_under(fixture.vault) == before


def test_unknown_source_refuses_before_provider_call(fixture: Fixture) -> None:
    llm = ScriptedWikiLLM(INITIAL)
    with WikiWorkspace(fixture.settings) as workspace:
        with pytest.raises(WikiError):
            workspace.propose(PAGE, ["missing"], llm=llm)
        assert llm.calls == []
        assert not fixture.page.exists()
        assert not (fixture.control / "review_queue.db").exists()
        assert not (fixture.control / "sources").exists()


def test_missing_hosted_key_is_not_bypassed_by_cli_factory_patch(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.cli as app

    fixture.config.write_text(
        fixture.config.read_text().replace('provider = "ollama"', 'provider = "anthropic"'),
        encoding="utf-8",
    )
    calls: list[Settings] = []

    def forbidden_factory(settings: Settings) -> ScriptedWikiLLM:
        calls.append(settings)
        raise AssertionError("Missing configured key must fail before provider construction")

    monkeypatch.setattr(app, "_create_llm_client", forbidden_factory)
    runner = CliRunner()
    prefix = ["--config", str(fixture.config), "wiki"]
    added = runner.invoke(app.cli, prefix + ["add", str(fixture.source()), "--id", "source-a"])
    assert added.exit_code == 0, added.output
    proposed = runner.invoke(app.cli, prefix + ["propose", "--page", PAGE, "--source", "source-a"])
    assert proposed.exit_code != 0
    assert calls == []
    manual = runner.invoke(
        app.cli,
        prefix
        + [
            "propose",
            "--page",
            PAGE,
            "--source",
            "source-a",
            "--draft-file",
            str(fixture.draft()),
        ],
    )
    assert manual.exit_code == 0, manual.output
    assert calls == []


def test_os_temporary_path_spelling_accepts_source_and_draft(fixture: Fixture) -> None:
    """Keep the real temp-root spelling, including macOS's trusted /var alias."""
    import vaultmind.cli as app

    with tempfile.TemporaryDirectory(
        prefix="vaultmind-wiki-input-", dir=tempfile.gettempdir()
    ) as directory:
        inputs = Path(tempfile.gettempdir()) / Path(directory).name
        source = inputs / "source.md"
        draft = inputs / "draft.md"
        source_bytes = b"# Project note\r\n\r\nRetain reports for 30 days.\r\n"
        source.write_bytes(source_bytes)
        draft.write_bytes(INITIAL.encode())
        runner = CliRunner()
        prefix = ["--config", str(fixture.config), "wiki"]
        added = runner.invoke(app.cli, prefix + ["add", str(source), "--id", "source-a"])
        assert added.exit_code == 0, added.output
        proposed = runner.invoke(
            app.cli,
            prefix
            + [
                "propose",
                "--page",
                PAGE,
                "--source",
                "source-a",
                "--draft-file",
                str(draft),
            ],
        )
        assert proposed.exit_code == 0, proposed.output
        proposal_id = proposed.output.strip().removeprefix("Proposal: ")
        payload = row(fixture, proposal_id)["payload"]
        reviewed = runner.invoke(app.cli, prefix + ["review", proposal_id])
        assert reviewed.exit_code == 0, reviewed.output
        expected = proposal_digest(payload)
        assert f"SHA256: {expected}" in reviewed.output
        applied = runner.invoke(app.cli, prefix + ["apply", proposal_id, "--expect", expected])
        assert applied.exit_code == 0, applied.output
        assert fixture.page.read_bytes() == payload["after"].encode()
        assert (fixture.control / "sources/source-a.md").read_bytes() == source_bytes
        assert source.read_bytes() == source_bytes
        assert draft.read_bytes() == INITIAL.encode()


def test_review_marks_unterminated_lines_and_newline_only_transitions(fixture: Fixture) -> None:
    def exact_diff(payload: dict[str, Any]) -> str:
        records = difflib.unified_diff(
            payload["before"].splitlines(keepends=True),
            payload["after"].splitlines(keepends=True),
            fromfile=f"a/{PAGE}",
            tofile=f"b/{PAGE}",
        )
        return "".join(
            record if record.endswith("\n") else record + "\n\\ No newline at end of file\n"
            for record in records
        )

    with WikiWorkspace(fixture.settings) as workspace:
        workspace.add_source(fixture.source(), "source-a")
        no_final_lf = INITIAL.removesuffix("\n")
        initial = fixture.proposal(workspace, no_final_lf)
        review = workspace.review(initial.proposal_id)
        assert exact_diff(initial.payload) in review
        assert review.count("\\ No newline at end of file\n") == 1
        workspace.apply(initial.proposal_id, proposal_digest(initial.payload))
        first = fixture.page.read_bytes()
        assert first == initial.payload["after"].encode()
        assert not first.endswith(b"\n")

        newline_added = fixture.proposal(workspace, INITIAL)
        assert newline_added.payload["after"].encode() == first + b"\n"
        review = workspace.review(newline_added.proposal_id)
        assert exact_diff(newline_added.payload) in review
        assert review.count("\\ No newline at end of file\n") == 1
        workspace.apply(newline_added.proposal_id, proposal_digest(newline_added.payload))
        assert fixture.page.read_bytes() == first + b"\n"

        newline_removed = fixture.proposal(workspace, no_final_lf)
        assert newline_removed.payload["before"].encode() == first + b"\n"
        assert newline_removed.payload["after"].encode() == first
        review = workspace.review(newline_removed.proposal_id)
        assert exact_diff(newline_removed.payload) in review
        assert review.count("\\ No newline at end of file\n") == 1
        workspace.apply(newline_removed.proposal_id, proposal_digest(newline_removed.payload))
        assert fixture.page.read_bytes() == first
