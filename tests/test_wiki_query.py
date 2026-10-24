"""Approved current-page query consumers, with synthetic sources and scripted providers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from vaultmind.config import Settings, load_settings
from vaultmind.llm.client import LLMResponse
from vaultmind.wiki.query import lint_pages, list_pages, query_pages
from vaultmind.wiki.workspace import WikiError, WikiWorkspace, proposal_digest

PAGE = "wiki/cache.md"
LATE = "The verified late retention limit is 19 days."
BODY = (
    "# Cache\n\n## Current understanding\n\n"
    + "Background context. " * 40
    + LATE
    + " [[_wiki/sources/source-a]]\n"
)
CONFLICT = (
    "# Cache\n\n## Current understanding\n\nRetention remains unresolved.\n\n"
    "## Conflicting information\n\n"
    "19 days versus 7 days. [[_wiki/sources/source-a]]\n"
)


class Provider:
    provider_name = "scripted-query-test"

    def __init__(self, response: dict[str, Any] | Exception, mutate: Any = None) -> None:
        self.response = response
        self.mutate = mutate
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, messages: Any, model: str, max_tokens: int = 4096, system: str | None = None
    ) -> LLMResponse:
        self.calls.append(
            dict(messages=messages, model=model, max_tokens=max_tokens, system=system)
        )
        if self.mutate:
            self.mutate()
        if isinstance(self.response, Exception):
            raise self.response
        return LLMResponse(text=json.dumps(self.response), model=model, usage={})

    def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise AssertionError("Unexpected multimodal query")


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Settings, Path]:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    config = tmp_path / "query.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = ["_wiki"]\n'
        '[llm]\nprovider = "ollama"\nthinking_model = "query-fixture"\nmax_tokens = 1729\n'
    )
    settings = load_settings(config)
    source = tmp_path / "input.md"
    source.write_text("# Retention\n\n19 days and an unresolved alternative of 7 days.\n")
    with WikiWorkspace(settings) as workspace:
        workspace.add_source(source, "source-a", uri="urn:retention")
    return settings, vault


def accept(world: tuple[Settings, Path], body: str = BODY) -> bytes:
    settings, vault = world
    draft = vault.parent / "draft.md"
    draft.write_text(body)
    with WikiWorkspace(settings) as workspace:
        proposal = workspace.propose(PAGE, ["source-a"], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    return (vault / PAGE).read_bytes()


def test_full_evidence_and_readonly_inspection(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    accept(world)
    before = snapshot(vault)
    result = query_pages(settings, "What is the limit?", [PAGE], evidence_only=True)
    assert result["status"] == "evidence"
    assert result["answer"] == "" and result["citations"] == []
    late = next(row for row in result["evidence"] if LATE in row["text"])
    assert late["text"].index(LATE) > 500
    assert late["page"] == PAGE
    assert late["sources"][0]["id"] == "source-a"
    assert late["sources"][0]["uri"] == "urn:retention"
    assert len(late["page_sha256"]) == 64
    assert [row["path"] for row in list_pages(settings)] == [PAGE]
    assert [row["status"] for row in lint_pages(settings)] == ["valid"]
    with WikiWorkspace(settings, read_only=True):
        pass
    assert snapshot(vault) == before


@pytest.mark.parametrize("status", ["supported", "unsupported"])
def test_generation_uses_configured_boundary(world: tuple[Settings, Path], status: str) -> None:
    settings, vault = world
    accept(world)
    evidence = query_pages(settings, "Limit?", [PAGE], evidence_only=True)["evidence"]
    citation = next(row["id"] for row in evidence if LATE in row["text"])
    client = Provider(
        dict(
            status=status,
            answer="19 days." if status == "supported" else "Not established.",
            citations=[citation] if status == "supported" else [],
        )
    )
    before = snapshot(vault)
    result = query_pages(settings, "Limit?", [PAGE], llm=client)
    assert result["status"] == status
    assert result["citations"] == ([citation] if status == "supported" else [])
    assert len(client.calls) == 1
    call = client.calls[0]
    assert (call["model"], call["max_tokens"]) == ("query-fixture", 1729)
    prompt = json.loads(call["messages"][0].content)
    assert prompt["question"] == "Limit?"
    assert any(LATE in row["text"] for row in prompt["evidence"])
    assert call["system"]
    assert snapshot(vault) == before


def test_unresolved_conflict_requires_attributed_conflict(world: tuple[Settings, Path]) -> None:
    settings, _ = world
    accept(world, CONFLICT)
    evidence = query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    ids = [row["id"] for row in evidence["evidence"] if "## Conflicting information" in row["text"]]
    assert evidence["unresolved_conflicts"] == [PAGE]
    assert lint_pages(settings)[0]["status"] == "unresolved"
    client = Provider(dict(status="conflicting", answer="19 and 7 days conflict.", citations=ids))
    assert query_pages(settings, "Limit?", [PAGE], llm=client)["citations"] == ids
    with pytest.raises(WikiError):
        query_pages(
            settings,
            "Limit?",
            [PAGE],
            llm=Provider(dict(status="supported", answer="19 days.", citations=ids)),
        )


@pytest.mark.parametrize(
    "response",
    [
        dict(status="supported", answer="19 days.", citations=["invented"]),
        dict(status="supported", answer="19 days.", citations=[]),
        dict(status="unsupported", answer="Unknown.", citations=["p1-s1"]),
        dict(status="unsupported", answer="Unknown.", citations=[], extra=True),
        RuntimeError("PRIVATE_PROVIDER_BODY"),
    ],
)
def test_invalid_generation_has_no_fallback_or_effects(
    world: tuple[Settings, Path], response: Any
) -> None:
    settings, vault = world
    accept(world)
    before = snapshot(vault)
    with pytest.raises(WikiError) as error:
        query_pages(settings, "Limit?", [PAGE], llm=Provider(response))
    assert "PRIVATE_PROVIDER_BODY" not in str(error.value)
    assert snapshot(vault) == before


def test_missing_source_lint_and_exact_restoration(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    accept(world)
    source = vault / "_wiki/sources/source-a.md"
    original = source.read_bytes()
    source.unlink()
    before = snapshot(vault)
    assert lint_pages(settings)[0]["status"] == "invalid"
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert snapshot(vault) == before
    source.write_bytes(original)
    assert lint_pages(settings)[0]["status"] == "valid"
    assert query_pages(settings, "Limit?", [PAGE], evidence_only=True)["evidence"]


@pytest.mark.parametrize(
    "pages",
    [
        [],
        [PAGE, PAGE],
        [PAGE] * 9,
        ["../cache.md"],
        ["_wiki/sources/source-a.md"],
        ["wiki/unknown.md"],
    ],
)
def test_explicit_page_bounds_precede_provider(
    world: tuple[Settings, Path], pages: list[str]
) -> None:
    settings, _ = world
    accept(world)
    client = Provider(dict(status="unsupported", answer="Unknown.", citations=[]))
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", pages, llm=client)
    assert client.calls == []


def test_stale_current_pending_and_unapproved_scope(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    old = accept(world)
    accept(world, BODY.replace("19 days", "21 days"))
    current = (vault / PAGE).read_bytes()
    (vault / PAGE).write_bytes(old)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    (vault / PAGE).write_bytes(current)
    draft = vault.parent / "pending.md"
    draft.write_text(BODY.replace("19 days", "23 days"))
    with WikiWorkspace(settings) as workspace:
        workspace.propose(PAGE, ["source-a"], draft_file=draft)
    result = query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    text = "\n".join(row["text"] for row in result["evidence"])
    assert "21 days" in text and "23 days" not in text
    (vault / "wiki/unapproved.md").write_bytes(current)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", ["wiki/unapproved.md"], evidence_only=True)


def test_question_and_context_bounds_before_call(
    world: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.wiki.query as module

    settings, _ = world
    accept(world)
    client = Provider(dict(status="unsupported", answer="Unknown.", citations=[]))
    with pytest.raises(WikiError):
        query_pages(settings, "x" * 4097, [PAGE], llm=client)
    monkeypatch.setattr(module, "MAX_QUERY_CONTEXT_BYTES", 128)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], llm=client)
    assert client.calls == []


def test_changes_during_completion_are_refused(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    accept(world)
    source = vault / "_wiki/sources/source-a.md"
    client = Provider(
        dict(status="unsupported", answer="Unknown.", citations=[]),
        mutate=lambda: source.write_bytes(b"changed while answering"),
    )
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], llm=client)
    assert len(client.calls) == 1


def test_empty_workspace_does_not_create_control_tree(tmp_path: Path) -> None:
    settings = Settings(vault={"path": tmp_path, "excluded_folders": ["_wiki"]})
    before = snapshot(tmp_path)
    assert list_pages(settings) == []
    assert lint_pages(settings) == []
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["operation", "metadata", "history"])
def test_approved_identity_corruption_is_not_queryable(
    world: tuple[Settings, Path], kind: str
) -> None:
    settings, vault = world
    accept(world)
    accept(world, BODY.replace("19 days", "21 days"))
    if kind == "operation":
        for path in (vault / "_wiki/operations").glob("*.json"):
            record = json.loads(path.read_bytes())
            record["state"] = "prepared"
            path.write_text(json.dumps(record))
    elif kind == "metadata":
        path = vault / "_wiki/sources/source-a.json"
        path.write_bytes(path.read_bytes() + b"\n ")
    else:
        path = next((vault / "_wiki/history/cache").glob("*.md"))
        path.write_bytes(b"unreviewed historical replacement")
    before = snapshot(vault)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert lint_pages(settings)[0]["status"] == "invalid"
    assert snapshot(vault) == before


def test_cli_readonly_and_configured_generation(
    world: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    import vaultmind.cli as app

    settings, vault = world
    accept(world)
    runner = CliRunner()
    prefix = ["--config", str(vault.parent / "query.toml"), "wiki"]
    before = snapshot(vault)

    def forbidden_factory(settings: Settings) -> Any:
        raise AssertionError("Read-only evidence/list/lint attempted generation")

    monkeypatch.setattr(app, "_create_llm_client", forbidden_factory)
    listed = runner.invoke(app.cli, prefix + ["list"])
    assert listed.exit_code == 0, listed.output
    assert [row["path"] for row in json.loads(listed.output)] == [PAGE]
    lint = runner.invoke(app.cli, prefix + ["lint"])
    assert lint.exit_code == 0, lint.output
    assert json.loads(lint.output)[0]["status"] == "valid"
    evidence = runner.invoke(
        app.cli, prefix + ["query", "Limit?", "--page", PAGE, "--evidence-only"]
    )
    assert evidence.exit_code == 0, evidence.output
    result = json.loads(evidence.output)
    assert any(LATE in row["text"] for row in result["evidence"])
    client = Provider(dict(status="unsupported", answer="Not established.", citations=[]))
    factories: list[Settings] = []

    def factory(config: Settings) -> Provider:
        factories.append(config)
        return client

    monkeypatch.setattr(app, "_create_llm_client", factory)
    generated = runner.invoke(app.cli, prefix + ["query", "Limit?", "--page", PAGE])
    assert generated.exit_code == 0, generated.output
    assert json.loads(generated.output)["status"] == "unsupported"
    assert len(factories) == len(client.calls) == 1
    assert factories[0].llm.thinking_model == settings.llm.thinking_model
    assert snapshot(vault) == before
    source = vault / "_wiki/sources/source-a.md"
    source.unlink()
    invalid = runner.invoke(app.cli, prefix + ["lint"])
    assert invalid.exit_code != 0
    assert json.loads(invalid.output)[0]["status"] == "invalid"


def test_independent_console_reads_same_accepted_state(
    world: tuple[Settings, Path],
) -> None:
    import subprocess
    import sys

    _, vault = world
    accept(world)
    before = snapshot(vault)
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    code = (
        "import socket,sys\n"
        "def guard(event,args):\n"
        " if event == 'socket.getaddrinfo' or "
        "(event == 'socket.connect' and args[0].family != socket.AF_UNIX):\n"
        "  raise RuntimeError('Offline console attempted network')\n"
        "sys.addaudithook(guard)\n"
        "from vaultmind.cli import cli\n"
        "cli()\n"
    )
    for args in (
        ["list"],
        ["lint"],
        ["query", "Limit?", "--page", PAGE, "--evidence-only"],
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                "--config",
                str(vault.parent / "query.toml"),
                "wiki",
                *args,
            ],
            cwd=vault.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        if args[0] == "list":
            assert [row["path"] for row in observed] == [PAGE]
        elif args[0] == "lint":
            assert [row["status"] for row in observed] == ["valid"]
        else:
            assert any(LATE in row["text"] for row in observed["evidence"])
    assert snapshot(vault) == before


def test_readonly_workspace_rejects_mutation(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    accept(world)
    before = snapshot(vault)
    with WikiWorkspace(settings, read_only=True) as workspace, pytest.raises(WikiError):
        workspace.add_source(vault.parent / "input.md", "another-source")
    assert snapshot(vault) == before


def test_fenced_resolution_cannot_hide_current_conflict(world: tuple[Settings, Path]) -> None:
    settings, _ = world
    accept(world, CONFLICT + "\n```markdown\n## Resolution\n\nExample only.\n```\n")
    result = query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert result["unresolved_conflicts"] == [PAGE]
    assert lint_pages(settings)[0]["status"] == "unresolved"
    citation = next(
        row["id"] for row in result["evidence"] if row["heading"] == "Current understanding"
    )
    with pytest.raises(WikiError):
        query_pages(
            settings,
            "Limit?",
            [PAGE],
            llm=Provider(dict(status="supported", answer="19 days.", citations=[citation])),
        )


def test_missing_latest_operation_cannot_resurrect_old_page(world: tuple[Settings, Path]) -> None:
    settings, vault = world
    old = accept(world)
    operations = vault / "_wiki/operations"
    earlier = set(operations.iterdir())
    accept(world, BODY.replace("19 days", "21 days"))
    latest_operation = next(path for path in operations.iterdir() if path not in earlier)
    latest_operation.unlink()
    (vault / PAGE).write_bytes(old)
    before = snapshot(vault)
    assert lint_pages(settings)[0]["status"] == "invalid"
    with pytest.raises(WikiError):
        list_pages(settings)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert snapshot(vault) == before


@pytest.mark.parametrize(
    "body",
    [
        CONFLICT + "\n```markdown\n``` extra\n## Resolution\n\nExample only.\n```\n",
        CONFLICT.replace("## Conflicting information", "```x```\n\n## Conflicting information"),
    ],
    ids=["closing-fence-with-text", "backtick-in-opener-info"],
)
def test_invalid_fence_cannot_hide_current_conflict(
    world: tuple[Settings, Path], body: str
) -> None:
    settings, vault = world
    accept(world, body)
    before = snapshot(vault)
    result = query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert result["unresolved_conflicts"] == [PAGE]
    assert lint_pages(settings)[0]["status"] == "unresolved"
    citation = next(
        row["id"] for row in result["evidence"] if row["heading"] == "Current understanding"
    )
    with pytest.raises(WikiError):
        query_pages(
            settings,
            "Limit?",
            [PAGE],
            llm=Provider(dict(status="supported", answer="19 days.", citations=[citation])),
        )
    assert snapshot(vault) == before


@pytest.mark.parametrize("queue_state", ["missing", "missing-schema"])
def test_lint_reports_unreadable_approval_queue(
    world: tuple[Settings, Path], queue_state: str
) -> None:
    from click.testing import CliRunner

    from vaultmind.cli import cli

    settings, vault = world
    accept(world)
    queue = vault / "_wiki/review_queue.db"
    if queue_state == "missing":
        queue.unlink()
    else:
        queue.write_bytes(b"")
    before = snapshot(vault)
    rows = lint_pages(settings)
    assert [(row["page"], row["status"]) for row in rows] == [(PAGE, "invalid")]
    assert LATE not in rows[0]["detail"]
    result = CliRunner().invoke(cli, ["--config", str(vault.parent / "query.toml"), "wiki", "lint"])
    assert result.exit_code == 1
    assert json.loads(result.output) == rows
    with pytest.raises(WikiError):
        list_pages(settings)
    with pytest.raises(WikiError):
        query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert snapshot(vault) == before


@pytest.mark.parametrize("target_change", ["remove-heading", "delete-page"])
def test_approved_query_does_not_revalidate_related_destinations(
    world: tuple[Settings, Path], target_change: str
) -> None:
    settings, vault = world
    draft = vault.parent / "background.md"
    draft.write_text(BODY + "\n## Details\n\nRELATED_PAGE_BODY\n")
    with WikiWorkspace(settings) as workspace:
        proposal = workspace.propose("wiki/background.md", ["source-a"], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    accepted = accept(world, BODY + "\nSee [[wiki/background#Details]].\n")
    if target_change == "remove-heading":
        draft.write_text(BODY + "\n## Overview\n\nRELATED_PAGE_BODY\n")
        with WikiWorkspace(settings) as workspace:
            proposal = workspace.propose("wiki/background.md", ["source-a"], draft_file=draft)
            workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    else:
        (vault / "wiki/background.md").unlink()
    before = snapshot(vault)
    result = query_pages(settings, "Limit?", [PAGE], evidence_only=True)
    assert any(LATE in row["text"] for row in result["evidence"])
    assert all(row["page"] == PAGE for row in result["evidence"])
    assert all("RELATED_PAGE_BODY" not in row["text"] for row in result["evidence"])
    assert PAGE in [row["path"] for row in list_pages(settings)]
    assert {row["page"]: row["status"] for row in lint_pages(settings)}[PAGE] == "valid"
    assert (vault / PAGE).read_bytes() == accepted
    assert snapshot(vault) == before
