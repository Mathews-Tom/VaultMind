"""Real registered MCP approved-query boundaries, with exact reviewed synthetic pages."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from vaultmind.config import MCPProfileConfig, Settings, VaultConfig
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.knowledge.query import query_knowledge
from vaultmind.llm.client import LLMResponse
from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool, create_mcp_server
from vaultmind.vault.parser import VaultParser
from vaultmind.wiki.workspace import WikiError, WikiWorkspace, proposal_digest

PAGE = "wiki/cache.md"
QUESTION = "QUESTION-PRIVATE-SENTINEL cache retention?"
EVIDENCE = "EVIDENCE-PRIVATE-SENTINEL retention is 19 days."
ANSWER = "ANSWER-PRIVATE-SENTINEL retention is 19 days."
PARENT = "PARENT-PRIVATE-SENTINEL"


class Provider:
    provider_name = "scripted-mcp-query"

    def __init__(self, response: Any = None, action: Any = None) -> None:
        self.response = response
        self.action = action
        self.calls: list[dict[str, Any]] = []
        self.factories = 0

    def factory(self) -> Provider:
        self.factories += 1
        return self

    def complete(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self.calls.append(dict(messages=messages, **kwargs))
        if self.action:
            self.action()
        if isinstance(self.response, Exception):
            raise self.response
        response = self.response
        if response is None:
            evidence = json.loads(messages[0].content)["evidence"]
            response = dict(
                status="supported",
                answer=ANSWER,
                citations=[row["id"] for row in evidence if EVIDENCE in row["text"]],
            )
        return LLMResponse(text=json.dumps(response), model=kwargs["model"], usage={})

    def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise AssertionError("Query must not call multimodal provider")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Settings, Path]:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "vault"
    root.mkdir()
    settings = Settings(
        _env_file=None,
        vault=VaultConfig(path=root, excluded_folders=["wiki", "_wiki"]),
        llm={"provider": "ollama", "thinking_model": "mcp-fixture", "max_tokens": 1729},
    )
    with WikiWorkspace(settings) as workspace:
        intake = SourceIntake.from_workspace(workspace)
        parent = intake.preserve_text(
            PARENT, original_name="parent.txt", origin="urn:private-parent", adapter="test"
        )
        attachment = intake.preserve_attachment(
            b"\x00\xff", original_name="parent.bin", origin="urn:attachment", adapter="test"
        )
        child = intake.preserve_text(
            "Derived retention observation",
            original_name="child.md",
            origin="urn:child",
            adapter="test",
            parents=[parent.source_id],
            attachments=[attachment.source_id],
        )
        draft = tmp_path / "draft.md"
        draft.write_text(
            "# Cache\n\n## Current understanding\n\n"
            f"{EVIDENCE} [[_wiki/sources/{child.source_id}]]\n"
        )
        proposal = workspace.propose(PAGE, [child.source_id], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    return settings, root


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def scope_for(root: Path, **changes: Any) -> ProfileEnforcer:
    return ProfileEnforcer(replace(load_profile("researcher"), **changes), root)


def dispatch(
    world: tuple[Settings, Path], provider: Provider, args: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    settings, root = world
    return _dispatch_tool(
        "knowledge_query",
        args,
        root,
        MagicMock(),
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=kwargs.pop("enforcer", scope_for(root)),
        settings=kwargs.pop("settings", settings),
        llm_factory=provider.factory,
        **kwargs,
    )


def make_server(world: tuple[Settings, Path], provider: Provider | None, **kwargs: Any) -> Any:
    pytest.importorskip("mcp")
    settings, root = world
    return create_mcp_server(
        root,
        MagicMock(),
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=kwargs.pop("enforcer", scope_for(root)),
        settings=settings,
        llm_factory=provider.factory if provider is not None else None,
        **kwargs,
    )


async def registered(server: Any, args: dict[str, Any], name: str = "knowledge_query") -> Any:
    from mcp.types import CallToolRequest, CallToolRequestParams

    response = await server.request_handlers[CallToolRequest](
        CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name=name, arguments=args)
        )
    )
    return response.root


def payload(response: Any) -> dict[str, Any]:
    assert not response.isError
    return json.loads(response.content[0].text)


def test_registered_generation_matches_shared_core_and_is_readonly(
    world: tuple[Settings, Path],
) -> None:
    settings, root = world
    provider = Provider()
    before = snapshot(root)
    actual = payload(asyncio.run(registered(make_server(world, provider), {"question": QUESTION})))
    expected = query_knowledge(settings, QUESTION, llm_factory=Provider().factory)
    assert actual == expected
    assert actual["status"] == "supported"
    cited = [row["text"] for row in actual["evidence"] if row["id"] in actual["citations"]]
    assert any(EVIDENCE in text for text in cited)
    dependencies = actual["sources"][0]["retained_dependencies"]
    assert len(dependencies) == 2
    assert {item["retention"]["kind"] for item in dependencies} == {"text", "attachment"}
    assert provider.factories == len(provider.calls) == 1
    assert provider.calls[0]["model"] == "mcp-fixture"
    assert provider.calls[0]["max_tokens"] == 1729
    assert snapshot(root) == before


def test_registered_schema_and_profile_replacement(world: tuple[Settings, Path]) -> None:
    from mcp.types import ListToolsRequest

    async def schemas(server: Any) -> Any:
        response = await server.request_handlers[ListToolsRequest](
            ListToolsRequest(method="tools/list")
        )
        return response.root.tools

    _, root = world
    server = make_server(world, Provider())
    tools = asyncio.run(schemas(server))
    schema = next(tool.inputSchema for tool in tools if tool.name == "knowledge_query")
    assert schema["required"] == ["question"] and schema["additionalProperties"] is False
    assert schema["properties"]["question"]["maxLength"] == 4096
    assert schema["properties"]["pages"]["uniqueItems"] is True
    for policy, allowed in [
        (load_profile("researcher"), True),
        (load_profile("full"), True),
        (load_profile("planner"), False),
        (
            load_profile(
                "researcher",
                {"researcher": MCPProfileConfig(allowed_tools=["vault_read"], folder_scope=["*"])},
            ),
            False,
        ),
    ]:
        client = Provider()
        result = payload(
            asyncio.run(
                registered(
                    make_server(world, client, enforcer=ProfileEnforcer(policy, root)),
                    {"question": QUESTION, "pages": [PAGE]},
                )
            )
        )
        assert ("error" not in result) is allowed
        assert client.factories == len(client.calls) == int(allowed)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"question": ""},
        {"question": " "},
        {"question": 7},
        {"question": "x" * 4097},
        {"question": QUESTION, "pages": []},
        {"question": QUESTION, "pages": None},
        {"question": QUESTION, "pages": PAGE},
        {"question": QUESTION, "pages": [PAGE, PAGE]},
        {"question": QUESTION, "pages": [1]},
        {"question": QUESTION, "pages": [f"wiki/{i}.md" for i in range(9)]},
        {"question": QUESTION, "evidence_only": "false"},
        {"question": QUESTION, "intent_questions": ["untrusted"]},
        {"question": QUESTION, "max_context_bytes": 1},
        {"question": QUESTION, "profile": "full"},
        {"question": QUESTION, "context": EVIDENCE},
        {"question": QUESTION, "trust": True},
    ],
)
def test_strict_registered_input_is_body_free_and_audited(
    world: tuple[Settings, Path], args: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    _, root = world
    client = Provider()
    audit = AuditLogger(root.parent / "audit.jsonl")
    before = snapshot(root)
    response = asyncio.run(registered(make_server(world, client, audit_logger=audit), args))
    assert "error" in payload(response)
    assert client.factories == len(client.calls) == 0
    assert snapshot(root) == before
    jsonl = (root.parent / "audit.jsonl").read_text()
    rows = audit.query()
    assert len(rows) == 1
    assert json.loads(jsonl)["params"] == {}
    durable = jsonl + json.dumps(rows) + caplog.text + response.model_dump_json()
    for sentinel in [QUESTION, EVIDENCE, ANSWER, PARENT, "untrusted", "x" * 4097]:
        assert sentinel not in durable


@pytest.mark.parametrize(
    "kind", ["missing-enforcer", "missing-settings", "settings-root", "scope-root"]
)
def test_direct_dispatch_requires_real_permission_and_root_agreement(
    world: tuple[Settings, Path], kind: str
) -> None:
    settings, root = world
    client = Provider()
    kwargs: dict[str, Any] = {}
    if kind == "missing-enforcer":
        kwargs["enforcer"] = None
    elif kind == "missing-settings":
        kwargs["settings"] = None
    elif kind == "settings-root":
        kwargs["settings"] = Settings(_env_file=None, vault=VaultConfig(path=root.parent))
    else:
        kwargs["enforcer"] = scope_for(root.parent)
    before = snapshot(root)
    with pytest.raises(ProfileError):
        dispatch(world, client, {"question": QUESTION, "pages": [PAGE]}, **kwargs)
    assert client.factories == len(client.calls) == 0
    assert snapshot(root) == before
    assert settings.vault.path == root


def test_direct_allowlist_is_not_optional(world: tuple[Settings, Path]) -> None:
    _, root = world
    client = Provider()
    with pytest.raises(ProfileError):
        dispatch(
            world,
            client,
            {"question": QUESTION},
            enforcer=scope_for(root, allowed_tools=frozenset({"vault_read"})),
        )
    assert client.factories == len(client.calls) == 0


@pytest.mark.parametrize("folders", [("wiki",), ("_wiki",)])
def test_explicit_scope_denies_but_automatic_scope_is_invisible(
    world: tuple[Settings, Path], folders: tuple[str, ...], caplog: pytest.LogCaptureFixture
) -> None:
    _, root = world
    provider = Provider()
    audit = AuditLogger(root.parent / "audit.jsonl")
    server = make_server(
        world, provider, enforcer=scope_for(root, folder_scope=folders), audit_logger=audit
    )
    explicit = payload(asyncio.run(registered(server, {"question": QUESTION, "pages": [PAGE]})))
    automatic = payload(asyncio.run(registered(server, {"question": QUESTION})))
    assert "error" in explicit
    assert automatic["status"] == "unsupported"
    assert automatic["evidence"] == automatic["sources"] == automatic["selected_pages"] == []
    assert provider.factories == len(provider.calls) == 0
    observed = (
        json.dumps(explicit)
        + json.dumps(automatic)
        + (root.parent / "audit.jsonl").read_text()
        + json.dumps(audit.query())
        + caplog.text
    )
    for sentinel in [PAGE, EVIDENCE, PARENT, "urn:private-parent", "retention is 19 days"]:
        assert sentinel not in observed


@pytest.mark.parametrize(
    "kind", ["denied-parent", "changed-parent", "missing-parent", "changed-page"]
)
def test_retained_closure_and_approved_page_tamper_refuse_before_factory(
    world: tuple[Settings, Path], kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    settings, root = world
    valid = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)
    parent = valid["sources"][0]["retained_dependencies"][0]
    parent_path = root / parent["path"]
    scope = scope_for(root)
    if kind == "denied-parent":

        class DenyParent(ProfileEnforcer):
            def allows_path(self, relative: str) -> bool:
                return relative != parent["path"] and super().allows_path(relative)

        scope = DenyParent(load_profile("researcher"), root)
    elif kind == "changed-parent":
        parent_path.write_bytes(b"altered retained original")
    elif kind == "missing-parent":
        parent_path.unlink()
    else:
        (root / PAGE).write_text("tampered approved page")
    before = snapshot(root)
    provider = Provider()
    audit = AuditLogger(root.parent / "audit.jsonl")
    response = asyncio.run(
        registered(
            make_server(world, provider, enforcer=scope, audit_logger=audit),
            {"question": QUESTION, "pages": [PAGE]},
        )
    )
    assert "error" in payload(response)
    assert provider.factories == len(provider.calls) == 0
    assert snapshot(root) == before
    observed = (
        response.model_dump_json()
        + (root.parent / "audit.jsonl").read_text()
        + json.dumps(audit.query())
        + caplog.text
    )
    for sentinel in [QUESTION, EVIDENCE, PARENT, "urn:private-parent", parent["path"]]:
        assert sentinel not in observed


def test_evidence_only_and_unsupported_never_construct_provider(
    world: tuple[Settings, Path],
) -> None:
    provider = Provider()
    server = make_server(world, provider)
    evidence = payload(
        asyncio.run(
            registered(server, {"question": QUESTION, "pages": [PAGE], "evidence_only": True})
        )
    )
    assert evidence["status"] == "evidence" and evidence["answer"] == ""
    unsupported = payload(asyncio.run(registered(server, {"question": "unmatchablexyz"})))
    assert unsupported["status"] == "unsupported" and unsupported["evidence"] == []
    assert provider.factories == len(provider.calls) == 0
    missing = payload(
        asyncio.run(registered(make_server(world, None), {"question": QUESTION, "pages": [PAGE]}))
    )
    assert missing == {"error": "Generated wiki queries require a configured LLM client factory"}


def test_utf8_whole_input_limit_and_stricter_complete_context_precede_factory(
    world: tuple[Settings, Path],
) -> None:
    _, root = world
    provider = Provider()
    args = {"question": "é" * 300, "pages": [PAGE]}
    assert len(json.dumps(args, ensure_ascii=False)) < 500
    assert len(json.dumps(args, ensure_ascii=False).encode("utf-8")) > 500
    with pytest.raises(ProfileError):
        dispatch(world, provider, args, enforcer=scope_for(root, max_note_size_bytes=500))
    args = {"question": QUESTION, "pages": [PAGE]}
    assert len(json.dumps(args).encode()) < 500
    with pytest.raises(WikiError):
        dispatch(world, provider, args, enforcer=scope_for(root, max_note_size_bytes=500))
    assert provider.factories == len(provider.calls) == 0


def test_actual_serialized_output_limit_never_crops_unicode_or_provenance(
    world: tuple[Settings, Path],
) -> None:
    settings, root = world
    expected = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)
    # Context fits, but a generated Unicode answer exceeds the limit only in
    # the actual indent=2/default=str/ensure_ascii=True wire representation.
    provider = Provider(dict(status="supported", answer="é" * 4000, citations=["p1-s1"]))
    limit = len(json.dumps(expected, indent=2, default=str).encode()) + 9000
    server = make_server(world, provider, enforcer=scope_for(root, max_note_size_bytes=limit))
    response = asyncio.run(registered(server, {"question": QUESTION, "pages": [PAGE]}))
    result = payload(response)
    assert result == {"error": "Knowledge query denied"}
    assert provider.factories == len(provider.calls) == 1
    assert "evidence" not in result and "citations" not in result and "sources" not in result


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("PROVIDER-PRIVATE-ERROR " + EVIDENCE),
        {"status": "supported", "answer": ANSWER, "citations": ["invented"]},
    ],
)
def test_no_retry_replay_and_both_audits_are_structural(
    world: tuple[Settings, Path], failure: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from vaultmind.mcp.retry import ToolRetryExecutor

    _, root = world
    provider = Provider(failure)
    retry = MagicMock(spec=ToolRetryExecutor)
    audit = AuditLogger(root.parent / "audit.jsonl")
    response = asyncio.run(
        registered(
            make_server(world, provider, retry_executor=retry, audit_logger=audit),
            {"question": QUESTION, "pages": [PAGE]},
        )
    )
    assert "error" in payload(response)
    retry.execute.assert_not_called()
    assert provider.factories == len(provider.calls) == 1
    observed = (
        response.model_dump_json()
        + (root.parent / "audit.jsonl").read_text()
        + json.dumps(audit.query())
        + caplog.text
    )
    for sentinel in [QUESTION, EVIDENCE, ANSWER, PARENT, "PROVIDER-PRIVATE-ERROR", "Traceback"]:
        assert sentinel not in observed


def test_success_and_denial_audits_never_persist_query_bodies(
    world: tuple[Settings, Path],
) -> None:
    _, root = world
    audit = AuditLogger(root.parent / "audit.jsonl")
    response = asyncio.run(
        registered(make_server(world, Provider(), audit_logger=audit), {"question": QUESTION})
    )
    assert payload(response)["answer"] == ANSWER
    denied = asyncio.run(
        registered(
            make_server(
                world,
                Provider(),
                audit_logger=audit,
                enforcer=scope_for(root, allowed_tools=frozenset()),
            ),
            {"question": QUESTION, "pages": [PAGE]},
        )
    )
    assert "error" in payload(denied)
    jsonl = (root.parent / "audit.jsonl").read_text()
    rows = audit.query()
    assert len(rows) == 2
    assert {row["result"] for row in rows} == {"OK", "DENIED"}
    for sentinel in [QUESTION, EVIDENCE, ANSWER, PARENT, PAGE, "urn:private-parent"]:
        assert sentinel not in jsonl + json.dumps(rows)


def test_query_worker_does_not_block_event_loop_or_move_sqlite_producers(
    world: tuple[Settings, Path],
) -> None:
    from vaultmind.memory.store import EpisodeStore

    owner = threading.get_ident()

    class ThreadBoundEpisodeStore(EpisodeStore):
        def query_resolved(self, limit: int = 20) -> Any:
            assert threading.get_ident() == owner, "Unrelated SQLite producer moved threads"
            return super().query_resolved(limit=limit)

    _, root = world
    entered = threading.Event()
    release = threading.Event()

    def block() -> None:
        entered.set()
        assert release.wait(3), "Event loop did not release query worker"

    provider = Provider(action=block)
    episodes = ThreadBoundEpisodeStore(root.parent / "episodes.db")
    server = make_server(world, provider, episode_store=episodes)

    async def run() -> dict[str, Any]:
        task = asyncio.create_task(registered(server, {"question": QUESTION, "pages": [PAGE]}))
        try:
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                # This real store was constructed on the current thread. An
                # indiscriminate to_thread(dispatch) would break this control.
                result = payload(await registered(server, {}, name="episode_query"))
                assert result["count"] == 0 and result["episodes"] == []
        finally:
            release.set()
        return payload(await task)

    assert asyncio.run(run())["status"] == "supported"
    assert provider.factories == len(provider.calls) == 1


def test_unrelated_sdk_schema_validation_remains_pre_callback(
    world: tuple[Settings, Path],
) -> None:
    _, root = world
    audit = AuditLogger(root.parent / "audit.jsonl")
    server = make_server(world, Provider(), audit_logger=audit)
    invalid = asyncio.run(
        registered(
            server, {"content": "ordinary input", "do_not_retain": True}, name="capture_note"
        )
    )
    assert invalid.isError
    assert "Input validation error" in invalid.content[0].text
    assert audit.query() == [] and not (root.parent / "audit.jsonl").exists()


def test_real_sdk_stdio_generation_rejection_and_raw_frames(
    world: tuple[Settings, Path],
) -> None:
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _, root = world
    launcher = root.parent / "mcp-query-launcher.py"
    audit_path = root.parent / "stdio-audit.jsonl"
    frames_path = root.parent / "stdio.frames.jsonl"
    launcher.write_text(
        "import asyncio,json,sys\n"
        "from pathlib import Path\n"
        "from mcp.server.stdio import stdio_server\n"
        "from vaultmind.config import Settings,VaultConfig\n"
        "from vaultmind.llm.client import LLMResponse\n"
        "from vaultmind.mcp.auth import AuditLogger,ProfileEnforcer\n"
        "from vaultmind.mcp.profiles import load_profile\n"
        "from vaultmind.mcp.server import create_mcp_server\n"
        "from vaultmind.vault.parser import VaultParser\n"
        "root=Path(sys.argv[1]); audit=Path(sys.argv[2])\n"
        "frames=Path(sys.argv[3]).open('w')\n"
        "class Output:\n"
        " async def write(self,text):\n"
        "  frames.write(text); frames.flush(); return sys.stdout.write(text)\n"
        " async def flush(self):\n"
        "  sys.stdout.flush()\n"
        "settings=Settings(_env_file=None,vault=VaultConfig(path=root))\n"
        "class Provider:\n"
        " def complete(self,messages,**kwargs):\n"
        "  return LLMResponse(text=json.dumps(dict(status='supported',"
        "answer='STDIO-ANSWER-SENTINEL',citations=['p1-s1'])),model=kwargs['model'],usage={})\n"
        "server=create_mcp_server(root,None,None,VaultParser(settings.vault),"
        "enforcer=ProfileEnforcer(load_profile('researcher'),root),"
        "audit_logger=AuditLogger(audit),settings=settings,llm_factory=Provider)\n"
        "async def main():\n"
        " async with stdio_server(stdout=Output()) as (read,write):\n"
        "  await server.run(read,write,server.create_initialization_options())\n"
        "asyncio.run(main())\n"
    )
    stderr_path = root.parent / "stdio.stderr"

    async def run() -> tuple[Any, Any, Any]:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(launcher), str(root), str(audit_path), str(frames_path)],
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(root.parent),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                "ANONYMIZED_TELEMETRY": "False",
            },
        )
        with stderr_path.open("w") as stderr:
            async with stdio_client(params, errlog=stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert "knowledge_query" in {tool.name for tool in listed.tools}
                    good = await session.call_tool(
                        "knowledge_query", {"question": QUESTION, "pages": [PAGE]}
                    )
                    # SDK maxLength/additionalProperties validation would echo
                    # the input before our callback, so exercise both via stdio.
                    bad = await session.call_tool(
                        "knowledge_query",
                        {"question": "SDK-REJECTED-SENTINEL" * 220, "trust": EVIDENCE},
                    )
                    unrelated = await session.call_tool("vault_read", {})
                    return good, bad, unrelated

    good, bad, unrelated = asyncio.run(run())
    assert payload(good)["answer"] == "STDIO-ANSWER-SENTINEL"
    assert payload(bad) == {"error": "Knowledge query failed"}
    assert unrelated.isError and "Input validation error" in unrelated.content[0].text
    with sqlite3.connect(audit_path.with_suffix(".db")) as db:
        rows = db.execute(
            "SELECT tool,result,reason,note_path,change_detail,output_summary FROM audit_events"
        ).fetchall()
    assert len(rows) == 2 and {row[1] for row in rows} == {"OK", "ERROR"}
    durable = audit_path.read_text() + json.dumps(rows) + stderr_path.read_text()
    rejected = [
        line
        for line in frames_path.read_text().splitlines()
        if "Knowledge query failed" in line or "Input validation error" in line
    ]
    assert len(rejected) == 2
    rejected_frames = "\n".join(rejected) + bad.model_dump_json() + unrelated.model_dump_json()
    for sentinel in [
        QUESTION,
        EVIDENCE,
        ANSWER,
        PARENT,
        "STDIO-ANSWER-SENTINEL",
        "SDK-REJECTED-SENTINEL",
    ]:
        assert sentinel not in durable + rejected_frames


def test_missing_registered_enforcer_is_structurally_audited(
    world: tuple[Settings, Path],
) -> None:
    _, root = world
    audit = AuditLogger(root.parent / "audit.jsonl")
    provider = Provider()
    response = asyncio.run(
        registered(
            make_server(world, provider, enforcer=None, audit_logger=audit),
            {"question": QUESTION, "pages": [PAGE]},
        )
    )
    assert payload(response) == {"error": "Knowledge query denied"}
    assert provider.factories == len(provider.calls) == 0
    assert [row["result"] for row in audit.query()] == ["DENIED"]
    assert json.loads((root.parent / "audit.jsonl").read_text())["params"] == {}


def test_automatic_selection_filters_forbidden_candidates_before_evidence(
    world: tuple[Settings, Path],
) -> None:
    settings, root = world
    hidden = "wiki/HIDDEN-PAGE-SENTINEL.md"
    source = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)["sources"][0]["id"]
    with WikiWorkspace(settings) as workspace:
        draft = root.parent / "hidden.md"
        draft.write_text(
            "# Cache retention\n\n## Current understanding\n\n"
            f"HIDDEN-BODY-SENTINEL {'cache retention ' * 100}"
            f" [[_wiki/sources/{source}]]\n"
        )
        proposal = workspace.propose(hidden, [source], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))

    class DenyHidden(ProfileEnforcer):
        def allows_path(self, relative: str) -> bool:
            return relative != hidden and super().allows_path(relative)

    provider = Provider()
    result = payload(
        asyncio.run(
            registered(
                make_server(world, provider, enforcer=DenyHidden(load_profile("researcher"), root)),
                {"question": "cache retention"},
            )
        )
    )
    assert result["selection"]["selected_pages"] == [PAGE]
    assert result["selection"]["limited"] is False
    assert hidden not in json.dumps(result) and "HIDDEN-BODY-SENTINEL" not in json.dumps(result)
    assert hidden not in provider.calls[0]["messages"][0].content


def test_conflict_citations_and_complete_evidence_are_not_adapter_rewritten(
    world: tuple[Settings, Path],
) -> None:
    settings, root = world
    source = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)["sources"][0]["id"]
    with WikiWorkspace(settings) as workspace:
        draft = root.parent / "conflict.md"
        draft.write_text(
            "# Cache\n\n## Current understanding\n\nRetention remains unresolved.\n\n"
            f"## Conflicting information\n\n19 versus 7 days. [[_wiki/sources/{source}]]\n"
        )
        proposal = workspace.propose(PAGE, [source], draft_file=draft)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    approved = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)
    citations = [row["id"] for row in approved["evidence"] if "19 versus 7 days." in row["text"]]
    response = dict(status="conflicting", answer="The sources disagree.", citations=citations)
    provider = Provider(response)
    result = payload(
        asyncio.run(
            registered(make_server(world, provider), {"question": QUESTION, "pages": [PAGE]})
        )
    )
    expected = query_knowledge(settings, QUESTION, [PAGE], llm_factory=Provider(response).factory)
    assert result == expected
    assert result["unresolved_conflicts"] == [PAGE]
    cited = [row["text"] for row in result["evidence"] if row["id"] in result["citations"]]
    assert any("19 versus 7 days." in text for text in cited)


def test_postgeneration_parent_change_refuses_the_entire_result(
    world: tuple[Settings, Path],
) -> None:
    settings, root = world
    initial = query_knowledge(settings, QUESTION, [PAGE], evidence_only=True)
    parent = initial["sources"][0]["retained_dependencies"][0]
    provider = Provider(action=lambda: (root / parent["path"]).write_bytes(b"changed"))
    result = payload(
        asyncio.run(
            registered(make_server(world, provider), {"question": QUESTION, "pages": [PAGE]})
        )
    )
    assert result == {"error": "Knowledge query failed"}
    assert provider.factories == len(provider.calls) == 1


def test_global_complete_context_bound_precedes_provider(
    world: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.knowledge.query as module

    monkeypatch.setattr(module, "MAX_QUERY_CONTEXT_BYTES", 500)
    provider = Provider()
    result = payload(
        asyncio.run(
            registered(make_server(world, provider), {"question": QUESTION, "pages": [PAGE]})
        )
    )
    assert result == {"error": "Knowledge query failed"}
    assert provider.factories == len(provider.calls) == 0
