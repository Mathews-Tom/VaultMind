"""MCP capture preserves exact sources before linked ordinary projections."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import frontmatter
import pytest

from vaultmind.config import Settings, VaultConfig
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool, create_mcp_server
from vaultmind.vault.parser import VaultParser
from vaultmind.vault.security import ReservedNamespaceError
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from pathlib import Path


def _settings(root: Path) -> Settings:
    return Settings(_env_file=None, vault=VaultConfig(path=root), search={"hybrid_enabled": False})


def _call(
    root: Path,
    args: dict[str, Any],
    *,
    name: str = "capture_note",
    scope: ProfileEnforcer | None = None,
    store: Any = None,
) -> dict[str, Any]:
    settings = _settings(root)
    return _dispatch_tool(
        name,
        args,
        root,
        store if store is not None else MagicMock(),
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=scope or ProfileEnforcer(load_profile("full"), root),
        settings=settings,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("name", ["capture", "capture_note"])
def test_exact_original_and_linked_projection(tmp_path: Path, name: str) -> None:
    raw = "# Exact input\r\n\r\nUnicode café; trailing spaces  \r\n"
    store = MagicMock()
    result = _call(
        tmp_path,
        {"content": raw, "title": "Title: with YAML", "tags": ["test", "x: y"]},
        name=name,
        store=store,
    )
    source_id = result["source_id"]
    assert result["status"] == "ok"
    assert result["preserved"] is True and result["approved"] is False
    assert result["original"]["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert (tmp_path / result["original"]["path"]).read_bytes() == raw.encode()
    projection = frontmatter.loads((tmp_path / result["path"]).read_text())
    assert projection["source_id"] == source_id
    assert projection["source_path"] == result["original"]["path"]
    assert projection["title"] == "Title: with YAML"
    assert projection["tags"] == ["test", "x: y"]
    assert projection["approved"] is False
    assert result["projection"]["status"] == "succeeded"
    assert result["index"]["status"] == "succeeded"
    store.index_single_note.assert_called_once()
    with SourceIntake(_settings(tmp_path)) as intake:
        record = intake.read(source_id)
    assert record.content == raw.encode()
    assert record.provenance["adapter"] == f"mcp:full:{name}"
    assert record.provenance["origin"] == "mcp"
    assert record.provenance["original_retained"] is True


def test_fresh_identity_not_content_dedup(tmp_path: Path) -> None:
    first = _call(tmp_path, {"content": "same body"})
    second = _call(tmp_path, {"content": "same body"})
    assert first["source_id"] != second["source_id"]
    assert first["path"] != second["path"]
    assert first["original"]["sha256"] == second["original"]["sha256"]


def test_stable_token_replay_and_revisions_across_fresh_instances(tmp_path: Path) -> None:
    store = MagicMock()
    args = {"content": "first revision", "idempotency_token": "event-123"}
    first = _call(tmp_path, args, store=store)
    before = _snapshot(tmp_path)
    replay = _call(tmp_path, args, store=store)
    assert replay["source_id"] == first["source_id"]
    assert replay["original"]["status"] == "unchanged"
    assert _snapshot(tmp_path) == before
    store.index_single_note.assert_called_once()
    revised = _call(tmp_path, {**args, "content": "second revision"}, store=store)
    assert revised["source_id"] != first["source_id"]
    metadata_revision = _call(tmp_path, {**args, "title": "Changed title"}, store=store)
    assert metadata_revision["source_id"] != first["source_id"]
    assert (tmp_path / first["original"]["path"]).read_text() == "first revision"


def test_index_failure_is_partial_and_never_retried_implicitly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "unretained diagnostic body must not escape"
    store = MagicMock()
    store.index_single_note.side_effect = RuntimeError(secret)
    args = {"content": "retained source", "idempotency_token": "partial-1"}
    result = _call(tmp_path, args, store=store)
    assert result["status"] == "partial"
    assert result["projection"]["status"] == "succeeded"
    assert result["index"]["status"] == "failed"
    assert (tmp_path / result["original"]["path"]).read_text() == "retained source"
    assert (tmp_path / result["path"]).exists()
    replay = _call(tmp_path, args, store=store)
    assert replay["index"]["status"] == "failed"
    store.index_single_note.assert_called_once()
    assert secret not in json.dumps(result) + json.dumps(replay) + caplog.text


def test_projection_failure_preserves_original_without_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vaultmind.wiki.workspace import WikiWorkspace

    writer = WikiWorkspace._atomic_bytes

    def fail_projection(self: Any, path: Path, content: bytes) -> None:
        if "_wiki" not in path.parts:
            raise OSError("private exception text")
        writer(self, path, content)

    monkeypatch.setattr(WikiWorkspace, "_atomic_bytes", fail_projection)
    store = MagicMock()
    args = {"content": "retained before projection", "idempotency_token": "projection-1"}
    result = _call(tmp_path, args, store=store)
    assert result["preserved"] is True
    assert result["projection"]["status"] == "failed"
    assert result["index"]["status"] == "skipped"
    assert not (tmp_path / result["path"]).exists()
    assert (tmp_path / result["original"]["path"]).read_text() == args["content"]
    _call(tmp_path, args, store=store)
    store.index_single_note.assert_not_called()


@pytest.mark.parametrize("name", ["capture", "capture_note"])
def test_planner_denied_before_body_lock_or_projection(tmp_path: Path, name: str) -> None:
    scope = ProfileEnforcer(load_profile("planner"), tmp_path)
    with pytest.raises(ProfileError):
        _call(tmp_path, {"content": "DO NOT RETAIN THIS BODY"}, name=name, scope=scope)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [
        {"content": "secret", "do_not_retain": True},
        {"content": "secret", "idempotency_token": "x" * 129},
        {"content": "secret", "idempotency_token": "contains whitespace"},
        {"content": "\ufeff# Unsupported BOM"},
        {"content": "Unsupported\x00control"},
        {"content": "secret", "tags": [7]},
        {"content": "secret", "note_type": "invalid"},
        {"content": "secret", "title": 7},
        {"content": "secret", "folder": "../outside"},
        {"content": "secret", "folder": "wiki"},
        {"content": "secret", "folder": "_wiki"},
    ],
)
def test_rejected_arguments_never_persist(tmp_path: Path, args: dict[str, Any]) -> None:
    with pytest.raises((ProfileError, ReservedNamespaceError, WikiError)):
        _call(tmp_path, args)
    assert list(tmp_path.iterdir()) == []


def test_original_and_generated_projection_size_preflight(tmp_path: Path) -> None:
    for body, limit in [("secret", 1), ("x", 150)]:
        scope = ProfileEnforcer(replace(load_profile("full"), max_note_size_bytes=limit), tmp_path)
        with pytest.raises(ProfileError):
            _call(tmp_path, {"content": body}, scope=scope)
        assert list(tmp_path.iterdir()) == []
    with pytest.raises(WikiError):
        _call(tmp_path, {"content": "x" * (256 * 1024 + 1)})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", [".md", ".json", ".intake.json", ".processing.json"])
def test_every_preservation_destination_preflight(tmp_path: Path, suffix: str) -> None:
    class DenyOneOutput(ProfileEnforcer):
        def allows_path(self, relative: str) -> bool:
            if relative.startswith("_wiki/sources/i-") and relative.endswith(suffix):
                return False
            return super().allows_path(relative)

    with pytest.raises(WikiError):
        _call(tmp_path, {"content": "secret"}, scope=DenyOneOutput(load_profile("full"), tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_projection_alias_into_reserved_storage_is_rejected_before_preservation(
    tmp_path: Path,
) -> None:
    (tmp_path / "_wiki").mkdir()
    (tmp_path / "ordinary-alias").symlink_to(tmp_path / "_wiki", target_is_directory=True)
    before = _snapshot(tmp_path)
    with pytest.raises(ReservedNamespaceError):
        _call(tmp_path, {"content": "secret", "folder": "ordinary-alias"})
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "_wiki/wiki.lock").exists()


@pytest.mark.parametrize(
    "name,args",
    [
        ("capture", {"content": "x"}),
        ("capture_note", {"content": "x"}),
        ("vault_search", {"query": "x"}),
    ],
)
def test_source_routes_require_real_settings(
    tmp_path: Path, name: str, args: dict[str, Any]
) -> None:
    with pytest.raises(ProfileError, match="selected Settings"):
        _dispatch_tool(
            name,
            args,
            tmp_path,
            MagicMock(),
            MagicMock(),
            VaultParser(VaultConfig(path=tmp_path)),
            enforcer=ProfileEnforcer(load_profile("full"), tmp_path),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("tool", ["capture", "capture_note", "vault_write"])
def test_registered_denial_is_body_free_in_both_audit_sinks(
    tmp_path: Path, tool: str, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams

    root = tmp_path / "vault"
    root.mkdir()
    settings = _settings(root)
    secret = "UNRETAINED-PRIVATE-BODY"
    audit = AuditLogger(tmp_path / "audit.jsonl")
    scope = ProfileEnforcer(load_profile("planner"), root)
    server = create_mcp_server(
        root,
        MagicMock(),
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=scope,
        audit_logger=audit,
        settings=settings,
    )
    args = {
        "content": secret,
        "title": secret,
        "idempotency_token": secret,
        "path": "04-resources/no.md",
    }

    async def request() -> Any:
        result = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call", params=CallToolRequestParams(name=tool, arguments=args)
            )
        )
        return result.root

    result = asyncio.run(request())
    assert "error" in json.loads(result.content[0].text)
    assert list(root.iterdir()) == []
    durable = (tmp_path / "audit.jsonl").read_text()
    jsonl_events = [json.loads(line) for line in durable.splitlines()]
    sqlite_events = audit.query()
    assert len(jsonl_events) == len(sqlite_events) == 1
    assert jsonl_events[0]["result"] == sqlite_events[0]["result"] == "DENIED"
    assert jsonl_events[0]["params"] == {}
    assert secret not in durable + json.dumps(sqlite_events) + caplog.text + result.content[0].text


def test_registered_capture_offloads_and_bypasses_retry_provider(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams

    settings = _settings(tmp_path)
    retry = MagicMock()
    store = MagicMock()
    server = create_mcp_server(
        tmp_path,
        store,
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=ProfileEnforcer(load_profile("full"), tmp_path),
        retry_executor=retry,
        settings=settings,
    )

    async def request() -> dict[str, Any]:
        import threading

        event_loop_thread = threading.get_ident()

        def index(*args: Any) -> None:
            assert threading.get_ident() != event_loop_thread

        store.index_single_note.side_effect = index
        response = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="capture_note",
                    arguments={"content": "offline input", "idempotency_token": "registered-1"},
                ),
            )
        )
        return json.loads(response.root.content[0].text)

    result = asyncio.run(request())
    assert result["preserved"] is True
    assert result["index"]["status"] == "succeeded"
    retry.execute.assert_not_called()


def test_selected_settings_root_mismatch_never_writes_either_vault(tmp_path: Path) -> None:
    root = tmp_path / "served"
    other = tmp_path / "selected"
    root.mkdir()
    other.mkdir()
    with pytest.raises(ProfileError, match="do not match"):
        _dispatch_tool(
            "capture_note",
            {"content": "private"},
            root,
            MagicMock(),
            MagicMock(),
            VaultParser(VaultConfig(path=root)),
            enforcer=ProfileEnforcer(load_profile("full"), root),
            settings=_settings(other),
        )
    assert list(root.iterdir()) == [] and list(other.iterdir()) == []


def test_projection_permission_precedes_original_persistence(tmp_path: Path) -> None:
    scope = ProfileEnforcer(replace(load_profile("full"), folder_scope=("_wiki",)), tmp_path)
    with pytest.raises(ProfileError):
        _call(tmp_path, {"content": "private", "folder": "00-inbox"}, scope=scope)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [
        {"content": "PRIVATE-REJECTED-BODY", "do_not_retain": True},
        {"content": "PRIVATE-REJECTED-BODY" * 15000},
    ],
)
def test_registered_rejection_is_body_free_at_sdk_and_handler_boundaries(
    tmp_path: Path, args: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams

    root = tmp_path / "vault"
    root.mkdir()
    settings = _settings(root)
    audit = AuditLogger(tmp_path / "audit.jsonl")
    server = create_mcp_server(
        root,
        MagicMock(),
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=ProfileEnforcer(load_profile("full"), root),
        audit_logger=audit,
        settings=settings,
    )

    async def request() -> Any:
        response = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="capture_note", arguments=args),
            )
        )
        return response.root

    response = asyncio.run(request())
    assert response.isError or "error" in json.loads(response.content[0].text)
    assert list(root.iterdir()) == []
    sqlite_events = audit.query()
    if args.get("do_not_retain") is True:
        # The SDK rejects the schema's forbidden retention flag before dispatch.
        assert response.isError
        assert not (tmp_path / "audit.jsonl").exists()
        assert sqlite_events == []
        durable = ""
    else:
        durable = (tmp_path / "audit.jsonl").read_text()
        jsonl_events = [json.loads(line) for line in durable.splitlines()]
        assert len(jsonl_events) == len(sqlite_events) == 1
        assert jsonl_events[0]["result"] == sqlite_events[0]["result"]
        assert sqlite_events[0]["result"] in {"DENIED", "ERROR"}
        assert jsonl_events[0]["params"] == {}
    assert "PRIVATE-REJECTED-BODY" not in (
        durable + json.dumps(sqlite_events) + caplog.text + response.content[0].text
    )


def test_registered_search_discovers_current_preserved_and_ordinary_sources(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams

    raw = "source discovery uniquephrase"
    capture = _call(tmp_path, {"content": raw})
    settings = _settings(tmp_path)
    store = MagicMock()
    store.search.return_value = []
    server = create_mcp_server(
        tmp_path,
        store,
        MagicMock(),
        VaultParser(settings.vault),
        enforcer=ProfileEnforcer(load_profile("full"), tmp_path),
        settings=settings,
    )
    before = _snapshot(tmp_path)

    async def request() -> dict[str, Any]:
        response = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="vault_search", arguments={"query": "uniquephrase"}
                ),
            )
        )
        return json.loads(response.root.content[0].text)

    result = asyncio.run(request())
    assert result["status"] == "source-only" and result["approved"] is False
    hits = {hit["path"]: hit for hit in result["results"]}
    assert set(hits) == {capture["path"], capture["original"]["path"]}
    assert hits[capture["original"]["path"]]["source_id"] == capture["source_id"]
    assert hits[capture["original"]["path"]]["content"] == raw
    for hit in hits.values():
        path = tmp_path / hit["path"]
        expected = (
            frontmatter.loads(path.read_text()).content if hit["source_kind"] == "ordinary" else raw
        )
        assert hit["content"] == expected
        assert hit["current_digest"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert hit["distance"] is None and hit["approved"] is False
    assert _snapshot(tmp_path) == before
