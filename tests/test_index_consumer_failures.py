"""Post-write index failures remain errors through the registered MCP callback."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams

from vaultmind.config import load_settings
from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import create_mcp_server

from .index_support import make_env

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("tool", ["vault_write", "capture_note"])
async def test_registered_mcp_mutation_reports_saved_index_failure(
    tmp_path: Path, tool: str
) -> None:
    env = make_env(tmp_path.resolve())
    audit = AuditLogger(env.data / "audit.jsonl")
    try:
        server = create_mcp_server(
            env.vault,
            env.store,
            KnowledgeGraph(load_settings(tmp_path / "config.toml").graph),
            env.parser,
            enforcer=ProfileEnforcer(load_profile("full"), env.vault),
            audit_logger=audit,
        )
        body = "Current shipyard evidence: indigoquay retains the pressure gauge."
        arguments = (
            {"path": "claim.md", "content": body}
            if tool == "vault_write"
            else {"title": "Shipyard claim", "content": body}
        )
        env.bm25.fail_on.add("replace_note")
        response = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=tool, arguments=arguments),
            )
        )
        result = json.loads(response.root.content[0].text)
        assert result["saved"] is True
        assert "error" in result and "status" not in result
        assert "keyword_replace" in result["error"]
        path = env.vault / result["path"]
        saved = path.read_bytes()
        assert body in saved.decode()
        events = audit.query(tool=tool)
        assert [event["result"] for event in events] == ["ERROR"]
        assert "keyword_replace" in events[0]["reason"]
        assert env.surfaces("indigoquay") == {"vector", "hybrid"}

        env.bm25.fail_on.clear()
        env.store.index_single_note(env.parser.parse_file(path), env.parser)
        assert env.surfaces("indigoquay") == {"vector", "keyword", "hybrid"}
        assert path.read_bytes() == saved
    finally:
        audit._conn.close()
        env.close()
