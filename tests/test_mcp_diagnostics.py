"""Real CLI assembly and registered search during an index population change."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

import frontmatter
import pytest
from click.testing import CliRunner


@pytest.mark.parametrize("profile", ["root-scope", "full"])
@pytest.mark.parametrize("fail_after_request", [False, True])
def test_concurrent_index_shrink_keeps_diagnostics_scoped_and_restores_sinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    fail_after_request: bool,
) -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams

    import vaultmind.cli as commands
    import vaultmind.config as configuration
    from vaultmind.indexer.embedder import Embedder

    for key in tuple(os.environ):
        if key.startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setenv("VAULTMIND_OPENAI_API_KEY", "offline-fixture-not-a-secret")
    monkeypatch.setattr(configuration, "VAULTMIND_HOME", tmp_path / "vaultmind-home")
    monkeypatch.chdir(tmp_path)

    def deny_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network forbidden in diagnostic regression")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_acceptance.py"
    spec = importlib.util.spec_from_file_location("diagnostic_acceptance_fixture", script)
    assert spec is not None and spec.loader is not None
    acceptance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acceptance)
    _, fixture, config = acceptance.mcp_root_fixture(tmp_path / "scenario", "root-scoped-small")
    # Both of these must remain discoverable without indexed candidates. The
    # shared small-index fixture deliberately leaves the private note unindexed.
    private = "03-private/closer-00.md"
    for relative in ("02-projects/suggestion.md", private):
        path = fixture["vault"] / relative
        path.write_text(path.read_text() + "\n" + acceptance.TEXT + "\n", encoding="utf-8")
    before = acceptance.mcp_root_snapshot(fixture)
    deleted = {
        "allowed-source",
        "source-second",
        "source-third",
        "allowed-suggestion",
        "safe-alias",
    }

    def concurrent_embeddings(self: Embedder, texts: list[str]) -> list[list[float]]:
        # A deterministic concurrent writer between the metadata preflight and query.
        # The tool remains real; neither its handler nor its result is replaced.
        fixture["store"]._collection.delete(ids=sorted(deleted))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(Embedder, "_embed_texts_uncached", concurrent_embeddings)
    payload: dict[str, Any] = {}
    failure_directory = tmp_path / "real-open-failure"
    failure_directory.mkdir()

    async def registered_request(server: Any, stdio_server: Any) -> None:
        request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="vault_search", arguments={"query": acceptance.TEXT, "n_results": 100}
            ),
        )
        result = (await server.request_handlers[CallToolRequest](request)).root
        payload.update(json.loads(result.content[0].text))
        logging.getLogger("vaultmind.diagnostics").warning("ordinary-warning-control")
        logging.getLogger("vaultmind.diagnostics").error("ordinary-error-control")
        if fail_after_request:
            with failure_directory.open("rb"):
                raise AssertionError("Opening a directory unexpectedly succeeded")

    # Only the outer transport loop is replaced here; the SDK stdio case separately
    # covers transport. This exercises real CLI logging and the registered SDK handler.
    monkeypatch.setattr(commands, "_run_mcp", registered_request)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [handler])
    monkeypatch.setattr(root_logger, "level", logging.DEBUG)
    original_level, original_filters = handler.level, handler.filters[:]
    result = CliRunner().invoke(
        commands.cli, ["-c", str(config), "mcp-serve", "--profile", profile]
    )
    if fail_after_request:
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0, (result.exception, result.output)
    assert payload["status"] == "source-only"
    assert payload["approved"] is False
    paths = {hit["path"] for hit in payload["results"]}
    assert payload["count"] == len(paths)
    allowed = {"02-projects/peer.md", "02-projects/source.md", "02-projects/suggestion.md"}
    if profile == "root-scope":
        assert paths == allowed
    else:
        assert paths == allowed | {private}
    for hit in payload["results"]:
        raw = (fixture["vault"] / hit["path"]).read_bytes()
        assert hit["content"] == frontmatter.loads(raw.decode("utf-8")).content
        assert hit["current_digest"] == hashlib.sha256(raw).hexdigest()
        assert hit["distance"] is None
        assert hit["approved"] is False and hit["status"] == "source-only"
    diagnostics = output.getvalue()
    if profile == "root-scope":
        assert "elements in index" not in diagnostics
        assert "n_results =" not in diagnostics
    assert "WARNING vaultmind.diagnostics ordinary-warning-control" in diagnostics
    assert "ERROR vaultmind.diagnostics ordinary-error-control" in diagnostics
    after = acceptance.mcp_root_snapshot(fixture)
    expected = {**before, "index": [row for row in before["index"] if row[0] not in deleted]}
    assert after == expected
    assert handler.level == original_level
    assert handler.filters == original_filters
