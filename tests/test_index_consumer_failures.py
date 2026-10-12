"""Post-write index failures remain errors through the registered MCP callback."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
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
    from typing import Any


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


@pytest.mark.parametrize("stage", ["embed", "keyword_replace"])
@pytest.mark.parametrize("failure_position", [0, 1])
async def test_research_partial_abort_preserves_saved_paths_and_reports_index_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, failure_position: int
) -> None:
    from youtube_transcript_api import YouTubeTranscriptApi

    from vaultmind.indexer.store import IndexUpdateError
    from vaultmind.research.pipeline import ResearchIndexError, run_research_pipeline
    from vaultmind.research.searcher import SearchResult

    from .index_support import ALL_SURFACES

    env = make_env(tmp_path.resolve())
    source_name = ("Alpha", "Beta")[failure_position]
    source = f"research/harbor/{source_name}.md"
    prior = "oldresearchtoken last-good research source"
    current = "newresearchtoken replacement transcript"
    env.write(source, prior)
    env.index(source)
    before = env.store._collection.get(
        where={"note_path": source}, include=["documents", "metadatas", "embeddings"]
    )

    def search(query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult("fixture-a", "Alpha", "https://example.invalid/a", "fixture", 1, 0, ""),
            SearchResult("fixture-b", "Beta", "https://example.invalid/b", "fixture", 1, 0, ""),
        ]

    def transcript(video_id: str, **kwargs: Any) -> list[dict[str, str]]:
        body = (
            current
            if video_id == ("fixture-a", "fixture-b")[failure_position]
            else ("earlierresearchtoken successfully indexed source")
        )
        return [{"text": body}]

    def complete(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            text=json.dumps({"summary": "summaryresearchtoken synthetic analysis"})
        )

    monkeypatch.setattr("vaultmind.research.searcher._run_yt_search", search)
    monkeypatch.setattr(YouTubeTranscriptApi, "get_transcript", transcript)
    if stage == "embed":
        embed = env.embedder.embed_texts

        def failed_embed(texts: list[str]) -> list[list[float]]:
            if any(current in text for text in texts):
                raise RuntimeError("synthetic research embedding outage")
            return embed(texts)

        monkeypatch.setattr(env.embedder, "embed_texts", failed_embed)
    else:
        replace = env.bm25.replace_note

        def failed_replace(note_path: str, rows: list[tuple[str, str, str, str]]) -> None:
            if note_path == source:
                raise sqlite3.OperationalError("synthetic research keyword outage")
            replace(note_path, rows)

        monkeypatch.setattr(env.bm25, "replace_note", failed_replace)

    try:
        with pytest.raises(ResearchIndexError) as raised:
            await run_research_pipeline(
                "harbor", env.vault, SimpleNamespace(complete=complete), env.store, env.parser
            )
        error = raised.value
        alpha = env.vault / "research/harbor/Alpha.md"
        beta = env.vault / "research/harbor/Beta.md"
        summary = env.vault / "research/harbor/summary.md"
        paths = (alpha, beta, summary)
        assert error.saved_paths == paths
        assert error.failed_path == paths[failure_position]
        assert error.indexed_paths == paths[:failure_position]
        assert error.unattempted_paths == paths[failure_position + 1 :]
        assert all(path.is_file() for path in error.saved_paths)
        assert current in error.failed_path.read_text()
        if failure_position == 1:
            assert "earlierresearchtoken" in alpha.read_text()
            assert env.surfaces("earlierresearchtoken") == ALL_SURFACES
        assert "summaryresearchtoken" in summary.read_text()
        assert isinstance(error.__cause__, IndexUpdateError)
        assert error.__cause__.stage.value == stage
        assert env.surfaces("summaryresearchtoken") == set()
        for path in error.unattempted_paths:
            assert (
                env.store._collection.get(where={"note_path": str(path.relative_to(env.vault))})[
                    "ids"
                ]
                == []
            )
        if stage == "embed":
            after = env.store._collection.get(
                where={"note_path": source}, include=["documents", "metadatas", "embeddings"]
            )
            assert before["ids"] == after["ids"]
            assert before["documents"] == after["documents"]
            assert before["metadatas"] == after["metadatas"]
            assert (before["embeddings"] == after["embeddings"]).all()
            assert env.surfaces("oldresearchtoken") == ALL_SURFACES
            assert env.surfaces("newresearchtoken") == set()
        else:
            assert env.surfaces("oldresearchtoken") == {"keyword", "hybrid"}
            assert env.surfaces("newresearchtoken") == {"vector", "hybrid"}
    finally:
        env.close()
