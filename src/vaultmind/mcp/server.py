"""MCP server — exposes vault read/write/search/graph as MCP tools.

Usage:
    vaultmind mcp-serve

This starts an MCP server that Claude Desktop, Claude Code, or any
MCP-compatible agent can connect to for vault operations.

Requires: pip install vaultmind[mcp]
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from vaultmind.vault.security import PathTraversalError, validate_vault_path

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)


def create_mcp_server(
    vault_path: Path,
    store: VaultStore,
    graph: KnowledgeGraph,
    parser: VaultParser,
) -> Any:  # Returns mcp.server.Server (optional dep)
    """Create and configure the MCP server with vault tools.

    Returns an MCP Server instance ready to run.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("vaultmind")

    @server.list_tools()  # type: ignore[misc,untyped-decorator,unused-ignore]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="vault_search",
                description=(
                    "Semantic search over the Obsidian vault."
                    " Returns relevant note chunks ranked by similarity."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                            "default": 5,
                        },
                        "note_type": {
                            "type": "string",
                            "description": "Filter by note type (optional)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="vault_read",
                description=(
                    "Read the full content of a specific note by its path relative to vault root."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Note path relative to vault root"
                                " (e.g., '02-projects/cairn/architecture.md')"
                            ),
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="vault_write",
                description=(
                    "Create or overwrite a note in the vault."
                    " Content should be valid markdown with YAML frontmatter."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Note path relative to vault root",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full markdown content including frontmatter",
                        },
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="vault_list",
                description=("List notes in a vault folder, optionally filtered by tag or type."),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": ("Folder path relative to vault root (default: root)"),
                            "default": "",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag (optional)",
                        },
                    },
                },
            ),
            Tool(
                name="graph_query",
                description=(
                    "Query the knowledge graph for an entity's"
                    " connections, neighbors, and relationships."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Entity name to look up",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Neighborhood depth (default 1)",
                            "default": 1,
                        },
                    },
                    "required": ["entity"],
                },
            ),
            Tool(
                name="graph_path",
                description=("Find the shortest path between two entities in the knowledge graph."),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name",
                        },
                    },
                    "required": ["source", "target"],
                },
            ),
            Tool(
                name="capture",
                description="Quick-capture a note to the vault inbox.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Note content to capture",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title (auto-generated if omitted)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags",
                        },
                    },
                    "required": ["content"],
                },
            ),
        ]

    @server.call_tool()  # type: ignore[misc,untyped-decorator,unused-ignore]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = _dispatch_tool(
                name,
                arguments,
                vault_path,
                store,
                graph,
                parser,
            )
            text = json.dumps(result, indent=2, default=str)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.exception("Tool %s failed", name)
            err = json.dumps({"error": str(e)})
            return [TextContent(type="text", text=err)]

    return server


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    vault_path: Path,
    store: VaultStore,
    graph: KnowledgeGraph,
    parser: VaultParser,
) -> dict[str, Any]:
    """Route tool calls to the appropriate handler."""

    if name == "vault_search":
        where = {}
        if args.get("note_type"):
            where["note_type"] = args["note_type"]
        results = store.search(
            args["query"],
            n_results=args.get("n_results", 5),
            where=where or None,
        )
        return {"results": results, "count": len(results)}

    elif name == "vault_read":
        try:
            filepath = validate_vault_path(args["path"], vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        if not filepath.exists():
            return {"error": f"Note not found: {args['path']}"}
        content = filepath.read_text(encoding="utf-8")
        return {"path": args["path"], "content": content}

    elif name == "vault_write":
        try:
            filepath = validate_vault_path(args["path"], vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(args["content"], encoding="utf-8")
        # Re-index the note
        try:
            note = parser.parse_file(filepath)
            store.index_single_note(note, parser)
        except Exception as e:
            logger.warning("Failed to re-index after write: %s", e)
        return {"status": "ok", "path": args["path"]}

    elif name == "vault_list":
        try:
            folder = validate_vault_path(args.get("folder", ""), vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        if not folder.exists():
            return {"error": f"Folder not found: {args.get('folder', '')}"}
        notes = []
        for md in folder.rglob("*.md"):
            rel = md.relative_to(vault_path)
            notes.append(str(rel))
        return {"notes": sorted(notes), "count": len(notes)}

    elif name == "graph_query":
        result = graph.get_neighbors(args["entity"], depth=args.get("depth", 1))
        return result

    elif name == "graph_path":
        path = graph.find_path(args["source"], args["target"])
        if path:
            return {"path": path, "length": len(path) - 1}
        return {"path": None, "message": "No path found between entities"}

    elif name == "capture":
        from datetime import datetime

        now = datetime.now()
        title = args.get("title", now.strftime("Capture %Y%m%d-%H%M%S"))
        tags = args.get("tags", ["capture"])
        slug = title.lower().replace(" ", "-")[:60]
        filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
        content = (
            f"---\ntype: fleeting\ntags: [{', '.join(tags)}]\n"
            f"created: {now.strftime('%Y-%m-%d %H:%M')}\nsource: mcp\nstatus: active\n---\n\n"
            f"# {title}\n\n{args['content']}\n"
        )
        filepath = vault_path / "00-inbox" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return {"status": "ok", "path": f"00-inbox/{filename}"}

    else:
        return {"error": f"Unknown tool: {name}"}
