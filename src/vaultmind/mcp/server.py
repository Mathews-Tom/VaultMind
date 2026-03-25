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

from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError
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
    duplicate_detector: object | None = None,
    note_suggester: object | None = None,
    enforcer: ProfileEnforcer | None = None,
    audit_logger: AuditLogger | None = None,
    # New introspection dependencies
    episode_store: object | None = None,
    procedural_memory: object | None = None,
    evolution_detector: object | None = None,
    preference_store: object | None = None,
    retry_executor: object | None = None,
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
                name="find_duplicates",
                description=(
                    "Find semantically similar or duplicate notes for a given note."
                    " Returns matches classified as 'duplicate' (≥92% similar)"
                    " or 'merge' (80-92% similar)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_path": {
                            "type": "string",
                            "description": "Note path relative to vault root",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matches (default 10)",
                            "default": 10,
                        },
                    },
                    "required": ["note_path"],
                },
            ),
            Tool(
                name="suggest_links",
                description=(
                    "Find notes that should be linked to a given note."
                    " Uses composite scoring: semantic similarity,"
                    " shared graph entities, and graph distance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_path": {
                            "type": "string",
                            "description": "Note path relative to vault root",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of suggestions (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["note_path"],
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
            Tool(
                name="capture_note",
                description=(
                    "Capture a structured note to the vault with full metadata control."
                    " Supports note type, tags, and custom target folder."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Note content (markdown)",
                        },
                        "title": {
                            "type": "string",
                            "description": "Note title (auto-generated from content if omitted)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for the note",
                        },
                        "note_type": {
                            "type": "string",
                            "enum": ["fleeting", "literature", "permanent", "project", "concept"],
                            "default": "fleeting",
                            "description": "Note classification type",
                        },
                        "folder": {
                            "type": "string",
                            "default": "00-inbox",
                            "description": "Target folder relative to vault root",
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="vault_stats",
                description=(
                    "Vault health metrics: note counts by type/folder, graph size, index coverage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="episode_query",
                description="Search episodic memory for past decisions and outcomes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Filter episodes by entity name (optional)",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "resolved"],
                            "description": "Filter by status (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10)",
                            "default": 10,
                        },
                    },
                },
            ),
            Tool(
                name="workflow_suggest",
                description="Find matching procedural workflow for a given context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to match against workflow trigger patterns",
                        },
                    },
                    "required": ["context"],
                },
            ),
            Tool(
                name="graph_evolution",
                description=(
                    "Belief evolution signals: confidence drift, relationship shifts, stale claims."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_severity": {
                            "type": "number",
                            "description": "Minimum severity threshold (0.0-1.0, default 0.0)",
                            "default": 0.0,
                        },
                    },
                },
            ),
            Tool(
                name="recent_activity",
                description="Recent vault activity: notes created/modified in the last N days.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Lookback period in days (default 7)",
                            "default": 7,
                        },
                    },
                },
            ),
        ]

    @server.call_tool()  # type: ignore[misc,untyped-decorator,unused-ignore]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        import time as _time

        _start = _time.perf_counter()
        try:
            # Profile enforcement
            if enforcer is not None:
                enforcer.check_tool(name)
                # Check write permission for write operations
                if name in ("vault_write", "capture", "capture_note"):
                    enforcer.check_write()
                # Check path scope for path-based operations
                if name in ("vault_read", "vault_write", "vault_list") and "path" in arguments:
                    from pathlib import Path as PathCls

                    enforcer.check_path(PathCls(arguments["path"]))
                if name == "vault_write" and "content" in arguments:
                    enforcer.check_size(arguments["content"])

            dispatch_kwargs: dict[str, Any] = {
                "vault_path": vault_path,
                "store": store,
                "graph": graph,
                "parser": parser,
                "duplicate_detector": duplicate_detector,
                "note_suggester": note_suggester,
                "episode_store": episode_store,
                "procedural_memory": procedural_memory,
                "evolution_detector": evolution_detector,
                "preference_store": preference_store,
            }

            if retry_executor is not None:
                from vaultmind.mcp.retry import ToolRetryExecutor

                assert isinstance(retry_executor, ToolRetryExecutor)
                result = retry_executor.execute(name, arguments, _dispatch_tool, dispatch_kwargs)
            else:
                result = _dispatch_tool(name, arguments, **dispatch_kwargs)
            text = json.dumps(result, indent=2, default=str)
            _duration_ms = int((_time.perf_counter() - _start) * 1000)

            # Build change detail for audit
            change_detail = _build_change_detail(name, arguments, result)
            output_summary = _build_output_summary(name, result)

            # Audit log success
            if audit_logger is not None and enforcer is not None:
                audit_logger.log(
                    enforcer.policy.name,
                    name,
                    arguments,
                    "OK",
                    duration_ms=_duration_ms,
                    change_detail=change_detail,
                    output_summary=output_summary,
                )

            return [TextContent(type="text", text=text)]
        except ProfileError as e:
            _duration_ms = int((_time.perf_counter() - _start) * 1000)
            # Audit log denial
            if audit_logger is not None and enforcer is not None:
                audit_logger.log(
                    enforcer.policy.name,
                    name,
                    arguments,
                    "DENIED",
                    reason=str(e),
                    duration_ms=_duration_ms,
                )
            err = json.dumps({"error": str(e)})
            return [TextContent(type="text", text=err)]
        except Exception as e:
            _duration_ms = int((_time.perf_counter() - _start) * 1000)
            logger.exception("Tool %s failed", name)
            if audit_logger is not None and enforcer is not None:
                audit_logger.log(
                    enforcer.policy.name,
                    name,
                    arguments,
                    "ERROR",
                    reason=str(e),
                    duration_ms=_duration_ms,
                )
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
    *,
    duplicate_detector: object | None = None,
    note_suggester: object | None = None,
    episode_store: object | None = None,
    procedural_memory: object | None = None,
    evolution_detector: object | None = None,
    preference_store: object | None = None,
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

    elif name == "find_duplicates":
        if duplicate_detector is None:
            return {"error": "Duplicate detection is not configured"}
        try:
            filepath = validate_vault_path(args["note_path"], vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        if not filepath.exists():
            return {"error": f"Note not found: {args['note_path']}"}
        note = parser.parse_file(filepath)
        from vaultmind.indexer.duplicate_detector import DuplicateDetector

        assert isinstance(duplicate_detector, DuplicateDetector)
        matches = duplicate_detector.find_duplicates(note, max_results=args.get("max_results", 10))
        return {
            "note_path": args["note_path"],
            "matches": [
                {
                    "path": m.match_path,
                    "title": m.match_title,
                    "similarity": m.similarity,
                    "type": m.match_type.value,
                }
                for m in matches
            ],
            "count": len(matches),
        }

    elif name == "suggest_links":
        if note_suggester is None:
            return {"error": "Note suggestions are not configured"}
        try:
            filepath = validate_vault_path(args["note_path"], vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        if not filepath.exists():
            return {"error": f"Note not found: {args['note_path']}"}
        note = parser.parse_file(filepath)
        from vaultmind.indexer.note_suggester import NoteSuggester

        assert isinstance(note_suggester, NoteSuggester)
        suggestions = note_suggester.suggest_links(note, max_results=args.get("max_results", 5))
        return {
            "note_path": args["note_path"],
            "suggestions": [
                {
                    "path": s.target_path,
                    "title": s.target_title,
                    "similarity": s.similarity,
                    "shared_entities": s.shared_entities,
                    "graph_distance": s.graph_distance,
                    "composite_score": s.composite_score,
                }
                for s in suggestions
            ],
            "count": len(suggestions),
        }

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

    elif name == "capture_note":
        import re
        from datetime import datetime

        now = datetime.now()
        raw_content: str = args["content"]

        # Auto-generate title from first non-empty line if not provided
        if args.get("title"):
            title = args["title"]
        else:
            first_line = next(
                (
                    line.strip().lstrip("#").strip()
                    for line in raw_content.splitlines()
                    if line.strip()
                ),
                now.strftime("Capture %Y%m%d-%H%M%S"),
            )
            title = first_line[:50]

        note_type = args.get("note_type", "fleeting")
        note_tags: list[str] = args.get("tags") or []
        target_folder: str = args.get("folder") or "00-inbox"

        # Build slug from title
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        filename = f"{slug}-{timestamp}.md"

        tags_yaml = "[" + ", ".join(note_tags) + "]" if note_tags else "[]"
        file_content = (
            f"---\n"
            f"title: {title}\n"
            f"type: {note_type}\n"
            f"tags: {tags_yaml}\n"
            f"created: {now.strftime('%Y-%m-%d %H:%M')}\n"
            f"source: mcp\n"
            f"status: active\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{raw_content}\n"
        )

        filepath = vault_path / target_folder / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(file_content, encoding="utf-8")

        # Parse and index the new note
        try:
            note = parser.parse_file(filepath)
            store.index_single_note(note, parser)
        except Exception as exc:
            logger.warning("Failed to index capture_note after write: %s", exc)

        rel_path = f"{target_folder}/{filename}"
        return {"status": "ok", "path": rel_path, "title": title}

    elif name == "vault_stats":
        by_type: dict[str, int] = {}
        by_folder: dict[str, int] = {}
        total = 0
        for md in vault_path.rglob("*.md"):
            rel = md.relative_to(vault_path)
            parts = rel.parts
            # Skip excluded folders
            if any(p.startswith(".") for p in parts):
                continue
            total += 1
            folder_name = str(parts[0]) if len(parts) > 1 else "(root)"
            by_folder[folder_name] = by_folder.get(folder_name, 0) + 1
            # Parse type from frontmatter (quick scan)
            try:
                text = md.read_text(encoding="utf-8")[:500]
                if text.startswith("---"):
                    for line in text.split("\n")[1:20]:
                        if line.startswith("type:"):
                            note_type = line.split(":", 1)[1].strip()
                            by_type[note_type] = by_type.get(note_type, 0) + 1
                            break
                        if line == "---":
                            break
            except Exception:
                pass
        graph_stats = graph.stats
        graph_info = {"entities": graph_stats["nodes"], "edges": graph_stats["edges"]}
        return {
            "total_notes": total,
            "by_type": by_type,
            "by_folder": by_folder,
            "graph": graph_info,
        }

    elif name == "episode_query":
        if episode_store is None:
            return {"error": "Episodic memory is not configured"}
        from vaultmind.memory.store import EpisodeStore

        assert isinstance(episode_store, EpisodeStore)
        limit = args.get("limit", 10)
        entity = args.get("entity")
        status = args.get("status")

        if entity:
            episodes = episode_store.search_by_entity(entity, limit=limit)
        elif status == "pending":
            episodes = episode_store.query_pending(limit=limit)
        elif status == "resolved":
            episodes = episode_store.query_resolved(limit=limit)
        else:
            episodes = episode_store.query_resolved(limit=limit)

        return {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "decision": ep.decision,
                    "context": ep.context,
                    "outcome": ep.outcome,
                    "status": ep.outcome_status.value,
                    "lessons": ep.lessons,
                    "entities": ep.entities,
                    "created": ep.created.isoformat() if ep.created else None,
                }
                for ep in episodes
            ],
            "count": len(episodes),
        }

    elif name == "workflow_suggest":
        if procedural_memory is None:
            return {"error": "Procedural memory is not configured"}
        from vaultmind.memory.procedural import ProceduralMemory

        assert isinstance(procedural_memory, ProceduralMemory)
        workflow = procedural_memory.suggest_workflow(args["context"])
        if workflow is None:
            return {"workflow": None, "message": "No matching workflow found"}
        return {
            "workflow": {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": workflow.steps,
                "trigger_pattern": workflow.trigger_pattern,
                "success_rate": workflow.success_rate,
                "usage_count": workflow.usage_count,
            }
        }

    elif name == "graph_evolution":
        if evolution_detector is None:
            return {"error": "Evolution detection is not configured"}
        from vaultmind.graph.evolution import EvolutionDetector

        assert isinstance(evolution_detector, EvolutionDetector)
        signals = evolution_detector.scan()
        min_sev = args.get("min_severity", 0.0)
        filtered = [s for s in signals if s.severity >= min_sev]
        return {
            "signals": [
                {
                    "evolution_id": s.evolution_id,
                    "entity_a": s.entity_a,
                    "entity_b": s.entity_b,
                    "signal_type": s.signal_type,
                    "detail": s.detail,
                    "severity": s.severity,
                    "source_notes": s.source_notes,
                }
                for s in filtered
            ],
            "count": len(filtered),
        }

    elif name == "recent_activity":
        from datetime import datetime

        days = args.get("days", 7)
        cutoff = datetime.now().timestamp() - (days * 86400)
        created: list[str] = []
        modified: list[str] = []
        for md in vault_path.rglob("*.md"):
            rel_path_parts = md.relative_to(vault_path).parts
            if any(p.startswith(".") for p in rel_path_parts):
                continue
            rel_str = str(md.relative_to(vault_path))
            stat = md.stat()
            # st_birthtime is macOS-only; fall back to st_ctime on Linux
            birth = getattr(stat, "st_birthtime", stat.st_ctime)
            if birth >= cutoff:
                created.append(rel_str)
            elif stat.st_mtime >= cutoff:
                modified.append(rel_str)
        return {
            "days": days,
            "created": sorted(created),
            "modified": sorted(modified),
            "created_count": len(created),
            "modified_count": len(modified),
        }

    else:
        return {"error": f"Unknown tool: {name}"}


def _build_change_detail(
    name: str, args: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any] | None:
    """Build tool-specific change detail for audit logging."""
    if name == "vault_write":
        return {
            "type": "vault_write",
            "note_path": args.get("path", ""),
            "size_bytes": len(args.get("content", "").encode()),
            "was_new": result.get("status") == "ok",
        }
    if name in ("capture", "capture_note"):
        return {
            "type": name,
            "note_path": result.get("path", ""),
            "title": result.get("title", ""),
        }
    if name == "vault_search":
        return {
            "type": "vault_search",
            "result_count": result.get("count", 0),
        }
    if name == "vault_read":
        return {
            "type": "vault_read",
            "note_path": args.get("path", ""),
        }
    if name in ("graph_query", "graph_path"):
        return {
            "type": name,
            "entity": args.get("entity", args.get("source", "")),
        }
    if name == "find_duplicates":
        return {
            "type": "find_duplicates",
            "note_path": args.get("note_path", ""),
            "matches_found": result.get("count", 0),
        }
    if name == "suggest_links":
        return {
            "type": "suggest_links",
            "note_path": args.get("note_path", ""),
            "suggestions_count": result.get("count", 0),
        }
    return None


def _build_output_summary(name: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Build a compact output summary for audit logging."""
    if name == "vault_search":
        return {"count": result.get("count", 0)}
    if name in ("vault_write", "capture", "capture_note"):
        return {"status": result.get("status", ""), "path": result.get("path", "")}
    if name == "vault_stats":
        return {"total_notes": result.get("total_notes", 0)}
    if name in ("find_duplicates", "suggest_links"):
        return {"count": result.get("count", 0)}
    if name in ("episode_query", "graph_evolution", "recent_activity"):
        return {"count": result.get("count", 0)}
    return None
