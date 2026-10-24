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
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError
from vaultmind.vault.security import (
    PathResolutionError,
    PathTraversalError,
    validate_ordinary_write_path,
    validate_vault_path,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vaultmind.config import Settings
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.models import Note
    from vaultmind.vault.parser import VaultParser

logger = logging.getLogger(__name__)
_QUERY_PUBLIC_ERRORS = frozenset(
    {
        "Generated wiki queries require a configured LLM client factory",
        "Selected wiki evidence is outside the authorized scope",
    }
)


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
    *,
    settings: Settings,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> Any:  # Returns mcp.server.Server (optional dep)
    """Create and configure the MCP server with vault tools.

    Returns an MCP Server instance ready to run.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("vaultmind")

    # MCP 1.26 leaves this factory untyped; its decorator preserves this callback.
    list_tools_factory = cast(
        "Callable[[], Callable[[Callable[[], Awaitable[list[Tool]]]], "
        "Callable[[], Awaitable[list[Tool]]]]]",
        server.list_tools,
    )

    @list_tools_factory()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="knowledge_query",
                description="Answer from approved current wiki evidence, with complete provenance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "pages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 8,
                            "uniqueItems": True,
                        },
                        "evidence_only": {"type": "boolean", "default": False},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="vault_search",
                description=(
                    "Discover current unapproved sources in the Obsidian vault."
                    " Source-only results are not approved wiki answers."
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
                name="read_frontmatter",
                description=(
                    "Peek a note's parsed frontmatter (type, tags, dates, authority)"
                    " without fetching the note body. Cheaper than vault_read for"
                    " metadata-only lookups."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Note path relative to vault root",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_folder_index",
                description=(
                    "List a folder's structured index: note titles and one-line"
                    " descriptions, without fetching full note bodies."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": ("Folder path relative to vault root (default: root)"),
                            "default": "",
                        },
                    },
                },
            ),
            Tool(
                name="follow_links",
                description=(
                    "Follow a note's wikilink neighborhood: outgoing [[links]] and"
                    " incoming backlinks from other notes, without fetching note bodies."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Note path relative to vault root",
                        },
                    },
                    "required": ["path"],
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
                description=(
                    "Preserve exact input as an unapproved source and project it to the inbox."
                ),
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
                        "idempotency_token": {
                            "type": "string",
                            "maxLength": 128,
                            "pattern": "^[A-Za-z0-9._:-]+$",
                            "description": (
                                "Optional caller event identity; omitted means a fresh source."
                            ),
                        },
                        "do_not_retain": {
                            "type": "boolean",
                            "const": False,
                            "description": (
                                "Capture requires retaining original input; true is rejected."
                            ),
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="capture_note",
                description=(
                    "Preserve exact input as an unapproved source and create a linked note."
                    " Returns separate preservation, projection, and indexing dispositions."
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
                        "idempotency_token": {
                            "type": "string",
                            "maxLength": 128,
                            "pattern": "^[A-Za-z0-9._:-]+$",
                            "description": (
                                "Optional caller event identity; omitted means a fresh source."
                            ),
                        },
                        "do_not_retain": {
                            "type": "boolean",
                            "const": False,
                            "description": (
                                "Capture requires retaining original input; true is rejected."
                            ),
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

    @server.call_tool(validate_input=False)  # type: ignore[misc,untyped-decorator,unused-ignore]
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
                if (
                    name
                    in (
                        "vault_read",
                        "vault_write",
                        "vault_list",
                        "read_frontmatter",
                        "follow_links",
                    )
                    and "path" in arguments
                ):
                    from pathlib import Path as PathCls

                    enforcer.check_path(PathCls(arguments["path"]))
                # list_folder_index scope is enforced per result in dispatch: a
                # restricted profile's omitted/empty/root folder is a virtual
                # aggregate of allowed roots, and an explicit forbidden folder is denied.
                if name == "vault_write" and "content" in arguments:
                    enforcer.check_size(arguments["content"])

            dispatch_kwargs: dict[str, Any] = {
                "vault_path": vault_path,
                "store": store,
                "graph": graph,
                "parser": parser,
                "enforcer": enforcer,
                "duplicate_detector": duplicate_detector,
                "note_suggester": note_suggester,
                "episode_store": episode_store,
                "procedural_memory": procedural_memory,
                "evolution_detector": evolution_detector,
                "preference_store": preference_store,
                "settings": settings,
                "llm_factory": llm_factory,
            }

            import asyncio

            if retry_executor is not None and name not in {
                "capture",
                "capture_note",
                "vault_write",
                "knowledge_query",
            }:
                from vaultmind.mcp.retry import ToolRetryExecutor

                assert isinstance(retry_executor, ToolRetryExecutor)
                if name == "vault_search":
                    result = await asyncio.to_thread(
                        retry_executor.execute, name, arguments, _dispatch_tool, dispatch_kwargs
                    )
                else:
                    result = retry_executor.execute(
                        name, arguments, _dispatch_tool, dispatch_kwargs
                    )
            elif name in {
                "capture",
                "capture_note",
                "vault_write",
                "vault_search",
                "knowledge_query",
            }:
                result = await asyncio.to_thread(_dispatch_tool, name, arguments, **dispatch_kwargs)
            else:
                # Existing optional producers own thread-affine SQLite state.
                # Offload the migrated source operations, not unrelated stores.
                result = _dispatch_tool(name, arguments, **dispatch_kwargs)
            text = json.dumps(result, indent=2, default=str)
            if name == "knowledge_query":
                _require_enforcer(name, enforcer).check_size(text)
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
                    "PARTIAL" if result.get("status") == "partial" else "OK",
                    duration_ms=_duration_ms,
                    change_detail=change_detail,
                    output_summary=output_summary,
                )

            return [TextContent(type="text", text=text)]
        except ProfileError as e:
            _duration_ms = int((_time.perf_counter() - _start) * 1000)
            # Audit log denial
            if audit_logger is not None and (enforcer is not None or name == "knowledge_query"):
                audit_logger.log(
                    enforcer.policy.name if enforcer is not None else "unavailable",
                    name,
                    arguments,
                    "DENIED",
                    reason="Knowledge query denied" if name == "knowledge_query" else str(e),
                    duration_ms=_duration_ms,
                )
            message = "Knowledge query denied" if name == "knowledge_query" else str(e)
            err = json.dumps({"error": message})
            return [TextContent(type="text", text=err)]
        except Exception as e:
            _duration_ms = int((_time.perf_counter() - _start) * 1000)
            if name in {"capture", "capture_note", "vault_write", "knowledge_query"}:
                logger.error("Body-bearing MCP tool failed (%s)", type(e).__name__)
            else:
                logger.exception("Tool %s failed", name)
            if audit_logger is not None and (enforcer is not None or name == "knowledge_query"):
                audit_logger.log(
                    enforcer.policy.name if enforcer is not None else "unavailable",
                    name,
                    arguments,
                    "ERROR",
                    reason="Knowledge query failed" if name == "knowledge_query" else str(e),
                    duration_ms=_duration_ms,
                )
            query_message = "Knowledge query failed"
            # Only fixed, non-disclosing core dispositions may cross this boundary.
            if name == "knowledge_query" and str(e) in _QUERY_PUBLIC_ERRORS:
                query_message = str(e)
            message = (
                query_message
                if name == "knowledge_query"
                else f"MCP input operation failed ({type(e).__name__})"
                if name in {"capture", "capture_note", "vault_write"}
                else str(e)
            )
            err = json.dumps({"error": message})
            return [TextContent(type="text", text=err)]

    # SDK schema errors echo rejected values before our callback and audit.
    # Route only the body-bearing query through our strict private validation;
    # preserve the SDK's normal validation/result handling for every other tool.
    from mcp.types import CallToolRequest

    query_handler = server.request_handlers[CallToolRequest]
    server.call_tool()(call_tool)
    validated_handler = server.request_handlers[CallToolRequest]

    async def route_call(request: Any) -> Any:
        handler = query_handler if request.params.name == "knowledge_query" else validated_handler
        return await handler(request)

    server.request_handlers[CallToolRequest] = route_call

    return server


def _one_line_description(note: Note, max_len: int = 160) -> str:
    """Derive a one-line description from frontmatter, else the note's first body line."""
    fm_description = note.frontmatter.get("description")
    if isinstance(fm_description, str) and fm_description.strip():
        return fm_description.strip()[:max_len]
    for line in note.body_without_frontmatter().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:max_len]
    return ""


def _require_enforcer(name: str, enforcer: ProfileEnforcer | None) -> ProfileEnforcer:
    """Return the enforcer, failing closed when a scoped tool has none."""
    if enforcer is None:
        raise ProfileError(f"Tool '{name}' requires profile enforcement but none was supplied")
    return enforcer


def _require_global_provenance(name: str, enforcer: ProfileEnforcer | None) -> None:
    """Deny producers without complete provenance before inspecting global data.

    Graph source lists can omit unknown contributions to merged attributes.
    Even a nonempty all-allowed list cannot authorize those records or aggregates.
    """
    scope = _require_enforcer(name, enforcer)
    if not scope.policy.allows_all_folders:
        raise ProfileError(
            f"Tool '{name}' is unavailable for folder-restricted profiles: "
            "complete note provenance is unavailable"
        )


def _result_note_paths(scope: ProfileEnforcer) -> list[Path]:
    """Canonicalize verified notes before counts; aliases never inflate aggregates."""
    paths: set[Path] = set()
    for key in scope.note_paths():
        path = scope.note_path(key)
        if path is not None and not any(part.startswith(".") for part in path.parts):
            paths.add(path)
    return sorted(paths)


def _require_settings(settings: Settings | None, vault_path: Path) -> Settings:
    """Never infer capture/search configuration from a root or parser."""
    from vaultmind.config import Settings

    if not isinstance(settings, Settings):
        raise ProfileError("Source operations require the selected Settings instance")
    if settings.vault.path.resolve() != vault_path.resolve():
        raise ProfileError("Source settings do not match the MCP vault root")
    return settings


def _capture_source(
    name: str,
    args: dict[str, Any],
    vault_path: Path,
    store: VaultStore,
    parser: VaultParser,
    scope: ProfileEnforcer,
    settings: Settings | None,
) -> dict[str, Any]:
    """Preserve first, then independently receipt projection and index effects."""
    import hashlib
    import re

    import yaml

    from vaultmind.knowledge.intake import SourceIntake

    scope.check_tool(name)
    scope.check_write()
    # Deny narrow profiles before inspecting a body or creating intake state.
    for relative in ("_wiki", "_wiki/wiki.lock", "_wiki/sources"):
        if not scope.allows_path(relative):
            raise ProfileError("Capture preservation is outside the active profile scope")
    selected = _require_settings(settings, vault_path)
    if args.get("do_not_retain", False) is not False:
        raise ProfileError("Capture requires explicit original retention")
    raw = args.get("content")
    if not isinstance(raw, str):
        raise ProfileError("Capture content must be a string")
    scope.check_size(raw)
    title = args.get("title")
    if title is None:
        title = next(
            (line.strip().lstrip("#").strip() for line in raw.splitlines() if line.strip()),
            "Capture",
        )[:50]
    if not isinstance(title, str):
        raise ProfileError("Capture title must be a string")
    tags = args.get("tags", ["capture"] if name == "capture" else [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ProfileError("Capture tags must be a string list")
    note_type = args.get("note_type", "fleeting")
    if note_type not in {"fleeting", "literature", "permanent", "project", "concept"}:
        raise ProfileError("Unsupported capture note type")
    folder = args.get("folder", "00-inbox") if name == "capture_note" else "00-inbox"
    if not isinstance(folder, str) or not folder:
        raise ProfileError("Capture folder must be a nonempty string")
    token = args.get("idempotency_token")
    if token is not None and (
        not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", token) is None
    ):
        raise ProfileError("Idempotency token must be 1-128 safe ASCII characters")
    projection_path: Path | None = None
    projection_text = ""
    relative = ""

    def preflight(source_id: str) -> None:
        nonlocal projection_path, projection_text, relative
        relative = (Path(folder) / f"{source_id}.md").as_posix()
        if not scope.allows_path(relative):
            raise ProfileError("Capture projection is outside the active profile scope")
        try:
            projection_path = validate_ordinary_write_path(relative, vault_path)
        except (PathTraversalError, PathResolutionError):
            raise ProfileError("Capture projection destination is not allowed") from None
        metadata = dict(
            title=title,
            type=note_type,
            tags=tags,
            source="mcp",
            status="active",
            source_id=source_id,
            source_path=f"_wiki/sources/{source_id}.md",
            approved=False,
        )
        projection_text = "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + raw
        scope.check_size(projection_text)

    with SourceIntake(selected, allows_path=scope.allows_path) as intake:
        capture = intake.preserve_text(
            raw,
            original_name="capture.md",
            origin="mcp",
            adapter=f"mcp:{scope.policy.name}:{name}",
            event_id=token,
            revision=hashlib.sha256(
                json.dumps(
                    dict(title=title, tags=tags, note_type=note_type, folder=folder), sort_keys=True
                ).encode("utf-8")
            ).hexdigest(),
            preflight=preflight,
        )
        assert projection_path is not None
        index_path = projection_path

        def project() -> str:
            path = validate_ordinary_write_path(relative, vault_path)
            if not scope.allows_path(relative):
                raise ProfileError("Capture projection is outside the active profile scope")
            if path.exists():
                raise ProfileError("Capture projection already exists")
            # Use the existing atomic writer; never introduce another source writer.
            intake.workspace._atomic_bytes(path, projection_text.encode("utf-8"))
            return relative

        projection = capture.run_stage("projection", project)
        index = None
        if projection.status == "succeeded":
            index = capture.run_stage(
                "index", lambda: store.index_single_note(parser.parse_file(index_path), parser)
            )
        original = capture.as_dict()
        return dict(
            status="ok" if index is not None and index.status == "succeeded" else "partial",
            source_id=capture.source_id,
            path=relative,
            title=title,
            approved=False,
            preserved=True,
            original=original,
            projection=projection.as_dict(),
            index=index.as_dict() if index is not None else {"stage": "index", "status": "skipped"},
        )


def _scoped_markdown_files(
    resolved_root: Path, enforcer: ProfileEnforcer, roots: list[Path]
) -> list[Path]:
    """Return sorted vault-relative markdown paths under ``roots`` allowed by the enforcer.

    Every candidate (symlinks included) is checked with ``ProfileEnforcer.check_path``
    before any caller parses or serializes it.
    """
    allowed: set[Path] = set()
    for root in roots:
        for md in root.rglob("*.md"):
            try:
                rel = md.relative_to(resolved_root)
                enforcer.check_path(rel)
            except (ValueError, ProfileError):
                continue
            if enforcer.note_path(str(rel)) is None:
                continue
            allowed.add(rel)
    return sorted(allowed)


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    vault_path: Path,
    store: VaultStore,
    graph: KnowledgeGraph,
    parser: VaultParser,
    *,
    enforcer: ProfileEnforcer | None,
    duplicate_detector: object | None = None,
    note_suggester: object | None = None,
    episode_store: object | None = None,
    procedural_memory: object | None = None,
    evolution_detector: object | None = None,
    preference_store: object | None = None,
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> dict[str, Any]:
    """Route tool calls to the appropriate handler."""

    if name == "knowledge_query":
        scope = _require_enforcer(name, enforcer)
        scope.check_tool(name)
        selected = _require_settings(settings, vault_path)
        if scope.vault_root.resolve() != vault_path.resolve():
            raise ProfileError("Knowledge query enforcement root does not match")
        # Measure the full UTF-8 request, not just the question or character count.
        scope.check_size(json.dumps(args, ensure_ascii=False, allow_nan=False))
        if (
            set(args) - {"question", "pages", "evidence_only"}
            or "question" not in args
            or ("pages" in args and args["pages"] is None)
            or ("evidence_only" in args and not isinstance(args["evidence_only"], bool))
        ):
            raise ValueError("Invalid knowledge query arguments")
        from vaultmind.knowledge.query import MAX_QUERY_CONTEXT_BYTES, query_knowledge
        from vaultmind.wiki.workspace import WikiError

        limit = scope.policy.max_note_size_bytes
        try:
            result = query_knowledge(
                selected,
                args["question"],
                args.get("pages"),
                evidence_only=args.get("evidence_only", False),
                llm_factory=llm_factory,
                allows_path=scope.allows_path,
                max_context_bytes=min(MAX_QUERY_CONTEXT_BYTES, limit)
                if limit is not None
                else None,
            )
        except WikiError as exc:
            message = str(exc) if str(exc) in _QUERY_PUBLIC_ERRORS else "Knowledge query failed"
            raise WikiError(message) from None
        # Match the outer handler's exact wire representation: no partial results.
        scope.check_size(json.dumps(result, indent=2, default=str))
        return result
    elif name == "vault_search":
        scope = _require_enforcer(name, enforcer)
        from vaultmind.knowledge.sources import search_sources

        selected = _require_settings(settings, vault_path)
        results = search_sources(
            selected,
            args["query"],
            store=store,
            n_results=args.get("n_results", 5),
            note_type=args.get("note_type"),
            allows_path=scope.allows_path,
        )
        return {
            "status": "source-only",
            "approved": False,
            "results": results,
            "count": len(results),
        }
    elif name == "vault_read":
        scope = _require_enforcer(name, enforcer)
        scope.check_path(Path(args["path"]))
        try:
            filepath = validate_vault_path(args["path"], vault_path)
        except PathTraversalError as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        if not filepath.exists():
            return {"error": f"Note not found: {args['path']}"}
        content = filepath.read_text(encoding="utf-8")
        return {"path": args["path"], "content": content}

    elif name == "vault_write":
        scope = _require_enforcer(name, enforcer)
        scope.check_write()
        scope.check_path(Path(args["path"]))
        scope.check_size(args["content"])
        try:
            filepath = validate_ordinary_write_path(args["path"], vault_path)
        except (PathTraversalError, PathResolutionError):
            raise ProfileError("Ordinary write destination is not allowed") from None
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(args["content"], encoding="utf-8")
        # Re-index the note
        try:
            note = parser.parse_file(filepath)
            store.index_single_note(note, parser)
        except Exception as e:
            logger.warning("Failed to re-index after write (%s)", type(e).__name__)
        return {"status": "ok", "path": args["path"]}

    elif name == "vault_list":
        scope = _require_enforcer(name, enforcer)
        try:
            folder = validate_vault_path(args.get("folder", ""), vault_path)
        except (PathTraversalError, PathResolutionError) as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        resolved_root = vault_path.resolve()
        if not scope.policy.allows_all_folders and folder == resolved_root:
            roots = [
                resolved_root / allowed for allowed in dict.fromkeys(scope.policy.folder_scope)
            ]
        else:
            scope.check_path(Path(args.get("folder", "")))
            if not folder.exists():
                return {"error": f"Folder not found: {args.get('folder', '')}"}
            roots = [folder]
        notes = [str(rel) for rel in _scoped_markdown_files(resolved_root, scope, roots)]
        return {"notes": notes, "count": len(notes)}

    elif name == "read_frontmatter":
        scope = _require_enforcer(name, enforcer)
        try:
            filepath = validate_vault_path(args["path"], vault_path)
        except (PathTraversalError, PathResolutionError) as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        scope.check_path(Path(args["path"]))
        if not filepath.exists():
            return {"error": f"Note not found: {args['path']}"}
        note = parser.parse_file(filepath)
        return {
            "path": args["path"],
            "title": note.title,
            "note_type": note.note_type.value,
            "tags": note.tags,
            "authority": note.authority,
            "status": note.status,
            "source": note.source,
            "created": note.created.isoformat(),
            "modified": note.modified.isoformat(),
            "frontmatter": note.frontmatter,
        }

    elif name == "list_folder_index":
        scope = _require_enforcer(name, enforcer)
        try:
            folder = validate_vault_path(args.get("folder", ""), vault_path)
        except (PathTraversalError, PathResolutionError) as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        resolved_root = vault_path.resolve()
        if not scope.policy.allows_all_folders and folder == resolved_root:
            # Omitted/empty/explicit root under a restricted profile is a virtual
            # aggregate of the allowed roots only, never the whole vault.
            roots = [
                resolved_root / allowed for allowed in dict.fromkeys(scope.policy.folder_scope)
            ]
        else:
            scope.check_path(Path(args.get("folder", "")))
            if not folder.exists():
                return {"error": f"Folder not found: {args.get('folder', '')}"}
            roots = [folder]
        entries = []
        for rel in _scoped_markdown_files(resolved_root, scope, roots):
            try:
                note = parser.parse_file(vault_path / rel)
            except Exception as e:
                logger.warning("Failed to parse %s for folder index: %s", rel, e)
                continue
            entries.append(
                {
                    "path": str(rel),
                    "title": note.title,
                    "description": _one_line_description(note),
                    "note_type": note.note_type.value,
                    "tags": note.tags,
                }
            )
        return {"folder": args.get("folder", ""), "notes": entries, "count": len(entries)}

    elif name == "follow_links":
        scope = _require_enforcer(name, enforcer)
        try:
            filepath = validate_vault_path(args["path"], vault_path)
        except (PathTraversalError, PathResolutionError) as e:
            return {"error": f"Path not allowed: {e.user_path}"}
        resolved_root = vault_path.resolve()
        scope.check_path(Path(args["path"]))
        if not filepath.exists():
            return {"error": f"Note not found: {args['path']}"}
        target_note = parser.parse_file(filepath)
        # Candidate notes are pre-filtered by the enforcer before they are parsed,
        # so forbidden notes never contribute titles, paths, links or counts.
        excluded = parser.config.excluded_folders
        roots = (
            [resolved_root]
            if scope.policy.allows_all_folders
            else [resolved_root / allowed for allowed in dict.fromkeys(scope.policy.folder_scope)]
        )
        all_notes: list[Note] = []
        for rel in _scoped_markdown_files(resolved_root, scope, roots):
            if any(part in excluded for part in rel.parts):
                continue
            try:
                all_notes.append(parser.parse_file(vault_path / rel))
            except Exception:
                logger.exception("Failed to parse %s", rel)
        title_to_path = {n.title: str(n.path) for n in all_notes}

        out_links = [
            {"title": link_title, "path": title_to_path.get(link_title)}
            for link_title in target_note.wikilinks
        ]
        if not scope.policy.allows_all_folders:
            # A restricted client cannot tell a private target from a truly
            # unresolved link without disclosing it, so only resolved in-scope
            # links are emitted.
            out_links = [link for link in out_links if link["path"] is not None]
        backlinks = [
            {"title": n.title, "path": str(n.path)}
            for n in all_notes
            if n.path != target_note.path and target_note.title in n.wikilinks
        ]

        return {
            "path": args["path"],
            "title": target_note.title,
            "out_links": out_links,
            "backlinks": backlinks,
            "out_link_count": len(out_links),
            "backlink_count": len(backlinks),
        }

    elif name == "graph_query":
        _require_global_provenance(name, enforcer)
        return graph.get_neighbors(args["entity"], depth=args.get("depth", 1))

    elif name == "graph_path":
        _require_global_provenance(name, enforcer)
        path = graph.find_path(args["source"], args["target"])
        if path:
            return {"path": path, "length": len(path) - 1}
        return {"path": None, "message": "No path found between entities"}

    elif name == "find_duplicates":
        scope = _require_enforcer(name, enforcer)
        scope.check_path(Path(args["note_path"]))
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
        matches = duplicate_detector.find_duplicates(
            note,
            max_results=args.get("max_results", 10),
            search=lambda query, count: scope.search(store, query, n_results=count),
            cache_results=False,
        )
        matches = [m for m in matches if scope.note_path(m.match_path) is not None]
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
        scope = _require_enforcer(name, enforcer)
        scope.check_path(Path(args["note_path"]))
        if not scope.policy.allows_all_folders:
            raise ProfileError(
                "Tool 'suggest_links' is unavailable for folder-restricted profiles: "
                "graph-derived suggestion scores lack scoped note provenance"
            )
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
        suggestions = note_suggester.suggest_links(
            note,
            max_results=args.get("max_results", 5),
            search=lambda query, count: scope.search(store, query, n_results=count),
            cache_results=False,
        )
        suggestions = [s for s in suggestions if scope.note_path(s.target_path) is not None]
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

    elif name in {"capture", "capture_note"}:
        return _capture_source(
            name, args, vault_path, store, parser, _require_enforcer(name, enforcer), settings
        )

    elif name == "vault_stats":
        _require_global_provenance(name, enforcer)
        scope = _require_enforcer(name, enforcer)
        by_type: dict[str, int] = {}
        by_folder: dict[str, int] = {}
        total = 0
        for rel in _result_note_paths(scope):
            md = vault_path.resolve() / rel
            parts = rel.parts
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
        _require_global_provenance(name, enforcer)
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
        _require_global_provenance(name, enforcer)
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
        _require_global_provenance(name, enforcer)
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

        scope = _require_enforcer(name, enforcer)

        days = args.get("days", 7)
        cutoff = datetime.now().timestamp() - (days * 86400)
        created: list[str] = []
        modified: list[str] = []
        for rel in _result_note_paths(scope):
            md = vault_path.resolve() / rel
            rel_str = str(rel)
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
    if name in ("vault_write", "capture", "capture_note"):
        return {"type": name, "status": result.get("status", "")}
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
        return {"status": result.get("status", "")}
    if name == "vault_stats":
        return {"total_notes": result.get("total_notes", 0)}
    if name in ("find_duplicates", "suggest_links"):
        return {"count": result.get("count", 0)}
    if name in ("episode_query", "graph_evolution", "recent_activity"):
        return {"count": result.get("count", 0)}
    return None
