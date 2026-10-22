# MCP Integration Guide

VaultMind exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that lets Claude Desktop, Claude Code, and other MCP-compatible agents interact with your vault programmatically.

## Setup

### Install the MCP extra

```bash
uv sync --extra mcp
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "vaultmind": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/vaultmind",
        "vaultmind",
        "mcp-serve"
      ]
    }
  }
}
```

### Claude Code

Add to `.claude/mcp.json` in your project or `~/.claude/mcp.json` globally:

```json
{
  "mcpServers": {
    "vaultmind": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/vaultmind",
        "vaultmind",
        "mcp-serve"
      ]
    }
  }
}
```

### Standalone server

```bash
uv run vaultmind mcp-serve                    # Default: researcher profile
uv run vaultmind mcp-serve --profile planner  # Planner profile
uv run vaultmind mcp-serve --profile full     # Full access
```

The CLI uses MCP stdio transport, not a listening HTTP port. Stdout contains only JSON-RPC protocol frames; startup diagnostics and logs go to stderr.

## Tools

### `vault_search`

Semantic search over the vault using ChromaDB embeddings.

**Parameters:**

- `query` (string, required) — natural language search query
- `n_results` (int, default 5) — number of results
- `note_type` (string, optional) — filter by note type (`fleeting`, `literature`, `permanent`, `project`, etc.)

**Returns:** Array of matching note chunks with similarity scores, source paths, and content.

Folder-restricted profiles search only verified existing Markdown notes in their allowed folders. The note-path whitelist is applied together with `note_type` before the store selects its top results; forbidden high-ranking chunks cannot crowd out allowed matches. Returned chunks are checked again for valid in-scope note provenance, and `count` includes only returned chunks.

### `vault_read`

Read the full content of a note.

**Parameters:**

- `path` (string, required) — note path relative to vault root (e.g., `02-projects/cairn/architecture.md`)

**Returns:** Full markdown content including frontmatter.

### `vault_write`

Create or overwrite a note. The note is automatically re-indexed in ChromaDB after writing.

**Parameters:**

- `path` (string, required) — note path relative to vault root
- `content` (string, required) — full markdown content including YAML frontmatter

**Returns:** Status confirmation.

### `vault_list`

List notes in a vault folder.

**Parameters:**

- `folder` (string, default `""`) — folder path relative to vault root. Omitted, empty, dot, or vault-root input returns a virtual aggregate of the active profile's allowed roots.
- `tag` (string, optional) — filter by tag

**Returns:** Sorted list of note paths and count.

Explicit forbidden folders are denied, whether or not they exist. Only allowed existing Markdown paths contribute entries or counts; symlinks do not bypass vault containment or folder scope.

### `graph_query`

Query an entity's connections in the knowledge graph.

**Parameters:**

- `entity` (string, required) — entity name to look up
- `depth` (int, default 1) — neighborhood traversal depth

**Returns:** Entity neighbors, relationships, and connection metadata.

### `graph_path`

Find the shortest path between two entities in the knowledge graph.

**Parameters:**

- `source` (string, required) — source entity name
- `target` (string, required) — target entity name

**Returns:** Path (list of entities) and path length.

### `find_duplicates`

Find semantically similar or duplicate notes.

**Parameters:**

- `note_path` (string, required) — note path relative to vault root
- `max_results` (int, default 10) — maximum number of matches

**Returns:** Matches classified into bands:

- **Duplicate** (>= 92% similarity) — nearly identical content
- **Merge candidate** (80-92% similarity) — related content worth consolidating

The source note must be allowed by the active profile. Candidate search is scoped before its result limit, and only verified in-scope matches contribute to the returned count. Restricted profiles can receive positive duplicate matches from their allowed folders.

The stdio server assembles the configured duplicate detector and note suggester when their corresponding features are enabled; disabled or unavailable producers return a not-configured error for otherwise authorized calls. MCP duplicate lookups do not populate the producer's notification/results cache.

### `suggest_links`

Find notes that should be linked to a given note using composite scoring.

**Parameters:**

- `note_path` (string, required) — note path relative to vault root
- `max_results` (int, default 5) — maximum number of suggestions

**Returns:** Suggestions with composite scores based on:

- Semantic similarity (70-80% band, below duplicate threshold)
- Shared graph entities (weight: 0.1 per entity)
- Graph path distance (weight: 0.05)

**Restricted profiles:** This tool is explicitly denied whenever folder scope is restricted, even when the source and proposed target notes are allowed. Shared entities, graph distance, and composite scores do not yet have scoped provenance, so filtering only target paths would not protect private graph information. Wildcard-folder profiles retain the existing composite-scoring behavior.

Authorized MCP suggestion lookups likewise do not populate the producer's notification/results cache. Their candidates and returned target paths still pass through the shared note-result boundary, including for wildcard profiles.

### `capture`

Quick-capture a note to the vault inbox.

**Parameters:**

- `content` (string, required) — note content
- `title` (string, optional) — note title (auto-generated timestamp if omitted)
- `tags` (array of strings, optional) — tags (defaults to `["capture"]`)

**Returns:** Status and path of the created note.

### `capture_note`

Rich note capture with full frontmatter control. Creates the note, writes full YAML frontmatter, and indexes immediately.

**Parameters:**

- `content` (string, required) — note content (markdown)
- `title` (string, optional) — note title (auto-generated from first line of content if omitted)
- `tags` (array of strings, optional) — tags for the note
- `note_type` (string, optional, default `"fleeting"`) — one of `fleeting`, `literature`, `permanent`, `project`, `concept`
- `folder` (string, optional, default `"00-inbox"`) — target folder relative to vault root

**Returns:** Status, path, and title of the created note.

Both capture tools authorize the computed destination and check the complete serialized note's UTF-8 size, including title, tags, and frontmatter, before creating directories, writing files, or indexing. `capture` always targets `00-inbox`; a profile excluding that folder cannot use it. Its title is sanitized into a single filename component, so slashes and traversal text cannot select another directory. `capture_note` can create nested folders only inside the active profile's allowed scope.

### `vault_stats`

Vault health metrics: note counts by type and folder, knowledge graph size.

**Parameters:** None

**Returns:**

| Field         | Type    | Description                                            |
| ------------- | ------- | ------------------------------------------------------ |
| `total_notes` | integer | Total markdown files in vault                          |
| `by_type`     | object  | Note counts grouped by frontmatter `type`              |
| `by_folder`   | object  | Note counts grouped by top-level folder                |
| `graph`       | object  | `{entities, edges}` — knowledge graph node/edge counts |

### `episode_query`

Search episodic memory for past decisions and outcomes.

**Parameters:**

| Parameter | Type    | Required | Description                    |
| --------- | ------- | -------- | ------------------------------ |
| `entity`  | string  | No       | Filter episodes by entity name |
| `status`  | string  | No       | `"pending"` or `"resolved"`    |
| `limit`   | integer | No       | Max results (default 10)       |

**Returns:** `{episodes: [...], count}` — each episode includes `episode_id`, `decision`, `context`, `outcome`, `status`, `lessons`, `entities`, `created`.

### `workflow_suggest`

Find a matching procedural workflow for a given context.

**Parameters:**

| Parameter | Type   | Required | Description                                        |
| --------- | ------ | -------- | -------------------------------------------------- |
| `context` | string | Yes      | Context to match against workflow trigger patterns |

**Returns:** `{workflow: {workflow_id, name, description, steps, trigger_pattern, success_rate, usage_count}}` or `{workflow: null}` if no match.

### `graph_evolution`

Belief evolution signals: confidence drift, relationship shifts, stale claims.

**Parameters:**

| Parameter      | Type   | Required | Description                                      |
| -------------- | ------ | -------- | ------------------------------------------------ |
| `min_severity` | number | No       | Minimum severity threshold 0.0–1.0 (default 0.0) |

**Returns:** `{signals: [...], count}` — each signal includes `evolution_id`, `entity_a`, `entity_b`, `signal_type`, `detail`, `severity`, `source_notes`.

### `recent_activity`

Recent vault activity: notes created or modified in the last N days.

**Parameters:**

| Parameter | Type    | Required | Description                         |
| --------- | ------- | -------- | ----------------------------------- |
| `days`    | integer | No       | Lookback period in days (default 7) |

**Returns:** `{days, created: [...], modified: [...], created_count, modified_count}` — file paths relative to vault root.

### `read_frontmatter`

Peek a note's parsed frontmatter (type, tags, dates, authority) without fetching the note body. Cheaper than `vault_read` for metadata-only lookups. Never returns body text.

**Parameters:**

- `path` (string, required) — note path relative to vault root

**Returns:** `{path, title, note_type, tags, authority, status, source, created, modified, frontmatter}` — `frontmatter` is the raw parsed YAML dict.

### `list_folder_index`

List a folder's structured index: note titles and one-line descriptions, without fetching full note bodies. Recurses into subfolders, like `vault_list`.

**Parameters:**

- `folder` (string, default `""`) — folder path relative to vault root. Omitted, empty, or explicit vault-root input aggregates only the active profile's allowed roots; unrestricted profiles list the whole vault.

**Returns:** `{folder, notes: [{path, title, description, note_type, tags}], count}`. `description` is the frontmatter `description` field if present, else the note's first non-heading body line.

Explicit out-of-scope folders are denied before checking existence. Every candidate is checked against the active profile before parsing its metadata, including resolved symlink targets. `count` includes only returned entries.

### `follow_links`

Follow a note's wikilink neighborhood: outgoing `[[links]]` and incoming backlinks from other notes, without fetching note bodies. Backlinks are resolved from an on-demand reverse index built over the vault's `Note.wikilinks` (title-exact-match) — not the knowledge graph, which only carries note-linkage edges when the opt-in, LLM-driven `graph-build` has already run.

**Parameters:**

- `path` (string, required) — note path relative to vault root

**Returns:** `{path, title, out_links: [{title, path}], backlinks: [{title, path}], out_link_count, backlink_count}`. Restricted profiles return only outgoing targets resolved inside their folder scope and backlinks from allowed notes; unresolved or forbidden titles are omitted, not disguised as `null` paths. An allowed note with a duplicate title remains resolvable even when a forbidden note has the same title. Unrestricted profiles retain `null` paths for genuinely unresolved titles. Counts include only returned links.

## Profiles

Profiles restrict what an agent can do. They control tool access, folder scope, and write permissions.

### `researcher` (default)

Read-only access for research and Q&A tasks.

- **Tools:** `vault_search`, `vault_read`, `vault_list`, `graph_query`, `graph_path`, `vault_stats`, `episode_query`, `workflow_suggest`, `graph_evolution`, `recent_activity`, `read_frontmatter`, `list_folder_index`, `follow_links`
- **Folders:** All
- **Write:** No

### `planner`

Read/write access scoped to project planning.

- **Tools:** All researcher tools + `vault_write`, `capture`, `capture_note`, `vault_stats`, `episode_query`, `workflow_suggest`, `graph_evolution`, `recent_activity`, `read_frontmatter`, `list_folder_index`, `follow_links`
- **Folders:** `02-projects`, `00-inbox`
- **Write:** Yes (max 50KB per note)

### `full`

Unrestricted access — all tools, all folders, write enabled.

- **Tools:** All
- **Folders:** All
- **Write:** Yes
- Requires explicit opt-in via `--profile full`

### Custom profiles

Define custom profiles in `config/default.toml`:

```toml
[mcp.profiles.reviewer]
description = "Read-only access to archive and resources"
allowed_tools = ["vault_search", "vault_read", "vault_list"]
folder_scope = ["04-resources", "05-archive"]
write_enabled = false
```

## Security

### Path traversal protection

Note-backed tools validate vault containment and folder scope, including resolved symlink targets. Wildcard folder access bypasses folder restrictions, never vault containment. Config-defined profiles override built-ins and use the same result boundary.

The shared `ProfileEnforcer` boundary covers `vault_search`, `vault_list`, `vault_read`, `vault_write`, `capture`, `capture_note`, `find_duplicates`, and the traversal tools (`read_frontmatter`, `list_folder_index`, `follow_links`). Direct dispatch of these tools requires an explicit enforcement context. For folder-restricted profiles, search and duplicate candidates are scoped before store truncation. Wildcard profiles preserve the original query filters and valid literal provenance. Both paths defensively verify candidates before returning content, paths, metadata, or counts. Missing, ill-typed, unresolvable, non-Markdown, and forbidden note provenance is omitted rather than treated as permission to disclose a result. Restricted `suggest_links` calls are denied rather than returning unscoped graph-derived scores.

**Derived/global privacy is not enforced:** This is a note-result boundary, not all-tool privacy. Derived/global tools — `vault_stats`, `graph_query`, `graph_path`, `episode_query`, `workflow_suggest`, `graph_evolution`, and `recent_activity` — are not secured by this change and may disclose out-of-scope information. Do not use planner or another folder-restricted profile as a whole-vault privacy sandbox. Denials can echo caller-supplied paths but must not disclose unknown resolved symlink targets.

Symlink loops fail closed with a caller-path-only error, on Python 3.12 and 3.13. Other strict-resolution filesystem errors fall back to ordinary non-strict lookup before containment and scope checks; allowed missing paths and children of files retain their normal not-found response. A genuine fallback-resolution failure remains an explicit caller-path-only error. Invalid or forbidden candidates do not contribute metadata or counts and do not prevent `list_folder_index` or `follow_links` from returning valid allowed notes. The offline receipt checks unknown targets on payloads, raw protocol frames, stderr, and audit reasons, and exercises each traversal tool through the registered SDK request handler with missing enforcement context and a valid-context control using the same arguments.

### Offline stdio verification

Run `uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case mcp --offline` to exercise the installed CLI entrypoint and MCP SDK transport against temporary synthetic vaults. The receipt records per-call expected and actual results, stdout protocol validation, scope counts, package origins, and network-denial instrumentation. No hosted embeddings, model calls, private vault input, or Telegram delivery are involved.

Embedding-boundary events are attributed to each completed tool call and recorded in the receipt. An allowed `vault_search` must exercise its own embedding boundary; events from `find_duplicates`, captures, or initialization cannot satisfy that check. The fixture replaces only the embedding provider's vector computation, not the search handler or returned results.

Folder-restricted MCP sessions omit INFO/DEBUG diagnostics, including with `--verbose`, so startup logs cannot disclose global index or graph counts. Warnings, errors, explicit profile diagnostics, and audit receipts remain available. Logging-handler thresholds are restored when the command closes, including on failure; wildcard profiles and other commands retain their logging behavior. The stdio fixture checks restricted startup output with verbose logging requested and retains a substantive unrestricted index-count control.

### Audit logging

Every MCP tool call is logged with:

- Profile name
- Tool name
- Arguments
- Outcome (OK / DENIED / ERROR)
- Denial reason (if applicable)

### Profile enforcement

The `ProfileEnforcer` checks:

1. Is this tool allowed for the active profile?
2. Is write access enabled (for `vault_write`, `capture`, `capture_note`)?
3. Is the target path within the allowed folder scope?
4. Is the complete written content within size limits (for `vault_write`, `capture`, and `capture_note`)?

Violations return a structured error response — the tool call does not execute.
