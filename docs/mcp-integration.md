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

For any nonempty metadata filter, the store first retrieves at most the requested number of matching chunk IDs, without documents, metadata, or embeddings. The backend query limit is bounded by that matching chunk population, not the number of notes or the global collection size. This prevents oversized scoped search and duplicate-candidate queries from triggering Chroma's global-size warning in a stable index while retaining multiple matching chunks from one note and their distance ordering. A filter with no matches returns an empty result without embedding or issuing a zero-limit query; invalid requested limits still fail. The lookup and vector query are not a transactional snapshot. If concurrent deletion invalidates the bound, restricted MCP diagnostics retain the backend warning but replace its global population with a count-free availability message.

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

The CLI opens the real episode store when `[episodic].enabled` is true, using its configured `db_path` or `~/.vaultmind/data/episodes.db`. Folder-restricted profiles are denied before querying it, including when an individual episode lists source notes: free-form decisions and linked entity names do not establish complete scoped ownership.

**Parameters:**

| Parameter | Type    | Required | Description                    |
| --------- | ------- | -------- | ------------------------------ |
| `entity`  | string  | No       | Filter episodes by entity name |
| `status`  | string  | No       | `"pending"` or `"resolved"`    |
| `limit`   | integer | No       | Max results (default 10)       |

**Returns:** `{episodes: [...], count}` — each episode includes `episode_id`, `decision`, `context`, `outcome`, `status`, `lessons`, `entities`, `created`.

### `workflow_suggest`

Find a matching procedural workflow for a given context.

Enable `[procedural].enabled` to open the real workflow store at its configured `db_path` or `~/.vaultmind/data/procedural.db`. Suggestions query existing workflows only; they do not synthesize workflows, call a model, increment usage, or update stored business state. Folder-restricted profiles are explicitly denied.

**Parameters:**

| Parameter | Type   | Required | Description                                        |
| --------- | ------ | -------- | -------------------------------------------------- |
| `context` | string | Yes      | Context to match against workflow trigger patterns |

**Returns:** `{workflow: {workflow_id, name, description, steps, trigger_pattern, success_rate, usage_count}}` or `{workflow: null}` if no match.

### `graph_evolution`

Belief evolution signals from the existing graph-only scan: confidence drift and relationship shifts. This MCP tool does not run file-age/stale-claim scanning. The CLI assembles the real detector when `[evolution].enabled` is true; folder-restricted profiles are explicitly denied before scanning.

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

The shared `ProfileEnforcer` boundary covers every registered tool. Direct dispatch requires an explicit enforcement context. Tool allowlists, write permission, note size, and vault containment remain independent checks. For folder-restricted profiles, search and duplicate candidates are scoped before store truncation. Wildcard profiles preserve the original query filters and valid literal provenance. Both paths defensively verify note-backed candidates before returning content, paths, metadata, or counts. Missing, ill-typed, unresolvable, non-Markdown, and forbidden note provenance never grants access.

### Complete result policy

The server registers 18 tools. The following policies apply after the profile's tool allowlist check; an allowlisted tool can still be explicitly denied when its producer cannot establish complete note provenance.

| Tool | Folder-restricted result policy | Wildcard-folder behavior |
| --- | --- | --- |
| `vault_search` | Filter verified note candidates before top-k, then validate returned chunks and counts. | Global search filters and valid note-backed results. |
| `vault_read` | Authorize the requested note before reading content. | Any contained note. |
| `vault_write` | Authorize destination, write permission, and size before writing/indexing. | Any contained destination, subject to write and size policy. |
| `vault_list` | List verified scoped notes; root input aggregates allowed roots only. | Contained vault listing. |
| `read_frontmatter` | Authorize note before parsing or returning metadata. | Any contained note. |
| `list_folder_index` | Authorize candidates before parsing metadata or counting. | Contained vault index. |
| `follow_links` | Resolve outgoing links and backlinks only among allowed notes; omit forbidden/unresolved targets. | Global contained links, including unresolved titles with null paths. |
| `find_duplicates` | Filter verified note candidates before ranking/truncation; do not update producer buffers. | Global valid note candidates; no producer-buffer writes. |
| `suggest_links` | Deny before producer invocation: graph-derived scoring lacks complete scoped provenance. | Global valid note candidates and graph scores; no producer-buffer writes. |
| `capture` | Authorize computed inbox destination, write permission, and size. | Contained inbox destination, subject to write and size policy. |
| `capture_note` | Authorize computed inbox destination, write permission, and size. | Contained inbox destination, subject to write and size policy. |
| `graph_query` | Deny before traversal: merged graph records lack complete contribution provenance. | Existing global graph query semantics. |
| `graph_path` | Deny before shortest-path computation for the same provenance limitation. | Existing global shortest paths. |
| `vault_stats` | Deny before aggregation because graph counts cannot be proved in scope; no partial or fabricated graph counts. | All verified canonical notes and existing global graph counts. |
| `recent_activity` | Inspect timestamps only for verified canonical allowed notes. | All verified canonical notes. |
| `episode_query` | Deny before store queries: free-form episode decisions, outcomes, lessons, and entities lack complete note ownership. | Configured global episodic queries. |
| `workflow_suggest` | Deny before suggestion: workflow text and source-episode references do not establish complete note ownership. | Configured global read-only suggestions; usage counts are not incremented. |
| `graph_evolution` | Deny before scanning: signal source lists do not establish complete provenance for merged entity labels and global relationships. | Configured global read-only evolution scan. |

Graph records combine labels, confidence, attributes, and relationships from multiple sources. The producer's source lists do not record unknown contributions: adding an allowed source after an unknown contribution, or an unknown contribution after an allowed source, leaves retained data with an apparently all-allowed source list. Neither a nonempty list nor filtering both endpoint and edge lists proves complete ownership. Folder-restricted profiles therefore receive explicit policy denials for `graph_query`, `graph_path`, and `vault_stats`, including when every listed source is allowed. Wildcard-folder profiles retain global results; restricted note search, listing, metadata, traversal, and activity remain available under their own guards.

Wildcard statistics and scoped activity deduplicate canonical note paths, so symlink aliases do not inflate counts. Hidden paths are excluded. Unconfigured optional dependencies still report their ordinary configuration errors to wildcard-folder profiles; restricted profiles receive policy denials first, not fabricated empty global results. These rules constrain MCP outputs; they do not provide a filesystem sandbox or atomicity guarantee. Denials can echo caller-supplied paths but must not disclose unknown resolved symlink targets.

Read-only tools do not write notes, graph snapshots, episode/workflow business state, dismissed signals, or duplicate/suggestion buffers. Existing audit logging is the intentional exception: every call can append its profile, caller arguments, result summary, and denial reason to the configured JSONL/SQLite audit stores.

Symlink loops fail closed with a caller-path-only error, on Python 3.12 and 3.13. Other strict-resolution filesystem errors fall back to ordinary non-strict lookup before containment and scope checks; allowed missing paths and children of files retain their normal not-found response. A genuine fallback-resolution failure remains an explicit caller-path-only error. Invalid or forbidden candidates do not contribute metadata or counts and do not prevent `list_folder_index` or `follow_links` from returning valid allowed notes. The offline receipt checks unknown targets on payloads, raw protocol frames, stderr, and audit reasons, and exercises each traversal tool through the registered SDK request handler with missing enforcement context and a valid-context control using the same arguments.

### Offline stdio verification

Run `uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case mcp --offline` to exercise the installed CLI entrypoint and MCP SDK transport against temporary synthetic vaults. The receipt records per-call expected and actual results, stdout protocol validation, scope counts, package origins, and network-denial instrumentation. No hosted embeddings, model calls, private vault input, or Telegram delivery are involved.

Embedding-boundary events are attributed to each completed tool call and recorded in the receipt. A `vault_search` with matching indexed chunks must exercise its own embedding boundary; an empty filtered search legitimately makes no provider call. Events from `find_duplicates`, captures, or initialization cannot satisfy another call's check. The fixture replaces only the embedding provider's vector computation, not the search handler or returned results.

Folder-restricted MCP sessions omit INFO/DEBUG diagnostics, including with `--verbose`, so startup logs cannot disclose global index or graph counts. Filtered search bounds backend requests before querying. A concurrent index shrink can still trigger Chroma's oversized-query warning; restricted sessions retain its WARNING severity and logger identity but render a count-free availability message instead of the global cardinality. Other warnings, errors, explicit profile diagnostics, and audit receipts remain available. Logging-handler thresholds and warning filters are restored when the command closes, including on failure; wildcard profiles retain their global index diagnostics. The stdio fixture scans complete stderr, including `elements in index` and `n_results` warnings, compares authorized search and duplicate results with fixed allowed chunks and different forbidden indexed populations, and retains seeded unrestricted global-count controls. Raw result distances must remain nondecreasing; exact-distance ties are not ordered by chunk ID. A separate real-CLI/registered-handler regression injects a deterministic concurrent index shrink and checks warning privacy, full-profile diagnostics, allowed results, state changes attributable only to the injected writer, and sink restoration on normal and exceptional exit; it does not replace the stdio transport verification.

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
