# Knowledge Graph

VaultMind builds a knowledge graph from your vault using LLM-powered entity extraction. The graph surfaces connections between concepts, people, projects, and ideas that might not be obvious from the file structure alone.

## How It Works

### Entity Extraction

When you run `vaultmind graph-build` or enable `reextract_graph` in watch mode:

1. Each note is sent to the configured LLM (default: `llm.thinking_model`)
2. The LLM identifies entities (people, concepts, tools, projects) and relationships between them
3. Each entity/relationship gets a confidence score (minimum threshold: `graph.min_confidence`, default 0.7)
4. Results are merged into the NetworkX graph with source-note attribution

### Storage

The graph is persisted as JSON at `~/.vaultmind/data/knowledge_graph.json`. This is a serialized NetworkX graph — no database server required.

### Graph Building

```bash
# Incremental build (only processes notes not yet in the graph)
uv run vaultmind graph-build

# Full rebuild from scratch
uv run vaultmind graph-build --full
```

### Watch Mode Integration

By default, graph re-extraction is disabled in watch mode because it makes an LLM call per note. Enable it with:

```bash
uv run vaultmind watch --graph
```

When enabled, changes are batched on a timer (`watch.batch_graph_interval_seconds`, default 300s) instead of triggering per-save. This coalesces rapid Obsidian edits into a single extraction batch.

## Querying

### Telegram Bot

```text
/graph Python
```

Returns the entity's neighbors, relationships, and connection metadata within the configured hop depth.

### MCP Tools

- `graph_query` — entity neighbors and relationships (configurable depth)
- `graph_path` — shortest path between two entities

Both tools require a profile with wildcard folder access (`folder_scope = ["*"]`) as well as the appropriate tool allowlist. Folder-restricted profiles, including the built-in planner, receive an explicit denial before graph traversal: merged labels, attributes, and relationships do not carry complete contribution provenance. Listing an allowed source note on a graph record does not prove all contributions are in scope. Do not widen a profile merely to bypass a denial; use scoped note search and traversal when global graph disclosure is not authorized. See the [complete MCP result policy](mcp-integration.md#complete-result-policy).

### CLI

```bash
uv run vaultmind graph-report
```

Generates analytics: top entities by degree, most connected clusters, relationship type distribution, and orphan entities.

## Graph Maintenance

Over time, notes get edited or deleted. The graph can accumulate stale references.

```bash
uv run vaultmind graph-maintain
```

This command:

1. **Prunes stale source references** — removes source-note attributions for notes that no longer exist
2. **Removes orphan entities** — deletes entities with no remaining source notes

Additionally, `GraphMaintainer` subscribes to `NoteDeletedEvent` on the event bus. When a note is deleted (via bot or watch mode), its graph references are cleaned up automatically.

## Graph and Knowledge Answering

The graph remains an independent extraction, traversal, maintenance and belief
evolution domain. It is not evidence for normal knowledge answers. Root `ask`,
selected `wiki query`, Telegram question/chat, `/think` (including sticky and
voice questions) and MCP `knowledge_query` use approved current wiki sections
with complete retained provenance through the shared query core.

No graph entity extraction, ego-subgraph context, episode context or Chroma
search result is injected into those answers. Source-only `/recall`,
`search-sources` and MCP `vault_search` are discovery tools, not approved answer
evidence. Thinking modes change presentation only; prior user questions resolve
intent without making assistant replies, summaries or graph facts evidence.

The former answer-only `[llm]` graph-context controls are removed. Keep `[graph]`
settings for actual graph operations: `extraction_model` selects their model
(empty uses `llm.thinking_model`), `min_confidence` filters extraction and
`persist_path` selects the graph file. `/graph`, MCP `graph_query`/`graph_path`,
graph maintenance and evolution retain their independent permissions and
behavior. A query does not update graph or episode state, mint a gap, distill a
session or create a proposal/page. See the [query policy](mcp-integration.md#knowledge_query)
for current-page validation, conflict citations and refusal boundaries.

## Belief Evolution

The belief evolution tracker (`graph/evolution.py`) monitors three signals across the knowledge graph:

### Confidence Drift

Detects entities whose confidence scores have changed significantly. A drift above `evolution.confidence_drift_threshold` (default: 0.3) indicates your understanding of a concept has shifted.

### Relationship Shifts

Identifies relationships that have been added, removed, or substantially modified between graph snapshots. Helps surface when your mental model of how things connect has changed.

### Stale Claims

Flags high-confidence entities (above `evolution.min_confidence_for_stale`, default: 0.8) that haven't been referenced or updated in `evolution.stale_days` (default: 180 days). These represent knowledge you were once confident about but haven't revisited — a prompt to re-evaluate.

### Access

- **Telegram:** `/evolve` command
- **Weekly digest:** Included when `evolution.include_in_digest = true`
- **Config:** `[evolution]` section in `default.toml`
