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

## Graph-Grounded Thinking

The `/think` command uses the knowledge graph as context:

1. The user's topic is sent to the LLM for entity extraction
2. Extracted entities are looked up in the graph
3. An ego subgraph (configurable depth via `llm.graph_hop_depth`) is built around matched entities
4. The subgraph's relationships are serialized and injected into the thinking prompt alongside ChromaDB search results

This gives the thinking partner awareness of how concepts in your vault relate to each other — not just what's semantically similar, but what's structurally connected.

Configuration in `[llm]`:

- `graph_context_enabled` — toggle graph context (default: true)
- `graph_hop_depth` — ego subgraph depth (default: 2)
- `graph_min_confidence` — minimum confidence for context entities (default: 0.6)
- `graph_max_relationships` — max relationships in context (default: 20)

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
