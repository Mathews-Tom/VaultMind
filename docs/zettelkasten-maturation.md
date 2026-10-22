# Zettelkasten Maturation

VaultMind includes a maturation pipeline that identifies clusters of fleeting and literature notes ready to be synthesized into permanent Zettelkasten notes. This implements the Zettelkasten principle that fleeting notes should eventually be refined into lasting knowledge.

## How It Works

### 1. Clustering

The pipeline uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) on existing ChromaDB embeddings to group semantically related notes:

- **Target note types:** `fleeting` and `literature` (configurable via `maturation.target_note_types`)
- **Minimum cluster size:** 3 notes (configurable via `maturation.min_cluster_size`)
- **Distance threshold (epsilon):** 0.25 (configurable via `maturation.cluster_eps`)

DBSCAN is chosen over K-Means because it doesn't require specifying the number of clusters upfront and naturally handles noise (notes that don't belong to any cluster).

### 2. Digest

Identified clusters are presented as a digest — a summary of which notes group together and what themes they represent. The digest uses metadata only (zero LLM cost):

- Note titles, types, and tags
- Cluster theme (derived from shared entities and tags)
- Creation dates and staleness indicators

### 3. Synthesis

Successful synthesis creates a single permanent note from the constituent notes:

- The permanent note includes `[[wikilinks]]` back to the source fleeting notes
- Frontmatter is set to `type: permanent` with tags derived from the source notes
- The synthesis uses `maturation.synthesis_model` (default: `llm.thinking_model`)
- Max output: `maturation.synthesis_max_tokens` (default: 1500)

Missing source notes, invalid generated frontmatter, or unresolved wikilinks fail synthesis rather than recording completion. A failed synthesis does not add the cluster fingerprint to the completed set. Direct `/mature synthesize <id>` requests report the error; scheduled review proposals use the failure/retry flow below.

### 4. Dismissal

Clusters you dismiss are tracked and won't resurface until `maturation.dismissed_cluster_expiry_days` (default: 90 days) have passed.

## Usage

### Telegram Bot

```text
/mature
```

Shows the top clusters ready for maturation (up to `maturation.max_clusters_per_digest`, default: 3), with note-title samples and an ID prefix. Discovery does not generate an LLM synthesis preview. An authorized user can run `/mature synthesize <id>` to synthesize directly, or `/mature dismiss <id>` to dismiss a cluster. The direct synthesis command is an explicit user action, not a scheduled queue proposal.

### Scheduled Proposal Review and Recovery

With `[autonomy].enabled = true`, scheduled discovery creates `MATURATION_SYNTHESIS` proposals in the shared review queue. They default to **SKIM**; `[autonomy].force_block = true` routes them to **BLOCK**. Review SKIM items through `/review`, or resolve individual BLOCK items through their Approve/Reject buttons. BLOCK notification delivery requires an enabled, available bot delivery path; undelivered pending proposals remain in `/review`.

Approval re-discovers the cluster by its fingerprint before calling the real synthesis operation. A cluster that has disappeared, been dismissed, or already been synthesized fails the operation instead of becoming `applied`. Successful operations record `applied` with the generated note path. Failed operations record `failed`, not a human rejection or a completed mutation.

Inspect the error and any partial effects before using the explicit Retry button in the failure response or `/review`'s failed-operation section. Failed proposals persist across bot restarts; approve-all does not retry them. A successful synthesis records its cluster fingerprint so discovery does not offer it again. No automatic retry service runs. Queue migration and rollback instructions are in the [Telegram review guide](telegram-bot.md#queue-migration-and-rollback).

### Scheduling

The bot registers a `maturation_digest` job at a seven-day interval while the maturation pipeline is enabled. A job without retained scheduler state is due on the next scheduler check; later runs use persisted last-run state. The process must be running for a job to execute.

`maturation.schedule_day`, `schedule_hour`, and `timezone` are parsed settings, but the current bot assembly does not use them to schedule this job. Their defaults (`sunday`, `9`, `UTC`) are not an enforced Sunday-at-09:00 schedule. The scheduler (`services/scheduler.py`) is asyncio-native and runs within the bot process.

## Configuration

The pipeline's settings live in `[maturation]`; the defaults below include the three reserved calendar fields. Scheduled proposal routing uses `[autonomy]`, and scheduler persistence uses `[scheduler]`.

```toml
[maturation]
enabled = true
schedule_day = "sunday"                      # Parsed only; not used by the bot job
schedule_hour = 9                            # Parsed only
timezone = "UTC"                             # Parsed only
min_cluster_size = 3
max_clusters_per_digest = 3
cluster_eps = 0.25
synthesis_max_tokens = 1500
synthesis_model = ""                         # Empty = use llm.thinking_model
target_note_types = ["fleeting", "literature"]
dismissed_cluster_expiry_days = 90
inbox_folder = "00-inbox"
```

## Pipeline Architecture

```text
src/vaultmind/pipeline/
├── clustering.py    # DBSCAN on ChromaDB embeddings
├── synthesis.py     # LLM synthesis → permanent note with wikilinks
└── maturation.py    # Orchestrator (clustering → digest → synthesis)

src/vaultmind/services/
└── scheduler.py     # Asyncio scheduler with persistent state
```

The `MaturationPipeline` in `pipeline/maturation.py` coordinates the flow:

1. Fetches embeddings for target note types from ChromaDB
2. Runs DBSCAN clustering via `pipeline/clustering.py`
3. Filters out dismissed clusters
4. Presents digest (via `/mature` command or scheduled notification)
5. On approval, synthesizes via `pipeline/synthesis.py`

The scheduler in `services/scheduler.py` retains last-run timestamps across bot restarts. This preserves progress for the assembled seven-day job; it does not activate the reserved calendar fields or execute jobs while the process is stopped.
