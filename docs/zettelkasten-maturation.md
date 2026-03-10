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

When you approve a cluster, the LLM synthesizes the constituent notes into a single permanent note:

- The permanent note includes `[[wikilinks]]` back to the source fleeting notes
- Frontmatter is set to `type: permanent` with tags derived from the source notes
- The synthesis uses `maturation.synthesis_model` (default: `llm.thinking_model`)
- Max output: `maturation.synthesis_max_tokens` (default: 1500)

### 4. Dismissal

Clusters you dismiss are tracked and won't resurface until `maturation.dismissed_cluster_expiry_days` (default: 90 days) have passed.

## Usage

### Telegram Bot

```text
/mature
```

Shows the top clusters ready for maturation (up to `maturation.max_clusters_per_digest`, default: 3). Each cluster shows its constituent notes and a synthesis preview.

### Scheduling

The maturation pipeline runs on a configurable schedule:

- **Day:** `maturation.schedule_day` (default: `sunday`)
- **Hour:** `maturation.schedule_hour` (default: 9, UTC)
- **Timezone:** `maturation.timezone` (default: `UTC`)

The scheduler (`services/scheduler.py`) is asyncio-native and runs within the bot process.

## Configuration

All settings live in `[maturation]` section of `config/default.toml`:

```toml
[maturation]
enabled = true
schedule_day = "sunday"
schedule_hour = 9
timezone = "UTC"
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

The scheduler in `services/scheduler.py` tracks last-run timestamps persistently, ensuring the pipeline runs at the configured interval even across bot restarts.
