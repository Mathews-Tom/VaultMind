# Configuration Reference

VaultMind uses a layered configuration system:

1. **`config/default.toml`** — all non-secret settings
2. **`.env`** — API keys and secrets only
3. **Environment variables** — `VAULTMIND_*` prefix overrides any TOML setting

Nested keys use `__` as delimiter in env vars: `VAULTMIND_TELEGRAM__BOT_TOKEN`.

---

## `[vault]`

Vault filesystem paths and behavior.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | string | `~/.vaultmind/vault` | Root directory of the Obsidian vault |
| `inbox_folder` | string | `00-inbox` | Folder for quick captures |
| `daily_folder` | string | `01-daily` | Folder for daily notes |
| `templates_folder` | string | `06-templates` | Folder for note templates |
| `meta_folder` | string | `_meta` | Folder for auto-generated reports |
| `sync_interval_seconds` | int | `300` | Full re-sync interval (seconds) |
| `excluded_folders` | list | `[".obsidian", ".git", ".trash"]` | Folders to skip during indexing |

## `[embedding]`

Embedding generation for vector search.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | `openai` | `"openai"` or `"voyage"` |
| `model` | string | `text-embedding-3-small` | Embedding model name |
| `dimensions` | int | `1536` | Embedding vector dimensions |
| `batch_size` | int | `64` | Notes per embedding batch |
| `cache_enabled` | bool | `true` | Enable SQLite embedding cache |

## `[chroma]`

ChromaDB vector store settings.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `persist_dir` | string | `~/.vaultmind/data/chromadb` | ChromaDB data directory |
| `collection_name` | string | `vault_chunks` | Collection name |
| `distance_fn` | string | `cosine` | Distance function (`cosine`, `l2`, `ip`) |

## `[graph]`

Knowledge graph extraction and persistence.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `persist_path` | string | `~/.vaultmind/data/knowledge_graph.json` | Graph JSON file |
| `extraction_model` | string | `""` | LLM model for entity extraction (empty = `llm.thinking_model`) |
| `min_confidence` | float | `0.7` | Minimum confidence for extracted entities |
| `enrichment_schedule` | string | `0 3 * * *` | Cron schedule for graph enrichment |

## `[llm]`

LLM provider configuration.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | `openai` | `"anthropic"`, `"openai"`, `"gemini"`, `"ollama"` |
| `thinking_model` | string | — | Primary model for thinking, analysis, extraction |
| `fast_model` | string | — | Lighter model for routing, tagging, chat |
| `max_context_notes` | int | `8` | Max notes injected as context |
| `max_tokens` | int | `4096` | Max response tokens |
| `ollama_base_url` | string | `http://localhost:11434` | Ollama server URL |
| `graph_context_enabled` | bool | `true` | Include graph context in thinking sessions |
| `graph_hop_depth` | int | `2` | Ego subgraph traversal depth |
| `graph_min_confidence` | float | `0.6` | Min confidence for graph context entities |
| `graph_max_relationships` | int | `20` | Max relationships in graph context |

**API keys** (set in `.env`):
- `VAULTMIND_ANTHROPIC_API_KEY` — Anthropic
- `VAULTMIND_OPENAI_API_KEY` — OpenAI (also used for embeddings)
- `VAULTMIND_GEMINI_API_KEY` — Google Gemini
- `VAULTMIND_VOYAGE_API_KEY` — Voyage (embeddings only)

Ollama requires no API key.

## `[telegram]`

Telegram bot behavior.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `thinking_session_ttl` | int | `3600` | Thinking session timeout (seconds) |

**Secrets** (set in `.env`):
- `VAULTMIND_TELEGRAM__BOT_TOKEN` — Bot token from @BotFather
- `VAULTMIND_TELEGRAM__ALLOWED_USER_IDS` — Comma-separated user ID whitelist (empty = allow all)

## `[routing]`

Message classification and routing behavior.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `chat_model` | string | `""` | Model for chat responses (empty = `llm.fast_model`) |
| `chat_max_tokens` | int | `1024` | Max tokens for chat responses |
| `vault_context_enabled` | bool | `true` | Include vault search results in conversational responses |
| `max_context_chunks` | int | `4` | Max chunks for context augmentation |
| `capture_all` | bool | `false` | Bypass routing — all text becomes a capture |

## `[watch]`

File watcher behavior for incremental indexing.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `debounce_ms` | int | `500` | Debounce window for rapid saves (ms) |
| `hash_stability_check` | bool | `true` | Two-stage hash check to avoid partial writes |
| `reextract_graph` | bool | `false` | Re-extract graph entities on file change (LLM call per note) |
| `batch_graph_interval_seconds` | int | `300` | Batch window for graph re-extraction when enabled |

## `[duplicate_detection]`

Semantic duplicate detection thresholds.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable automatic duplicate detection |
| `min_content_length` | int | `100` | Skip notes shorter than this |

Similarity bands: duplicate (>= 92%), merge candidate (80-92%).

## `[note_suggestions]`

Link suggestion scoring.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable note suggestions |
| `min_content_length` | int | `100` | Skip notes shorter than this |
| `entity_weight` | float | `0.1` | Score boost per shared graph entity |
| `graph_weight` | float | `0.05` | Score boost from graph path proximity |

## `[search]`

Search pagination and session management.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `page_size` | int | `5` | Results per page |
| `max_results` | int | `25` | Maximum total results |
| `session_ttl` | int | `300` | Paginated session expiry (seconds) |

## `[ranking]`

Post-retrieval ranking with note-type multipliers and temporal decay.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable post-retrieval ranking |

## `[maturation]`

Zettelkasten maturation pipeline — clusters fleeting/literature notes for synthesis into permanent notes.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable maturation pipeline |
| `schedule_day` | string | `sunday` | Day of the week for scheduled maturation |
| `schedule_hour` | int | `9` | Hour (UTC) for scheduled maturation |
| `timezone` | string | `UTC` | Timezone for scheduling |
| `min_cluster_size` | int | `3` | Minimum notes per cluster |
| `max_clusters_per_digest` | int | `3` | Max clusters per maturation digest |
| `cluster_eps` | float | `0.25` | DBSCAN epsilon (distance threshold) |
| `synthesis_max_tokens` | int | `1500` | Max tokens for synthesis output |
| `synthesis_model` | string | `""` | Model for synthesis (empty = `llm.thinking_model`) |
| `target_note_types` | list | `["fleeting", "literature"]` | Note types eligible for clustering |
| `dismissed_cluster_expiry_days` | int | `90` | Days before dismissed clusters resurface |
| `inbox_folder` | string | `00-inbox` | Folder to check for maturation candidates |

## `[evolution]`

Belief evolution tracking — detects confidence drift, relationship shifts, and stale claims.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable evolution tracking |
| `confidence_drift_threshold` | float | `0.3` | Minimum drift to flag |
| `stale_days` | int | `180` | Days before a high-confidence claim is flagged stale |
| `min_confidence_for_stale` | float | `0.8` | Minimum confidence to consider for staleness |
| `max_results` | int | `10` | Max evolution signals per query |
| `include_in_digest` | bool | `true` | Include evolution signals in weekly digest |

## `[digest]`

Weekly digest generation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable weekly digest |
| `period_days` | int | `7` | Digest period (days) |
| `schedule_hour` | int | `8` | Hour (UTC) for scheduled digest |
| `timezone` | string | `UTC` | Timezone |
| `save_to_vault` | bool | `true` | Save digest as a vault note |
| `send_telegram` | bool | `true` | Send digest via Telegram |
| `max_trending` | int | `10` | Max trending topics |
| `max_suggestions` | int | `5` | Max suggestions per digest |
| `connection_threshold_low` | float | `0.70` | Lower bound for connection suggestions |
| `connection_threshold_high` | float | `0.85` | Upper bound for connection suggestions |

## `[auto_tag]`

LLM-based automatic tag suggestions.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable auto-tagging |
| `max_tags_per_note` | int | `2` | Max tags suggested per note |
| `min_content_length` | int | `100` | Skip notes shorter than this |
| `tagging_model` | string | `""` | Model for tagging (empty = `llm.fast_model`) |

New tags are quarantined at `~/.vaultmind/data/tag_quarantine.json` for user approval.

## `[voice]`

Voice message transcription.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable voice transcription |
| `whisper_model` | string | `whisper-1` | OpenAI Whisper model |
| `language` | string | `""` | Language hint (empty = auto-detect) |

Requires `VAULTMIND_OPENAI_API_KEY` in `.env`.

## `[ingest]`

URL ingestion settings (YouTube transcripts, article content).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable URL ingestion |
| `youtube_language` | string | `en` | Preferred transcript language |
| `max_content_length` | int | `100000` | Max content length (characters) |
| `inbox_folder` | string | `00-inbox` | Target folder for ingested notes |

## `[research]`

Research pipeline settings (`vaultmind research` command).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_results` | int | `5` | Max YouTube results per query |
| `output_folder` | string | `research` | Vault subfolder for research notes |
| `youtube_language` | string | `en` | Preferred transcript language |

## `[tracking]`

User preference and usage tracking.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable interaction tracking |
| `db_path` | string | `""` | SQLite path (empty = `~/.vaultmind/data/preferences.db`) |

## `[mcp]`

MCP server settings and profile definitions.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable MCP server |
| `host` | string | `127.0.0.1` | Bind address |
| `port` | int | `8765` | Bind port |

### Profile definitions

Profiles are defined under `[mcp.profiles.<name>]`:

```toml
[mcp.profiles.researcher]
description = "Read-only access for research and Q&A tasks"
allowed_tools = ["vault_search", "vault_read", "vault_list", "graph_query", "graph_path"]
folder_scope = ["*"]
write_enabled = false

[mcp.profiles.planner]
description = "Read/write access for project planning"
allowed_tools = ["vault_search", "vault_read", "vault_write", "vault_list", "graph_query", "graph_path", "capture"]
folder_scope = ["02-projects", "00-inbox"]
write_enabled = true
max_note_size_bytes = 50000

[mcp.profiles.full]
description = "Unrestricted access — requires explicit opt-in"
allowed_tools = ["*"]
folder_scope = ["*"]
write_enabled = true
requires_confirmation = true
```

Activate with: `vaultmind mcp-serve --profile <name>`. Default (no flag) uses `researcher`.
