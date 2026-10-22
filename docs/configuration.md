# Configuration Reference

Configuration precedence, highest priority first:

1. **Deliberate constructor/call-site overrides** — `Settings(...)` or `Settings.from_toml(path, **overrides)`.
2. **Environment variables** — `VAULTMIND_*`.
3. **`.env`** — read from the current working directory, with the same key names as environment variables. Keep secrets here or in the process environment, not in TOML.
4. **TOML** — the explicit `--config PATH` file, or `config/default.toml` relative to the current working directory.
5. **Model defaults** — used for keys absent from all higher-priority sources.

Nested keys use `__`, for example `VAULTMIND_VAULT__PATH=/absolute/vault` and `VAULTMIND_SEARCH__HYBRID_ENABLED=false`. Overrides merge by nested key: changing the vault path does not discard a TOML `inbox_folder`. Lists use JSON syntax, for example `VAULTMIND_TELEGRAM__ALLOWED_USER_IDS='[123,456]'`. Invalid selected values raise a configuration error; they do not fall back to a valid lower-priority value.

MCP profile names are dictionary keys, including names that match policy fields such as `allowed_tools`, `description`, or `write_enabled`. For example, `VAULTMIND_MCP__PROFILES__ALLOWED_TOOLS__FOLDER_SCOPE='["00-inbox"]'` changes the `allowed_tools` profile's folder scope without changing its allowed tools or write policy. The same JSON-list syntax applies in cwd `.env`.

Settings loading requires `pydantic-settings>=2.13.1,<3`. The environment/dotenv source adapter preserves that API's resolved per-load controls; older settings-library releases do not expose all of those controls.

All CLI, bot, and MCP assemblies use [`load_settings`](../src/vaultmind/config.py), which delegates to `Settings.from_toml`. TOML is a separate lower-priority Pydantic source, not a dictionary of constructor overrides. Plain `Settings(...)` reads environment/`.env` and defaults without implicitly selecting a TOML file. A missing TOML file retains the established environment/default loading behavior; malformed TOML and unreadable files fail.

For an isolated run, set `HOME`, use a working directory without a personal `.env`, pass an explicit temporary TOML file, and configure every persistent component the command uses. Overriding `vault.path` alone does not relocate `chroma.persist_dir`, `graph.persist_path`, or `search.bm25_db_path`. The offline `scripts/verify_acceptance.py --case config --offline` case exercises temporary targets through runtime consumers with deterministic external embeddings; it is not hosted-provider qualification.

## Declared versus consumed controls

The following sections are **not wired into the shipped CLI/bot assemblies**. Their values are parsed and validated, but changing them does not activate the described runtime behavior. The first five reserve future capabilities; the remaining three have standalone library implementations without a settings-driven assembly. Field descriptions for these sections describe declared intent, not operational guarantees.

| Section | Declaration in [`config.py`](../src/vaultmind/config.py) | Runtime status |
| --- | --- | --- |
| `memory_profile` | `MemoryProfileConfig`, `Settings.memory_profile` | Dormant; no identity-context or daily-log consumer |
| `proactivity` | `ProactivityConfig`, `Settings.proactivity` | Dormant; no runtime policy enforcement |
| `heartbeat` | `HeartbeatConfig`, `Settings.heartbeat` | Dormant; no heartbeat orchestrator |
| `drafts` | `DraftsConfig`, `Settings.drafts` | Dormant; no draft lifecycle consumer |
| `integrations` | `IntegrationsConfig`, `Settings.integrations` | Dormant; no integration-policy consumer |
| `consolidation` | `ConsolidationConfig`, `Settings.consolidation` | No assembled consolidation job; `MemoryConsolidator` requires explicit construction |
| `health` | `HealthConfig`, `Settings.health` | No assembled periodic monitor; bot `/health` uses separate on-demand checks |
| `mcp_retry` | `MCPRetryConfig`, `Settings.mcp_retry` | CLI `mcp-serve` does not construct or pass a retry executor |

Consumer inventory: [`cli.py`](../src/vaultmind/cli.py) assembles CLI commands, bot jobs, and MCP dependencies; [`mcp/server.py`](../src/vaultmind/mcp/server.py) implements MCP tools. The CLI does not consume the eight sections above. The MCP server accepts an explicitly supplied `ToolRetryExecutor`, but does not load `[mcp_retry]` itself. Existing `autonomy` review queues, `sources` connectors, scheduler jobs, and MCP profiles use their own consumed controls.

---

## `[vault]`

Vault filesystem paths and behavior.

| Key                     | Type   | Default                           | Description                          |
| ----------------------- | ------ | --------------------------------- | ------------------------------------ |
| `path`                  | string | `~/.vaultmind/vault`              | Root directory of the Obsidian vault |
| `inbox_folder`          | string | `00-inbox`                        | Folder for quick captures            |
| `daily_folder`          | string | `01-daily`                        | Folder for daily notes               |
| `templates_folder`      | string | `06-templates`                    | Folder for note templates            |
| `meta_folder`           | string | `_meta`                           | Folder for auto-generated reports    |
| `sync_interval_seconds` | int    | `300`                             | Full re-sync interval (seconds)      |
| `excluded_folders`      | list   | `[".obsidian", ".git", ".trash"]` | Folders to skip during indexing      |

## `[memory_profile]`

Reserved identity-memory layout and promotion settings; dormant, with no always-available context or daily-log pipeline driven by this section.

| Key                         | Type   | Default              | Description                                          |
| --------------------------- | ------ | -------------------- | ---------------------------------------------------- |
| `enabled`                   | bool   | `true`               | Reserved identity-layout toggle; no runtime effect   |
| `identity_folder`           | string | `_meta/identity`     | Vault folder holding curated identity-memory notes   |
| `user_note_name`            | string | `user.md`            | User profile note filename                           |
| `system_note_name`          | string | `system.md`          | System rules and operating-boundaries note filename  |
| `promoted_memory_note_name` | string | `memory.md`          | Promoted long-term memory note filename              |
| `priorities_note_name`      | string | `priorities.md`      | Current priorities note filename                     |
| `daily_log_folder`          | string | `_meta/daily-logs`   | Append-only daily interaction log folder             |
| `recent_daily_logs`         | int    | `3`                  | Number of recent daily logs to include in context    |
| `promoted_memory_max_chars` | int    | `12000`              | Ceiling for curated promoted-memory size             |

## `[proactivity]`

Reserved autonomy-policy values; dormant. These fields do not enforce confirmation or side-effect boundaries.

| Key                                          | Type   | Default      | Description                                         |
| -------------------------------------------- | ------ | ------------ | --------------------------------------------------- |
| `mode`                                       | string | `observer`   | `observer`, `advisor`, `assistant`, or `partner`    |
| `require_confirmation_for_external_send`     | bool   | `true`       | Require approval before any outbound external send  |
| `require_confirmation_for_destructive_actions` | bool | `true`       | Require approval before delete or destructive writes |
| `allow_local_automation`                     | bool   | `false`      | Allow local vault-side automation                   |
| `allow_external_side_effects`                | bool   | `false`      | Allow external writes when a later adapter supports them |

## `[heartbeat]`

Reserved proactive-state collection settings; dormant. No heartbeat job is scheduled from this section.

| Key                   | Type   | Default    | Description                                             |
| --------------------- | ------ | ---------- | ------------------------------------------------------- |
| `enabled`             | bool   | `false`    | Reserved heartbeat toggle; no runtime effect             |
| `schedule`            | string | `*/30 * * * *` | Cron schedule for heartbeat runs                    |
| `timezone`            | string | `UTC`      | Timezone for schedule interpretation                    |
| `active_hours_start`  | int    | `8`        | Start hour for active monitoring window                 |
| `active_hours_end`    | int    | `18`       | End hour for active monitoring window                   |
| `snapshot_state_path` | string | `""`       | Snapshot state path (empty = default under `~/.vaultmind/data/`) |
| `max_snapshot_items`  | int    | `50`       | Maximum items gathered into one heartbeat snapshot      |
| `max_actions_per_run` | int    | `5`        | Maximum suggested actions emitted per heartbeat run     |

## `[drafts]`

Reserved draft lifecycle settings; dormant. No draft storage, expiration, approval, or send workflow consumes this section.

| Key                      | Type   | Default                 | Description                                       |
| ------------------------ | ------ | ----------------------- | ------------------------------------------------- |
| `enabled`                | bool   | `false`                 | Reserved draft toggle; no runtime effect           |
| `active_folder`          | string | `_meta/drafts/active`   | Vault folder for live drafts                      |
| `sent_folder`            | string | `_meta/drafts/sent`     | Vault folder for user-sent replies                |
| `expired_folder`         | string | `_meta/drafts/expired`  | Vault folder for expired drafts                   |
| `expiry_hours`           | int    | `24`                    | Draft expiration window                           |
| `require_human_approval` | bool   | `true`                  | Require review before any send-capable workflow   |
| `voice_match_limit`      | int    | `5`                     | Max prior sent drafts used for tone matching      |

## `[integrations]`

Reserved external-adapter policy settings; dormant. They do not configure the existing source connectors or enforce capability limits.

| Key                      | Type   | Default | Description                                                   |
| ------------------------ | ------ | ------- | ------------------------------------------------------------- |
| `enabled`                | bool   | `false` | Reserved integration toggle; no runtime effect                 |
| `manifest_path`          | string | `""`    | Integration manifest path (empty = default under data dir)    |
| `default_capability`     | string | `read`  | Default adapter capability ceiling: `read`, `draft`, `write`  |
| `sanitize_external_text` | bool   | `true`  | Sanitize provider payload text before prompt assembly         |
| `audit_retention_days`   | int    | `90`    | Retention window for integration audit records                |

## `[embedding]`

Embedding generation for vector search.

| Key             | Type   | Default                  | Description                   |
| --------------- | ------ | ------------------------ | ----------------------------- |
| `provider`      | string | `openai`                 | `"openai"` or `"voyage"`      |
| `model`         | string | `text-embedding-3-small` | Embedding model name          |
| `dimensions`    | int    | `1536`                   | Embedding vector dimensions   |
| `batch_size`    | int    | `64`                     | Notes per embedding batch     |
| `cache_enabled` | bool   | `true`                   | Enable SQLite embedding cache |

## `[chroma]`

ChromaDB vector store settings.

| Key               | Type   | Default                      | Description                              |
| ----------------- | ------ | ---------------------------- | ---------------------------------------- |
| `persist_dir`     | string | `~/.vaultmind/data/chromadb` | ChromaDB data directory                  |
| `collection_name` | string | `vault_chunks`               | Collection name                          |
| `distance_fn`     | string | `cosine`                     | Distance function (`cosine`, `l2`, `ip`) |

## `[scheduler]`

Scheduler

```toml
[scheduler]
state_path = ""  # default: ~/.vaultmind/data/scheduler_state.json
```

Persistent state file for the compound loop scheduler. Stores last-run timestamps, run counts, job state dicts, and completion flags. Uses atomic writes (write-to-tmp, rename) to prevent corruption.

## `[graph]`

Knowledge graph extraction and persistence.

| Key                   | Type   | Default                                  | Description                                                    |
| --------------------- | ------ | ---------------------------------------- | -------------------------------------------------------------- |
| `persist_path`        | string | `~/.vaultmind/data/knowledge_graph.json` | Graph JSON file                                                |
| `extraction_model`    | string | `""`                                     | LLM model for entity extraction (empty = `llm.thinking_model`) |
| `min_confidence`      | float  | `0.7`                                    | Minimum confidence for extracted entities                      |
| `enrichment_schedule` | string | `0 3 * * *`                              | Cron schedule for graph enrichment                             |

## `[llm]`

LLM provider configuration.

| Key                       | Type   | Default                  | Description                                       |
| ------------------------- | ------ | ------------------------ | ------------------------------------------------- |
| `provider`                | string | `openai`                 | `"anthropic"`, `"openai"`, `"gemini"`, `"ollama"` |
| `thinking_model`          | string | —                        | Primary model for thinking, analysis, extraction  |
| `fast_model`              | string | —                        | Lighter model for routing, tagging, chat          |
| `max_context_notes`       | int    | `8`                      | Max notes injected as context                     |
| `max_tokens`              | int    | `4096`                   | Max response tokens                               |
| `ollama_base_url`         | string | `http://localhost:11434` | Ollama server URL                                 |
| `graph_context_enabled`   | bool   | `true`                   | Include graph context in thinking sessions        |
| `graph_hop_depth`         | int    | `2`                      | Ego subgraph traversal depth                      |
| `graph_min_confidence`    | float  | `0.6`                    | Min confidence for graph context entities         |
| `graph_max_relationships` | int    | `20`                     | Max relationships in graph context                |

**API keys** (set in `.env`):

- `VAULTMIND_ANTHROPIC_API_KEY` — Anthropic
- `VAULTMIND_OPENAI_API_KEY` — OpenAI (also used for embeddings)
- `VAULTMIND_GEMINI_API_KEY` — Google Gemini
- `VAULTMIND_VOYAGE_API_KEY` — Voyage (embeddings only)

Ollama requires no API key.

## `[telegram]`

Telegram bot behavior.

| Key                    | Type | Default | Description                        |
| ---------------------- | ---- | ------- | ---------------------------------- |
| `thinking_session_ttl` | int  | `3600`  | Thinking session timeout (seconds) |

**Secrets** (set in `.env`):

- `VAULTMIND_TELEGRAM__BOT_TOKEN` — Bot token from @BotFather
- `VAULTMIND_TELEGRAM__ALLOWED_USER_IDS` — JSON user ID list, e.g. `[123,456]` (empty = allow all)

### Proactive Notifications

```toml
[telegram]
notification_chat_id = 0  # Telegram chat ID for proactive messages; 0 = disabled
```

When set to a non-zero chat ID, scheduled loop jobs can send proactive notifications to this chat when they detect significant changes (new search trends, belief drift, workflow patterns). The chat ID is the numeric Telegram chat identifier.

## `[routing]`

Message classification and routing behavior.

| Key                     | Type   | Default | Description                                              |
| ----------------------- | ------ | ------- | -------------------------------------------------------- |
| `chat_model`            | string | `""`    | Model for chat responses (empty = `llm.fast_model`)      |
| `chat_max_tokens`       | int    | `1024`  | Max tokens for chat responses                            |
| `vault_context_enabled` | bool   | `true`  | Include vault search results in conversational responses |
| `max_context_chunks`    | int    | `4`     | Max chunks for context augmentation                      |
| `capture_all`           | bool   | `false` | Bypass routing — all text becomes a capture              |

## `[watch]`

File watcher behavior for incremental indexing.

| Key                            | Type | Default | Description                                                  |
| ------------------------------ | ---- | ------- | ------------------------------------------------------------ |
| `debounce_ms`                  | int  | `500`   | Debounce window for rapid saves (ms)                         |
| `hash_stability_check`         | bool | `true`  | Two-stage hash check to avoid partial writes                 |
| `reextract_graph`              | bool | `false` | Re-extract graph entities on file change (LLM call per note) |
| `batch_graph_interval_seconds` | int  | `300`   | Batch window for graph re-extraction when enabled            |

## `[duplicate_detection]`

Semantic duplicate detection thresholds.

| Key                  | Type | Default | Description                          |
| -------------------- | ---- | ------- | ------------------------------------ |
| `enabled`            | bool | `true`  | Enable automatic duplicate detection |
| `min_content_length` | int  | `100`   | Skip notes shorter than this         |

Similarity bands: duplicate (>= 92%), merge candidate (80-92%).

## `[note_suggestions]`

Link suggestion scoring.

| Key                  | Type  | Default | Description                           |
| -------------------- | ----- | ------- | ------------------------------------- |
| `enabled`            | bool  | `true`  | Enable note suggestions               |
| `min_content_length` | int   | `100`   | Skip notes shorter than this          |
| `entity_weight`      | float | `0.1`   | Score boost per shared graph entity   |
| `graph_weight`       | float | `0.05`  | Score boost from graph path proximity |

## `[search]`

Search pagination, session management, and hybrid retrieval.

| Key              | Type   | Default | Description                                            |
| ---------------- | ------ | ------- | ------------------------------------------------------ |
| `page_size`      | int    | `5`     | Results per page                                       |
| `max_results`    | int    | `25`    | Maximum total results                                  |
| `session_ttl`    | int    | `300`   | Paginated session expiry (seconds)                     |
| `hybrid_enabled` | bool   | `true`  | Enable BM25 + vector hybrid search with RRF            |
| `bm25_db_path`   | string | `""`    | BM25 SQLite path (empty = `~/.vaultmind/data/bm25.db`) |

## `[ranking]`

Post-retrieval ranking with note-type multipliers, mode multipliers, activation scoring, temporal decay, and source-authority weighting.

| Key                 | Type            | Default                                     | Description                                                              |
| -------------------- | --------------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| `enabled`            | bool            | `true`                                       | Enable post-retrieval ranking                                            |
| `authority_default`  | int             | `3`                                          | Neutral authority level applied to notes with no stamped `authority`      |
| `authority_weight`   | table (int→float) | `{1:0.85, 2:0.92, 3:1.0, 4:1.08, 5:1.15}`   | Multiplier applied to `/recall`/`vaultmind bench` results by note `authority` (1-5, stamped at creation: authored=5, distilled=4, research=3, web clip=2, auto-ingested=1) |

```toml
[ranking.authority_weight]
1 = 0.85
2 = 0.92
3 = 1.0
4 = 1.08
5 = 1.15
```

## `[bench]`

Retrieval self-benchmark (`vaultmind bench`) — deterministic recall@k/MRR scoring of the live `/recall` path against a golden question set. See [`benchmarks/golden.yaml`](../benchmarks/golden.yaml) for the schema.

| Key                         | Type   | Default                  | Description                                                     |
| ---------------------------- | ------ | ------------------------- | ----------------------------------------------------------------- |
| `enabled`                    | bool   | `true`                    | Enable the bench subsystem                                       |
| `golden_path`                | string | `benchmarks/golden.yaml`  | Path to the golden question set                                  |
| `k`                          | int    | `5`                       | Top-k for retrieval scoring                                      |
| `score_floor`                | float  | `0.5`                     | Distance threshold above which a hit counts as "not confidently relevant" (also used by M5's gap-minting for weak-retrieval misses) |
| `recall_at_k_threshold`      | float  | `0.8`                     | Minimum recall@k to pass                                         |
| `mrr_threshold`              | float  | `0.6`                     | Minimum MRR to pass                                               |
| `retrieval_decline_threshold`| float  | `0.8`                     | Minimum honest-decline accuracy for `answerable: false` questions |
| `llm_decline_threshold`      | float  | `0.8`                     | Minimum `--llm` cite-or-decline accuracy                          |
| `llm_model`                  | string | `""`                      | Model for `--llm` scoring (empty = `llm.fast_model`)              |
| `trend_path`                 | string | `""`                      | JSONL trend record path (empty = `~/.vaultmind/data/bench/trend.jsonl`) |

## `[activation]`

Activation-based note decay — notes that are accessed, edited, or referenced stay warm. Unused notes decay.

| Key              | Type   | Default | Description                                              |
| ---------------- | ------ | ------- | -------------------------------------------------------- |
| `enabled`        | bool   | `true`  | Enable activation tracking                               |
| `half_life_days` | float  | `14.0`  | Exponential decay half-life (days)                       |
| `db_path`        | string | `""`    | SQLite path (empty = `~/.vaultmind/data/activations.db`) |

## `[maturation]`

Zettelkasten maturation pipeline — clusters fleeting/literature notes for synthesis into permanent notes.

The bot currently registers maturation discovery at a seven-day interval using persistent scheduler state. `schedule_day`, `schedule_hour`, and `timezone` are parsed but not consumed by that assembly; they do not set its calendar schedule. See the [maturation guide](zettelkasten-maturation.md#scheduling) for scheduled review and explicit retry behavior.

| Key                             | Type   | Default                      | Description                                        |
| ------------------------------- | ------ | ---------------------------- | -------------------------------------------------- |
| `enabled`                       | bool   | `true`                       | Enable maturation pipeline                         |
| `schedule_day`                  | string | `sunday`                     | Reserved calendar day; not consumed by the bot job |
| `schedule_hour`                 | int    | `9`                          | Reserved calendar hour; not consumed by the bot job |
| `timezone`                      | string | `UTC`                        | Reserved calendar timezone; not consumed by the bot job |
| `min_cluster_size`              | int    | `3`                          | Minimum notes per cluster                          |
| `max_clusters_per_digest`       | int    | `3`                          | Max clusters per maturation digest                 |
| `cluster_eps`                   | float  | `0.25`                       | DBSCAN epsilon (distance threshold)                |
| `synthesis_max_tokens`          | int    | `1500`                       | Max tokens for synthesis output                    |
| `synthesis_model`               | string | `""`                         | Model for synthesis (empty = `llm.thinking_model`) |
| `target_note_types`             | list   | `["fleeting", "literature"]` | Note types eligible for clustering                 |
| `dismissed_cluster_expiry_days` | int    | `90`                         | Days before dismissed clusters resurface           |
| `inbox_folder`                  | string | `00-inbox`                   | Folder to check for maturation candidates          |

## `[evolution]`

Belief evolution tracking — detects confidence drift, relationship shifts, and stale claims.

| Key                          | Type  | Default | Description                                          |
| ---------------------------- | ----- | ------- | ---------------------------------------------------- |
| `enabled`                    | bool  | `true`  | Enable evolution tracking                            |
| `confidence_drift_threshold` | float | `0.3`   | Minimum drift to flag                                |
| `stale_days`                 | int   | `180`   | Days before a high-confidence claim is flagged stale |
| `min_confidence_for_stale`   | float | `0.8`   | Minimum confidence to consider for staleness         |
| `max_results`                | int   | `10`    | Max evolution signals per query                      |
| `include_in_digest`          | bool  | `true`  | Include evolution signals in weekly digest           |

## `[digest]`

Weekly digest generation.

| Key                         | Type   | Default    | Description                             |
| --------------------------- | ------ | ---------- | --------------------------------------- |
| `enabled`                   | bool   | `true`     | Enable weekly digest                    |
| `period_days`               | int    | `7`        | Digest period (days)                    |
| `schedule_hour`             | int    | `8`        | Hour (UTC) for scheduled digest         |
| `timezone`                  | string | `UTC`      | Timezone                                |
| `save_to_vault`             | bool   | `true`     | Save digest as a vault note             |
| `send_telegram`             | bool   | `true`     | Send digest via Telegram                |
| `max_trending`              | int    | `10`       | Max trending topics                     |
| `max_suggestions`           | int    | `5`        | Max suggestions per digest              |
| `connection_threshold_low`  | float  | `0.70`     | Lower bound for connection suggestions  |
| `connection_threshold_high` | float  | `0.85`     | Upper bound for connection suggestions  |
| `inbox_folder`              | string | `00-inbox` | Folder to scan for inbox triage         |
| `inbox_age_warning_days`    | int    | `7`        | Age threshold to flag stale inbox notes |
| `max_inbox_shown`           | int    | `10`       | Max inbox notes shown in triage section |

## `[auto_tag]`

LLM-based automatic tag suggestions.

| Key                  | Type   | Default | Description                                  |
| -------------------- | ------ | ------- | -------------------------------------------- |
| `enabled`            | bool   | `true`  | Enable auto-tagging                          |
| `max_tags_per_note`  | int    | `2`     | Max tags suggested per note                  |
| `min_content_length` | int    | `100`   | Skip notes shorter than this                 |
| `tagging_model`      | string | `""`    | Model for tagging (empty = `llm.fast_model`) |

New tags are quarantined at `~/.vaultmind/data/tag_quarantine.json` for user approval.

## `[voice]`

Voice message transcription.

| Key             | Type   | Default     | Description                         |
| --------------- | ------ | ----------- | ----------------------------------- |
| `enabled`       | bool   | `true`      | Enable voice transcription          |
| `whisper_model` | string | `whisper-1` | OpenAI Whisper model                |
| `language`      | string | `""`        | Language hint (empty = auto-detect) |

Requires `VAULTMIND_OPENAI_API_KEY` in `.env`.

## `[ingest]`

URL ingestion settings (YouTube transcripts, article content).

| Key                  | Type   | Default    | Description                      |
| -------------------- | ------ | ---------- | -------------------------------- |
| `enabled`            | bool   | `true`     | Enable URL ingestion             |
| `youtube_language`   | string | `en`       | Preferred transcript language    |
| `max_content_length` | int    | `100000`   | Max content length (characters)  |
| `inbox_folder`       | string | `00-inbox` | Target folder for ingested notes |

## `[research]`

Research pipeline settings (`vaultmind research` command).

| Key                | Type   | Default    | Description                        |
| ------------------ | ------ | ---------- | ---------------------------------- |
| `max_results`      | int    | `5`        | Max YouTube results per query      |
| `output_folder`    | string | `research` | Vault subfolder for research notes |
| `youtube_language` | string | `en`       | Preferred transcript language      |

## `[distill]`

Conversation distillation — finished thinking-partner sessions and the manual `/distill` command become `qa-artifact` notes. **Disabled by default.**

| Key             | Type   | Default        | Description                                                    |
| ---------------- | ------ | -------------- | ---------------------------------------------------------------- |
| `enabled`        | bool   | `false`        | Enable automatic session-end distillation (`/distill` always works) |
| `min_turns`      | int    | `3`            | Minimum session turns before auto-distillation triggers          |
| `model`          | string | `""`           | Model for distillation (empty = `llm.thinking_model`)            |
| `output_folder`  | string | `qa-artifacts` | Vault subfolder for `qa-artifact` notes                          |
| `max_tokens`     | int    | `800`          | Max tokens for the distillation LLM call                         |

## `[tracking]`

User preference and usage tracking.

| Key       | Type   | Default | Description                                              |
| --------- | ------ | ------- | -------------------------------------------------------- |
| `enabled` | bool   | `true`  | Enable interaction tracking                              |
| `db_path` | string | `""`    | SQLite path (empty = `~/.vaultmind/data/preferences.db`) |

## `[image]`

Image/photo capture via Telegram bot.

| Key                    | Type   | Default    | Description                                                   |
| ---------------------- | ------ | ---------- | ------------------------------------------------------------- |
| `enabled`              | bool   | `true`     | Enable photo capture                                          |
| `vision_model`         | string | `""`       | Vision model for image description (empty = `llm.fast_model`) |
| `max_image_size_bytes` | int    | `10000000` | Maximum image size (bytes)                                    |
| `save_originals`       | bool   | `true`     | Save original images to vault                                 |

## `[episodic]`

Episodic memory — decision-outcome tracking.

| Key                | Type   | Default | Description                                           |
| ------------------ | ------ | ------- | ----------------------------------------------------- |
| `enabled`          | bool   | `true`  | Enable episodic memory                                |
| `db_path`          | string | `""`    | SQLite path (empty = `~/.vaultmind/data/episodes.db`) |
| `auto_extract`     | bool   | `false` | LLM extraction of episodes from notes (opt-in)        |
| `extraction_model` | string | `""`    | Model for extraction (empty = `llm.fast_model`)       |

## `[gaps]`

Knowledge gap ledger — unanswered questions, weak-retrieval misses, and contradiction escalations recorded as deduplicated, lifecycle-tracked gaps instead of discarded.

| Key               | Type   | Default | Description                                                          |
| ------------------ | ------ | ------- | ------------------------------------------------------------------ |
| `enabled`          | bool   | `true`  | Enable the knowledge gap ledger                                    |
| `db_path`          | string | `""`    | SQLite path (empty = `~/.vaultmind/data/gaps.db`)                  |
| `stale_after_days` | int    | `30`    | Days an open gap sits untouched before auto-transitioning to `stale` |
| `max_shown`        | int    | `10`    | Max gaps shown by `/gaps` and the digest gap section                |

## `[contradiction]`

Contradiction detection + deterministic resolution for notes landing in the duplicate-detector's 80-92% merge band. **Escalate-only by default** — flip `auto_resolve` only after `vaultmind eval contradict` shows the detector beating a trivial always-escalate baseline on your own vault (Kosha's own real-model Gate-0 run recorded contradiction routing accuracy 0.17 vs. a 1.00 prompt-only baseline).

| Key                | Type   | Default                              | Description                                                  |
| -------------------- | ------ | -------------------------------------- | ---------------------------------------------------------- |
| `enabled`            | bool   | `true`                                 | Enable contradiction detection                             |
| `auto_resolve`       | bool   | `false`                                | Apply the deterministic policy automatically instead of always escalating |
| `detection_model`    | string | `""`                                    | Model for LLM conflict detection (empty = `llm.fast_model`) |
| `max_tokens`         | int    | `300`                                   | Max tokens for the detection LLM call                       |
| `eval_path`          | string | `benchmarks/contradict_eval.yaml`       | Labeled eval set for `vaultmind eval contradict`             |

## `[autonomy]`

Graduated autonomy review queue — every automated mutation proposal (tag application, duplicate-merge suggestion, contradiction resolution, maturation synthesis) routes through one `AUTO`/`SKIM`/`BLOCK` lane by confidence × impact. Thresholds mirror Kosha's own `AutonomyThresholds` defaults.

| Key           | Type   | Default | Description                                                      |
| -------------- | ------ | ------- | ------------------------------------------------------------------ |
| `enabled`      | bool   | `true`  | Enable the review queue                                          |
| `db_path`      | string | `""`    | SQLite path (empty = `~/.vaultmind/data/review_queue.db`)        |
| `block_below`  | float  | `0.4`   | Confidence below this always routes to `BLOCK`                    |
| `skim_below`   | float  | `0.9`   | Confidence below this (and impact not `HIGH`) routes to `SKIM`; at or above auto-applies |
| `force_block`  | bool   | `false` | Force every proposal to `BLOCK`, bypassing confidence routing     |

## `[sources]`

Explicit-registry source connector framework (`rss`, `youtube-channel`, `github-activity`) with durable per-instance cursors. Per-instance connector definitions (name, kind, target, schedule, `enabled`) live in `config/sources.toml`, a separate file — see `vaultmind source list|status|run`.

| Key           | Type   | Default                | Description                                             |
| -------------- | ------ | ----------------------- | --------------------------------------------------------- |
| `enabled`      | bool   | `true`                  | Global toggle for the source connector framework          |
| `db_path`      | string | `""`                    | SQLite path (empty = `~/.vaultmind/data/sources.db`)      |
| `config_path`  | string | `""`                    | Path to the instance-definition file (empty = `config/sources.toml`) |

Every ingested item is stamped `authority=1` (auto-ingested tier), passed through M4-style distillation, evaluated by the existing dedup/contradiction event-bus path, and any auto-applying proposal still routes through `[autonomy]`'s review queue. Each connector instance defaults `enabled: false` in `config/sources.toml`.

## `[procedural]`

Procedural memory — self-evolving workflow patterns mined from episodic memory. **Experimental, disabled by default.**

| Key                        | Type   | Default | Description                                             |
| -------------------------- | ------ | ------- | ------------------------------------------------------- |
| `enabled`                  | bool   | `false` | Enable procedural memory (experimental)                 |
| `db_path`                  | string | `""`    | SQLite path (empty = `~/.vaultmind/data/procedural.db`) |
| `min_episodes_for_pattern` | int    | `3`     | Minimum resolved episodes to form a workflow            |
| `synthesis_model`          | string | `""`    | Model for workflow synthesis (empty = `llm.fast_model`) |

## `[consolidation]`

Parsed configuration only in the shipped assemblies. [`MemoryConsolidator`](../src/vaultmind/memory/consolidation.py) can be constructed explicitly by library callers; `enabled = true` does not register it with the bot scheduler, and `schedule` does not schedule a job.

| Key | Type | Default | Declared intent |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Reserve opt-in consolidation |
| `retention_days` | int | `365` | Episode archival age |
| `min_references_for_promotion` | int | `3` | Promotion reference threshold |
| `schedule` | string | `"0 0 1 * *"` | Monthly cron expression; not consumed by the shipped scheduler |

## Compound Loops

```toml
[loops]
insight_enabled = true        # Usage pattern analysis loop
insight_interval_days = 7
insight_schedule = ""         # Cron expression; empty derives it from interval_days
evolution_enabled = true       # Belief evolution tracking loop
evolution_interval_days = 7
evolution_schedule = ""
procedural_enabled = false     # Workflow synthesis loop (requires [procedural].enabled)
procedural_interval_days = 7
procedural_schedule = ""
```

Per-loop enable flags, cron expressions, and fallback intervals are consumed by the bot assembly. A nonempty schedule selects its cron expression; an empty schedule derives one from `interval_days`. The procedural loop additionally requires `[procedural].enabled = true`. All loops are state-aware — each run reads and extends its prior-run state, enabling cross-cycle pattern recognition.

## `[mcp]`

MCP server settings and profile definitions.

| Key       | Type   | Default     | Description       |
| --------- | ------ | ----------- | ----------------- |
| `enabled` | bool   | `false`     | Enable MCP server |
| `host`    | string | `127.0.0.1` | Bind address      |
| `port`    | int    | `8765`      | Bind port         |

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
allowed_tools = ["vault_search", "vault_read", "vault_write", "vault_list", "graph_query", "graph_path", "capture", "capture_note"]
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

Profile values are validated during settings loading, before the MCP server starts. `allowed_tools` and `folder_scope` must be lists of strings; nested environment/`.env` overrides use JSON arrays, for example `VAULTMIND_MCP__PROFILES__PLANNER__FOLDER_SCOPE='["00-inbox"]'`. `write_enabled` and `requires_confirmation` are booleans: environment/`.env` values such as `false` and `true` are parsed as booleans, not string truthiness. Invalid booleans, lists, and note-size limits fail validation. `max_note_size_bytes` is a nonnegative integer, or `None` through programmatic configuration for no size limit; omitting it also means no limit.

The environment example narrows the scope of a planner already defined in TOML. Without a `[mcp.profiles.planner]` table, that variable alone creates a custom `planner` with the defaults below: it allows no tools and replaces, rather than inherits, the built-in planner policy.

A configured profile replaces the same-named built-in profile rather than inheriting its policy. Omitted profile fields retain the established custom-profile defaults: empty description, no allowed tools, folder scope `["*"]`, writes disabled, no note-size limit, and confirmation disabled. Built-in researcher, planner, and full policies remain unchanged when no same-named profile is configured.

## `[health]`

Parsed configuration only in the shipped assemblies. [`HealthMonitor`](../src/vaultmind/services/health.py) is available for explicit library construction, but the bot does not create or periodically run it from this section. The on-demand `/health` command reports vector-store, graph, vault, configured LLM-key, and graph-file status independently; it does not test live provider authentication or consume these flags.

| Key | Type | Default | Declared intent |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Reserve periodic health monitoring |
| `check_interval_seconds` | int | `60` | Periodic check interval |
| `check_chromadb` | bool | `true` | Vector-store checks |
| `check_sqlite` | bool | `true` | SQLite checks |
| `check_graph_file` | bool | `true` | Graph-file checks |
| `check_llm` | bool | `true` | LLM checks |
| `check_bot` | bool | `true` | Bot checks |
| `check_vault_access` | bool | `true` | Vault-access checks |
| `chromadb_latency_warn_ms` | int | `1000` | Vector-store latency warning threshold |
| `retention_days` | int | `30` | Health-history retention |

## `[mcp_retry]`

Parsed configuration only in the shipped CLI. `mcp-serve` does not construct or pass [`ToolRetryExecutor`](../src/vaultmind/mcp/retry.py), so these settings do not enable automatic tool retries or LLM correction. Library callers can explicitly supply an executor to `create_mcp_server`; tool authorization and result-scope policy remain separate requirements.

| Key | Type | Default | Declared intent |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Reserve MCP correction/retry behavior |
| `max_retries` | int | `1` | Intended retry limit |
| `use_llm_correction` | bool | `true` | Permit correction when an executor and LLM are supplied |
| `correction_model` | string | `""` | Intended correction-model selection |
| `timeout_seconds` | int | `30` | Intended execution timeout |
| `retryable_errors` | list[string] | `["ValueError", "LLMError", "TimeoutError", "ConnectionError"]` | Error class names considered for correction |

The standalone executor performs at most one corrected dispatch. It does not consume `max_retries`, `timeout_seconds`, or `correction_model`; the caller supplies its `llm_model` separately. These fields are not enforced retry, timeout, or model-selection controls.
