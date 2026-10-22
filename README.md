[![VaultMind](assets/banner.png)](https://github.com/Mathews-Tom/VaultMind)

---

# VaultMind

Turns your Obsidian vault into an intelligent second brain — hybrid search
(vector + BM25), knowledge graph, Telegram bot with photo capture, AI thinking
partner, Zettelkasten maturation pipeline, belief evolution tracking, episodic
and procedural memory, MCP integration for connecting Claude and other
agents directly to your notes, compound loop engine for proactive insights,
conversation distillation into structured Q&A notes, a knowledge gap ledger,
contradiction detection with deterministic resolution, graduated autonomy
review lanes, background source connectors, and non-destructive edit lineage.

## Why VaultMind?

- **Obsidian is the source of truth** — plain markdown files, always
- **AI augments, never replaces** — hybrid search (vector + BM25) and knowledge graphs surface what you'd miss
- **Mobile-first capture** — Telegram bot gives full PKM access from your phone, including photo capture via vision models
- **Agent-native** — Claude Desktop/Code and other AI agents read/write your vault via MCP
- **Multi-provider LLM** — Anthropic, OpenAI, Gemini, or Ollama for thinking + extraction
- **Hybrid retrieval** — Reciprocal Rank Fusion merges vector (ChromaDB) and keyword (SQLite FTS5) search results
- **Contextual embeddings** — chunks carry document-level context (title, type, section, tags) for higher-quality vectors
- **Zettelkasten maturation** — DBSCAN clustering surfaces fleeting notes ready for permanent synthesis
- **Belief evolution** — tracks confidence drift, relationship shifts, and stale claims across your knowledge
- **Episodic memory** — decision-outcome tracking with lessons learned and entity linking
- **Procedural memory** — mines recurring decision patterns into reusable workflows (experimental)
- **Compound loops** — scheduler accumulates state across runs, detects usage shifts, belief drift trends, and new workflow patterns; pushes insights to Telegram proactively
- **Provenance-aware ranking** — notes carry a stamped `authority` (1-5) at creation; `/recall` reweights results by source trust, backfilling a neutral default for older notes
- **Retrieval self-benchmark** — `vaultmind bench` scores recall@k/MRR (and optionally cite-or-decline accuracy) against a golden question set, gating ranking/detector changes before they ship
- **Conversation distillation** — finished thinking-partner sessions and `/distill` requests become `qa-artifact` notes (question, resolution, systems, participants), indexed immediately and mined for episodic memory
- **Knowledge gap ledger** — unanswered questions, weak-retrieval misses, and contradiction escalations are deduplicated into trackable gaps (`/gaps`), auto-staled, and closed by running `research` on them
- **Contradiction detection** — new notes landing in the duplicate-detector's merge band are checked by an eval-gated LLM surface; conflicts resolve by temporal-then-authority-then-escalate policy and never edit or delete the losing note
- **Graduated autonomy** — every automated mutation proposal (tags, merges, contradiction resolutions, maturation synthesis) routes through one AUTO/SKIM/BLOCK review queue with an approval-fatigue metric in `vaultmind learn`
- **Source connectors** — scheduled `rss`, `youtube-channel`, and `github-activity` connectors ingest with durable cursors, run every item through distillation and the review queue, never bypassing governance
- **Non-destructive edits** — bot-initiated `/delete`/`/edit` write a dated `> [!superseded]` callout and a lineage edge instead of deleting text, preserving full edit history in the note
- **OKF export** — `vaultmind export-okf` exports permanent/concept notes as a conformant [OKF](https://github.com/Mathews-Tom/Kosha) bundle for governed serving from Kosha
- **Note modes** — learning vs. operational mode with activation-based decay scoring
- **Secure by default** — path traversal protection, input sanitization, injection detection on all vault operations
- **Smart caching** — SQLite-backed embedding cache eliminates redundant API calls during re-indexing
- **Your data, your infra** — self-hosted, local-first, no cloud dependency

**Delivery limits:** Non-destructive edit history is implemented, but consistent exclusion of retired or superseded text after full/watch reindexing and the original native-watcher notification failure remain unresolved; [PR96](https://github.com/Mathews-Tom/VaultMind/pull/96) is held. SQLite/Chroma batching does not provide cross-index atomicity. Offline fixture checks are not native-provider, private-vault, or real Telegram qualification.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- One of: Anthropic / OpenAI / Gemini API key, or local Ollama
- OpenAI or Voyage API key (for embeddings)
- Telegram bot token (from [@BotFather](https://t.me/botfather))

### 1. Clone and install

```bash
git clone https://github.com/Mathews-Tom/vaultmind.git
cd vaultmind
uv sync --extra dev
```

This installs VaultMind into a local virtualenv. All commands run through `uv run`.

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# LLM provider key (set the one matching your provider choice)
VAULTMIND_ANTHROPIC_API_KEY=sk-ant-...
# VAULTMIND_OPENAI_API_KEY=sk-...
# VAULTMIND_GEMINI_API_KEY=...

# Embedding key (required for default OpenAI embeddings)
VAULTMIND_OPENAI_API_KEY=sk-...
# VAULTMIND_VOYAGE_API_KEY=pa-...   # If using Voyage embeddings

# Telegram bot
VAULTMIND_TELEGRAM__BOT_TOKEN=123456:ABC-DEF...
VAULTMIND_TELEGRAM__ALLOWED_USER_IDS=[123456789]  # Optional whitelist
```

### 3. Configure settings

Edit `config/default.toml`:

```toml
[llm]
provider = "anthropic"        # "anthropic", "openai", "gemini", "ollama"
thinking_model = "claude-sonnet-4-20250514"

[vault]
path = "~/.vaultmind/vault"   # Or path to your existing Obsidian vault

[telegram]
thinking_session_ttl = 3600
```

See [`config/default.toml`](config/default.toml) for all options.

### 4. Initialize

```bash
uv run vaultmind init
```

Creates the `~/.vaultmind/` directory structure:

```text
~/.vaultmind/
├── vault/
│   ├── 00-inbox/
│   ├── 01-daily/
│   ├── 02-projects/
│   ├── 03-areas/
│   ├── 04-resources/
│   ├── 05-archive/
│   ├── 06-templates/
│   └── _meta/
└── data/
    ├── chromadb/
    ├── bm25.db
    ├── embedding_cache.db
    ├── knowledge_graph.json
    ├── activations.db
    ├── episodes.db
    ├── preferences.db
    └── sessions.db
```

### 5. Index your vault

```bash
uv run vaultmind index
```

Parses all markdown, chunks by headings, generates embeddings, stores in ChromaDB.

### 6. Build the knowledge graph

```bash
uv run vaultmind graph-build
```

Uses your configured LLM to extract entities and relationships from notes.
Add `--full` to rebuild from scratch.

### 7. Start the Telegram bot

```bash
uv run vaultmind bot
```

The bot starts with watch mode enabled — vault changes are indexed incrementally.

## CLI Commands

| Command                              | Description                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `vaultmind init`                     | Create vault folder structure                                           |
| `vaultmind index`                    | Full vault index into ChromaDB                                          |
| `vaultmind watch`                    | Watch vault, incrementally re-index on change                           |
| `vaultmind watch --graph`            | Watch + batched graph re-extraction                                     |
| `vaultmind bot`                      | Start Telegram bot (includes watch + duplicate detection + suggestions) |
| `vaultmind graph-build`              | Build knowledge graph (add `--full` to rebuild)                         |
| `vaultmind graph-maintain`           | Prune stale refs + remove orphan entities                               |
| `vaultmind graph-report`             | Generate graph analytics report                                         |
| `vaultmind scan-duplicates`          | Scan vault for duplicate/merge candidates                               |
| `vaultmind suggest-links`            | Suggest links between notes                                             |
| `vaultmind digest`                   | Generate weekly digest (add `--save` to write to vault)                 |
| `vaultmind auto-tag`                 | LLM-based tag suggestions (dry-run by default)                          |
| `vaultmind auto-tag --apply`         | Apply suggested tags to frontmatter                                     |
| `vaultmind research "query"`         | Search YouTube, analyze transcripts, create vault notes                 |
| `vaultmind learn`                    | Analyze usage patterns, generate insights report                        |
| `vaultmind learn --save`             | Save insights report to vault                                           |
| `vaultmind tag-synonyms`             | Detect likely tag synonyms and suggest merges                           |
| `vaultmind synthesize-workflows`     | Mine episodic memory for workflow patterns                              |
| `vaultmind stats`                    | Show vault + graph statistics                                           |
| `vaultmind stats --metadata-audit`   | Audit frontmatter completeness                                          |
| `vaultmind mcp-serve`                | Start MCP server (default profile: researcher)                          |
| `vaultmind mcp-serve --profile full` | Start MCP with full access                                              |
| `vaultmind bench`                    | Score `/recall` against `benchmarks/golden.yaml` (`--llm` for cite-or-decline) |
| `vaultmind eval contradict`          | Score the contradiction detector vs. an always-escalate baseline        |
| `vaultmind source list`              | List configured connector instances and cursor state                    |
| `vaultmind source status [name]`     | Show cursor state + recent run history                                  |
| `vaultmind source run <name>`        | Run one connector instance once, outside the scheduler                  |
| `vaultmind export-okf`               | Export permanent/concept notes as an OKF bundle for Kosha                |

## LLM Providers

Set `[llm].provider` in `config/default.toml` and the matching API key in `.env`.

| Provider  | Config                   | Model examples                                       |
| --------- | ------------------------ | ---------------------------------------------------- |
| Anthropic | `provider = "anthropic"` | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| OpenAI    | `provider = "openai"`    | `gpt-4.1`, `gpt-4.1-mini`                            |
| Gemini    | `provider = "gemini"`    | `gemini-2.5-flash`, `gemini-2.5-pro`                 |
| Ollama    | `provider = "ollama"`    | `llama3.3`, `qwen3`, `deepseek-r1`                   |

Ollama requires no API key. Set `ollama_base_url` if not running on `localhost:11434`.

## Telegram Bot

### Smart Message Routing

Plain text messages are classified automatically — no command prefix needed:

| You send                    | What happens                                               |
| --------------------------- | ---------------------------------------------------------- |
| `note: buy groceries`       | Captured to inbox (prefix stripped)                        |
| `save: meeting notes...`    | Captured to inbox (prefix stripped)                        |
| Multiline paste (3+ lines)  | Captured to inbox (pasted content = intentional)           |
| Long text (500+ chars)      | Captured to inbox                                          |
| "Hi", "thanks", "ok"        | Static greeting response (no LLM call)                     |
| "What did I write about X?" | Vault-context-aware answer via LLM                         |
| "Tell me about my projects" | Conversational response with vault context                 |
| Follow-up after `/think`    | Continues thinking session (sticky)                        |
| URL in message              | Auto-ingests YouTube transcript or article content         |
| Photo/image                 | Described via vision model, saved as note with image embed |

Capture prefixes: `note:`, `save:`, `capture:`, `remember:`, `jot:`, `log:`

Set `capture_all = true` in `[routing]` config to restore old behavior (all text → capture).

### Commands

| Command                         | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| `/recall <query>`               | Semantic search over vault (paginated)                          |
| `/think <topic>`                | Start thinking partner session (persists across restarts)       |
| `/think explore: <topic>`       | Divergent ideation mode                                         |
| `/think critique: <topic>`      | Stress-test an idea                                             |
| `/think synthesize: <topic>`    | Connect dots across domains                                     |
| `/think plan: <topic>`          | Create execution plan                                           |
| `/graph <entity>`               | Query knowledge graph connections                               |
| `/daily`                        | Get/create today's daily note                                   |
| `/notes <date>`                 | Find notes by date (natural language: `yesterday`, `last week`) |
| `/read <note>`                  | Read full note content                                          |
| `/edit <note> <instruction>`    | AI-assisted edit with confirmation                              |
| `/delete <note>`                | Delete note with confirmation                                   |
| `/bookmark <title>`             | Save thinking session or last Q&A to vault                      |
| `/suggest <note>`               | Find notes worth linking (composite scoring)                    |
| `/duplicates <note>`            | Find duplicate/similar notes                                    |
| `/review`                       | Weekly review, pending SKIM/BLOCK decisions, and failed-operation recovery |
| `/evolve`                       | Belief evolution signals (confidence drift, stale claims)       |
| `/mature`                       | Zettelkasten maturation — clusters ready for synthesis          |
| `/decide <decision>`            | Record a decision (creates pending episode)                     |
| `/outcome <id> <status> <desc>` | Resolve a decision with outcome and lessons                     |
| `/episodes [entity]`            | List episodes, optionally filtered by entity                    |
| `/workflows`                    | List active workflow patterns with success rates                |
| `/workflow <id>`                | Show workflow steps and details                                 |
| `/health`                       | On-demand component/configuration status, not live provider qualification |
| `/stats`                        | Vault and graph statistics                                      |
| `/help`                         | Quick reference for registered bot commands                     |
| `/distill`                      | Distill the current thinking session into a `qa-artifact` note  |
| `/gaps`                         | List open knowledge gaps by age                                 |
| Send voice message              | Transcribe via Whisper and route as capture or question         |
| Send photo                      | Describe via vision model and capture as note with image embed  |

`/review` lists pending SKIM items for approve-all, pending BLOCK items for an individual decision, and failed operations for explicit retry. Review output uses plain text and bounded message chunks; each item's actions accompany its own text, including when long summaries require multiple messages. Inspect partial side effects before retrying a failed mutation. Listing the queue does not apply or retry operations.

## MCP Integration

For Claude Desktop, add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vaultmind": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/vaultmind",
        "vaultmind",
        "mcp-serve"
      ]
    }
  }
}
```

Install the MCP extra first:

```bash
uv sync --extra mcp
```

### MCP Tools

| Tool               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `vault_search`     | Semantic search with optional note type filter                   |
| `vault_read`       | Read full note by relative path                                  |
| `vault_write`      | Create/overwrite note + auto-reindex                             |
| `vault_list`       | List folder contents with optional tag/type filter               |
| `graph_query`      | Entity neighbors and relationships (configurable depth)          |
| `graph_path`       | Shortest path between two entities                               |
| `find_duplicates`  | Semantic duplicate detection (duplicate/merge bands)             |
| `suggest_links`    | Note suggestions (similarity + shared entities + graph distance) |
| `capture`          | Quick-capture to inbox with optional title and tags              |
| `capture_note`     | Rich capture with note type, tags, and target folder             |
| `vault_stats`      | Vault health metrics: note counts by type/folder, graph size     |
| `episode_query`    | Search episodic memory for past decisions and outcomes           |
| `workflow_suggest` | Find matching procedural workflow for a given context            |
| `graph_evolution`  | Belief evolution signals: confidence drift, relationship shifts  |
| `recent_activity`  | Notes created or modified in the last N days                     |
| `read_frontmatter` | Peek parsed frontmatter (type, tags, dates, authority) — no body  |
| `list_folder_index`| Folder listing with titles + one-line descriptions — no body     |
| `follow_links`     | Wikilink out-links + backlinks for a note — no body               |

### MCP Profiles

Profiles control per-agent access. Activate with `--profile <name>`:

| Profile                | Access                                                          | Folders                   |
| ---------------------- | --------------------------------------------------------------- | ------------------------- |
| `researcher` (default) | Read-only: search, read, list, graph, introspection tools       | All                       |
| `planner`              | Scoped note reads/writes and capture; derived tools restricted | `02-projects`, `00-inbox` |
| `full`                 | Unrestricted (requires explicit opt-in)                         | All                       |

Tool allowlisting does not override result policy. Folder-restricted profiles, including the built-in planner, cannot use `graph_query`, `graph_path`, `vault_stats`, `episode_query`, `workflow_suggest`, `graph_evolution`, or `suggest_links`: their producers cannot prove complete in-scope provenance. Note-backed results are checked before returning content, metadata, and counts. See the [complete MCP result policy](docs/mcp-integration.md#complete-result-policy) for per-tool behavior and read-only/audit boundaries.

## Vault Structure

VaultMind expects (and creates via `init`) this structure:

```text
~/.vaultmind/vault/
├── 00-inbox/          # Quick captures, unsorted
├── 01-daily/          # Daily notes
├── 02-projects/       # Active project notes
├── 03-areas/          # Life areas (health, finance, career)
├── 04-resources/      # Reference material, articles, book notes
├── 05-archive/        # Completed/inactive
├── 06-templates/      # Note templates
└── _meta/             # Auto-generated reports and indexes
```

Notes use YAML frontmatter:

```yaml
---
type: project # fleeting | literature | permanent | daily | project | area | person | concept
mode: operational # learning (default) | operational — affects retrieval ranking
tags: [python, ai]
created: 2026-01-15
entities: [CAIRN, MCP]
status: active
---
```

## Configuration

Precedence, highest priority first: deliberate constructor/call-site overrides → `VAULTMIND_*` environment variables → cwd `.env` → selected TOML → model defaults. `--config PATH` selects TOML; it does not promote TOML above environment variables. Nested overrides use `__`, and list values use JSON arrays. Invalid selected values fail configuration validation rather than falling back.

Keep credentials in `.env` or the process environment, not in TOML. To isolate a run, configure all persistent paths; changing `vault.path` does not relocate ChromaDB, BM25, graph, or other stores.

Sections with runtime consumers: `[vault]`, `[llm]`, `[telegram]`, `[routing]`, `[embedding]`, `[chroma]`, `[graph]`, `[watch]`, `[duplicate_detection]`, `[note_suggestions]`, `[search]`, `[ranking]`, `[bench]`, `[activation]`, `[maturation]`, `[evolution]`, `[digest]`, `[auto_tag]`, `[voice]`, `[ingest]`, `[research]`, `[distill]`, `[tracking]`, `[image]`, `[episodic]`, `[gaps]`, `[contradiction]`, `[autonomy]`, `[sources]`, `[procedural]`, `[mcp]`, `[scheduler]`, `[loops]`. A consumed section does not imply that every declared field is wired.

Additional parsed sections are not wired into the shipped CLI/bot assemblies: `[memory_profile]`, `[proactivity]`, `[heartbeat]`, `[drafts]`, `[integrations]`, `[consolidation]`, `[health]`, `[mcp_retry]`. Enabling them does not activate jobs, outbound policy, periodic health monitoring, or MCP retries. Standalone library components and the bot's on-demand `/health` command do not make these settings operational.

See the [Configuration Reference](docs/configuration.md) for defaults, consumed-versus-declared controls, profile replacement semantics, and isolation requirements.

## Docker

```bash
cp .env.example .env
# Edit .env with your API keys

cd docker
docker compose up -d vaultmind-bot
docker compose --profile indexer run --rm vaultmind-indexer
docker compose --profile mcp up -d   # Optional MCP server on port 8765
```

## Development

```bash
uv sync --extra dev --extra mcp
uv run --extra dev --extra mcp ruff check .
uv run --extra dev --extra mcp ruff format --check .
uv run --extra dev --extra mcp mypy --strict src/ --ignore-missing-imports
uv run --extra dev --extra mcp pytest -v
```

### Retrieval confidence and the acceptance smoke

`/recall`, `vaultmind bench`, and the thinking partner's gap minting share one relevance judgement, `retrieval_confidence(hits, score_floor)` in `indexer/ranking.py`. It reads **only the raw vector distance** (`hit["distance"]`, lower is closer) against `[bench].score_floor` (default `0.5`) and returns one of three states:

| State     | When                                                                                           | Effect                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `strong`  | any hit has a raw distance strictly below the floor                                            | no gap is minted; no decline credit (a strong result is not an honest decline)               |
| `weak`    | no hits at all, or every hit has a raw distance at or above the floor                          | mints a weak-retrieval gap (`/recall`) or unanswered-question gap (thinking); earns decline credit in `vaultmind bench` |
| `unknown` | non-empty results where no hit is below the floor and at least one hit has no raw distance     | **no gap is minted and no decline credit is given**; a hit without a raw distance renders as `relevance unknown` instead of a percentage |

Raw distance is never replaced by a fusion pseudo-distance. In hybrid search a lexical-only (BM25) hit has `distance: None` — it has keyword evidence but no vector evidence — so it can never prove relevance and never proves irrelevance. `rrf_score` (fusion) and `authority_score` (source-trust reweighting) only change *ordering*; they never change a hit's `distance` or the result set's confidence.

`vaultmind bench --bundle <dir>` reads `<dir>/retrieval.yaml`, a mapping of exact golden-question text to an ordered hit list. Each hit needs `note_path` and may carry `distance`, `authority`, `rrf_score`, `content`. A hit whose `distance` is **missing or `null` is explicit unknown evidence** (it is not defaulted to a "good" or "bad" distance), so a bundle can model lexical-only hits and the unknown path deterministically.

`scripts/verify_acceptance.py` is a small real-surface smoke for this behaviour:

```bash
uv run python scripts/verify_acceptance.py --case retrieval --offline
```

It builds a fresh temporary `HOME`, vault, ChromaDB, BM25 index, gap ledger and graph from explicit TOML files, indexes a small corpus through the real parser, then drives the real `/recall` handler, the real bench runner, and the real thinking partner. The lexical-only row is deliberately inserted directly at the BM25 fixture boundary, separately from the parser-indexed notes; it is never added to the vector index. It asserts exact self-match is `strong` in vector and hybrid modes, empty retrieval is `weak`, that lexical-only BM25 chunk is `unknown` (and mints no gap), recall and bench retrieve identical result sets, the floor boundary (`floor - ε` strong, `floor` weak, `None`/missing unknown), and that authority re-ordering changes order but never distance or confidence. It prints a machine-readable JSON receipt (git revision, working-tree dirty flag, package/API-origin/corpus digests, boundary labels, every check's expected and observed value) and exits non-zero on any failed check or error, including under `python -O`.

Smoke limits — what it does **not** prove: embeddings are a deterministic hash-bucket stand-in, the LLM is a scripted client, and Telegram delivery is an in-process recorder, so it demonstrates plumbing and raw-distance semantics, not retrieval quality or live-provider behaviour. The direct BM25 fixture insert tests missing vector evidence, not native-provider quality or parser creation of a lexical-only note. Exact floor-boundary values come from a clearly labeled scripted store kept separate from the real storage path. Thinking retrieves with vector-only search, so its `unknown` state is exercised only through that scripted store. No network or paid calls are made (the run fails if a connect is attempted).

## Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                       User Interfaces                         │
│  Obsidian Desktop │ Obsidian Mobile │ Telegram Bot │ MCP      │
└────────┬──────────┴────────┬────────┴──────┬───────┴──┬───────┘
         │                   │               │          │
         ▼                   ▼               ▼          ▼
┌───────────────────────────────────────────────────────────────┐
│                        VaultMind Core                         │
│                                                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Parser       │  │ Hybrid Search    │  │ Knowledge Graph │  │
│  │ (contextual  │  │ (ChromaDB +      │  │ (nx) + Belief   │  │
│  │  chunks)     │  │  BM25 FTS5 + RRF)│  │   Evolution     │  │
│  └──────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────────────┐   │
│  │ LLM Abstraction      │  │ Maturation Pipeline          │   │
│  │ (Anthropic/OpenAI/   │  │ (DBSCAN clustering →         │   │
│  │  Gemini/Ollama)      │  │  LLM synthesis)              │   │
│  │ + Multimodal (vision)│  └──────────────────────────────┘   │
│  └──────────────────────┘                                     │
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────────────┐   │
│  │ Memory System        │  │ Research Pipeline            │   │
│  │ (Episodic: decisions │  │ (YouTube → vault)            │   │
│  │  + Procedural:       │  ├──────────────────────────────┤   │
│  │    workflows)        │  │ Preference Tracking          │   │
│  └──────────────────────┘  │ + Activation Scoring         │   │
│                            └──────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
         │
         ▼
   ~/.vaultmind/
   ├── vault/     (Obsidian markdown files)
   └── data/      (ChromaDB + BM25 + graph + episodes + activations + sessions)
```

## Tech Stack

| Layer           | Technology                                            |
| --------------- | ----------------------------------------------------- |
| Language        | Python 3.12+                                          |
| LLM             | Anthropic, OpenAI, Gemini, Ollama (provider-agnostic) |
| Embeddings      | OpenAI / Voyage                                       |
| Vector Store    | ChromaDB                                              |
| Knowledge Graph | NetworkX                                              |
| Clustering      | scikit-learn (DBSCAN)                                 |
| Telegram Bot    | aiogram 3.x                                           |
| Agent Protocol  | MCP                                                   |
| CLI             | Click + Rich                                          |
| Config          | Pydantic Settings + TOML                              |
| Packaging       | Hatch + uv                                            |

## Documentation

- [Configuration Reference](docs/configuration.md)
- [Architecture Guide](docs/architecture.md)
- [Telegram Bot Guide](docs/telegram-bot.md)
- [MCP Integration Guide](docs/mcp-integration.md)
- [Knowledge Graph](docs/knowledge-graph.md)
- [Zettelkasten Maturation](docs/zettelkasten-maturation.md)

## License

MIT — see [LICENSE](LICENSE).
