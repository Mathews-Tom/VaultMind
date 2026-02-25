# VaultMind — Project Instructions

## Overview

AI-powered personal knowledge management built on Obsidian. Turns an Obsidian vault into an intelligent second brain with semantic search (ChromaDB), knowledge graph (NetworkX), Telegram bot (aiogram), thinking partner (Claude API), and MCP server for agent integration.

## Architecture

```
src/vaultmind/
├── vault/       # Markdown parser, models, file watcher, event bus, path security
├── indexer/     # Embedding pipeline + ChromaDB vector store + embedding cache + duplicate detection + auto-tagging
├── graph/       # NetworkX knowledge graph + LLM entity extraction + graph maintenance
├── bot/         # Telegram bot (aiogram 3.x) — commands, router, thinking partner
│   ├── commands.py      # Thin facade delegating to handlers/ package
│   ├── handlers/        # Decomposed handler modules (capture, recall, think, etc.)
│   │   ├── context.py   # HandlerContext dataclass shared by all handlers
│   │   ├── utils.py     # Shared utilities (_is_authorized, _split_message, etc.)
│   │   └── *.py         # Individual handler modules (capture, recall, think, graph, etc.)
│   ├── router.py        # Heuristic message classifier (Intent enum)
│   ├── sanitize.py      # Input sanitization (length limits, null bytes, injection detection)
│   ├── session_store.py # SQLite-backed thinking session persistence
│   ├── telegram.py      # aiogram Router, handler registration, callback queries
│   ├── thinking.py      # Multi-turn thinking partner (RAG + graph)
│   └── transcribe.py    # OpenAI Whisper API voice transcription
├── llm/         # Provider-agnostic LLM abstraction (Protocol-based)
│   ├── client.py        # LLMClient Protocol, Message, LLMResponse, factory
│   └── providers/       # Anthropic, OpenAI, Gemini, Ollama implementations
├── mcp/         # MCP server for Claude Desktop/Code integration
├── config.py    # Pydantic Settings — TOML + env vars (incl. WatchConfig, RoutingConfig)
└── cli.py       # Click CLI entry point
```

## Commands

```bash
uv run vaultmind init          # Create vault folder structure
uv run vaultmind index         # Full vault index into ChromaDB
uv run vaultmind watch         # Watch vault, incrementally re-index on change
uv run vaultmind watch --graph # Watch + batched graph re-extraction
uv run vaultmind scan-duplicates # Scan vault for duplicate/merge candidates
uv run vaultmind suggest-links   # Scan vault for link suggestions between notes
uv run vaultmind bot           # Start Telegram bot (includes watch mode + duplicate detection + suggestions)
uv run vaultmind graph-build    # Build knowledge graph
uv run vaultmind graph-maintain # Prune stale refs + remove orphan entities
uv run vaultmind graph-report   # Generate graph analytics
uv run vaultmind auto-tag       # LLM-based tag suggestions (dry-run)
uv run vaultmind auto-tag --apply # Apply suggested tags to frontmatter
uv run vaultmind mcp-serve      # Start MCP server
uv run vaultmind stats          # Show vault statistics
```

## Tech Stack

- **Python 3.12+** with `uv` package manager
- **Pydantic v2** for models and config
- **aiogram 3.x** for Telegram bot
- **ChromaDB** for vector storage
- **NetworkX** for knowledge graph
- **Multi-provider LLM** — Anthropic, OpenAI, Gemini, Ollama via `llm/` abstraction
- **OpenAI SDK** for embeddings + Gemini/Ollama compatible endpoints
- **Click + Rich** for CLI
- **Hatch** for build system

## Testing

```bash
uv run pytest                           # Run all tests
uv run pytest -v --cov=vaultmind        # With coverage
uv run mypy src/ --ignore-missing-imports
uv run ruff check src/
```

## Config

- Structural config: `config/default.toml`
- Secrets: `.env` (never commit)
- All env vars prefixed `VAULTMIND_`
- Nested config via `__` delimiter: `VAULTMIND_TELEGRAM__BOT_TOKEN`

## LLM Provider Configuration

Set provider in `config/default.toml` under `[llm]`:

```toml
provider = "anthropic"  # or "openai", "gemini", "ollama"
thinking_model = "claude-sonnet-4-20250514"
```

Each provider needs its env var: `VAULTMIND_ANTHROPIC_API_KEY`, `VAULTMIND_OPENAI_API_KEY`, `VAULTMIND_GEMINI_API_KEY`. Ollama needs no key.

The `llm/` package uses a `Protocol`-based abstraction. Gemini and Ollama reuse the OpenAI SDK via their OpenAI-compatible endpoints — no extra SDK dependencies.

## Key Design Decisions

- Obsidian vault is the canonical store — all other systems are indexes
- Heading-aware chunking for embeddings (not arbitrary token splits)
- NetworkX + JSON persistence for graph (zero-infra, upgradable to Neo4j)
- MCP server uses optional `mcp` extra — don't import at top level in non-MCP code
- LLM providers implement `LLMClient` Protocol — system prompt handling varies per provider (Anthropic: dedicated param; others: prepended message)
- Smart message routing: heuristic-first classification in `bot/router.py` (zero LLM cost for greetings/capture), LLM only for questions/conversational
- Delete/edit commands use aiogram inline keyboards for confirmation flow (callback queries)
- Natural language date resolution: common keywords handled locally, complex expressions fall back to LLM
- Path traversal protection via `vault/security.py` — all user-supplied paths validated before filesystem access
- Thinking sessions persisted to SQLite (`~/.vaultmind/data/sessions.db`) with `(user_id, session_name)` composite key for future named sessions
- Embedding cache (`indexer/embedding_cache.py`) — SQLite-backed, content-hash keyed with `(content_hash, provider, model)` composite key, eliminates redundant API calls during re-indexing
- Bot handlers decomposed into `handlers/` package — `HandlerContext` dataclass shared state, `commands.py` is a thin facade preserving the public interface
- Input sanitization (`bot/sanitize.py`) — null byte stripping, length limits per operation, log-only injection detection (never blocks)
- Incremental watch mode (`vault/watch_handler.py`) — two-stage hash stability check eliminates partial-write bugs, debounce coalesces rapid Obsidian saves, async event bus (`vault/events.py`) decouples downstream consumers (duplicate detection, suggestions) from the watcher
- Graph re-extraction defaults to False in watch mode — it's an LLM call per note; when enabled, changes are batched on a timer (`batch_graph_interval_seconds`) instead of per-save
- Semantic duplicate detection (`indexer/duplicate_detector.py`) — reuses existing ChromaDB embeddings (zero additional API cost), classifies matches into bands: duplicate (≥92%), merge (80–92%). Subscribes to event bus for automatic fire-and-forget detection on every index update
- Context-aware note suggestions (`indexer/note_suggester.py`) — composite scoring: vector similarity (0.70–0.80 band) + shared graph entities (entity_weight=0.1) + graph path distance (graph_weight=0.05). Operates below duplicate/merge thresholds with clean band separation. Available via `/suggest` command, `suggest_links` MCP tool, and `suggest-links` CLI
- posthog pinned to `<4` — chromadb uses the 3-arg `capture()` API removed in posthog 4.x+
- Async I/O pipeline — all blocking calls (ChromaDB, embedding API, LLM) wrapped with `asyncio.to_thread()` at handler call sites, keeping the event loop responsive without protocol changes
- Graph maintenance (`graph/maintenance.py`) — `GraphMaintainer` prunes stale source-note references, removes orphan entities, and subscribes to `NoteDeletedEvent` for incremental cleanup on note deletion
- Auto-tagging (`indexer/auto_tagger.py`) — LLM classifies notes using existing vault tag vocabulary. New tags go to quarantine (`~/.vaultmind/data/tag_quarantine.json`) pending user approval. Default dry-run with `--apply` opt-in for frontmatter writes
- Voice note capture (`bot/transcribe.py`) — OpenAI Whisper API transcription (uses existing `openai` dependency). Voice messages → transcription → route to capture (default) or question (if ends with `?`). Requires `VAULTMIND_OPENAI_API_KEY`
