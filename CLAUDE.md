# VaultMind — Project Instructions

## Overview

AI-powered personal knowledge management built on Obsidian. Turns an Obsidian vault into an intelligent second brain with semantic search (ChromaDB), knowledge graph (NetworkX), Telegram bot (aiogram), thinking partner (Claude API), and MCP server for agent integration.

## Architecture

```
src/vaultmind/
├── vault/       # Markdown parser, models, file watcher, path security
├── indexer/     # Embedding pipeline + ChromaDB vector store
├── graph/       # NetworkX knowledge graph + LLM entity extraction
├── bot/         # Telegram bot (aiogram 3.x) — commands, router, thinking partner
│   ├── commands.py    # All command handler logic
│   ├── router.py      # Heuristic message classifier (Intent enum)
│   ├── session_store.py # SQLite-backed thinking session persistence
│   ├── telegram.py    # aiogram Router, handler registration, callback queries
│   └── thinking.py    # Multi-turn thinking partner (RAG + graph)
├── llm/         # Provider-agnostic LLM abstraction (Protocol-based)
│   ├── client.py        # LLMClient Protocol, Message, LLMResponse, factory
│   └── providers/       # Anthropic, OpenAI, Gemini, Ollama implementations
├── mcp/         # MCP server for Claude Desktop/Code integration
├── config.py    # Pydantic Settings — TOML + env vars (incl. RoutingConfig)
└── cli.py       # Click CLI entry point
```

## Commands

```bash
uv run vaultmind init          # Create vault folder structure
uv run vaultmind index         # Full vault index into ChromaDB
uv run vaultmind bot           # Start Telegram bot
uv run vaultmind graph-build   # Build knowledge graph
uv run vaultmind graph-report  # Generate graph analytics
uv run vaultmind mcp-serve     # Start MCP server
uv run vaultmind stats         # Show vault statistics
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
- posthog pinned to `<4` — chromadb uses the 3-arg `capture()` API removed in posthog 4.x+
