# Architecture Guide

## Overview

VaultMind is structured as a set of composable modules around a central principle: **the Obsidian vault is the canonical store**. All other systems (ChromaDB, knowledge graph, preferences DB) are derived indexes that can be rebuilt from the vault at any time.

Architectural prep is now in place for five additional runtime capabilities that are not fully implemented yet: curated identity memory, explicit proactivity policy, bounded heartbeat orchestration, draft lifecycle management, and permission-scoped integrations. The configuration and decision records for these capabilities are part of the codebase so later implementation can land against stable boundaries.

## Module Map

```text
src/vaultmind/
├── cli.py           # Click CLI entry point
├── config.py        # Pydantic Settings — TOML + env vars
├── errors.py        # VaultMindError base exception hierarchy
│
├── vault/           # Vault I/O layer
│   ├── parser.py        # Markdown + YAML frontmatter parsing, heading-aware chunking
│   ├── models.py        # Note, NoteChunk, NoteType data classes
│   ├── projection.py    # Shared current-content and bot history boundary grammar
│   ├── watcher.py       # Watchdog-based file system watcher
│   ├── watch_handler.py # Two-stage hash stability, debounce, event dispatch
│   ├── events.py        # Async event bus (NoteIndexedEvent, NoteDeletedEvent, etc.)
│   ├── security.py      # Path traversal validation
│   └── ingest.py        # URL detection, YouTube transcript + article ingestion
│
├── indexer/         # Embedding + vector search layer
│   ├── embedder.py        # Embedding generation (batched)
│   ├── embedding_cache.py # SQLite-backed cache (content-hash + provider + model key)
│   ├── store.py           # ChromaDB vector store + hybrid search (vector + BM25)
│   ├── bm25.py            # SQLite FTS5-backed BM25 keyword index
│   ├── hybrid.py          # Reciprocal Rank Fusion (RRF) combiner
│   ├── activation.py      # SQLite-backed activation tracking (note access/edit scoring)
│   ├── duplicate_detector.py  # Semantic duplicate detection (92%/80% bands)
│   ├── note_suggester.py  # Composite link suggestions (similarity + entities + graph)
│   ├── auto_tagger.py     # LLM tag classification, routes suggestions through the review queue
│   ├── tag_analyzer.py    # Tag synonym/merge detection (string similarity + co-occurrence)
│   ├── digest.py          # Weekly digest with inbox triage (zero LLM cost)
│   └── ranking.py         # Post-retrieval ranking (type + mode + activation + decay + authority)
│
├── graph/           # Knowledge graph layer
│   ├── knowledge_graph.py # NetworkX graph with JSON persistence
│   ├── extractor.py       # LLM entity/relationship extraction from notes
│   ├── context.py         # Graph context builder (ego subgraph, entity extraction from queries)
│   ├── evolution.py       # Belief evolution (confidence drift, relationship shifts, stale claims)
│   └── maintenance.py     # Orphan cleanup, stale source pruning, event-driven deletion
│
├── llm/             # Provider-agnostic LLM abstraction
│   ├── client.py          # LLMClient Protocol, Message, MultimodalMessage, LLMResponse, factory
│   └── providers/
│       ├── anthropic.py   # Anthropic provider (system prompt as dedicated param, multimodal)
│       ├── openai.py      # OpenAI provider (multimodal via image_url content parts)
│       ├── gemini.py      # Gemini via OpenAI-compatible endpoint
│       └── ollama.py      # Ollama via OpenAI-compatible endpoint
│
├── bot/             # Telegram bot (aiogram 3.x)
│   ├── telegram.py      # Router, handler registration, callback queries
│   ├── commands.py      # Thin facade delegating to handlers/ package
│   ├── router.py        # Heuristic message classifier (Intent enum)
│   ├── thinking.py      # Multi-turn thinking partner (RAG + graph context)
│   ├── session_store.py # SQLite-backed thinking session persistence
│   ├── sanitize.py      # Input sanitization (length, null bytes, injection detection)
│   ├── transcribe.py    # OpenAI Whisper API voice transcription
│   ├── notifier.py      # Proactive Telegram notifications from scheduler
│   ├── formatter.py     # Rich formatting utilities
│   └── handlers/        # Decomposed handler modules
│       ├── context.py       # HandlerContext dataclass (shared state)
│       ├── utils.py         # Shared utilities (_is_authorized, _split_message)
│       ├── capture.py       # Note capture + URL ingestion + photo capture
│       ├── recall.py        # Semantic search with pagination
│       ├── think.py         # Thinking partner sessions
│       ├── graph.py         # Knowledge graph queries
│       ├── daily.py         # Daily note management
│       ├── notes.py         # Date-based note lookup
│       ├── read.py          # Read full note content
│       ├── edit.py          # AI-assisted note editing
│       ├── delete.py        # Note deletion with confirmation
│       ├── bookmark.py      # Session/Q&A bookmarking to vault
│       ├── suggestions.py   # Note link suggestions
│       ├── duplicates.py    # Duplicate detection interface
│       ├── review.py        # Weekly review with graph insights
│       ├── voice.py         # Voice message handling
│       ├── stats.py         # Vault/graph statistics
│       ├── health.py        # System health check
│       ├── evolve.py        # Belief evolution signals
│       ├── maturation.py    # Zettelkasten maturation clusters
│       ├── memory.py        # Episodic memory commands (/decide, /outcome, /episodes, /workflows)
│       └── routing.py       # Message intent routing
│
├── pipeline/        # Zettelkasten maturation + conversation distillation
│   ├── clustering.py    # DBSCAN clustering of semantically related notes
│   ├── synthesis.py     # LLM synthesis (cluster → permanent Zettelkasten note)
│   ├── maturation.py    # Pipeline orchestrator (clustering → digest → synthesis)
│   └── distill.py       # distill_conversation() — session/`/distill` → qa-artifact note
│
├── services/        # Background job scheduling + graduated autonomy review queue
│   ├── scheduler.py     # State-aware compound loop scheduler with event triggering
│   ├── review_queue.py  # AUTO/SKIM/BLOCK proposal routing (Lane, ProposalKind, ReviewQueue)
│   └── loops/           # Compound loop jobs
│       ├── insight_loop.py     # Usage pattern shift detection
│       ├── evolution_loop.py   # Belief drift trend accumulation
│       └── procedural_loop.py  # Workflow synthesis from episodes
│
├── contradiction/   # Duplicate-band conflict detection + deterministic resolution
│   ├── detection.py     # LLM material-conflict detector, gated by `vaultmind eval contradict`
│   ├── policy.py        # Temporal → authority → escalate resolver (Kosha `policy.py` port)
│   ├── marking.py       # write_superseded_block() — non-destructive frontmatter/callout writer
│   ├── detector.py      # Event-bus orchestrator (detection + policy + marking + escalation)
│   └── eval.py          # Scores the detector vs. a trivial always-escalate baseline
│
├── bench/           # Retrieval self-benchmark (`vaultmind bench`)
│   ├── runner.py        # recall@k/MRR + retrieval-confidence scoring against the live hybrid+ranking path
│   ├── golden.py        # benchmarks/golden.yaml schema loader
│   ├── fixture_store.py # Deterministic FixtureStore for `--bundle` CI-safe self-tests (missing/null distance = unknown)
│   ├── llm_score.py     # `--llm` opt-in cite-or-decline scoring
│   └── trend.py         # Per-run JSONL trend record storage
│
├── sources/         # Explicit-registry connector framework with durable cursors
│   ├── registry.py      # Closed rss/youtube-channel/github-activity connector registry
│   ├── cursor.py        # ConnectorState cursor model
│   ├── store.py         # sources.db cursor + bounded run-summary SQLite store
│   ├── pipeline.py      # ingest_item() — distillation + review-queue hookup per fetched item
│   └── connectors/      # rss.py, youtube_channel.py, github_activity.py
│
├── export/          # OKF bundle export for Kosha interop
│   └── okf.py           # export_okf_bundle() — permanent/concept notes → conformant OKF bundle
│
├── drafts/          # Planned draft lifecycle subsystem (active/sent/expired)
├── integrations/    # Planned permission-scoped external adapters
│
├── research/        # External source research
│   ├── pipeline.py      # YouTube search → transcript → LLM analysis → vault notes
│   ├── searcher.py      # YouTube search backend (yt-dlp)
│   └── analyzer.py      # Comparative analysis of sources
│
├── tracking/        # Usage analytics
│   ├── preferences.py   # SQLite interaction store
│   └── analyzer.py      # Usage pattern analysis + insights generation
│
├── memory/          # Episodic + procedural memory + knowledge gap ledger
│   ├── models.py        # Episode dataclass, OutcomeStatus enum
│   ├── store.py         # SQLite-backed episode store (CRUD + entity search)
│   ├── extractor.py     # LLM-based episode extraction from notes
│   ├── procedural.py    # Workflow synthesis from episodic patterns (experimental)
│   └── gaps.py          # GapStore — deduplicated, lifecycle-tracked knowledge gaps
│
└── mcp/         # MCP server — 18 tools for vault CRUD, graph, memory introspection, traversal
    ├── server.py        # MCP server with 18 vault tools
    ├── auth.py          # Profile enforcement + audit logging
    └── profiles.py      # Profile loading (researcher/planner/full)
```

## Data Flow

### Indexing Pipeline

```text
Markdown files → Parser (contextual chunks) → Embedder → ChromaDB
                    ↓                              ↓          ↓
              Frontmatter metadata         Embedding Cache   BM25 Index (FTS5)
```

1. `VaultParser` retains the complete frontmatter-free Markdown body in `Note.content`. `Note.current_content()` projects that parsed body directly, including authored leading thematic rules, before heading-aware chunking, contextual prefixes, inline tags and importance scoring.
2. `Embedder` generates vectors, checking `EmbeddingCache` first (keyed by content hash + provider + model). All supplied notes are parsed/projected/chunked and all embeddings are prepared before replacement starts.
3. `VaultStore` reconciles its configured ChromaDB collection and optional BM25 FTS5 index sequentially. Incremental replacement removes obsolete chunks for that path; a zero-chunk soft-delete retires both indexes. Full indexing treats the supplied notes as the canonical snapshot: absent paths, shorter revisions and an empty snapshot remove obsolete derived rows. Other Chroma collections, Markdown files and lineage are not cleared.

### Preserved History and Current Evidence

Bot `/delete` and `/edit` preserve the complete previous body under dated supersede callouts. Full-note reads remain historical reads; vector, keyword and hybrid retrieval all use the same current-content projection. A bot-deleted note produces no chunks, and two edits index only the latest claim while retaining both older revisions on disk. Ordinary authored `[!superseded]` callouts are not retirement instructions.

New bot mutations pair `supersede_boundaries: ["<32hex-id>=<superseded_at-value>", ...]` with `<!-- vaultmind:supersede id=<32hex-id> -->` directly above the generated callout. Paired markers must be unique and in newest-first body order; the random ID prevents accidental collisions with authored lookalikes, not deliberate forgery. Unpaired marker-looking text remains authored content. Legacy deletion requires `superseded_at: bot:delete` plus the exact dated `Deleted via bot` callout/rationale; legacy edit UTC ISO stamps pair one-to-one with dated `Edited via bot` / `Instruction:` callouts. Ambiguous or malformed boundaries raise an indexing error without rewriting the source or silently indexing preserved history. A full parse failure aborts snapshot reconciliation rather than interpreting an omitted note as a deletion.

Only reconciling full-index callers request `iter_notes(strict=True)`; a malformed file aborts the complete snapshot before replacement. Non-reconciling iteration keeps valid notes and logs rejected paths. The standalone watch and MCP command assemblies maintain the configured keyword index alongside Chroma, as do the bot, research and source-run index mutations; read-only store consumers do not acquire a keyword mutator.

Embedding preparation failure leaves the last-good vector and keyword evidence unchanged, for both incremental and full indexing. Chroma and SQLite are separate stores: a failure after replacement starts can leave partially updated derived state. Such errors are reported explicitly, not as successful indexing or cross-database atomicity. Bot/MCP writes retain their saved Markdown and report the indexing failure; a failed edit/delete still retains its mutation lineage. Recovery is an explicit subsequent index operation, not a new retry service.

The watcher advances its last-indexed hash and publishes successful note events only after indexing succeeds. Another filesystem event for unchanged failed content retries it instead of being suppressed as already indexed. `scripts/verify_acceptance.py --case lineage --offline` exercises installed `index`, standalone `watch` and `mcp-serve` commands over isolated absolute TOML fixtures. The commands construct their configured indexes; only the external embedding factory is substituted. Watch readiness and mutation barriers require actual filesystem events and observable indexed state, and registered MCP mutation tools run over real SDK stdio. Separately labeled direct-watcher scenarios expose handler hash/event and injected-error state; they are not proof of command assembly. The case also exercises real parser, Chroma, SQLite BM25/parameterized FTS matches, hybrid retrieval, bot edit/delete handlers and lineage with scripted LLM responses and recorded Telegram delivery. Receipts are fixture plumbing/failure evidence; native provider, hosted embedding quality and production vault operations remain **NOT QUALIFIED**.

### Hybrid Search

```text
Query → Vector search (ChromaDB) ─┐
     → BM25 keyword search (FTS5) ┼→ Reciprocal Rank Fusion → Ranked results
                                   │        ↓
                             Activation scoring + mode multipliers
```

When hybrid search is enabled (default), both vector and keyword results are merged via Reciprocal Rank Fusion (RRF, k=60). Items appearing in both lists get boosted. Post-retrieval ranking applies note-type multipliers, mode multipliers (operational > learning), activation-based scoring, and temporal decay.

### Retrieval Confidence

Fusion and relevance are separate signals. `hybrid_search()` returns, per hit:

| Key            | Meaning                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `distance`     | **Raw vector distance** (`float`, lower is closer) or `None` for a lexical-only (BM25) hit      |
| `rrf_score`    | Reciprocal Rank Fusion score — ordering only, never relevance evidence                        |
| `authority_score` | Added by `apply_authority()` — `rrf_score` (or `1 - distance`) × the stamped-authority weight; ordering only |

`retrieval_confidence(hits, score_floor)` (`indexer/ranking.py`) is the single classifier used by `/recall` gap minting, `vaultmind bench` decline scoring, and the thinking partner's gap minting. It looks only at raw `distance` and returns a `RetrievalConfidence` (`StrEnum`):

1. empty result set → `weak`
2. any raw distance `< score_floor` (`[bench].score_floor`, default `0.5`) → `strong`
3. otherwise, any hit with a `None`/missing distance → `unknown`
4. otherwise → `weak`

`unknown` means "the results are non-empty, but no hit has vector evidence of relevance and at least one has no vector evidence at all" — typically lexical-only BM25 hits. It is deliberately not weak: no knowledge gap is minted (`GapKind.WEAK_RETRIEVAL` / `UNANSWERED_QUESTION`), `vaultmind bench` withholds deterministic decline credit and counts the question as unknown confidence, and `formatter.relevance_label()` renders `relevance unknown` rather than a fabricated percentage. Authority re-ranking can reorder hits but never rewrites `distance`, so it cannot change the confidence of a result set. Raw-vector consumers (duplicate detection, note suggestions, digest) are unchanged.

The bench `--bundle` fixture (`retrieval.yaml`: golden query text → ordered hit list) follows the same contract: a hit's `distance` is optional, and a **missing or `null` distance is explicit unknown evidence**, never defaulted to a number. Optional `rrf_score`, `authority` and `content` pass through unchanged.

`scripts/verify_acceptance.py --case retrieval --offline` exercises this contract through the real parser → ChromaDB + BM25 → `handle_recall` / `run_bench` / `ThinkingPartner.think` → `GapStore` path in a fresh temporary HOME configured by explicit TOML, and emits a JSON receipt (git revision, dirty flag, code/API-origin/corpus digests, boundary labels, per-assertion expected vs observed). Its boundary substitutions are labeled in the receipt: a deterministic hash embedder (no embedding provider), a scripted LLM client, an in-process fake Telegram message, a scripted store used *only* for exact floor-boundary values (`floor − ε`, `floor`, `None`, missing key), and a direct BM25 insert that creates the lexical-only chunk. It is a plumbing and raw-distance-semantics check, not retrieval-quality or live-provider evidence; thinking retrieves with vector-only `store.search()`, so its `unknown` branch is covered only through the scripted store.

### Watch Pipeline

```text
Filesystem event → Debounce → Hash stability check → Index update → Event bus
                                                                       ↓
                                                     Duplicate detection (fire-and-forget)
                                                     Note suggestions (fire-and-forget)
                                                     Graph re-extraction (if enabled, batched)
```

Two-stage hash stability eliminates partial-write bugs from Obsidian's rapid save behavior. The async event bus (`vault/events.py`) decouples downstream consumers from the watcher.

### Message Routing

```text
User message → Sanitize → Intent classifier (heuristic-first)
                              ↓
                   ┌──────────┼──────────────┐
                   ↓          ↓              ↓
                Capture    Greeting      Question/Chat
               (inbox)    (static)      (LLM + vault context)
```

The router uses heuristics (regex, length, line count) before falling back to LLM classification. Greetings and explicit captures never invoke the LLM — zero cost for common operations.

### Thinking Partner

```text
User topic → Identity memory (planned always-loaded tier)
          → Entity extraction (LLM) → Ego subgraph (NetworkX)
                                           ↓
                                    Vault search (ChromaDB)
                                           ↓
                                    Context assembly → LLM response
                                           ↓
                            Session store (SQLite) → promotion pipeline (planned)
```

Multi-turn sessions persist to SQLite with `(user_id, session_name)` composite key. Follow-up messages within the TTL continue the same session automatically.

The architectural boundary is now explicit:

- identity memory: a small curated vault-backed context tier
- retrieval memory: search and graph-derived context assembled per request
- episodic memory: decision and outcome ledger
- promoted memory: distilled durable facts and priorities derived from raw interactions

### Zettelkasten Maturation

```text
Fleeting/literature notes → DBSCAN clustering (embeddings)
                                    ↓
                            Cluster digest (user review)
                                    ↓
                            LLM synthesis → Permanent note with [[wikilinks]]
```

The pipeline identifies clusters of semantically related fleeting notes ready to be synthesized into permanent Zettelkasten notes. Dismissed clusters are tracked and re-surface after a configurable expiry.

### Belief Evolution

```text
Knowledge graph snapshots → Confidence drift detection
                         → Relationship shift detection
                         → Stale claim detection (> 180 days)
                                    ↓
                            Evolution signals → Digest / /evolve command
```

### Episodic Memory

```text
/decide <text> → EpisodeStore (SQLite) → Pending episode
/outcome <id> <status> <desc> → Resolved episode with lessons
                                        ↓
                              ProceduralMemory scans for patterns
                                        ↓
                              LLM synthesizes → Workflow (steps + trigger)
```

Episodic memory tracks decision-outcome pairs. Each episode records the decision, context, outcome, lessons, and linked graph entities. The procedural memory layer (experimental, disabled by default) mines resolved episodes for recurring patterns and synthesizes reusable workflows.

### Proactivity Policy

Architectural prep defines a single system-wide autonomy model:

- `observer` — notify only
- `advisor` — propose drafts and suggestions
- `assistant` — allow bounded vault-side changes and low-risk automations
- `partner` — reserved for controlled external actions with confirmation gates still enforced for destructive or irreversible operations

This policy is configuration-driven and intended to become the single source of truth for proactive behavior, draft creation, and future integration actions.

### Architectural Control Plane

```text
config/default.toml
    ↓
Settings
    ├─ memory_profile  -> identity note layout, daily-log layout, promotion bounds
    ├─ proactivity     -> autonomy mode and confirmation boundaries
    ├─ heartbeat       -> schedule and snapshot limits for proactive orchestration
    ├─ drafts          -> active/sent/expired folders and approval defaults
    └─ integrations    -> default capability ceiling and sanitation policy
```

These settings do not, by themselves, add new behavior. They define the stable boundaries future runtime modules must obey. This keeps later implementation work additive: identity memory can be introduced without changing retrieval, drafts can be added without bypassing policy checks, and integrations can be added under a shared capability model instead of per-adapter special cases.

## Key Design Decisions

### Why heading-aware chunking with contextual prefixes?

Arbitrary token-count splits break semantic boundaries. Heading-aware chunking preserves the author's logical structure — a section about "Authentication" stays as one chunk rather than being split mid-paragraph. Each chunk is prepended with a contextual prefix (`note: {title} (type: {type}) | section: {heading} | tags: ...`) that gives the embedding model document-level context, producing higher-quality vectors.

### Why NetworkX + JSON persistence?

Zero infrastructure. The graph runs in-process with JSON serialization. No Neo4j or database server to manage. When the vault grows beyond what NetworkX can handle, the `KnowledgeGraph` interface is swappable to a graph database.

### Why heuristic-first routing?

Most messages are obvious: capture prefixes, greetings, long pastes. Using regex and simple rules first means zero LLM calls for ~60% of messages. The LLM classifier only activates for ambiguous cases (questions vs. conversational).

### Why Protocol-based LLM abstraction?

The `LLMClient` Protocol in `llm/client.py` allows swapping providers without touching calling code. Gemini and Ollama reuse the OpenAI SDK via their OpenAI-compatible endpoints — no extra SDK dependencies. System prompt handling varies per provider (Anthropic uses a dedicated parameter; others prepend a system message).

### Why async I/O with `asyncio.to_thread()`?

All blocking calls (ChromaDB, embedding API, LLM) are wrapped with `asyncio.to_thread()` at handler call sites. This keeps the Telegram bot's event loop responsive without requiring protocol-level async support in ChromaDB.

### Why SQLite for everything?

Sessions, embedding cache, preferences, tag quarantine, BM25 FTS5 index, activation tracking, and episodic memory all use SQLite. It's embedded, requires no server, handles concurrent reads well, and the data volumes are small enough that a full database would be over-engineering.

## Compound Loop Engine

The scheduler (`services/scheduler.py`) supports state-aware compound loops where each run reads prior-run output as input. This enables cross-cycle pattern recognition.

**Architecture:**

```text
Vault Events → Event Bus → Scheduler (threshold check) → Loop Job (state in → state out) → Notifier (Telegram)
                                                              ↕
                                                     JSON state file (atomic writes)
```

**Three loop jobs:**

| Loop       | Input              | Detects                                               | Notifies When                                           |
| ---------- | ------------------ | ----------------------------------------------------- | ------------------------------------------------------- |
| Insight    | PreferenceStore    | Search trends, acceptance rate shifts, volume changes | >15% rate shift or >50% volume change                   |
| Evolution  | EvolutionDetector  | Belief drift signals, escalating trends               | High-severity signal or 3+ consecutive scan appearances |
| Procedural | EpisodeStore + LLM | Recurring decision patterns → workflows               | New workflow synthesized from episodes                  |

**Event-driven triggering:** Vault events (NoteCreated, NoteModified) accumulate per-job. When a configurable threshold is met (default 10) and cooldown period has passed (default 1 hour), the loop fires early outside its normal schedule.

Architectural prep also reserves a bounded heartbeat orchestration model:

```text
Deterministic snapshots → diffing → policy check → bounded LLM reasoning → notifications, drafts, or local updates
```

The existing scheduler remains the execution backbone. Later implementation should add heartbeat snapshot builders and policy-aware action planning rather than replacing the scheduler with a separate runtime.

## Security Model

- **Path traversal protection** — `vault/security.py` validates all user-supplied paths before filesystem access
- **Input sanitization** — `bot/sanitize.py` strips null bytes, enforces length limits per operation, and logs injection patterns (never blocks — defense in depth)
- **MCP profiles** — `mcp/auth.py` enforces tool access, folder scope, write permissions, and size limits per agent profile
- **Audit logging** — all MCP tool calls are logged with profile, tool, arguments, and outcome (OK/DENIED/ERROR)
- **Telegram auth** — configurable user ID whitelist; empty list allows all users
- **Integration capability boundaries** — architectural prep introduces read, draft, and write capability tiers for external systems
- **Outbound confirmation gates** — draft creation and external send are intentionally separated so future integrations can keep human approval in the loop
- **Autonomy policy boundary** — outbound sends, destructive actions, and future integration writes must flow through the global proactivity and capability policy rather than ad hoc handler logic
