# Architecture Guide

## Overview

VaultMind is structured as a set of composable modules around a central principle: **the Obsidian vault is the canonical store**. All other systems (ChromaDB, knowledge graph, preferences DB) are derived indexes that can be rebuilt from the vault at any time.

Architectural prep is now in place for five additional runtime capabilities that are not fully implemented yet: curated identity memory, explicit proactivity policy, bounded heartbeat orchestration, draft lifecycle management, and permission-scoped integrations. The configuration and decision records for these capabilities are part of the codebase so later implementation can land against stable boundaries.

## Manual wiki boundary

`wiki/cli.py` registers the `wiki add/propose/review/apply/list/query/lint` Click group on the existing root CLI. `wiki/workspace.py` owns a bounded, synchronous source-backed page workflow, not a replacement runtime. Settings are loaded through the normal `load_settings` precedence. Add/review/apply and draft-file proposals construct no LLM, embedding, graph, bot, connector or index collaborator. Generated proposals use the existing configured LLM client for one completion, with no fallback/retry.

Current pages are one-level `wiki/SLUG.md` Markdown files. `_wiki/sources/ID.md` preserves exact source bytes (including CRLF); adjacent JSON records the ID, original name, origin/URI, capture time and full SHA-256. The URI is provenance, not a fetch instruction. `_wiki` must be explicitly excluded from vault indexing before any persistence. Source/history/pending state is not fed to Chroma, FTS or the event bus. Page application itself does not index the current page.

A candidate has a title, `Current understanding`, resolvable source/page links, and wiki-controlled frontmatter/provenance. Drafts cannot supply frontmatter. Existing sources remain in provenance on updates. A new-source update must explicitly retain attributed disagreements in `Conflicting information`, or provide a reviewed `Resolution` with its reason and source citations. These are structural checks, not automatic conflict discovery or factual entailment. Reviewers must check actual statements against original sources.

Obsidian wikilinks use vault-relative identities, such as `[[_wiki/sources/source-a]]`. Ordinary Markdown destinations are relative to the current page: from `wiki/report-cache.md`, use `[source](../_wiki/sources/source-a.md)` or `[related](background.md)`. The resolved target must remain a selected source or an existing current wiki page; history, outside paths and invented destinations are refused.

Each proposal is a BLOCK `WIKI_PAGE` in the existing review queue, held in `_wiki/review_queue.db` with a real page applier. Plain review output includes the exact unified diff, source IDs/full revisions, base identity, status and canonical-payload SHA-256. Propose/review never replace a current page. Explicit `apply ID --expect SHA256` checks the persisted identity, source bytes/metadata and current base again, preserves complete prior bytes in `_wiki/history`, then atomically replaces one page. The operation record tracks before/after and payload identities and progress; success is recorded only after verification. Completed reapplication is a no-op only while the recorded current identity still holds. Failed rows stay failed; incomplete/ambiguous operations are reported on reopen rather than replayed. There is no automatic retry.

Completed proposals are historical evidence: their payload identity, hashes, source snapshots and operation/history records remain checked, but their links are not resolved again against today's related-page headings or existence. Historical review and a completed no-op therefore survive unrelated page changes. Pending candidates still require resolvable links at proposal and apply time; a fresh reviewed update can remove an obsolete related link.

A prepared operation that was not confirmed complete holds the page against both new proposals and reapplication. There is no supported in-place resolution command in this command set. Stop maintenance on the affected page and preserve its current bytes, sources, prior history, queue and operation records for manual diagnosis; do not edit those records to force a retry. A failed proposal with no operation record can instead be replaced with a fresh reviewed proposal after correcting its cause.

Path checks reject traversal, unsafe IDs, nonregular files and user-controlled symlinks, including control storage. Explicit source/draft paths may expand administrator-owned ancestor aliases with protected root-owned parent directories, such as macOS's system temporary-directory spelling; leaf symlinks remain refused. Retention-marked inputs fail before source/proposal persistence. Wiki-owned locking serializes cooperating wiki commands and paths/identities are rechecked immediately before replacement. This is neither an all-writer lock nor global multi-file atomicity: an editor or script ignoring the lock can still race replacement. Keep edits quiescent while applying; a stale proposal must not overwrite an observed edit. These local guarantees do not repair held legacy reindex/watcher behavior or qualify native providers.

`wiki/query.py` supplies read-only list, query and lint over one-level current pages. Read-only workspace/queue access does not initialize control directories, migrate databases, or write accepted state. Empty list/lint do not create a workspace. Approved current inspection requires exact latest completed applied bytes and matching source/metadata, operation and history identities. Pending/unapproved/manual-changed pages, restored older versions, incomplete operations and missing preserved sources are refused; list never presents them as accepted. Lint instead returns per-page `valid`, `unresolved` or `invalid` diagnostics, with body-redacted details; CLI prints all rows and exits 1 only for invalid rows. Structurally valid unresolved conflicts remain inspectable.

Inspection keeps the completed proposal's approval-time link validation; it does not follow or re-resolve related-page destinations against today's pages. Removing a related page or heading therefore does not invalidate an unchanged selected page. Preserved source identities remain mandatory, and fresh proposals and pending applies still validate every candidate link. A missing or schema-incomplete approval queue produces per-page `invalid` lint diagnostics without creating or repairing queue state.

Query selects 1–8 unique explicit current page paths, without hidden traversal, raw-source bodies, pending/history content, Chroma or FTS. A question is at most 4096 characters and the complete UTF-8 question/evidence envelope is at most 128 KiB; oversize fails before completion, never truncates. Full selected sections, not prefixes, become evidence rows with deterministic page-order/section-order IDs (`p1-s1`, etc.), page path/hash, heading/text and preserved source ID/path/hash/metadata hash/URI citations. These result-local section IDs are not permanent identifiers. `--evidence-only` constructs no provider or writer and returns `status: evidence` with empty answer/citations.

Generated query calls the existing configured client once with structured question/evidence/conflict data, treated as untrusted content, and validates a strict `status`/`answer`/`citations` response. `supported` and `conflicting` require unique known section IDs; `unsupported` has no citations. A selected unresolved `Conflicting information` section without a reviewed `Resolution` requires conflicting (citing every unresolved conflict section) or unsupported output, not an implicit winner. Invalid schema/citations or provider failure raises a redacted error, without fallback/retry. Current/source/operation identities are revalidated after completion before returning. The returned object includes evidence and unresolved page paths.

Neither citation membership nor structural conflict checks prove factual truth or semantic entailment. Readers must inspect sections and preserved sources. These commands never promote answers, submit proposals, apply changes or mutate page/history/accepted state. Legacy RAG/defaults remain unchanged. The [README procedure](../README.md#manual-source-backed-wiki) gives real SOURCE commands; synthetic offline adapters do not establish independently installed-wheel, live-model, private-vault, Telegram or native-watcher qualification.

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

1. `VaultParser` reads markdown files, splits by headings into `NoteChunk` objects, and prepends a contextual prefix (`note: {title} (type: {type}) | section: {heading} | tags: {t1, t2}`) for embedding quality
2. `Embedder` generates vectors, checking `EmbeddingCache` first (keyed by content hash + provider + model)
3. `VaultStore` upserts chunks with their metadata and deletes prior chunks in batches bounded by ChromaDB's client-reported record cap, then synchronizes the BM25 FTS5 index after the vector operations succeed. The cap is read from the method in ChromaDB 0.5.1+ or the property in 0.5.0, not a guessed limit.

An indexing run is not atomic across ChromaDB batches or across the vector and keyword stores. A later vector-batch failure propagates to the caller and can leave earlier vector writes committed before BM25 synchronization. Incremental replacement still deletes the target note's prior rows before embedding; batching does not change that ordering or provide last-good-content preservation. Unrelated notes remain outside that replacement's scope.

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

### Offline Acceptance and CI

Run each standalone consumer case from a tracked checkout after installing `dev` and `mcp`:

```bash
uv sync --extra dev --extra mcp
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case retrieval --offline
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case mcp --offline
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case approval --offline
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case config --offline
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case wiki --offline
uv run --extra dev --extra mcp python scripts/verify_acceptance.py --case wiki-query --offline
```

Keep committed Git history available: approval reconstructs its legacy migration schema from a prior committed revision. Both the full-test and standalone offline CI jobs use `fetch-depth: 0` because the full suite also exercises this committed-history migration; neither copies an ignored ledger or private historical files.

The verifier creates temporary HOME/vault/index/SQLite/graph state and supplies explicit fixture TOML before loading application settings. All cases share one offline environment boundary before artifact discovery, application imports and in-process consumers: only a required, nonempty caller `PATH` is inherited; `HOME` comes from the explicit synthetic fixture root and `ANONYMIZED_TELEMETRY=False` is fixed. Unlisted ambient provider/API/Telegram/GitHub/cloud credentials, proxies, Git/SSH configuration and helper/socket overrides, and source/interpreter overrides are excluded by construction, not a provider-prefix blacklist. The prior environment is restored on success or failure. Config's deliberately created fixture environment/dotenv overrides remain available after clean ingress so the case still exercises real precedence and selected-target computation.

Every verifier Git child and fresh persisted Chroma snapshot reader uses the same construction. MCP additionally fixes `COLUMNS=2000` for stable diagnostics and scopes the parent environment while the SDK constructs and runs its stdio child: the SDK merges parent HOME/LOGNAME/PATH/SHELL/TERM/USER before its supplied environment, so passing a clean child map alone is insufficient. Direct SDK and snapshot helper callers receive the same bounded isolation without relying on the command entrypoint. MCP still executes the actual `vaultmind` console script beside the running Python executable with `-I`, records the consumed script's execution path and digest, imported package origin/code digest and interpreter, and checks real persisted vectors/documents/metadata. Python can create locale-coercion `LC_CTYPE`, and macOS can create `__CF_USER_TEXT_ENCODING`, after process creation; neither is inherited as an offline runtime input. Regression checks inspect the actual pre-exec environment separately from these runtime-created fields. Approval still exercises the real CLI/review queue/appliers, committed legacy-schema migration/reopen and Telegram router with recorded delivery rather than external effects.

Direct fixture initialization is scoped before creating or populating Chroma: cold PostHog OS discovery can start `lsb_release` or `uname` on Linux before the SDK child exists. Those helpers receive the same fixture environment, with the caller's state restored afterward. Regression observations reject injected caller credentials across actual subprocesses without forbidding deliberately created fixture configuration or dependency-owned locale controls.

Required runtime inputs are the nonempty `PATH` with usable Git, the actual Python environment with installed application and MCP dependencies (`dev+mcp` for source gates) and its registered sibling console script, reviewed public source/dependency assets, and committed migration history. Missing inputs fail explicitly; missing required history or execution controls cannot become optional reference identities or a passing empty result. The interpreter path retains its lexical virtual-environment `bin` spelling rather than resolving to a base Python. No source import path is injected; source checks use their actual editable runtime, while wheel checks require the actual noneditable installed runtime.

These controls are credential hygiene, **not a filesystem, process or network sandbox**. They do not make PATH executables or repository-local Git configuration trustworthy, block inherited file descriptors or system credential stores, or replace OS isolation. Existing network guards are fixture instrumentation. Hosting/shipping operations and separately authorized coding-agent authentication remain outside this offline-consumer boundary; production commands are unchanged.

Reviewed assets resolve relative to the verifier, not the caller's working directory. Receipts identify the Git revision, verifier, dependencies, reviewed source and actual imported runtime; private planning files and ignored history are not inputs. The imported package is fingerprinted separately from the reviewed checkout; differing code fails acceptance. Missing required public assets also fail explicitly. The globally ignored `.docs` directory stays local and must not be force-added to support verification. Add `--receipt-out /temporary/path/receipt.json` to retain a JSON outcome; consumer failures emit a failed outcome and exit nonzero.

CI retains lint, strict source typing, full pytest and formatting checks. A separate fail-fast matrix runs each of `retrieval`, `mcp`, `approval`, `config`, `wiki` and `wiki-query` offline on Python 3.12 and 3.13 with `dev+mcp`, explicit interpreter selection and a temporary HOME. The query case exercises actual Click list/query/lint, approved current-page/source validation and the configured generation boundary with a scripted client. These are synthetic source-consumer checks, not hosted-model qualification or installed-wheel evidence. Wheel verification must invoke this reviewed script with the wheel environment's Python from outside the checkout, without source-path injection or an editable install.

All CI jobs inherit explicit workflow-level `contents: read` permissions, with metadata read implicit and no additional write grant. Every checkout sets `persist-credentials: false`. Immediately after each checkout, a presence-only local Git configuration check looks for HTTP extraheader keys, including checkout authentication headers, with both names and values suppressed. A match fails the job; only Git's no-match status succeeds and prints `Checkout authentication persistence: absent`; every other Git error fails with a generic message. This checks local checkout persistence, not global configuration or credential stores. Verify effective token scopes, rendered checkout inputs and this absence message in fresh logs for all jobs; workflow declarations or a green status alone do not establish the observed behavior. This is defense in depth, not remediation of an observed write grant.

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

Large BM25 replacements stage chunk IDs in a keyed temporary table with one host parameter per insert, then delete matching FTS5 rows in a single scan rather than rescanning the virtual table per ID batch. Writes on the shared BM25 connection are serialized; staging, replacement and staging cleanup remain in one transaction, so a failed insert rolls back the replacement without leaking IDs into the next call. Embedding-cache lookups split their `IN` clauses to support SQLite builds with a 999-host-parameter limit, including the fixed provider/model/access-time parameters. Cache lookups retain provider/model isolation and update access times only for matching hits. These are local SQLite guarantees, not cross-index atomicity or preservation of the old note during the separate delete-before-embedding path.

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
