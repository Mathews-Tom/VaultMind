# Architecture Guide

## Overview

VaultMind is structured as a set of composable modules around a central principle: **the Obsidian vault is the canonical store**. All other systems (ChromaDB, knowledge graph, preferences DB) are derived indexes that can be rebuilt from the vault at any time.

Architectural prep is now in place for five additional runtime capabilities that are not fully implemented yet: curated identity memory, explicit proactivity policy, bounded heartbeat orchestration, draft lifecycle management, and permission-scoped integrations. The configuration and decision records for these capabilities are part of the codebase so later implementation can land against stable boundaries.

## Manual wiki boundary

`wiki/cli.py` registers the `wiki add/propose/review/apply/list/query/lint` Click group on the existing root CLI. `wiki/workspace.py` owns a bounded, synchronous source-backed page workflow, not a replacement runtime. Settings are loaded through the normal `load_settings` precedence. Add/review/apply and draft-file proposals construct no LLM, embedding, graph, bot, connector or index collaborator. Generated proposals use the existing configured LLM client for one completion, with no fallback/retry.

Current pages are one-level `wiki/SLUG.md` Markdown files. `_wiki/sources/ID.md` preserves exact source bytes (including CRLF); adjacent JSON records the ID, original name, origin/URI, capture time and full SHA-256. The URI is provenance, not a fetch instruction. Both `wiki` and `_wiki` must be excluded from ordinary indexing before persistence. Source/history/pending state is not fed to Chroma, FTS or the event bus. Page application itself does not index the current page.

A candidate has a title, `Current understanding`, resolvable source/page links, and wiki-controlled frontmatter/provenance. Drafts cannot supply frontmatter. Existing sources remain in provenance on updates. A new-source update must explicitly retain attributed disagreements in `Conflicting information`, or provide a reviewed `Resolution` with its reason and source citations. These are structural checks, not automatic conflict discovery or factual entailment. Reviewers must check actual statements against original sources.

Obsidian wikilinks use vault-relative identities, such as `[[_wiki/sources/source-a]]`. Ordinary Markdown destinations are relative to the current page: from `wiki/report-cache.md`, use `[source](../_wiki/sources/source-a.md)` or `[related](background.md)`. The resolved target must remain a selected source or an existing current wiki page; history, outside paths and invented destinations are refused.

Each proposal is a BLOCK `WIKI_PAGE` in the existing review queue, held in `_wiki/review_queue.db` with a real page applier. Plain review output includes the exact unified diff, source IDs/full revisions, base identity, status and canonical-payload SHA-256. Propose/review never replace a current page. Explicit `apply ID --expect SHA256` checks the persisted identity, source bytes/metadata and current base again, preserves complete prior bytes in `_wiki/history`, then atomically replaces one page. The operation record tracks before/after and payload identities and progress; success is recorded only after verification. Completed reapplication is a no-op only while the recorded current identity still holds. Failed rows stay failed; incomplete/ambiguous operations are reported on reopen rather than replayed. There is no automatic retry.

Completed proposals are historical evidence: their payload identity, hashes, source snapshots and operation/history records remain checked, but their links are not resolved again against today's related-page headings or existence. Historical review and a completed no-op therefore survive unrelated page changes. Pending candidates still require resolvable links at proposal and apply time; a fresh reviewed update can remove an obsolete related link.

A prepared operation that was not confirmed complete holds the page against both new proposals and reapplication. There is no supported in-place resolution command in this command set. Stop maintenance on the affected page and preserve its current bytes, sources, prior history, queue and operation records for manual diagnosis; do not edit those records to force a retry. A failed proposal with no operation record can instead be replaced with a fresh reviewed proposal after correcting its cause.

Content rollback in a consistent completed workspace is a new reviewed update, not replacement of the current file with old bytes. Copy the desired historical Markdown body into an external draft, omit its controlled YAML frontmatter, preserve every currently retained source citation, and retain attributed conflicts or supply an explicit reviewed resolution with a reason. Propose against the current page, review the new diff/digest, then apply; this preserves the pre-restore current page as another history revision. The [README restoration procedure](../README.md#restore-earlier-wording-through-review) gives concrete commands. It does not repair incomplete operations or missing provenance.

Path checks reject traversal, unsafe IDs, nonregular files and user-controlled symlinks, including control storage. Explicit source/draft paths may expand administrator-owned ancestor aliases with protected root-owned parent directories, such as macOS's system temporary-directory spelling; leaf symlinks remain refused. Retention-marked inputs fail before source/proposal persistence. Wiki-owned locking serializes cooperating wiki commands and paths/identities are rechecked immediately before replacement. This is neither an all-writer lock nor global multi-file atomicity: an editor or script ignoring the lock can still race replacement. Keep edits quiescent while applying; a stale proposal must not overwrite an observed edit. These local guarantees do not repair held legacy reindex/watcher behavior or qualify native providers.

`wiki/query.py` owns read-only list and lint; `knowledge/query.py` owns the shared
answer/evidence operation used by root CLI `ask`, explicit `wiki query`, Telegram
question/chat and `/think` (including sticky and voice questions), and MCP
`knowledge_query`. There is no alternate conversational evidence engine.
Read-only workspace/queue access does not initialize control directories,
migrate databases or write accepted state. Empty list/lint create no workspace.
Approved inspection requires exact latest completed applied bytes and matching
source/metadata, operation and history identities. Pending, unapproved or
manually changed pages, restored older versions, incomplete operations and
missing sources are refused. Lint returns body-redacted per-page `valid`,
`unresolved` or `invalid` diagnostics and exits 1 only for invalid rows.

Inspection keeps the completed proposal's approval-time link validation; it does not follow or re-resolve related-page destinations against today's pages. Removing a related page or heading therefore does not invalidate an unchanged selected page. Preserved source identities remain mandatory, and fresh proposals and pending applies still validate every candidate link. A missing or schema-incomplete approval queue produces per-page `invalid` lint diagnostics without creating or repairing queue state.

`knowledge/discovery.py` automatically discovers approved current pages for
`ask`. Effective adapter path/provenance permissions precede body reads, ranking,
counts and output. All permitted current candidates are validated before
deterministic lexical ranking over title, headings and body; invalid permitted
state fails closed even when unrelated to the question. Selection is limited to
eight pages, with deterministic path tie-breaks and disclosed lexical/selection
limits. No lexical match or an empty corpus returns deterministic unsupported
without a provider; missed synonyms do not establish absence from the wiki.
Explicit `wiki query --page` selects 1–8 unique current paths through the same
core, bypassing ranking only.

No answering route adds linked pages, raw-source bodies, pending/history content,
Chroma or FTS. Questions and each selected prior user question are at most 4096
characters; the complete UTF-8 question/intent/evidence envelope is at most
128 KiB. Overflow fails without truncation or dropping a selected page.
Complete sections become evidence rows with
result-local page-order/section-order IDs (`p1-s1`, etc.), page path/hash,
heading/text and preserved source ID/path/hash/metadata hash/URI citations.
Evidence-only returns `status: evidence` for a nonempty selection and empty
answer/citations. Lazy provider construction occurs only after valid complete
evidence is assembled; empty, no-match, invalid, denied and oversized requests
construct no provider.

Generated queries invoke the per-query factory only after evidence validation,
then call `[llm].thinking_model` with `[llm].max_tokens` once. The structured
question/intent/evidence/conflict envelope is untrusted content. The core validates
an exact `status`/`answer`/`citations` schema: `supported` and `conflicting` need
unique known section IDs; `unsupported` has no citations. An unresolved
`Conflicting information` section without a reviewed `Resolution` requires
conflicting output citing every unresolved conflict section, or unsupported,
never an implicit winner. Invalid schema/citations and provider failures are
errors, without fallback or hidden retry. Selected page/source/operation identities
are revalidated after completion. Results retain complete evidence, provenance,
unresolved page paths and lexical-selection qualifications.

Citation membership and structural conflict checks do not prove factual truth or
semantic entailment. Readers must inspect sections and preserved sources.
Querying creates no proposal, approval, page/history update, graph/episode record,
gap, extraction or distillation and never falls back to raw notes, graph, memory,
web or previous answers. Session and audit persistence are separate, permitted
effects, not evidence. Async Telegram and MCP adapters offload the synchronous
operation to a worker; they do not run a second answer stage.

`knowledge/render.py` preserves the accepted answer and full citation,
section-to-page/source mapping, selected revisions, source identities, unresolved
conflicts and selection qualifications in plain text. Evidence-only rendering also
includes complete section text. Telegram sends every character in ordered chunks
of at most 4000 characters with Markdown parsing disabled; it does not clip answers,
rewrite citations or strip boundary whitespace. CLI/MCP return the structured result.

The [README procedure](../README.md#manual-source-backed-wiki) uses checkout commands; a noneditable local wheel runs the same system and answering policy through its registered console script. Source CI and installed-runtime verification measure different execution origins, not different knowledge policies. Synthetic offline adapters do not establish live-model, private-vault, Telegram or native qualification.

### Source intake and discovery

`knowledge/intake.py::SourceIntake` is the provider-free preservation boundary
shared by adapters and the workspace source reader. Legacy exact source metadata
and proposal snapshots remain readable. New sources additionally bind a versioned
manifest through the metadata origin marker's hash. The manifest records actual
origin, adapter, event/revision, role, original-retention disposition and complete
parent/attachment identities. Readers validate that closure before returning text
or permitting curation; changing an attachment or parent invalidates its derivative.
Metadata is published last, so an incomplete write is not a valid source.

Capture separates immutable preservation from persisted provider, derivative,
ordinary projection and index stages. A source is never an approval. Projections
link back to preserved source IDs but do not silently become wiki curation input.
Event/revision/role replay avoids duplicate effects; changed bytes receive a new
identity and distinct events do not collapse merely because their bytes match.
An attempted stage is recorded before its effect. Failed or incomplete stages
remain explicit and are not retried automatically; preservation, projection,
Chroma, FTS and approval are not one atomic transaction.

Authenticated replay lookup validates adjacent event identities before reading matching retained closures. Invalid provenance or a matching incomplete/unauthorized closure refuses replay rather than becoming a missing event; unrelated bodies do not consume the matching closure's byte budget.

For a capture/replay refusal, malformed or orphan adjacent provenance can block a
later event even when its source body is unrelated. Stop affected captures and
preserve all files and state. An authorized operator can identify candidate IDs
from adjacent `_wiki/sources/ID.intake.json` manifest filenames and diagnose each
using the existing API: open `WikiWorkspace(settings, read_only=True,
allows_path=allows_path)`, then call
`SourceIntake.from_workspace(workspace, allows_path=allows_path).read(ID)` inside
its reading context. Use the operator's actual permission predicate, not a
permissive replacement; complete retained dependencies must remain authorized.
Record only the ID and failure class privately, never source bodies, hidden paths
or exception messages in responses/logs. This diagnoses, not repairs. Do not delete
or quarantine records, rewrite hashes, skip provenance, mark success, reset
reservations, or restart to force replay. Reconciliation requires separate
authorization; preservation promises neither automatic retry nor index completion.

Text capture preserves submitted text. URL submission and extracted text have
separate parent-linked identities; extraction is not wire-response preservation.
Photo-original retention follows `save_originals`; audio is transient and voice
questions do not create captured sources. A provider-derived description or
transcript does not establish original retention or factual accuracy.

Research, connector ingestion, explicit conversation distillation, enabled non-query idle finalization and maturation now use the same preserved-source boundary before model transforms. Research retains the available search/transcript-segment representation and complete joined transcript, not an HTTP wire response; connectors retain the fetched `SourceItem` strings. Exact generation requests retain the selected messages, system prompt where used, model and token limit. Parsed/rendered generated artifacts are separate parent-linked, explicitly unapproved sources, not a promise to archive raw provider responses.

`pipeline/distill.py::generate_distillation`, `pipeline/synthesis.py::generate_synthesis` and `research/analyzer.py::analyze_sources` generate content in memory without writing notes. Their caller-owned orchestration preserves inputs and requests, validates retained dependency closures, then retains and projects successful output. Reported source IDs can enter the existing selected-page `wiki propose → review → apply` workflow. Updating an existing page still preserves prior provenance and conflicting inputs and requires the exact reviewed digest. A generated `permanent` type, authority value or source/maturation approval cannot publish approved wiki knowledge.

Connector validation runs before new queue persistence as well as before source capture, so do-not-retain, malformed or oversized original fields cannot evade policy inside a JSON payload. Existing eight-field `SOURCE_INGESTION` rows retain their source-capture meaning and the applier still returns a relative ordinary-note path. Provider failure can retain the existing plain-source projection UX, labeled `generation_status: failed` and unapproved; it is not successful synthesis. New per-item outcomes distinguish retained IDs, verified projection, generation and finalizer disposition. A returned asynchronous finalizer has no durable index/event receipt, so those fields remain unknown rather than inferred from an applied proposal. Existing cursor/run-summary, delayed-approval, failure and restart limitations remain; neither capture nor approval supplies cross-store atomicity or exactly-once delivery.

A prequeue policy rejection creates neither a proposal nor a preserved source. It is reported as an item/run error with a body-redacted rejected-item identifier. The existing fetched cursor/ETag can still advance past that item: the cursor is not a durable completion checkpoint, and rejected-item recovery remains unsupported. Do not preserve forbidden bodies in failed proposals, relax retention rules, reset the cursor or infer an automatic retry.

Checkpoint storage retains fetched upstream tokens, which can include an item's GUID. Diagnostic/run-error redaction is not a guarantee that rejected strings are absent from checkpoint metadata. This existing connector limitation requires a separately scoped retention/checkpoint decision; it is not repaired by preserved-source capture.

Retained-projection verification annotates the outcome; it does not prevent the existing raw-note finalizer from indexing a legitimately edited, permitted ordinary projection. Such a mismatch remains `unverified`, generation remains unknown where unproven, and index/event fields remain unknown without a durable receipt. Destination authorization still runs before this diagnostic boundary: reserved paths and out-of-vault aliases refuse before indexing or event publication. Raw finalization never repairs provenance or approves wiki knowledge.

Manual `/distill` copies the complete available selected conversation revision, participants and authenticated transport identity before awaiting. Last-exchange timestamps are identified as monotonic, not occurrence dates. Idle finalization retains archived history and stored summaries before completion; summaries are labeled historical generated material, not replacement model turns. Query handling still never starts a producer. Independent episode and gap effects remain separately enabled domain operations, not answer evidence or page approval.

Maturation retains every selected note's exact bytes and declared retained parents before synthesis; missing, nonregular, forbidden, oversized or over-closure selections refuse before a completion instead of silently dropping notes. Its existing fingerprint-only review operation creates only an unapproved ordinary projection, and completion state changes only after that projection succeeds. Failed or interrupted stages are not automatically replayed. Bookmarking and other independent domain commands retain their guarded ordinary-note paths and limits; this migration does not claim that every domain artifact is intake-preserved or wiki-approved.

`knowledge/sources.py::search_sources(settings, query, *, store=None, n_results=5, note_type=None, allows_path=None)` serves CLI `search-sources`, Telegram `/recall`, MCP `vault_search` and benchmark queries. It enumerates authorized current ordinary Markdown and validated preserved text before ranking/limits/counts, independently of approval. Results carry `source-only`, `approved: false`, current-byte digests and provenance. Current wiki pages, history, queue/control files and unsafe aliases are excluded. Optional indexed paths assist ranking; displayed content always comes from current bytes. Raw distance is retained only for an exact current indexed chunk. The shared parser orders tags deterministically in chunk prefixes so unchanged tagged sources have the same indexed/current representation across processes; old differing prefix variants remain unknown rather than supplying stale confidence. Lexical-only or stale chunks remain unknown. Permissions apply before body reads, including the full retained-source dependency closure.

Search refuses excessive scopes rather than silently scanning a prefix: at most
512 candidates, 4096 indexed chunks, 1 MiB per searched source and 64 MiB total
search bytes. Lexical matching is not exhaustive semantic retrieval. Benchmarks
measure source discovery, not approved-answer quality. Offline bundles contain
real `vault/` Markdown plus candidate identities/distances in `retrieval.yaml`;
the loader snapshots chunks once and the same discovery path independently
hydrates the current bundle corpus.

An unreadable, malformed or oversized authorized ordinary source aborts source discovery with a redacted error rather than silently omitting evidence. Correct the source or explicitly narrow the authorized scope; query text alone does not narrow enumeration. Ordinary and preserved frontmatter authority values share the same 1–5 conversion/range validation; invalid values use the configured neutral-default ranking behavior, never approval.

Ordinary writers reject lexical and resolved `wiki`/`_wiki` targets, including
symlinks and configuration aliases into those trees. Only the narrow wiki/intake
writers own their respective namespace operations. This application boundary is
not an OS sandbox or a repair of old indexed data. Retired/superseded-content
reindexing and native watcher qualification remain unresolved; batching still
does not make Chroma and FTS atomic or recover interrupted indexing.

## Module Map

```text
src/vaultmind/
├── cli.py           # Click CLI entry point
├── config.py        # Pydantic Settings — TOML + env vars
├── errors.py        # VaultMindError base exception hierarchy
│
├── knowledge/       # Shared evidence policy and source preservation/discovery
│   ├── query.py         # Approved-current answering/evidence; one completion boundary
│   ├── discovery.py     # Authorized deterministic lexical page selection
│   ├── intake.py        # Immutable source intake and retained-closure validation
│   ├── sources.py       # Explicit unapproved source-only exploration
│   └── render.py        # Lossless plain-text answer/provenance presentation
│
├── wiki/            # Explicit manual curation; never automatic query promotion
│   ├── workspace.py     # Preservation, proposal, review and page application
│   ├── query.py         # Read-only current-page list/lint
│   └── cli.py           # CLI ask and wiki command adapters
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
│   ├── thinking.py      # Approved-current session intent + independent idle finalization
│   ├── session_store.py # SQLite-backed thinking session persistence
│   ├── sanitize.py      # Input sanitization (length, null bytes, injection detection)
│   ├── transcribe.py    # OpenAI Whisper API voice transcription
│   ├── notifier.py      # Proactive Telegram notifications from scheduler
│   ├── formatter.py     # Rich formatting utilities
│   └── handlers/        # Decomposed handler modules
│       ├── context.py       # HandlerContext dataclass (shared state)
│       ├── utils.py         # Shared utilities (_is_authorized, _split_message)
│       ├── capture.py       # Note capture + URL ingestion + photo capture
│       ├── recall.py        # Current-file source-only discovery with pagination
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
│   ├── synthesis.py     # Pure in-memory generate_synthesis()
│   ├── maturation.py    # Retained snapshots → unapproved synthesis/projection
│   └── distill.py       # Retained conversation orchestration + pure generation
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
└── mcp/         # MCP server — approved answering, vault CRUD, graph, memory and traversal
    ├── server.py        # 19 registered tools with profile-scoped result policy
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

`retrieval_confidence(hits, score_floor)` (`indexer/ranking.py`) classifies raw
source-discovery relevance for `/recall` and `vaultmind bench` decline scoring.
Approved-current answering does not use it or mint retrieval gaps. It looks only
at raw `distance` and returns a `RetrievalConfidence` (`StrEnum`):

1. empty result set → `weak`
2. any raw distance `< score_floor` (`[bench].score_floor`, default `0.5`) → `strong`
3. otherwise, any hit with a `None`/missing distance → `unknown`
4. otherwise → `weak`

`unknown` means no hit establishes vector relevance and at least one lacks vector
evidence, typically a lexical-only BM25 hit. Bench withholds deterministic decline
credit and `formatter.relevance_label()` renders `relevance unknown`, not a
fabricated percentage. Authority ranking can reorder hits but never rewrites
`distance`. Independent raw-vector consumers such as duplicate detection,
suggestions and digest remain available; none supplies approved-answer evidence.

Benchmark bundles pair `golden.yaml` with `retrieval.yaml` candidates and actual `vault/` Markdown. Candidate YAML supplies paths, optional chunk indices, raw distances and fusion scores only; content and authority come from real parsed files. Missing/null distance remains unknown. The loader snapshots indexed chunks once, and shared source queries independently hydrate current corpus bytes; stale chunk text cannot lend confidence to changed content.

`scripts/verify_acceptance.py --case retrieval --offline` exercises parser →
Chroma/BM25 → shared source discovery → real recall/benchmark consumers in
synthetic HOME/state. Deterministic embeddings, recorded Telegram transport and
labeled exact-threshold substitutions are not live-quality evidence. Boundary
hits use pre-created Markdown chunks; a separate lexical-only current file is
discoverable without vectors. The case checks authority/distance invariants,
authorized top-k and current-byte hydration against stale indexed excerpts.
Approved-current answering has separate `wiki-query` and `mcp` scripted-provider
coverage, not vector-based conversational gap coverage.

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

The public verifier, guides and `config/default.toml` are checkout assets, not wheel contents. A wheel installs the Python package, typed settings defaults and registered `vaultmind` console script. Supply an explicit external TOML when reproducing checkout configuration: a working directory without `config/default.toml` otherwise uses typed defaults (including their different LLM provider/model defaults). The external verifier below intentionally consumes reviewed public checkout assets while importing the noneditable installed package; running a console help command alone is not that proof.

For a noneditable local-wheel check, build once into a fresh external directory and install the frozen runtime dependencies plus `mcp`. From the repository root, with Python 3.12 and `uv` available:

```bash
(
  set -e
  REPO="$PWD"
  T="$(mktemp -d)"
  mkdir -p "$T/home"
  uv export --frozen --no-dev --extra mcp --no-emit-project --format requirements-txt --output-file "$T/dependencies.txt"
  uv build --wheel --out-dir "$T/dist"
  uv venv --python 3.12 "$T/installed"
  uv pip install --python "$T/installed/bin/python" --requirement "$T/dependencies.txt" "$T"/dist/*.whl
  cd "$T"
  vm() { env -i PATH="$PATH" HOME="$T/home" ANONYMIZED_TELEMETRY=False "$T/installed/bin/vaultmind" "$@"; }
  vm --version
  vm --help
  env -i PATH="$PATH" HOME="$T/home" ANONYMIZED_TELEMETRY=False "$T/installed/bin/python" -I -c 'import vaultmind; print(vaultmind.__file__)'
  for CASE in retrieval mcp approval config wiki wiki-query; do
    env -i PATH="$PATH" HOME="$T/home" ANONYMIZED_TELEMETRY=False "$T/installed/bin/python" -I "$REPO/scripts/verify_acceptance.py" --case "$CASE" --offline --receipt-out "$T/$CASE.json"
  done
)
```

The subshell stops at the first failed command without changing the caller's directory, variables, functions or shell options. The fresh `dist` directory must contain only the newly built wheel; record its full SHA-256 and source revision rather than selecting an old artifact. The printed import must be under the new environment's `site-packages`, not `src/`; receipts must show matching reviewed/runtime package hashes and the sibling installed console where used. The public verifier still needs the checkout's reviewed scripts/assets and committed migration history: this is not a claim that those verification assets ship inside the wheel. For manual coverage, add the README's source/create/update/query and reviewed-restore commands inside the subshell before it exits, using `vm --config /absolute/synthetic/config.toml wiki ...` outside the checkout on fresh synthetic state. Scripted generator/query consumers establish engineering behavior, not live-model quality, native-watcher repair, private-vault readiness or release publication.

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
User message → capture_all? → source capture
             → active /think session? → approved-current query
             → heuristic classifier → capture / static greeting / approved query
```

Routing uses local patterns, length and line count; question versus chat changes
neither evidence nor model. Capture may independently invoke URL/vision providers;
it must not be described as universally model-free. Static greetings make no
completion. `/recall` explicitly selects raw exploration rather than answering.

### Thinking Partner

```text
Current question + last 3 complete user questions (intent only)
    → fresh approved-current selection and complete sections
    → one configured answer completion + validation and revalidation
    → plain-text result → session history (SQLite)

Existing scheduler, only when distill.enabled
    → claim expired histories → independent distillation/index/episode/gap producer
```

Sessions persist full exchanges with `(user_id, session_name)` identities.
Follow-ups within `[telegram].thinking_session_ttl` continue the active session.
Only three complete recent user questions can supply intent and lexical selection
terms; assistant text, historical summaries, graph facts and episodes never enter
query evidence. Selected legacy questions are validated in full, not truncated or
silently discarded. `explore:`, `critique:`, `synthesize:` and `plan:` are
presentation prefixes, not prompts for alternate evidence engines. The former
answer-only graph-context builder and XML response-extraction parser are removed;
independent graph extraction and memory domains remain.

Query expiry leaves the original record untouched and starts a fresh generation;
it does not summarize, archive, finalize or dispatch a producer. When
`[distill].enabled` is true, the bot registers a separate one-minute job on its
existing scheduler. It archives full expired histories and existing summaries,
claims pending work before generation, then attempts distillation for records
meeting `min_turns`. Its ordinary note/index/episode/gap effects do not approve a
wiki page. Completed, skipped, failed and interrupted claimed records remain
retained; failed or claimed work is not automatically retried after polling or
restart. Partial producer effects may already exist, so retention is not atomic
index completion. Explicit `/distill` and bookmarking remain separate actions.

The per-query factory is lazy, but the bot's general domain client is still
assembled at startup for other commands/producers. `[routing].chat_model` remains
for real edit/date consumers, not answering; `fast_model` remains for independent
domain work. Old RAG/graph-answer, XML-extraction and session-summary controls
have been removed rather than left as alternate engines.

### Zettelkasten Maturation

```text
Fleeting/literature notes → DBSCAN clustering (embeddings)
                                    ↓
                            Cluster digest (user review)
                                    ↓
                            LLM synthesis → Permanent note with [[wikilinks]]
```

The pipeline identifies clusters of semantically related fleeting notes ready for unapproved permanent-note synthesis. Dismissed clusters re-surface after a configurable expiry. Cluster review authorizes this producer operation only; promoting its reported retained source ID into approved knowledge requires the separate exact wiki review/apply workflow described above.

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

Capture patterns, greetings and question/chat distinctions are handled locally.
No model classifier chooses a different answering policy; ambiguous question/chat
messages use the same approved-current core.

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
