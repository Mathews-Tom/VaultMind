[![VaultMind](assets/banner.png)](https://github.com/Mathews-Tom/VaultMind)

---

# VaultMind

Turn captured material into reviewed, source-backed wiki pages, then ask questions
against their approved current revisions from the CLI, Telegram or MCP. Raw-note
and preserved-source search stays available for exploration, but is never silently
promoted into answer evidence. VaultMind also keeps its independent Obsidian tools:
hybrid search, knowledge graph, photo capture, Zettelkasten maturation, belief
evolution, episodic/procedural memory, compound loops, conversation distillation,
contradiction detection, graduated autonomy review and non-destructive edit lineage.

## Why VaultMind?

- **Obsidian is the source of truth** — plain markdown files, always
- **One answering policy** — CLI `ask`, `wiki query`, Telegram questions/chat/thinking and MCP `knowledge_query` use complete approved-current sections and retained provenance, with explicit supported/conflicting/unsupported outcomes
- **AI augments, never replaces** — hybrid search (vector + BM25) and knowledge graphs surface what you'd miss
- **Mobile-first capture** — Telegram bot gives full PKM access from your phone, including photo capture via vision models
- **Agent-native** — Claude Desktop/Code and other AI agents read/write your vault via MCP
- **Multi-provider LLM** — Anthropic, OpenAI, Gemini, or Ollama for thinking + extraction
- **Hybrid retrieval** — Reciprocal Rank Fusion merges vector (ChromaDB) and keyword (SQLite FTS5) search results
- **Contextual embeddings** — chunks carry document-level context (title, type, section, tags) for higher-quality vectors
- **Zettelkasten maturation** — DBSCAN clustering selects note snapshots for explicitly unapproved permanent-note synthesis; publishing answer evidence still requires exact wiki review
- **Belief evolution** — tracks confidence drift, relationship shifts, and stale claims across your knowledge
- **Episodic memory** — decision-outcome tracking with lessons learned and entity linking
- **Procedural memory** — mines recurring decision patterns into reusable workflows (experimental)
- **Compound loops** — scheduler accumulates state across runs, detects usage shifts, belief drift trends, and new workflow patterns; pushes insights to Telegram proactively
- **Provenance-aware ranking** — notes carry a stamped `authority` (1-5) at creation; `/recall` reweights results by source trust, backfilling a neutral default for older notes
- **Retrieval self-benchmark** — `vaultmind bench` scores recall@k/MRR (and optionally cite-or-decline accuracy) against a golden question set, gating ranking/detector changes before they ship
- **Conversation distillation** — explicit `/distill` and an independently enabled idle-session job can produce ordinary `qa-artifact` notes and episodic records; asking does not run those producers or approve their output
- **Knowledge gap ledger** — independent enabled producers and contradiction escalations maintain trackable gaps (`/gaps`); approved-current questions do not mint gaps or trigger research
- **Contradiction detection** — new notes landing in the duplicate-detector's merge band are checked by an eval-gated LLM surface; conflicts resolve by temporal-then-authority-then-escalate policy and never edit or delete the losing note
- **Graduated autonomy** — every automated mutation proposal (tags, merges, contradiction resolutions, maturation synthesis) routes through one AUTO/SKIM/BLOCK review queue with an approval-fatigue metric in `vaultmind learn`
- **Source connectors** — scheduled `rss`, `youtube-channel`, and `github-activity` connectors preserve fetched items before distillation; source-ingestion approval authorizes capture, not approved wiki publication
- **Non-destructive edits** — bot-initiated `/delete`/`/edit` write a dated `> [!superseded]` callout and a lineage edge instead of deleting text, preserving full edit history in the note
- **OKF export** — `vaultmind export-okf` exports permanent/concept notes as a conformant [OKF](https://github.com/Mathews-Tom/Kosha) bundle for governed serving from Kosha
- **Note modes** — learning vs. operational mode with activation-based decay scoring
- **Secure by default** — path traversal protection, input sanitization, injection detection on all vault operations
- **Smart caching** — SQLite-backed embedding cache eliminates redundant API calls during re-indexing
- **Your data, your infra** — self-hosted, local-first, no cloud dependency

**Delivery limits:** Non-destructive edit history is implemented, but consistent exclusion of retired or superseded text after full/watch reindexing and the original native-watcher notification failure remain unresolved; [PR96](https://github.com/Mathews-Tom/VaultMind/pull/96) is held. SQLite/Chroma batching does not provide cross-index atomicity. Offline fixture checks are not native-provider, private-vault, or real Telegram qualification.

## Manual source-backed wiki

`wiki add`, `propose`, `review`, and `apply` maintain a selected Markdown page without starting the bot, watcher, graph or retrieval indexes. Preserved sources, pending proposals and prior versions live under `_wiki`; both `wiki` and `_wiki` must be excluded from ordinary indexing. Defaults include both; an explicitly configured older list requires a deliberate setup edit and is never rewritten automatically. This explicit draft workflow needs no model or embedding key. The examples use `uv run` from a checkout. A locally built, noneditable wheel provides the same commands and knowledge policy through its installed `vaultmind` console script; these are usage choices, not separate answering systems. This is unreleased source/local-wheel use, not a claim of PyPI publication. See the [external wheel recipe](docs/architecture.md#offline-acceptance-and-ci) for installed-runtime verification.

Use a disposable vault for this example. Create `/tmp/wiki-demo.toml` with an absolute, existing vault path:

```toml
[vault]
path = "/absolute/path/to/synthetic-vault"
excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]
```

Save `source-a.md` outside the vault with a project note recommending 30-day report-cache retention. Preserve it:

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki add source-a.md --id source-a --uri urn:example:project-note
```

In a Markdown editor, create `draft-a.md` outside the vault, without frontmatter:

```markdown
# Report cache policy

## Current understanding

The project note recommends retaining cached reports for 30 days. [[_wiki/sources/source-a]]
```

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki propose --page wiki/report-cache.md --source source-a --draft-file draft-a.md
uv run vaultmind --config /tmp/wiki-demo.toml wiki review PROPOSAL_ID
uv run vaultmind --config /tmp/wiki-demo.toml wiki apply PROPOSAL_ID --expect REVIEWED_SHA256
```

Replace `PROPOSAL_ID` with the exact `Proposal:` output and `REVIEWED_SHA256` with the full `SHA256:` from review. Inspect the source hashes and complete plain diff before approving. Propose/review do not create or change the current page. After apply, open the vault in Obsidian or your editor and follow the source link from `wiki/report-cache.md` to the preserved source.

The complete user path is capture → explicit unapproved discovery (`search-sources`, Telegram `/recall` or MCP `vault_search`) → select preserved source IDs for a page proposal → inspect the exact diff and digest → apply that reviewed proposal → ask normally. CLI `ask`, Telegram questions and MCP `knowledge_query` then select approved current pages automatically; `wiki query --page` offers explicit selection under the same policy. Capturing or discovering a contradictory source does not change the approved answer evidence. Only a reviewed page update can do that.

For an update, save `source-b.md` with an operations note recommending 7 days, whose scope relative to the project note is unresolved, then add it:

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki add source-b.md --id source-b --uri urn:example:operations-note
```

Create `draft-b.md` in your editor:

```markdown
# Report cache policy

## Current understanding

Retention duration is unresolved across the two sources.

## Conflicting information

- The project note recommends 30 days. [[_wiki/sources/source-a]]
- The operations note recommends 7 days; its scope relative to the project note is unresolved. [[_wiki/sources/source-b]]

Do not infer that either source silently supersedes the other. A later human decision must state the chosen scope and reason.
```

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki propose --page wiki/report-cache.md --source source-a --source source-b --draft-file draft-b.md
uv run vaultmind --config /tmp/wiki-demo.toml wiki review NEW_PROPOSAL_ID
uv run vaultmind --config /tmp/wiki-demo.toml wiki apply NEW_PROPOSAL_ID --expect NEW_REVIEWED_SHA256
```

The same page is updated; both original source byte sequences remain intact, and the prior complete page is preserved under `_wiki/history`. Repeating an identical source addition or completed apply is a no-op (unless the accepted state has since changed). Changing source bytes under the same ID is rejected. New-source updates retain earlier provenance even when only the new ID is passed. Structural citation checks do **not** establish truth or automatically detect semantic disagreements: read both sources and preserve attributed uncertainty, or supply an explicit reviewed `Resolution` with its reason.

A wrong digest, changed source/metadata, tampered payload or stale page fails explicitly without overwriting accepted content. Failed proposals are never retried automatically. If no operation record was prepared, correct the underlying problem and prepare a fresh reviewed proposal. A prepared but unconfirmed operation holds the entire page: new proposals and applies are refused, and this command set has no supported in-place recovery command. Stop maintenance on that page and preserve its current bytes, sources, history, queue and operation evidence for manual diagnosis; do not edit control records or create another proposal to bypass the hold. The wiki lock coordinates wiki commands, not Obsidian, external scripts or every OS writer; a noncooperating writer can still race the final replacement.

Omit `--draft-file` only when deliberately requesting one bounded completion from the configured LLM provider. This requires the normal provider key and sends selected/retained sources and the current page to that provider; errors are not converted into a draft fallback. The scripted offline consumer exercises this adapter, not live-model factual quality or native/private-vault operation. See [architecture](docs/architecture.md#manual-wiki-boundary) and [configuration](docs/configuration.md#manual-wiki-configuration) for limits.

### Capture sources before curating

Telegram text, photos, voice captures, URL ingestion and MCP `capture`/`capture_note`
use shared source preservation before downstream projection, indexing or analysis.
Capture reports an immutable source ID/path/hash separately from derivative,
provider, projection and index outcomes. `preserved` is not `indexed`, and neither
means `approved`: the linked inbox note is a convenient ordinary projection, not
an accepted wiki page. Supply a returned source ID explicitly to
`wiki propose --source SOURCE_ID`, inspect the complete review and apply its exact
digest to curate it.

Submitted URL text and successfully extracted text are distinct parent-linked
sources; extraction is not an archive of the HTTP response. Fetch failure leaves
the submitted text preserved and reports the failed derivative instead of inventing
an extracted source. Photos retain their original only when `save_originals` is
enabled; vision text records that disposition. Voice audio is transient, not an
archived original; a voice question is answered rather than captured as a source.
These distinctions describe preservation, not provider accuracy.

Where an adapter supplies an event/revision identity, replay does not repeat
completed or failed downstream effects. Changed bytes create a new immutable
source; separate events with equal bytes remain distinct. Failed or interrupted
provider/projection/index stages are not automatically retried. Partial completion
can leave a valid preserved source without an inbox note or usable index; capture
is not an atomic multi-store transaction or an interrupted-index recovery system.

Research, connectors, explicit conversation distillation, enabled non-query idle finalization and maturation now preserve their selected inputs and actual generation requests before model transforms. Generated analysis, `qa-artifact` and `permanent` notes are separate, explicitly unapproved derivatives with retained parent identities. Use a reported source ID in the existing selected-page proposal → exact review → apply workflow; a research result, source-ingestion approval or maturation approval cannot publish answer evidence.

Provider failure leaves preserved originals available. Retained source IDs, ordinary-note projection, indexing/event delivery and wiki approval are distinct outcomes. Connector cursor/run-summary and delayed-approval/restart limitations remain: an applied queue operation does not prove index/event finalization. Bookmarking and other independent domain writers retain their existing limits.

For malformed/orphan provenance blocking a later capture or replay, stop affected
captures, preserve state and follow the
[read-only diagnostic boundary](docs/architecture.md#source-intake-and-discovery);
do not force replay or treat preservation as index completion.

```bash
uv run vaultmind --config /tmp/wiki-demo.toml search-sources "report cache"
```

`search-sources`, Telegram `/recall`, MCP `vault_search` and `vaultmind bench`
share current ordinary-note and preserved-source discovery. Results are
`source-only`, `approved: false`, with current-byte digests/provenance; index text
never substitutes for current content. Permission and namespace filtering happen
before ranking/top-k/counts. Preserved history, pending/control files and current
`wiki` pages are not raw-search candidates. Missing or stale vector evidence has
unknown relevance, not a fabricated distance of zero. Source discovery is bounded
and lexical matching can miss synonyms.

Root `ask`, explicit `wiki query`, normal Telegram question/chat, `/think` and its
sticky follow-ups, voice questions, and MCP `knowledge_query` now share one
approved-current answering policy. Raw discovery is a separate exploration action,
not an alternate evidence mode or fallback when an answer is unsupported.
The built-in planner profile still covers inbox/projects only and therefore
denies capture requiring `_wiki` storage before writing a source or projection.
Do not grant blanket `_wiki` access as a workaround; capturing through a restricted
profile requires a deliberate permission decision for its preservation dependencies.

Inspect the accepted page without constructing a provider or writer:

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki list
uv run vaultmind --config /tmp/wiki-demo.toml ask "What retention policy is supported?" --evidence-only
uv run vaultmind --config /tmp/wiki-demo.toml wiki lint
```

These commands print pretty JSON. List returns approved current pages with their title, full SHA-256 and unresolved-conflict flag; invalid or unapproved current files fail instead of appearing accepted. Lint prints `page`, `status` and body-redacted `detail` for each page. Its statuses are `valid`, `unresolved` and `invalid`; only `invalid` makes the command exit 1. An unresolved but structurally valid disagreement is not a lint failure.

Evidence-only ask returns `status: "evidence"` when matching approved pages exist,
an empty answer/citations, and complete current-page sections. Each evidence row
has a deterministic ID such as `p1-s1`, page path/hash, heading, full section text,
and preserved source identities/hashes/metadata hashes/URI. IDs follow selected
page order; they are references within that result, not permanent global section
identifiers. Source citations describe preserved provenance, not retrieval of raw
source bodies.

For one deliberately configured model completion, omit `--evidence-only`:

```bash
uv run vaultmind --config /tmp/wiki-demo.toml ask "What retention policy is supported?"
```

`ask` automatically selects up to eight approved current pages using deterministic
lexical matches in titles, headings and bodies. The result discloses selection
and its limits: synonyms without shared terms can be missed, and a bounded
selection is not proof of complete wiki coverage. Empty or no-match results are
deterministic `unsupported` results without constructing a provider.
Results include selected page revisions and deduplicated source identities,
plus `selection.mode`, `selection.selected_pages`, `selection.limited` and a
lexical limitation. Empty automatic results distinguish `no_eligible_pages`
from `no_lexical_support` in `reason`.

For explicit scope, use the same implementation through
`wiki query "What retention policy is supported?" --page wiki/report-cache.md`,
optionally adding `--evidence-only`. Repeat `--page` for 1–8 unique approved
current pages. Explicit selection bypasses ranking, not approval validation.
Automatic discovery validates every permitted current page before ranking; even
an unrelated invalid page fails closed rather than disappearing from the answer.

Questions and each of up to three prior session questions are limited to 4096
characters; the complete UTF-8 question/intent/evidence envelope is limited to
128 KiB. Oversized input fails without truncation, dropping
selected pages or constructing a provider. No linked pages, pending drafts,
history, raw-source bodies or indexes are silently added. Current bytes must
match the latest completed applied proposal, including source/metadata and
operation/history identities; manual edits, copied-back old versions, missing
sources and incomplete operations fail closed.

Generated results use `supported`, `conflicting` or `unsupported`. Supported/conflicting answers require unique citations to supplied section IDs; unsupported answers have no citations. An unresolved `Conflicting information` section without a reviewed `Resolution` cannot silently become a winner: the result must be conflicting (citing every unresolved conflict section) or unsupported. Invalid output, invented IDs and provider failures are errors, not fallback answers. Selected identities are checked again after completion to reject evidence that changed during generation.

The query operation creates no proposal, approval, page/history update, graph or
episode record, gap, extraction, distillation or automatic promotion. Session
history and MCP audit records are separate permitted persistence, not answer
evidence. The per-query provider factory is lazy: evidence-only, empty, no-match,
invalid and oversized requests make no answering completion. Generated answering
uses `[llm].thinking_model` and `[llm].max_tokens`, at most once per request, with
no hidden retry or raw-note, graph, memory, web or prior-answer fallback. The bot
still assembles its shared domain client at startup for unrelated commands.

`/think` retains sessions across restarts, but only the last three complete user
questions can resolve conversational intent and influence fresh page selection.
Prior assistant answers, stored summaries, graph and episode facts never become
evidence. Mode prefixes are presentation conventions, not different engines or
evidence policies. Telegram sends the accepted answer, citations, complete
section-to-page/source mapping and selection qualifications as plain text, split
without clipping or dropping boundary characters. CLI/MCP retain structured results.

When `[distill].enabled` is true, a separate one-minute job on the existing bot
scheduler claims expired sessions before attempting distillation. It retains
original histories/summaries and completed, skipped, failed or interrupted claims;
failed/claimed work is not automatically retried. Asking neither dispatches that
producer nor deletes expired history. These ordinary producer effects and explicit
`/distill`/`/bookmark` actions require wiki review before becoming answer evidence.

Citation/schema validation does **not** prove that a claim follows from the text
or is true: inspect the complete sections and preserved sources yourself.
Checkout CI/offline results do not by themselves qualify a noneditable installed runtime or a live provider. The [architecture verification recipe](docs/architecture.md#offline-acceptance-and-ci) distinguishes source checks from installed-runtime evidence; neither establishes live-model quality or native/private-vault assurance.

### Restore earlier wording through review

Do not copy a historical page over `wiki/report-cache.md`: older accepted bytes fail current-approval validation. Open the desired `_wiki/history/report-cache/*.md` revision in an editor, copy its Markdown body into an external `restore-draft.md`, and remove the historical YAML frontmatter. Keep every currently retained source citation. In the example above, restore the original 30-day project-note paragraph but retain the attributed 30-day/7-day `Conflicting information` block from `draft-b.md`; restoring wording does not settle that disagreement. An intentional resolution instead needs an explicit reason and citations to both sources.

```bash
uv run vaultmind --config /tmp/wiki-demo.toml wiki propose --page wiki/report-cache.md --source source-a --source source-b --draft-file restore-draft.md
uv run vaultmind --config /tmp/wiki-demo.toml wiki review RESTORE_PROPOSAL_ID
uv run vaultmind --config /tmp/wiki-demo.toml wiki apply RESTORE_PROPOSAL_ID --expect RESTORE_REVIEWED_SHA256
uv run vaultmind --config /tmp/wiki-demo.toml wiki lint
```

Use the newly printed proposal ID and reviewed digest. Propose/review leave the current page unchanged; apply preserves that current page as another complete historical revision, retains both original sources and stamps a fresh accepted revision. This restores selected content, not old approval metadata or a byte-for-byte snapshot. The example remains `unresolved` with lint exit 0. This procedure applies only to a consistent completed workspace; it cannot clear an incomplete-operation hold or replace missing sources/control records.

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

Parses eligible ordinary Markdown, chunks by headings, generates embeddings and
stores in ChromaDB. Keep `wiki` and `_wiki` excluded, including when upgrading an
older explicit exclusion list. Indexing and graph building are for raw discovery
and independent domain tools; neither makes a note approved answer evidence.

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

The bot starts with watch mode enabled — ordinary vault changes are indexed
incrementally. Capture and curate sources through the [manual wiki workflow](#manual-source-backed-wiki)
to establish answer evidence; an empty wiki gives an explicit unsupported answer,
not an automatic search of raw notes.

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
| "What did I write about X?" | Answer from matching approved current wiki pages, or unsupported |
| "Tell me about my projects" | Same approved-current answering policy                    |
| Follow-up after `/think`    | Continues thinking session (sticky)                        |
| URL in message              | Auto-ingests YouTube transcript or article content         |
| Photo/image                 | Described via vision model, saved as note with image embed |

Capture prefixes: `note:`, `save:`, `capture:`, `remember:`, `jot:`, `log:`

Set `capture_all = true` in `[routing]` config to restore old behavior (all text → capture).

The table describes routing outside an active thinking session; sticky follow-ups
use approved-current answering rather than recapture, unless `capture_all` is set.
This is a breaking evidence change: old conversational answers and unapproved notes
are not factual authority. See [session and idle-producer boundaries](docs/telegram-bot.md#thinking-partner).

### Commands

| Command                         | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| `/recall <query>`               | Unapproved source-only discovery (paginated)                     |
| `/think <topic>`                | Approved-current answering with persisted conversational intent |
| `/think explore: <topic>`       | Presentation prefix; same approved evidence policy               |
| `/think critique: <topic>`      | Presentation prefix; same approved evidence policy               |
| `/think synthesize: <topic>`    | Presentation prefix; same approved evidence policy               |
| `/think plan: <topic>`          | Presentation prefix; same approved evidence policy               |
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
| `knowledge_query`  | Approved-current answer or evidence-only result, with citations and selection limits |
| `vault_search`     | Unapproved source-only discovery with optional note type filter   |
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
| `researcher` (default) | Read-only: approved answering, source search, read, list, graph, introspection | All |
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

Precedence, highest priority first: deliberate constructor/call-site overrides → `VAULTMIND_*` environment variables → cwd `.env` → selected TOML → model defaults. `--config PATH` selects TOML; it does not promote TOML above environment variables. Nested overrides use `__`, and list values use JSON arrays. Values rejected by model field types or validators raise a configuration error rather than falling back to lower-priority sources. Loop schedule strings (`insight_schedule`, `evolution_schedule`, `procedural_schedule`) are not cron-validated during settings loading: job assembly accepts valid cron expressions or converts recognized natural-language schedules, while an invalid nonempty schedule logs a warning and uses the cron expression derived from its corresponding `*_interval_days` setting.

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

`/recall` and `vaultmind bench` share source-relevance classification,
`retrieval_confidence(hits, score_floor)` in `indexer/ranking.py`. It reads **only
the raw vector distance** (`hit["distance"]`, lower is closer) against
`[bench].score_floor` (default `0.5`). This is raw exploration, not the
approved-current answering policy, and answering no longer uses it to mint gaps:

| State     | When                                                                                           | Effect                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `strong`  | any hit has a raw distance strictly below the floor                                            | no decline credit (a strong result is not an honest decline) |
| `weak`    | no hits at all, or every hit has a raw distance at or above the floor                          | earns deterministic decline credit in `vaultmind bench` |
| `unknown` | non-empty results where no hit is below the floor and at least one hit has no raw distance     | no decline credit; missing raw distance renders as `relevance unknown`, not a percentage |

Raw distance is never replaced by a fusion pseudo-distance. In hybrid search a lexical-only (BM25) hit has `distance: None` — it has keyword evidence but no vector evidence — so it can never prove relevance and never proves irrelevance. `rrf_score` (fusion) and `authority_score` (source-trust reweighting) only change *ordering*; they never change a hit's `distance` or the result set's confidence.

`vaultmind bench --bundle <dir>` reads `<dir>/golden.yaml`, `<dir>/retrieval.yaml` and actual Markdown under `<dir>/vault/`. Retrieval YAML maps exact questions to indexed candidates with `note_path`, optional `chunk_index` (default 0), optional `distance` and optional `rrf_score`; it cannot supply content or authority. The loader parses/chunks actual files once, while the shared query independently reads their current bytes. A missing or `null` distance is unknown evidence, never a default good/bad distance. The bundle vault replaces the operator vault for this run; fixture scores measure source discovery, not approved-answer quality.

`scripts/verify_acceptance.py` is a small real-surface smoke for this behaviour:

```bash
uv run python scripts/verify_acceptance.py --case retrieval --offline
```

It builds temporary synthetic state from explicit TOML, indexes real Markdown
through the parser, then drives `/recall` and the benchmark runner. A real
lexical-only current file has a deliberately inserted BM25 row but no vector.
It checks exact self-match, empty/unknown evidence, threshold boundaries, shared
recall/bench results, authority ordering with unchanged raw distances, scope
filtering before top-k and current-byte hydration instead of stale indexed
excerpts. Boundary-distance substitutions use pre-created corpus chunks. The
JSON receipt records expected/observed checks and runtime/corpus identities.

Smoke limits — embeddings are deterministic stand-ins and Telegram delivery is an
in-process recorder: these checks demonstrate source-discovery plumbing and
raw-distance semantics, not retrieval quality, approved-answer quality or live
provider behaviour. Exact threshold substitutions are labeled separately from
the real storage path. The separate `wiki-query` and `mcp` offline cases exercise
approved-current answering with scripted providers; none qualifies live-model
entailment or factual quality. Network attempts fail the offline run.

## Architecture

```text
Capture → preserved source → explicit propose/review/apply → current wiki page
CLI ask / Telegram questions and thinking / MCP knowledge_query
    → fresh approved-current validation and selection → one bounded completion
Raw /recall / vault_search / search-sources → unapproved exploration only
```

The independent domain components below remain available; they are not additional
answer-evidence inputs.

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
