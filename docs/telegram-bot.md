# Telegram Bot Guide

## Setup

1. Create a bot via [@BotFather](https://t.me/botfather) on Telegram
2. Set the bot token in `.env`:
   ```text
   VAULTMIND_TELEGRAM__BOT_TOKEN=123456:ABC-DEF...
   ```
3. Optionally restrict access to specific users:
   ```text
   VAULTMIND_TELEGRAM__ALLOWED_USER_IDS=[123456789,987654321]
   ```
4. Start the bot:
   ```bash
   uv run vaultmind bot
   ```

The bot starts with watch mode enabled — vault changes are indexed incrementally in the background.

Before answering, curate approved current wiki pages through the
[source → propose → review → apply workflow](../README.md#manual-source-backed-wiki).
Set `[vault].excluded_folders` to include both `wiki` and `_wiki`; defaults already
do, but an older explicit list needs a deliberate setup edit. The application
does not rewrite that list or fall back to raw notes. Capture/index/graph
operations remain independent tools, not approval steps.

After capture, use `/recall` for explicitly unapproved discovery, select returned preserved source IDs in CLI `wiki propose`, then inspect `wiki review` and apply with its exact digest. Source-ingestion, maturation or other ordinary-note approvals do not approve a wiki page. A new contradictory capture leaves approved answer evidence unchanged until that separate reviewed page update is applied. Once applied, normal questions and `/think` automatically use eligible approved current pages; a selected unresolved conflict must remain conflicting (or unsupported), just as on CLI and MCP.

Do not restore earlier wording by overwriting the current page from history. Use a new reviewed proposal against the current revision, retaining source citations and attributed conflicts or an explicit reasoned resolution; [the restore example](../README.md#restore-earlier-wording-through-review) preserves the exact pre-restore historical bytes. Manually changed current pages and incomplete apply operations are refused, not silently used or repaired. See [configuration migration](configuration.md#llm) for removed answering controls and their environment/dotenv counterparts.

All generated knowledge answers use the selected provider's
`[llm].thinking_model` and `[llm].max_tokens`; there is no separate chat answering
model. The bot still creates its shared domain client at startup, and unrelated
vision, transcription, editing and scheduled producers retain their own provider
requirements.

The configured `allowed_user_ids` rule applies to mutation callbacks as well as commands, including approve, reject, approve-all, retry, delete/edit confirmation and cancellation. An empty list continues to allow all users. Edit confirmation/cancellation additionally requires the payload owner to match the callback actor; another user's pending edit is never consumed.

## Message Routing

Plain text messages are classified automatically using a heuristic-first router. No command prefix is needed for most interactions.

### Capture

Messages matching these patterns enter source capture, followed by an ordinary inbox projection:

- **Explicit prefix**: `note:`, `save:`, `capture:`, `remember:`, `jot:`, `log:`
- **Multiline paste**: 3+ lines (pasting = intentional capture)
- **Long text**: 500+ characters
- **URL in message**: Preserves submitted text first; a successful YouTube transcript or article extraction becomes a separate parent-linked source
- **Photo/image**: Preserves the original only when `save_originals` is enabled, then records provider-derived vision text and a linked ordinary note

Text preservation precedes downstream projection and indexing. Replies distinguish
the source ID/hash from provider/derivative, projection and index dispositions;
preservation is not approval and does not promise that every later stage succeeded.
Use a returned source ID explicitly with `wiki propose --source SOURCE_ID` before
reviewing and applying any accepted page. The ordinary projection is for note UX,
not an automatic curation decision.

URL extraction preserves extracted text, not original HTTP response bytes.
Fetch failure retains the submitted text and reports a failed derivative without
fabricating extracted content. Voice capture preserves a transcript with explicit
audio nonretention; downloaded audio is transient. Voice questions follow the
question route and do not create source captures.

Where Telegram supplies a stable event/revision identity, replay does not repeat
provider, derivative, projection or index effects. Changed bytes get a new immutable
source and separate equal-byte events remain distinct. Failed/incomplete stages
are visible and not automatically retried. Partial success can leave preserved
evidence without an indexed note; this is not an atomic write/index transaction.
Text retention restrictions are checked before preservation, and source readers
verify the complete parent/attachment chain before discovery or curation.

Capture, photo and transcription refusals log only the exception class, not input bodies or exception messages. Invalid retained replay evidence refuses the event without another transcription/provider attempt; it is not treated as an uncaptured event.

Malformed or orphan provenance may also block a later capture/replay. Stop
affected captures and preserve all files/state. An authorized operator can use
the existing read-only workspace/source reader to identify the failing retained
ID, with actual dependency permissions; see
[source replay diagnosis](architecture.md#source-intake-and-discovery). Record the
ID and failure class privately, never bodies, hidden paths or exception messages
in responses/logs. Diagnosis does not repair: do not delete/quarantine files,
rewrite hashes, skip records, mark success, reset reservations or restart to force
replay. Reconciliation needs separate authorization; preservation guarantees
neither automatic retry nor index completion.

### Greetings

Short messages matching common greetings ("hi", "thanks", "ok", "hello") get a static response. No LLM call is made.

### Questions and Chat

Question and conversational classifications use the same approved-current query
core as CLI `ask`, selected `wiki query` and MCP `knowledge_query`:

1. Validate eligible approved current pages and all retained provenance.
2. Automatically select at most eight pages with disclosed deterministic lexical
   matching; synonyms without shared terms can be missed.
3. Supply complete selected sections to one configured answering completion,
   validate status/citations/conflicts, then revalidate identities before return.

No matching approved evidence gives deterministic `unsupported`, without calling
the answering factory. There is no raw-note, vector, graph, episode, web or
old-answer fallback. `supported` and `conflicting` outcomes cite supplied section
IDs; unresolved conflicts cannot silently become a winner. Invalid provider output
and changed evidence are explicit refusals, not shortened or fallback answers.
Citation/schema checks are structural, not proof of factual entailment.

Replies retain the accepted answer, citations, complete section-to-page/source
mapping, selected revisions, source identities, unresolved conflicts and lexical
selection qualifications. They use plain text (`parse_mode=None`), split into
ordered chunks of at most 4000 characters with no clipping, citation rewriting or
loss of boundary whitespace. The core uses complete sections, not index excerpts.

Set `capture_all = true` in `[routing]` config to bypass routing and capture all plain text.

## Commands

### Capture and Notes

| Command     | Usage                            | Description                                                                                              |
| ----------- | -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `/daily`    | `/daily`                         | Get or create today's daily note                                                                         |
| `/notes`    | `/notes yesterday`               | Find notes by date. Accepts natural language: `yesterday`, `last week`, `over the weekend`, `2026-02-20` |
| `/read`     | `/read my-note.md`               | Read full note content. Accepts path, filename, or search term                                           |
| `/edit`     | `/edit my-note.md add a summary` | AI-assisted edit. Shows a diff preview with confirm/cancel buttons. On confirm, the new text is prepended above a dated `> [!superseded]` callout — the prior body is preserved, never overwritten |
| `/delete`   | `/delete my-note.md`             | Soft-delete a note. Shows confirmation with inline keyboard. On confirm, the note is marked with a `> [!superseded]` callout and dropped from `/recall` results — the file and its full original content remain on disk |
| `/bookmark` | `/bookmark My Insights`          | Save the current thinking session or last Q&A exchange to vault as a permanent note                      |

### Search and Discovery

| Command       | Usage                      | Description                                                                                                    |
| ------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `/recall`     | `/recall machine learning` | Source-only discovery over current ordinary notes and preserved text, with paginated results |
| `/suggest`    | `/suggest my-note.md`      | Find notes worth linking. Uses composite scoring: semantic similarity + shared graph entities + graph distance |
| `/duplicates` | `/duplicates my-note.md`   | Find duplicate or merge-candidate notes. Bands: duplicate (>= 92%), merge (80-92%)                             |

Note lookup tries a relative path, a `.md` suffix, matching filenames and then one semantic result. Symlink loops, unreadable candidates and overlong path components do not prevent a valid later lookup strategy. Paths escaping the vault remain explicit containment errors; unexpected filesystem stat errors are not suppressed.

`/recall` shares its current-file policy with CLI `search-sources`, MCP
`vault_search` and `vaultmind bench`. Hits are explicitly unapproved, with
current-byte digests/provenance; missing or stale indexed distance is unknown,
not zero. History, pending/control state and current `wiki` pages are excluded.
Changing a paginated source invalidates its snapshot and requires a fresh search.
These results are `source-only`, `approved: false`: use them to inspect and
curate, not as answer evidence. Normal chat, `/think` and voice questions already
use the same approved-current policy as root `ask` and explicit `wiki query`.

### Thinking Partner

| Command              | Usage                                | Description                                                                                |
| -------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `/think`             | `/think What is our cache policy?`  | Approved-current answering with persisted conversational intent |
| `/think explore:`    | `/think explore: cache alternatives` | Presentation prefix; unchanged evidence/model policy |
| `/think critique:`   | `/think critique: cache policy`    | Presentation prefix; unchanged evidence/model policy |
| `/think synthesize:` | `/think synthesize: cache policies` | Presentation prefix; unchanged evidence/model policy |
| `/think plan:`       | `/think plan: cache migration`     | Presentation prefix; unchanged evidence/model policy |

After `/think`, follow-ups within `[telegram].thinking_session_ttl` (default one
hour) continue the active session, including after restart. Sticky session routing
takes precedence over ordinary capture/greeting heuristics; `capture_all` instead
captures all plain text.

Only the last three complete prior user questions supply intent and fresh lexical selection terms. Their claims are not evidence. Prior assistant answers, stored summaries, graph and episode facts never enter the answering envelope. Current and selected prior questions must each fit 4096 characters, and the complete question/intent/evidence envelope must fit 128 KiB. Newlines, carriage returns and tabs remain intact; whitespace-only input and other Unicode control/format characters are rejected. Invalid or oversized input fails without truncating history or silently dropping pages. Old sessions therefore do not preserve their former raw-note factual authority after this cutover.

Asking does not summarize, extract entities or episodes, mint gaps, distill,
create proposals, promote answers or mutate accepted pages. Session-history
persistence is separate from the evidence boundary. An expired session remains
intact; a new query starts a fresh generation without archiving or scheduling a
producer.

Idle finalization is independent: only `[distill].enabled = true` registers a one-minute job on the bot's existing scheduler. It archives full expired histories and existing summaries, claims records before generation, and attempts ordinary `qa-artifact` distillation for sessions meeting `[distill].min_turns` (default 3). Every in-flight question keeps its conversation generation active across TTL expiry; overlapping questions must all finish or unwind before idle claiming. Explicit clearing prevents a late completion from restoring deleted history. The producer preserves the selected revision and exact generation request before completion, retains historical summaries without treating them as current answer evidence, and creates only an unapproved parent-linked derivative. Indexing, episodic extraction and unresolved-gap production are separate downstream producer effects, not query effects or wiki approval. Completed/skipped/failed archives and interrupted claims retain their originals; failed or claimed work is not automatically retried on polling or restart. Distillation/indexing is not atomic and a failure may have partial effects.

`/distill` explicitly distills the active session, or the last Q&A exchange when there is no active session, even when automatic distillation is disabled. It copies the selected conversation and participants before yielding, retains the authenticated capture identity and generation request, and reports the unapproved derivative source ID and index disposition. Provider failure leaves the input source preserved; it does not approve or fabricate a generated note. Select the reported source in a wiki proposal, inspect the exact diff/digest and apply it separately to publish answer evidence.

`/bookmark` remains an explicit ordinary-note writer with its existing limits. Neither command turns output into approved-current evidence without wiki review.

### Knowledge Graph

| Command   | Usage           | Description                                                                                               |
| --------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| `/graph`  | `/graph Python` | Query an entity's connections, neighbors, and relationships in the knowledge graph                        |
| `/evolve` | `/evolve`       | Show belief evolution signals: confidence drift, relationship shifts, stale claims                        |
| `/mature` | `/mature`       | Show Zettelkasten maturation clusters — groups of fleeting notes ready for synthesis into permanent notes |

### Episodic Memory

| Command      | Usage                                              | Description                                                                              |
| ------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `/decide`    | `/decide Use PostgreSQL for the auth service`      | Record a decision (creates a pending episode). Returns episode ID                        |
| `/outcome`   | `/outcome a1b2c3 success Worked well, low latency` | Resolve a decision with outcome status (`success`, `failure`, `partial`) and description |
| `/episodes`  | `/episodes` or `/episodes Python`                  | List recent episodes (pending first), optionally filtered by entity                      |
| `/workflows` | `/workflows`                                       | List active workflow patterns with success rates (requires procedural memory enabled)    |
| `/workflow`  | `/workflow w1x2y3`                                 | Show workflow steps and trigger pattern                                                  |

### Knowledge Gaps

| Command | Usage   | Description                          |
| ------- | ------- | ------------------------------------- |
| `/gaps` | `/gaps` | List existing open gaps by age, including independent producer/contradiction escalations. Auto-stale after `[gaps].stale_after_days` (default 30); approved-current answering does not mint gaps. Research can close its originating gap |

### System

| Command   | Usage     | Description                                                                                        |
| --------- | --------- | --------------------------------------------------------------------------------------------------- |
| `/review` | `/review` | Weekly review with graph insights, trending topics, connection suggestions, and a "Pending Review" section listing SKIM-lane autonomy proposals with a one-tap approve-all button |
| `/health` | `/health` | On-demand vector-store, graph, vault, LLM-key configuration, and graph-file status; not a live provider-authentication or watcher-liveness check |
| `/stats`  | `/stats`  | Vault and graph statistics (note counts, types, entities, relationships)                             |
| `/help`   | `/help`   | Quick reference for all commands                                                                     |

## Contradiction Detection & Autonomy Review

New notes landing in the duplicate-detector's 80-92% merge band are checked for material conflict by an LLM detection surface. `[contradiction].auto_resolve` is `false` by default, so every detected conflict escalates: with the gap ledger enabled, VaultMind mints a knowledge gap (`/gaps`); with autonomy enabled, it also queues an informational BLOCK proposal. With bot notifications enabled, the proposal sends an Approve/Reject keyboard; Approve acknowledges the escalation, not a vault mutation. With autonomy disabled there is no queue proposal or review notification; the enabled gap ledger remains the durable escalation record. Resolved conflicts only ever mark the losing note with a `contradicted_by` frontmatter field and a `> [!warning]` callout — the note's body is never edited or deleted.

Every automated proposal (tag application/vocabulary, duplicate-merge suggestions, contradiction resolution/escalation, maturation synthesis and source ingestion) uses the shared review queue. Confidence-based proposals route by confidence × impact; duplicate suggestions and maturation use SKIM by default, while explicit `[autonomy].force_block = true` sends them to BLOCK too.

- **AUTO** — applies and logs without interaction (high confidence, low impact)
- **SKIM** — batches into the weekly digest and `/review`'s "Pending Review" section for one-tap approve-all
- **BLOCK** — stays pending and unapplied until an authorized user resolves it. A newly pending proposal on an enabled bot path sends one Approve/Reject keyboard after the queue transaction commits.

Run `vaultmind learn` to see the approval-fatigue percentage — how often a proposal needed a human in the loop.

CLI and bot approvals use the same tag operations: tag application merges frontmatter tags without discarding note text or existing metadata; vocabulary approval updates the existing `~/.vaultmind/data/tag_vocabulary.json` file (`{"approved": [...]}`). Note targets must pass the vault path-security boundary before writing.

`vaultmind auto-tag --approve-skim` processes only pending tag-application and tag-vocabulary proposals. Other proposal kinds remain pending for their owning bot operations. The command reports actual applied, acknowledged and failed counts; a failed tag operation exits with an error rather than claiming approval.

`vaultmind auto-tag --apply` reports unique proposal identities, newly applied operations, previously resolved proposals without a new mutation, failures and pending review. Several notes suggesting the same new vocabulary item do not inflate queue counts. Existing human rejection or completed application remains terminal; a repeated suggestion does not claim a fresh mutation. Operational failures exit nonzero.

Only contradiction escalation and duplicate-merge suggestions are informational: approving them records `acknowledged`, not an executed merge. Mutation kinds require an actual applier. Missing operations and raised operations record `failed` with an error result, never `applied` or human `rejected`. Approve-all reports applied, acknowledged and failed counts separately. Failed operations can have partial effects: inspect the error and state before retrying. A stale maturation cluster and an unsuccessful synthesis are failures, not applied notes.

Use the explicit Retry button on a failure response or `/review`'s failed-operation section to retry one failed proposal, including after a bot restart. `/review` lists every FAILED proposal in bounded plain-text message chunks, so permanently failing older items cannot hide newer recoverable ones. BLOCK and FAILED actions use distinct display ordinals (`B1`, `F1`) matched to the summary in the same message; summaries are clipped with an ellipsis, and each message has at most 80 buttons. Approve-all reports counts and directs failed items to `/review`, rather than attaching ambiguous retry buttons to a count-only response. The authorized callback decodes a lossless base64url token back to the unchanged SHA-256 proposal ID; keyboard payloads remain within Telegram's 64-byte limit. Reopening, producer deduplication and approve-all never retry failures. Applied, acknowledged and rejected proposals are terminal; replay does not repeat the mutation. A very large backlog can require several messages; live Telegram rate-limit recovery is not qualified.

BLOCK delivery reports `sent`, `disabled` (`notification_chat_id = 0`), `unavailable` (no active bot delivery collaborator, including CLI-created proposals), or `failed` (transport error). Deduplicated and terminal proposals do not send again. Undelivered pending BLOCK proposals remain available with per-item Approve/Reject buttons in `/review`; no polling service or automatic delivery retry runs. Generic proposal messages use plain text so vault-controlled summaries cannot inject Telegram Markdown.

Source proposal notifications use this same delivery path, but source write/index/event finalization and durable cursor-completion guarantees remain separate connector-recovery work. An applied queue operation is not evidence that those downstream source steps completed. Offline approval verification uses temporary fixture state, scripted LLM responses, synthetic embeddings and a recorded Telegram transport; it does not qualify native providers, embeddings, production migrations or actual Telegram delivery.

### Queue migration and rollback

Opening a queue upgrades a copied or live database additively: schema version 1 adds `legacy_status` and `legacy_result`. Three deterministic pre-upgrade execution failures become `failed`: old `rejected` rows whose result is exactly `applier raised — see logs`; known mutation kinds recorded as `acknowledged` with an empty result by the old missing-applier path; and `applied` maturation rows matching the old no-source, missing-frontmatter, frontmatter-parse-error or unresolved-wikilink failure messages case-sensitively, provided the result does not end in `.md`. Every successful old synthesis returns a `.md` path, including a configured inbox beginning with a failure prefix, so possible success paths remain unchanged. A failure message ending in `.md`, the old `cluster no longer available (dismissed or already synthesized)` marker, unknown results, informational acknowledgements, nonempty acknowledgements and genuine rejections remain unchanged as ambiguous history. Original status/result remain in the evidence columns; proposal IDs, payloads, results, creation/last-seen/resolution timestamps and occurrence counts remain intact. Opening the database never retries a mutation. Fixture verification does not migrate production data.

Back up the queue before a separately authorized deployment migration. Retain the upgraded database and its evidence if reverting code; the old code cannot decode new `failed` statuses. Use a pre-upgrade backup for an old-code rollback rather than relabeling operational failures as human rejections. Restoring that backup omits decisions made after it, so stop queue writers and preserve both copies before recovery.

## Proactive Notifications

VaultMind can send proactive messages to your Telegram chat when scheduled compound loops detect significant changes. This requires `notification_chat_id` to be set in config.

### Configuration

```toml
[telegram]
notification_chat_id = 123456789  # Your Telegram chat ID; 0 = disabled
```

### What Gets Notified

| Loop       | Triggers Notification When                                                            |
| ---------- | ------------------------------------------------------------------------------------- |
| Insight    | New trending searches, acceptance rate shift >15%, interaction volume change >50%     |
| Evolution  | High-severity belief drift signal, or entity appearing in 3+ consecutive weekly scans |
| Procedural | New workflow pattern synthesized from episodic memory                                 |

Notifications are gated by a significance filter — trivial results (< 20 chars) are dropped. Messages exceeding Telegram's 4096-char limit are automatically split.

## Photo Capture

Send a photo and VaultMind processes it using a vision model (Anthropic or OpenAI):

1. Download and validate the photo/caption. With `save_originals = true`, preserve
   the original attachment and attempt an ordinary
   `{inbox_folder}/images/{source_id}.jpg` projection; otherwise retain a textual
   event descriptor explicitly stating that the binary original was not retained.
2. Attempt one vision-description stage and preserve successful generated text as
   a separate parent-linked source. Vision text is not evidence of original retention.
3. Attempt an ordinary fleeting-note projection, adding an image embed only when
   the original projection succeeded, then attempt indexing.

Replies distinguish preservation, original projection, vision, derivative and
index outcomes. A failed stage is not automatically retried or reported as an
indexed note, and none of these stages approves a wiki page.

Configure the vision model in `[image].vision_model` (default: uses `llm.fast_model`).

## Voice Messages

Send a voice message and VaultMind transcribes it using OpenAI Whisper:

- If the transcription ends with `?` → the same approved-current answer route as text questions, without source capture
- Otherwise → preserve the transcript, then attempt an ordinary inbox projection and indexing

Requires `VAULTMIND_OPENAI_API_KEY` in `.env`.

## Confirmation Flows

Destructive operations (`/edit`, `/delete`) and BLOCK-lane autonomy proposals (including contradiction resolutions) use aiogram inline keyboards:

1. You send the command, or an automated proposal routes to the BLOCK lane
2. VaultMind shows a preview with Confirm/Cancel (or Approve/Reject) buttons
3. You tap to proceed or abort

`/edit`/`/delete` never overwrite existing note content on confirm — see [Capture and Notes](#capture-and-notes) above for the non-destructive supersede-callout behavior.

## Session Persistence

Thinking sessions are stored in SQLite at `~/.vaultmind/data/sessions.db`, with
`(user_id, session_name)` identities and a TTL controlled by
`[telegram].thinking_session_ttl`. Active sessions survive restarts; expired
histories are not deleted by a later question. Existing summaries are retained
for inspection/producer use, never answer evidence. The enabled independent idle
finalizer retains original histories and summaries in archives with actual
completion/failure/skip/claimed state. Explicit session clearing is distinct from
expiry and removes the user's retained conversation generations.
