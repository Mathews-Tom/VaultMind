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

The configured `allowed_user_ids` rule applies to mutation callbacks as well as commands, including approve, reject, approve-all, retry, delete/edit confirmation and cancellation. An empty list continues to allow all users. Edit confirmation/cancellation additionally requires the payload owner to match the callback actor; another user's pending edit is never consumed.

## Message Routing

Plain text messages are classified automatically using a heuristic-first router. No command prefix is needed for most interactions.

### Capture

Messages matching any of these patterns are saved directly to the inbox:

- **Explicit prefix**: `note:`, `save:`, `capture:`, `remember:`, `jot:`, `log:`
- **Multiline paste**: 3+ lines (pasting = intentional capture)
- **Long text**: 500+ characters
- **URL in message**: Creates a vault note from the YouTube transcript or article content
- **Photo/image**: Described via vision model, saved as note with `![[images/...]]` embed

The prefix is stripped before saving. The note gets `type: fleeting` frontmatter and is indexed immediately.

### Greetings

Short messages matching common greetings ("hi", "thanks", "ok", "hello") get a static response. No LLM call is made.

### Questions and Chat

Everything else goes through vault-augmented LLM response:

1. The message is searched against ChromaDB for relevant vault chunks
2. Relevant chunks are injected as context
3. The LLM responds with vault-aware answers

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

`/edit` generates its replacement from current content only; `/read` still returns the complete saved body and prior revisions within the normal Telegram message limit. Full-body readers preserve authored `---` rules: YAML frontmatter is removed once by the parser, not guessed from the parsed body. After a successful incremental, watcher or full index, only the latest edit is searchable; soft-deleted notes stay absent from vector, keyword and hybrid results. Ordinary authored supersede-looking callouts remain searchable. Matching legacy bot markers are recognized, but ambiguous history boundaries fail indexing rather than guessing or rewriting the note.

Saving Markdown and indexing are separate operations. If indexing fails, capture/bookmark/distill/edit/delete replies state the failure instead of claiming complete success. Saved text/history and edit/delete lineage remain available. A delete confirmation replay retries retirement without appending another callout or lineage edge. An explicit later index operation or a new watcher event can repair derived state; no background retry service or cross-database atomicity is promised.

### Search and Discovery

| Command       | Usage                      | Description                                                                                                    |
| ------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `/recall`     | `/recall machine learning` | Semantic search over the vault. Results are paginated (5 per page) with navigation buttons                     |
| `/suggest`    | `/suggest my-note.md`      | Find notes worth linking. Uses composite scoring: semantic similarity + shared graph entities + graph distance |
| `/duplicates` | `/duplicates my-note.md`   | Find duplicate or merge-candidate notes. Bands: duplicate (>= 92%), merge (80-92%)                             |

Note lookup tries a relative path, a `.md` suffix, matching filenames and then one semantic result. Symlink loops, unreadable candidates and overlong path components do not prevent a valid later lookup strategy. Paths escaping the vault remain explicit containment errors; unexpected filesystem stat errors are not suppressed.

### Thinking Partner

| Command              | Usage                                | Description                                                                                |
| -------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `/think`             | `/think Should I learn Rust?`        | Start a thinking partner session. The session persists across bot restarts (SQLite-backed) |
| `/think explore:`    | `/think explore: future of PKM`      | Divergent ideation — generates multiple angles and possibilities                           |
| `/think critique:`   | `/think critique: my startup idea`   | Stress-tests an idea, finds weaknesses and counterarguments                                |
| `/think synthesize:` | `/think synthesize: AI + education`  | Connects dots across domains, finds non-obvious relationships                              |
| `/think plan:`       | `/think plan: migrate to Kubernetes` | Creates a structured execution plan with steps and dependencies                            |

After starting a `/think` session, follow-up messages within the TTL (default: 1 hour) continue the same session automatically. No need to use `/think` again.

A session that reaches `[distill].min_turns` (default 3) and then idles out is automatically distilled by an LLM into a `qa-artifact` note (question, resolution, systems, participants) — indexed immediately and run through episodic-memory extraction. Auto-distillation is opt-in (`[distill].enabled = false` by default). Run `/distill` any time to distill the current session manually, regardless of the config flag.

The thinking partner draws context from both ChromaDB (semantic search) and the knowledge graph (entity relationships, ego subgraph).

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
| `/gaps` | `/gaps` | List open knowledge gaps by age — unanswered questions, weak-retrieval `/recall` misses, and contradiction escalations. Auto-stale after `[gaps].stale_after_days` (default 30). Run `vaultmind research "<question>"` on a gap to close it |

### System

| Command   | Usage     | Description                                                                                        |
| --------- | --------- | --------------------------------------------------------------------------------------------------- |
| `/review` | `/review` | Weekly review with graph insights, trending topics, connection suggestions, and a "Pending Review" section listing SKIM-lane autonomy proposals with a one-tap approve-all button |
| `/health` | `/health` | System health check — reports status of ChromaDB, graph, watcher, and LLM                           |
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

Source proposal notifications use this same delivery path, but source write/index/event finalization and durable cursor-completion guarantees remain a separate M15 remediation. An applied queue operation is not evidence that those downstream source steps completed. Offline approval verification uses temporary fixture state, scripted LLM responses, synthetic embeddings and a recorded Telegram transport; it does not qualify native providers, embeddings, production migrations or actual Telegram delivery.

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

1. The photo is downloaded and sent to the configured vision model
2. The model generates a description of the image content
3. The original image is saved to `00-inbox/images/{timestamp}.jpg` (configurable via `[image].save_originals`)
4. A fleeting note is created with an `![[images/{filename}]]` embed and the AI description
5. The note is indexed immediately

Configure the vision model in `[image].vision_model` (default: uses `llm.fast_model`).

## Voice Messages

Send a voice message and VaultMind transcribes it using OpenAI Whisper:

- If the transcription ends with `?` → routed as a question (vault-augmented LLM response)
- Otherwise → captured to inbox as a fleeting note

Requires `VAULTMIND_OPENAI_API_KEY` in `.env`.

## Confirmation Flows

Destructive operations (`/edit`, `/delete`) and BLOCK-lane autonomy proposals (including contradiction resolutions) use aiogram inline keyboards:

1. You send the command, or an automated proposal routes to the BLOCK lane
2. VaultMind shows a preview with Confirm/Cancel (or Approve/Reject) buttons
3. You tap to proceed or abort

`/edit`/`/delete` never overwrite existing note content on confirm — see [Capture and Notes](#capture-and-notes) above for the non-destructive supersede-callout behavior.

## Session Persistence

Thinking sessions are stored in SQLite at `~/.vaultmind/data/sessions.db` with a composite key of `(user_id, session_name)`. Sessions survive bot restarts. The TTL is configurable via `[telegram].thinking_session_ttl`.
