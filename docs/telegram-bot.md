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

### Search and Discovery

| Command       | Usage                      | Description                                                                                                    |
| ------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `/recall`     | `/recall machine learning` | Semantic search over the vault. Results are paginated (5 per page) with navigation buttons                     |
| `/suggest`    | `/suggest my-note.md`      | Find notes worth linking. Uses composite scoring: semantic similarity + shared graph entities + graph distance |
| `/duplicates` | `/duplicates my-note.md`   | Find duplicate or merge-candidate notes. Bands: duplicate (>= 92%), merge (80-92%)                             |

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

New notes landing in the duplicate-detector's 80-92% merge band are checked for material conflict by an LLM detection surface. `[contradiction].auto_resolve` is `false` by default, so every detected conflict escalates: VaultMind sends a Telegram message with an inline "Acknowledge" button and mints a knowledge gap (`/gaps`). Resolved conflicts only ever mark the losing note with a `contradicted_by` frontmatter field and a `> [!warning]` callout — the note's body is never edited or deleted.

Every automated mutation proposal (tag suggestions, duplicate-merge candidates, contradiction resolutions, maturation synthesis) is routed through one of three autonomy lanes by confidence x impact:

- **AUTO** — applies and logs without interaction (high confidence, low impact)
- **SKIM** — batches into the weekly digest and `/review`'s "Pending Review" section for one-tap approve-all
- **BLOCK** — sends an immediate Telegram inline-keyboard confirmation (approve/reject) and withholds application until answered

Run `vaultmind learn` to see the approval-fatigue percentage — how often a proposal needed a human in the loop.

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
