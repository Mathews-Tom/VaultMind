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
| `/edit`     | `/edit my-note.md add a summary` | AI-assisted edit. Shows a diff preview with confirm/cancel buttons                                       |
| `/delete`   | `/delete my-note.md`             | Delete a note. Shows confirmation with inline keyboard                                                   |
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

### System

| Command   | Usage     | Description                                                                    |
| --------- | --------- | ------------------------------------------------------------------------------ |
| `/review` | `/review` | Weekly review with graph insights, trending topics, and connection suggestions |
| `/health` | `/health` | System health check — reports status of ChromaDB, graph, watcher, and LLM      |
| `/stats`  | `/stats`  | Vault and graph statistics (note counts, types, entities, relationships)       |
| `/help`   | `/help`   | Quick reference for all commands                                               |

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

Destructive operations (`/edit`, `/delete`) use aiogram inline keyboards:

1. You send the command
2. VaultMind shows a preview with Confirm/Cancel buttons
3. You tap Confirm to proceed or Cancel to abort

This prevents accidental edits or deletions.

## Session Persistence

Thinking sessions are stored in SQLite at `~/.vaultmind/data/sessions.db` with a composite key of `(user_id, session_name)`. Sessions survive bot restarts. The TTL is configurable via `[telegram].thinking_session_ttl`.
