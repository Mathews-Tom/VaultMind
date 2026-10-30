# AGENTS.md

## Code Review Rules

- **What failure means here:** `src/vaultmind/knowledge/query.py` answers from raw/unapproved notes, invents section citations, or presents unresolved conflicting evidence as supported; CLI, Telegram and MCP must share the approved-current answering policy.
- `src/vaultmind/wiki/workspace.py` applies an unreviewed/stale digest, overwrites accepted content without preserving history, or accepts tampered source/approval evidence.
- `src/vaultmind/knowledge/intake.py` or `src/vaultmind/sources/pipeline.py` loses original inputs before derived-note generation, or confuses preservation/indexing success with wiki approval.
- `src/vaultmind/mcp/auth.py` or `src/vaultmind/mcp/server.py` exposes notes, retained provenance or derived results outside the active profile's folder/tool/write permissions.
- **Do not report:** style, formatting or naming nits; diagnostics already enforced by Ruff check/format or mypy --strict; speculative hardening without a reachable failing scenario. Keep unrelated unchanged-code issues out of required fixes; tag any necessary context Pre-existing. Do not repeat findings already raised on an earlier commit of the same PR.
- **Paths to skip:** static data under `tests/fixtures/` (including synthetic benchmark vaults and source feeds), unless needed to establish a concrete changed-behavior failure; test code consuming those fixtures remains in scope. `uv.lock` stays in scope for dependency changes: flag packages added or version-changed without a matching `pyproject.toml` change; do not review it line by line.
- **Evidence and format:** each finding names `file:line`, the bug class and a concrete failing scenario grounded in the changed code; trace behavior rather than guessing from a name. Tag each finding Important / Nit / Pre-existing so the author can fix every instance of the same bug class.
- Start the summary with the count of Important findings; when zero, say plainly that there is nothing to fix. Omit excluded nits rather than filling the review with them.
- **Recurring rule:** completed wiki approvals validate accepted bytes, payload/source identities and operation/history evidence, not today's links or anchors on related pages; retain full candidate-link validation for pending proposals and apply (PRs #108, #109).
