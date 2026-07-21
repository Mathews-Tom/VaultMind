"""Source connector framework — explicit registry, durable cursors, closed set.

`rss`, `youtube-channel`, and `github-activity` are the only connector kinds
(no dynamic/plugin-based loading — `registry.ALLOWED_KINDS` is a closed set).
Each configured `SourceInstance` (from `config/sources.toml`) tracks durable
per-instance cursor state (`ConnectorState`) in `~/.vaultmind/data/sources.db`
so reruns ingest only items newer than the stored cursor.
"""

from __future__ import annotations
