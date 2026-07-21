"""Connector implementations for the closed registry.

Importing this package registers all three connectors
(`rss`/`youtube-channel`/`github-activity`) into `sources.registry.REGISTRY`
— the explicit, closed bootstrap point (no dynamic/plugin discovery).
`sources/pipeline.py::run_connector_once` and the `vaultmind source` CLI
both import this package before calling `registry.get_connector()`.
"""

from __future__ import annotations

from vaultmind.sources.connectors.github_activity import GITHUB_ACTIVITY_CONNECTOR
from vaultmind.sources.connectors.rss import RSS_CONNECTOR
from vaultmind.sources.connectors.youtube_channel import YOUTUBE_CHANNEL_CONNECTOR
from vaultmind.sources.registry import register_connector

register_connector(RSS_CONNECTOR)
register_connector(YOUTUBE_CHANNEL_CONNECTOR)
register_connector(GITHUB_ACTIVITY_CONNECTOR)

__all__ = [
    "GITHUB_ACTIVITY_CONNECTOR",
    "RSS_CONNECTOR",
    "YOUTUBE_CHANNEL_CONNECTOR",
]
