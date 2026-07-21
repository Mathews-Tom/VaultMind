"""GitHub-activity connector — own repos' commits (DEVELOPMENT_PLAN.md M8).

Uses GitHub's public REST API via stdlib `urllib.request` + `json.loads` —
matching `vault/ingest.py::fetch_article()`'s existing "urllib, no extra
dependencies" convention. Unauthenticated requests work for public repos
(the plan's own scope) at GitHub's 60/hr unauthenticated rate limit,
acceptable for a periodic scheduled job. An optional token is honored via
`instance.options["token_env"]` naming an environment variable to read —
the token itself is never stored in `config/sources.toml` or `sources.db`.

Cursor is ID-position based (`sources/cursor.py::filter_new_by_id`): GitHub's
commits API returns commits newest-first by default.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import TYPE_CHECKING, Any

from vaultmind.sources.cursor import filter_new_by_id
from vaultmind.sources.models import ConnectorDefinition, FetchResult, SourceItem

if TYPE_CHECKING:
    from vaultmind.sources.models import ConnectorState, SourceInstance

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_DEFAULT_PER_PAGE = 30
_TIMEOUT_SECONDS = 15


def _github_request(url: str, token: str) -> list[dict[str, Any]]:
    """Fetch and parse one GitHub REST API JSON-array endpoint synchronously
    (called via `to_thread`). Returns an empty list for a non-list response
    (e.g. an error object) rather than raising — a repo with no commits, or
    an unexpected API shape, should not crash the whole connector run."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "VaultMind-Sources/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
        raw_bytes: bytes = resp.read()
    data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    return data if isinstance(data, list) else []


def parse_commits(raw_commits: list[dict[str, Any]], repo: str) -> list[SourceItem]:
    """Parse GitHub commits-API JSON entries into `SourceItem`s. Commits
    without a `sha` are skipped — there is nothing stable to key cursor
    advancement on."""
    items: list[SourceItem] = []
    for entry in raw_commits:
        sha = str(entry.get("sha", ""))
        if not sha:
            continue
        commit_info = entry.get("commit", {}) or {}
        message = str(commit_info.get("message", "")).strip()
        title = message.splitlines()[0] if message else sha[:7]
        author_date = str((commit_info.get("author") or {}).get("date", ""))
        html_url = str(entry.get("html_url", f"https://github.com/{repo}/commit/{sha}"))
        items.append(
            SourceItem(
                item_id=sha,
                title=f"{repo}: {title}",
                content=message,
                url=html_url,
                published_at=author_date,
            )
        )
    return items


async def fetch(instance: SourceInstance, state: ConnectorState) -> FetchResult:
    """Fetch new commits on `instance.target` (an `owner/repo` string)
    since `state.last_seen_id`."""
    token_env = instance.options.get("token_env", "")
    token = os.environ.get(token_env, "") if token_env else ""
    per_page = int(instance.options.get("max_listing", str(_DEFAULT_PER_PAGE)))

    url = f"{_API_BASE}/repos/{instance.target}/commits?per_page={per_page}"
    raw_commits = await asyncio.to_thread(_github_request, url, token)
    all_items = parse_commits(raw_commits, instance.target)
    new_items = filter_new_by_id(all_items, state.last_seen_id)

    next_cursor_id = all_items[0].item_id if all_items else None
    return FetchResult(items=new_items, next_cursor_id=next_cursor_id)


GITHUB_ACTIVITY_CONNECTOR = ConnectorDefinition(
    kind="github-activity",
    fetch=fetch,
    description="Own repos' commit activity as a daily activity note",
)


__all__ = ["GITHUB_ACTIVITY_CONNECTOR", "fetch", "parse_commits"]
