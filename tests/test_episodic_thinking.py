"""Legacy graph/episode/session assertions cannot become approved query facts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from tests.test_capture import make_message, registered_callback
from tests.test_knowledge_query import CACHE, cache_body, snapshot
from tests.test_knowledge_transports import delivered
from vaultmind.config import GraphConfig
from vaultmind.graph.knowledge_graph import KnowledgeGraph
from vaultmind.knowledge.query import query_knowledge
from vaultmind.memory.gaps import GapStore
from vaultmind.memory.store import EpisodeStore

if TYPE_CHECKING:
    from tests.test_knowledge_transports import Transport


@pytest.mark.asyncio
async def test_legacy_ninety_day_sources_do_not_override_current_conflict(
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = transport
    (t.corpus.vault / "legacy-cache.md").write_text(
        "# Cache retention\n\nAuthoritative raw-note policy is 90 days.\n"
    )
    queue = MagicMock()
    t.handlers._ctx.review_queue = queue
    graph_path = t.corpus.vault.parent / "legacy-graph.json"
    graph = KnowledgeGraph(GraphConfig(persist_path=graph_path))
    graph.add_entity("Cache retention", "concept", description="Authoritative policy is 90 days.")
    graph.save()
    episodes = EpisodeStore(t.corpus.vault.parent / "episodes.db")
    episode = episodes.create("Use 90-day cache retention", entities=["cache", "retention"])
    gaps = GapStore(t.corpus.vault.parent / "gaps.db")
    t.handlers._ctx.graph = graph
    t.handlers._ctx.episode_store = episodes
    t.handlers._ctx.gap_store = gaps
    t.sessions.save(42, [{"user": "cache retention", "assistant": "The policy is 90 days."}])
    t.sessions.save_summary(42, "default", 0, 0, 0, "Settled on 90 days.", ["cache"], [])
    body = cache_body() + (
        "\n## Conflicting information\n\nRetention is 30 days or 7 days; unresolved. "
        "[[_wiki/sources/cache-source]]\n"
    )
    t.corpus.publish(CACHE, body)
    approved = query_knowledge(t.corpus.settings, "cache retention", evidence_only=True)
    cited = [row["id"] for row in approved["evidence"] if "30 days" in row["text"]]
    t.provider.response = dict(
        status="conflicting",
        answer="The 30-day and 7-day claims conflict.",
        citations=cited,
    )
    before = snapshot(t.corpus.vault)
    before_graph = graph_path.read_bytes()
    before_episode = episodes.get(episode.episode_id)
    before_gaps = list(gaps._conn.iterdump())
    message = make_message("What is the cache retention policy?")
    await registered_callback(t.handlers, "handle_text", monkeypatch)(message)
    prompt = t.provider.calls[0][0].content
    assert "90 days" not in prompt and "90-day" not in prompt
    assert json.loads(prompt)["intent_questions"] == ["cache retention"]
    assert "Status: conflicting" in delivered(message)
    assert all(section in delivered(message) for section in cited)
    assert snapshot(t.corpus.vault) == before
    assert graph_path.read_bytes() == before_graph
    assert episodes.get(episode.episode_id) == before_episode
    assert list(gaps._conn.iterdump()) == before_gaps
    assert len(t.provider.calls) == 1
    assert queue.mock_calls == []
