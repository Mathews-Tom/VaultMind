"""Historical summaries persist for inspection, not knowledge-answer authority."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.test_knowledge_transports import Transport


@pytest.mark.asyncio
async def test_historical_summaries_preserved_but_not_sent_or_regenerated(
    transport: Transport,
) -> None:
    t = transport
    t.sessions.save(42, [{"user": "cache retention", "assistant": "90-day policy"}])
    t.sessions.save_summary(
        42, "default", 0, 0, 0, "The authoritative policy is 90 days.", ["cache"], ["Why 90?"]
    )
    before = t.sessions.get_summaries(42)
    result = await t.thinking.think(42, "What is the current policy?")
    assert result["status"] == "supported"
    assert len(t.provider.calls) == 1
    prompt = t.provider.calls[0][0].content
    assert "90 days" not in prompt and "90-day" not in prompt
    assert json.loads(prompt)["intent_questions"] == ["cache retention"]
    assert t.sessions.get_summaries(42) == before
    assert t.thinking._get_session(42).history[0]["assistant"] == "90-day policy"


def test_summary_roundtrip_replacement_and_session_isolation(transport: Transport) -> None:
    store = transport.sessions
    store.save_summary(42, "alpha", 0, 0, 1, "Original", ["topic"], ["question"])
    store.save_summary(42, "beta", 0, 0, 1, "Other", [], [])
    store.save_summary(42, "alpha", 0, 0, 1, "Revised", ["topic"], ["question"])
    summaries = store.get_summaries(42, "alpha")
    assert len(summaries) == 1
    assert summaries[0] == dict(
        batch_number=0,
        start_turn=0,
        end_turn=1,
        summary="Revised",
        topics=["topic"],
        questions=["question"],
    )
    store.delete(42, "alpha")
    assert store.get_summaries(42, "alpha") == []
    assert store.get_summaries(42, "beta")[0]["summary"] == "Other"
