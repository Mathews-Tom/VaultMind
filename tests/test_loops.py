"""Tests for compound loop jobs."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaultmind.services.loops.evolution_loop import create_evolution_executor
from vaultmind.services.loops.insight_loop import create_insight_executor


class TestInsightLoop:
    @pytest.mark.asyncio
    async def test_first_run_no_notification(self) -> None:
        """First run has no prior state, so no shifts to detect."""
        store = MagicMock()
        store.get_counts.return_value = {}
        store.get_top_searches.return_value = []
        store.get_top_tags.return_value = []
        store.get_capture_topics.return_value = []
        store.get_active_hours.return_value = []

        execute = create_insight_executor(store)
        result = await execute({})

        assert result["notification"] is None
        assert "top_searches" in result
        assert "acceptance_rate" in result

    @pytest.mark.asyncio
    async def test_detects_new_search_trend(self) -> None:
        """Second run detects new top searches not in prior state."""
        store = MagicMock()
        store.get_counts.return_value = {}
        store.get_top_searches.return_value = [("neural networks", 5), ("transformers", 3)]
        store.get_top_tags.return_value = []
        store.get_capture_topics.return_value = []
        store.get_active_hours.return_value = []

        execute = create_insight_executor(store)
        prior: dict[str, Any] = {
            "top_searches": ["python", "rust"],
            "acceptance_rate": 0.5,
            "total_interactions": 10,
        }
        result = await execute(prior)

        assert result["notification"] is not None
        assert "neural networks" in result["notification"]

    @pytest.mark.asyncio
    async def test_detects_acceptance_rate_shift(self) -> None:
        store = MagicMock()
        store.get_counts.return_value = {}
        store.get_top_searches.return_value = [("python", 5)]
        store.get_top_tags.return_value = []
        store.get_capture_topics.return_value = []
        store.get_active_hours.return_value = []

        execute = create_insight_executor(store)
        prior: dict[str, Any] = {
            "top_searches": ["python"],
            "acceptance_rate": 0.9,
            "total_interactions": 10,
        }

        with patch("vaultmind.tracking.analyzer.analyze_preferences") as mock_analyze:
            from vaultmind.tracking.analyzer import PreferenceInsights

            mock_analyze.return_value = PreferenceInsights(
                total_interactions=15,
                period_days=30,
                top_searches=[("python", 5)],
                top_tags_approved=[],
                top_tags_rejected=[],
                capture_topics=[],
                interaction_counts={},
                suggestions_acceptance_rate=0.3,
                active_hours=[],
                recommendations=[],
            )
            result = await execute(prior)
            assert result["notification"] is not None
            assert "acceptance rate" in result["notification"].lower()


class TestEvolutionLoop:
    @pytest.mark.asyncio
    async def test_first_run_no_notification_when_no_signals(self) -> None:
        detector = MagicMock()
        detector.scan.return_value = []

        execute = create_evolution_executor(detector)
        result = await execute({})

        assert result["notification"] is None
        assert result["signal_ids"] == []

    @pytest.mark.asyncio
    async def test_detects_new_high_severity(self) -> None:
        signal = MagicMock()
        signal.evolution_id = "ev1"
        signal.entity_a = "Python"
        signal.entity_b = "FastAPI"
        signal.signal_type = "confidence_drift"
        signal.detail = "confidence changed"
        signal.severity = 0.8

        detector = MagicMock()
        detector.scan.return_value = [signal]

        execute = create_evolution_executor(detector)
        result = await execute({})

        assert result["notification"] is not None
        assert "high-severity" in result["notification"]
        assert result["new_signal_count"] == 1
