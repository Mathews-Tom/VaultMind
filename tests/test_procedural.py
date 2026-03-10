"""Tests for procedural memory — ProceduralMemory and query_resolved on EpisodeStore."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime in tmp_path fixture signatures
from unittest.mock import MagicMock

import pytest

from vaultmind.llm.client import LLMResponse
from vaultmind.memory.models import OutcomeStatus
from vaultmind.memory.procedural import ProceduralMemory
from vaultmind.memory.store import EpisodeStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_procedural(tmp_path: Path) -> ProceduralMemory:
    return ProceduralMemory(tmp_path / "procedural.db")


def _make_episode_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.db")


def _make_fake_llm(response_data: object) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = LLMResponse(
        text=json.dumps(response_data),
        model="fake-model",
        usage={"total_tokens": 50},
    )
    return client


# ---------------------------------------------------------------------------
# ProceduralMemory CRUD tests
# ---------------------------------------------------------------------------


class TestCreateAndListWorkflow:
    def test_create_and_list_workflow(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf = pm.create_workflow(
            name="Code Review Workflow",
            description="Steps for reviewing PRs effectively",
            steps=["Read the diff", "Run tests", "Leave comments", "Approve or request changes"],
            trigger_pattern="code review pull request PR",
            source_episodes=["abc123", "def456", "ghi789"],
        )

        assert wf.workflow_id
        assert len(wf.workflow_id) == 12
        assert all(c in "0123456789abcdef" for c in wf.workflow_id)
        assert wf.name == "Code Review Workflow"
        assert wf.success_rate == 0.0
        assert wf.usage_count == 0
        assert wf.active is True

        active = pm.list_active()
        assert len(active) == 1
        assert active[0].workflow_id == wf.workflow_id
        assert active[0].name == "Code Review Workflow"

        pm.close()

    def test_list_active_ordered_by_usage_count(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf_a = pm.create_workflow("Low Usage", "", [], "low trigger")
        wf_b = pm.create_workflow("High Usage", "", [], "high trigger")

        # Record several usages for wf_b
        pm.record_usage(wf_b.workflow_id, success=True)
        pm.record_usage(wf_b.workflow_id, success=True)

        active = pm.list_active()
        assert active[0].workflow_id == wf_b.workflow_id
        assert active[1].workflow_id == wf_a.workflow_id

        pm.close()


class TestRecordUsageUpdatesRate:
    def test_first_success_sets_rate_to_one(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)
        wf = pm.create_workflow("Test", "", [], "")

        pm.record_usage(wf.workflow_id, success=True)

        updated = pm.get(wf.workflow_id)
        assert updated is not None
        assert updated.usage_count == 1
        assert updated.success_rate == pytest.approx(1.0)

        pm.close()

    def test_first_failure_sets_rate_to_zero(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)
        wf = pm.create_workflow("Test", "", [], "")

        pm.record_usage(wf.workflow_id, success=False)

        updated = pm.get(wf.workflow_id)
        assert updated is not None
        assert updated.usage_count == 1
        assert updated.success_rate == pytest.approx(0.0)

        pm.close()

    def test_running_average_two_successes_one_failure(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)
        wf = pm.create_workflow("Test", "", [], "")

        pm.record_usage(wf.workflow_id, success=True)
        pm.record_usage(wf.workflow_id, success=True)
        pm.record_usage(wf.workflow_id, success=False)

        updated = pm.get(wf.workflow_id)
        assert updated is not None
        assert updated.usage_count == 3
        # (1 + 1 + 0) / 3 = 2/3
        assert updated.success_rate == pytest.approx(2 / 3)

        pm.close()

    def test_record_usage_unknown_id_is_noop(self, tmp_path: Path) -> None:
        """Recording usage for an unknown workflow does not raise."""
        pm = _make_procedural(tmp_path)
        pm.record_usage("nonexistent123", success=True)  # must not raise
        pm.close()


class TestInactiveWorkflowExcluded:
    def test_deactivated_workflow_not_in_list_active(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf = pm.create_workflow("Will be deactivated", "", [], "")
        pm.deactivate(wf.workflow_id)

        active = pm.list_active()
        assert all(w.workflow_id != wf.workflow_id for w in active)

        pm.close()

    def test_deactivated_workflow_still_retrievable_by_get(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf = pm.create_workflow("Deactivated", "", [], "")
        pm.deactivate(wf.workflow_id)

        fetched = pm.get(wf.workflow_id)
        assert fetched is not None
        assert fetched.active is False

        pm.close()


class TestSuggestWorkflow:
    def test_suggest_workflow_word_match(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        pm.create_workflow(
            "Deploy Checklist",
            "Steps before deploying to production",
            ["Run tests", "Review diff", "Notify team"],
            trigger_pattern="deploy production release ship",
        )

        result = pm.suggest_workflow("I need to deploy the production release today")
        assert result is not None
        assert result.name == "Deploy Checklist"

        pm.close()

    def test_suggest_workflow_no_match(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        pm.create_workflow(
            "Deploy Checklist",
            "",
            [],
            trigger_pattern="deploy production release",
        )

        result = pm.suggest_workflow("how do I bake bread today")
        assert result is None

        pm.close()

    def test_suggest_workflow_empty_context(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)
        pm.create_workflow("Any Workflow", "", [], "deploy production")
        result = pm.suggest_workflow("")
        assert result is None
        pm.close()

    def test_suggest_workflow_returns_highest_usage_on_tie(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf_low = pm.create_workflow("Low Use", "", [], trigger_pattern="review code quality")
        wf_high = pm.create_workflow("High Use", "", [], trigger_pattern="review code quality")

        pm.record_usage(wf_high.workflow_id, success=True)
        pm.record_usage(wf_high.workflow_id, success=True)

        result = pm.suggest_workflow("code review quality check")
        assert result is not None
        assert result.workflow_id == wf_high.workflow_id

        _ = wf_low  # suppress unused warning
        pm.close()


class TestGetWorkflow:
    def test_get_workflow_by_id(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)

        wf = pm.create_workflow(
            "My Workflow",
            "A detailed description",
            ["Step one", "Step two"],
            "trigger phrase",
            source_episodes=["ep1", "ep2"],
        )

        fetched = pm.get(wf.workflow_id)
        assert fetched is not None
        assert fetched.workflow_id == wf.workflow_id
        assert fetched.name == "My Workflow"
        assert fetched.description == "A detailed description"
        assert fetched.steps == ["Step one", "Step two"]
        assert fetched.trigger_pattern == "trigger phrase"
        assert fetched.source_episodes == ["ep1", "ep2"]

        pm.close()

    def test_get_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        pm = _make_procedural(tmp_path)
        assert pm.get("doesnotexist") is None
        pm.close()


# ---------------------------------------------------------------------------
# EpisodeStore.query_resolved tests
# ---------------------------------------------------------------------------


class TestQueryResolvedEpisodes:
    def test_query_resolved_excludes_pending(self, tmp_path: Path) -> None:
        store = _make_episode_store(tmp_path)

        pending = store.create(decision="Still pending")
        resolved_ep = store.create(decision="Already resolved")
        store.resolve(resolved_ep.episode_id, "Done", OutcomeStatus.SUCCESS, [])

        results = store.query_resolved()
        result_ids = {ep.episode_id for ep in results}

        assert resolved_ep.episode_id in result_ids
        assert pending.episode_id not in result_ids

        store.close()

    def test_query_resolved_includes_all_non_pending_statuses(self, tmp_path: Path) -> None:
        store = _make_episode_store(tmp_path)

        ep_success = store.create(decision="Success")
        ep_failure = store.create(decision="Failure")
        ep_partial = store.create(decision="Partial")
        ep_unknown = store.create(decision="Unknown")

        store.resolve(ep_success.episode_id, "All good", OutcomeStatus.SUCCESS, [])
        store.resolve(ep_failure.episode_id, "Bad", OutcomeStatus.FAILURE, [])
        store.resolve(ep_partial.episode_id, "Partial", OutcomeStatus.PARTIAL, [])
        store.resolve(ep_unknown.episode_id, "?", OutcomeStatus.UNKNOWN, [])

        results = store.query_resolved()
        result_ids = {ep.episode_id for ep in results}

        assert ep_success.episode_id in result_ids
        assert ep_failure.episode_id in result_ids
        assert ep_partial.episode_id in result_ids
        assert ep_unknown.episode_id in result_ids

        store.close()

    def test_query_resolved_respects_limit(self, tmp_path: Path) -> None:
        store = _make_episode_store(tmp_path)

        for i in range(10):
            ep = store.create(decision=f"Decision {i}")
            store.resolve(ep.episode_id, f"Outcome {i}", OutcomeStatus.SUCCESS, [])

        results = store.query_resolved(limit=5)
        assert len(results) == 5

        store.close()

    def test_query_resolved_ordered_by_created_desc(self, tmp_path: Path) -> None:
        store = _make_episode_store(tmp_path)

        ep_first = store.create(decision="First")
        store.resolve(ep_first.episode_id, "done", OutcomeStatus.SUCCESS, [])
        ep_last = store.create(decision="Last")
        store.resolve(ep_last.episode_id, "done", OutcomeStatus.SUCCESS, [])

        results = store.query_resolved()
        ids = [ep.episode_id for ep in results]
        # Most recent first
        assert ids.index(ep_last.episode_id) < ids.index(ep_first.episode_id)

        store.close()

    def test_query_resolved_returns_empty_when_all_pending(self, tmp_path: Path) -> None:
        store = _make_episode_store(tmp_path)
        store.create(decision="Still pending")
        assert store.query_resolved() == []
        store.close()


# ---------------------------------------------------------------------------
# synthesize_workflows integration test (with mock LLM)
# ---------------------------------------------------------------------------


class TestSynthesizeWorkflows:
    def test_synthesize_creates_workflow_from_episodes(self, tmp_path: Path) -> None:
        episode_store = _make_episode_store(tmp_path)

        # Create 3 resolved episodes sharing an entity
        for i in range(3):
            ep = episode_store.create(
                decision=f"Decision {i} about deployment",
                entities=["deployment", "production"],
            )
            episode_store.resolve(ep.episode_id, f"Outcome {i}", OutcomeStatus.SUCCESS, [])

        llm = _make_fake_llm(
            {
                "name": "Deployment Workflow",
                "description": "Steps to safely deploy to production",
                "steps": ["Run tests", "Deploy to staging", "Smoke test", "Deploy to prod"],
                "trigger_pattern": "deploy production release",
            }
        )

        pm = ProceduralMemory(tmp_path / "procedural.db")
        workflows = pm.synthesize_workflows(
            episode_store=episode_store,
            llm_client=llm,  # type: ignore[arg-type]
            model="fake-model",
            min_episodes=3,
        )

        assert len(workflows) == 1
        assert workflows[0].name == "Deployment Workflow"
        assert len(workflows[0].steps) == 4
        assert workflows[0].trigger_pattern == "deploy production release"
        assert workflows[0].usage_count == 0
        assert workflows[0].success_rate == 0.0

        episode_store.close()
        pm.close()

    def test_synthesize_skips_groups_below_min_episodes(self, tmp_path: Path) -> None:
        episode_store = _make_episode_store(tmp_path)

        # Only 2 episodes with shared entity — below default min_episodes=3
        for i in range(2):
            ep = episode_store.create(decision=f"Decision {i}", entities=["shared_entity"])
            episode_store.resolve(ep.episode_id, "Done", OutcomeStatus.SUCCESS, [])

        llm = _make_fake_llm(
            {
                "name": "Should not appear",
                "description": "",
                "steps": [],
                "trigger_pattern": "",
            }
        )

        pm = ProceduralMemory(tmp_path / "procedural.db")
        workflows = pm.synthesize_workflows(
            episode_store=episode_store,
            llm_client=llm,  # type: ignore[arg-type]
            model="fake-model",
            min_episodes=3,
        )

        assert workflows == []
        llm.complete.assert_not_called()

        episode_store.close()
        pm.close()

    def test_synthesize_handles_null_llm_response(self, tmp_path: Path) -> None:
        episode_store = _make_episode_store(tmp_path)

        for i in range(3):
            ep = episode_store.create(decision=f"Decision {i}", entities=["shared_entity"])
            episode_store.resolve(ep.episode_id, "Done", OutcomeStatus.SUCCESS, [])

        # LLM returns null — no pattern found
        llm = _make_fake_llm(None)

        pm = ProceduralMemory(tmp_path / "procedural.db")
        workflows = pm.synthesize_workflows(
            episode_store=episode_store,
            llm_client=llm,  # type: ignore[arg-type]
            model="fake-model",
            min_episodes=3,
        )

        assert workflows == []

        episode_store.close()
        pm.close()

    def test_synthesize_returns_empty_with_no_resolved_episodes(self, tmp_path: Path) -> None:
        episode_store = _make_episode_store(tmp_path)
        # Only pending episodes
        episode_store.create(decision="Pending decision", entities=["entity"])

        llm = _make_fake_llm({"name": "x", "description": "", "steps": [], "trigger_pattern": ""})
        pm = ProceduralMemory(tmp_path / "procedural.db")
        workflows = pm.synthesize_workflows(
            episode_store=episode_store,
            llm_client=llm,  # type: ignore[arg-type]
            model="fake-model",
            min_episodes=1,
        )

        assert workflows == []
        llm.complete.assert_not_called()

        episode_store.close()
        pm.close()
