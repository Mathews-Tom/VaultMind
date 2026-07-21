"""Tests for the retrieval scoring runner — recall@k/MRR and score-floor
honest-decline scoring, with zero LLM calls and zero live embeddings."""

from __future__ import annotations

from typing import Any

import pytest

from vaultmind.bench.golden import GoldenQuestion
from vaultmind.bench.runner import (
    DEFAULT_SCORE_FLOOR,
    BenchReport,
    aggregate,
    passes_thresholds,
    run_bench,
    run_query,
    score_question,
)
from vaultmind.config import RankingConfig


class FakeStore:
    """Deterministic RetrievalStore double, mirroring the repo's `FakeStore`
    test convention (see `tests/test_duplicate_detector.py`)."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self._results = results_by_query
        self.search_calls: list[tuple[str, int]] = []
        self.hybrid_calls: list[tuple[str, int]] = []

    def search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.search_calls.append((query, n_results))
        return self._results.get(query, [])[:n_results]

    def hybrid_search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.hybrid_calls.append((query, n_results))
        return self._results.get(query, [])[:n_results]


def _hit(note_path: str, distance: float = 0.1, authority: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"note_path": note_path}
    if authority is not None:
        metadata["authority"] = authority
    return {
        "chunk_id": f"{note_path}::0",
        "content": "x",
        "metadata": metadata,
        "distance": distance,
    }


def _q(
    qid: str, question: str = "q", answerable: bool = True, expected: tuple[str, ...] = ("a.md",)
) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=question,
        answerable=answerable,
        expected_notes=expected if answerable else (),
    )


class TestRunQuery:
    def test_hybrid_enabled_calls_hybrid_search(self) -> None:
        store = FakeStore({"q": [_hit("a.md")]})
        run_query(store, "q", k=5, hybrid_enabled=True)
        assert store.hybrid_calls == [("q", 5)]
        assert store.search_calls == []

    def test_hybrid_disabled_calls_plain_search(self) -> None:
        store = FakeStore({"q": [_hit("a.md")]})
        run_query(store, "q", k=5, hybrid_enabled=False)
        assert store.search_calls == [("q", 5)]
        assert store.hybrid_calls == []

    def test_authority_reranks_hits_when_ranking_config_given(self) -> None:
        store = FakeStore(
            {
                "q": [
                    _hit("low.md", distance=0.20, authority=1),
                    _hit("high.md", distance=0.30, authority=5),
                ]
            }
        )
        hits = run_query(store, "q", k=5, hybrid_enabled=True, ranking_config=RankingConfig())
        assert hits[0]["metadata"]["note_path"] == "high.md"

    def test_no_ranking_config_still_authority_neutral_no_reorder(self) -> None:
        store = FakeStore({"q": [_hit("a.md", distance=0.1), _hit("b.md", distance=0.2)]})
        hits = run_query(store, "q", k=5, hybrid_enabled=True)
        assert [h["metadata"]["note_path"] for h in hits] == ["a.md", "b.md"]


class TestScoreQuestionAnswerable:
    def test_hit_at_rank_1(self) -> None:
        question = _q("q1", expected=("a.md",))
        hits = [_hit("a.md"), _hit("b.md")]
        result = score_question(question, hits, DEFAULT_SCORE_FLOOR)
        assert result.hit is True
        assert result.rank == 1

    def test_hit_at_rank_3(self) -> None:
        question = _q("q1", expected=("c.md",))
        hits = [_hit("a.md"), _hit("b.md"), _hit("c.md")]
        result = score_question(question, hits, DEFAULT_SCORE_FLOOR)
        assert result.hit is True
        assert result.rank == 3

    def test_no_hit(self) -> None:
        question = _q("q1", expected=("z.md",))
        hits = [_hit("a.md"), _hit("b.md")]
        result = score_question(question, hits, DEFAULT_SCORE_FLOOR)
        assert result.hit is False
        assert result.rank is None

    def test_empty_hits(self) -> None:
        question = _q("q1", expected=("a.md",))
        result = score_question(question, [], DEFAULT_SCORE_FLOOR)
        assert result.hit is False
        assert result.rank is None
        assert result.retrieved_paths == ()

    def test_multiple_expected_notes_matches_any(self) -> None:
        question = _q("q1", expected=("a.md", "b.md"))
        hits = [_hit("z.md"), _hit("b.md")]
        result = score_question(question, hits, DEFAULT_SCORE_FLOOR)
        assert result.hit is True
        assert result.rank == 2


class TestScoreQuestionUnanswerable:
    def test_low_confidence_hits_score_correct_decline(self) -> None:
        question = _q("q1", answerable=False)
        hits = [_hit("random.md", distance=0.9)]
        result = score_question(question, hits, score_floor=0.5)
        assert result.retrieval_decline_correct is True

    def test_high_confidence_hit_scores_incorrect_decline(self) -> None:
        question = _q("q1", answerable=False)
        hits = [_hit("random.md", distance=0.1)]
        result = score_question(question, hits, score_floor=0.5)
        assert result.retrieval_decline_correct is False

    def test_no_hits_scores_correct_decline(self) -> None:
        question = _q("q1", answerable=False)
        result = score_question(question, [], score_floor=0.5)
        assert result.retrieval_decline_correct is True

    def test_boundary_distance_equal_to_floor_is_correct(self) -> None:
        question = _q("q1", answerable=False)
        hits = [_hit("random.md", distance=0.5)]
        result = score_question(question, hits, score_floor=0.5)
        assert result.retrieval_decline_correct is True

    def test_unanswerable_never_sets_hit_or_rank(self) -> None:
        question = _q("q1", answerable=False)
        hits = [_hit("random.md", distance=0.1)]
        result = score_question(question, hits, score_floor=0.5)
        assert result.hit is False
        assert result.rank is None


class TestAggregate:
    def test_recall_and_mrr_over_answerable_only(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",)), _q("q2", "q2", True, ("b.md",))]
        results = [
            score_question(golden[0], [_hit("a.md")], DEFAULT_SCORE_FLOOR),  # rank 1
            score_question(golden[1], [_hit("z.md"), _hit("b.md")], DEFAULT_SCORE_FLOOR),  # rank 2
        ]
        report = aggregate(results, k=5)
        assert report.n_answerable == 2
        assert report.n_unanswerable == 0
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_recall_penalizes_misses(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",)), _q("q2", "q2", True, ("b.md",))]
        results = [
            score_question(golden[0], [_hit("a.md")], DEFAULT_SCORE_FLOOR),  # hit
            score_question(golden[1], [_hit("z.md")], DEFAULT_SCORE_FLOOR),  # miss
        ]
        report = aggregate(results, k=5)
        assert report.recall_at_k == pytest.approx(0.5)
        assert report.mrr == pytest.approx(0.5)

    def test_retrieval_decline_accuracy_over_unanswerable_only(self) -> None:
        golden = [_q("q1", "q1", False), _q("q2", "q2", False)]
        results = [
            score_question(golden[0], [_hit("x.md", distance=0.9)], score_floor=0.5),  # correct
            score_question(golden[1], [_hit("y.md", distance=0.1)], score_floor=0.5),  # incorrect
        ]
        report = aggregate(results, k=5)
        assert report.n_answerable == 0
        assert report.n_unanswerable == 2
        assert report.retrieval_decline_accuracy == pytest.approx(0.5)
        assert report.recall_at_k == 0.0
        assert report.mrr == 0.0

    def test_no_unanswerable_questions_gives_none_decline_accuracy(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",))]
        results = [score_question(golden[0], [_hit("a.md")], DEFAULT_SCORE_FLOOR)]
        report = aggregate(results, k=5)
        assert report.retrieval_decline_accuracy is None

    def test_no_llm_scoring_gives_none_llm_accuracy(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",))]
        results = [score_question(golden[0], [_hit("a.md")], DEFAULT_SCORE_FLOOR)]
        report = aggregate(results, k=5)
        assert report.llm_cite_or_decline_accuracy is None


class TestRunBench:
    def test_zero_llm_calls_by_default(self) -> None:
        """No decline_scorer passed -> llm_cite_or_decline_accuracy stays None."""
        golden = [_q("q1", "q1", True, ("a.md",))]
        store = FakeStore({"q1": [_hit("a.md")]})
        report = run_bench(golden, store, k=5, hybrid_enabled=True)
        assert report.llm_cite_or_decline_accuracy is None
        assert isinstance(report, BenchReport)

    def test_mirrors_recall_path_hybrid_branch(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",))]
        store = FakeStore({"q1": [_hit("a.md")]})
        run_bench(golden, store, k=5, hybrid_enabled=True)
        assert store.hybrid_calls == [("q1", 5)]

    def test_decline_scorer_is_invoked_and_recorded(self) -> None:
        golden = [_q("q1", "q1", True, ("a.md",))]
        store = FakeStore({"q1": [_hit("a.md")]})
        calls: list[str] = []

        def scorer(question: GoldenQuestion, hits: list[dict[str, Any]]) -> bool:
            calls.append(question.id)
            return True

        report = run_bench(golden, store, k=5, hybrid_enabled=True, decline_scorer=scorer)
        assert calls == ["q1"]
        assert report.llm_cite_or_decline_accuracy == pytest.approx(1.0)

    def test_authority_moves_mrr_for_authority_differentiated_golden_case(self) -> None:
        """The authority signal is observable through `run_bench`'s aggregate
        MRR, not only through `run_query`'s raw hit order — this is the exact
        acceptance the milestone names ("`vaultmind bench` run before/after
        ... shows the authority signal moving ranked order")."""
        golden = [_q("q1", "q1", True, ("authored.md",))]
        hits = [
            _hit("draft.md", distance=0.20, authority=1),
            _hit("authored.md", distance=0.30, authority=5),
        ]
        store = FakeStore({"q1": hits})

        no_op_config = RankingConfig(authority_weight={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
        before = run_bench(golden, store, k=5, hybrid_enabled=True, ranking_config=no_op_config)
        assert before.mrr == pytest.approx(0.5)  # authored.md sits at rank 2

        after = run_bench(golden, store, k=5, hybrid_enabled=True, ranking_config=RankingConfig())
        assert after.mrr == pytest.approx(1.0)  # authority promotes it to rank 1


class TestPassesThresholds:
    def _report(self, **overrides: Any) -> BenchReport:
        base = dict(
            k=5,
            n_answerable=2,
            n_unanswerable=1,
            recall_at_k=0.9,
            mrr=0.8,
            retrieval_decline_accuracy=0.9,
            llm_cite_or_decline_accuracy=None,
            results=(),
        )
        base.update(overrides)
        return BenchReport(**base)  # type: ignore[arg-type]

    def test_passes_when_all_thresholds_met(self) -> None:
        report = self._report()
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is True

    def test_fails_on_low_recall(self) -> None:
        report = self._report(recall_at_k=0.5)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is False

    def test_fails_on_low_mrr(self) -> None:
        report = self._report(mrr=0.1)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is False

    def test_fails_on_low_retrieval_decline_accuracy(self) -> None:
        report = self._report(retrieval_decline_accuracy=0.2)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is False

    def test_skips_recall_mrr_when_no_answerable_questions(self) -> None:
        report = self._report(n_answerable=0, recall_at_k=0.0, mrr=0.0)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is True

    def test_skips_retrieval_decline_when_none(self) -> None:
        report = self._report(retrieval_decline_accuracy=None)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is True

    def test_llm_threshold_ignored_when_not_provided(self) -> None:
        report = self._report(llm_cite_or_decline_accuracy=0.1)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, None) is True

    def test_fails_on_low_llm_accuracy_when_threshold_given(self) -> None:
        report = self._report(llm_cite_or_decline_accuracy=0.1)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, 0.8) is False

    def test_passes_on_sufficient_llm_accuracy(self) -> None:
        report = self._report(llm_cite_or_decline_accuracy=0.9)
        assert passes_thresholds(report, 0.8, 0.6, 0.8, 0.8) is True
