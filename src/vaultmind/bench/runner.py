"""Retrieval scoring runner — exercises the same retrieval call
`bot/handlers/recall.py::handle_recall` uses for `/recall`, and scores it
against a golden question set.

Deterministic scoring (no LLM call):
  - recall@k / MRR over `answerable: true` questions, using each hit's
    `metadata["note_path"]` against the question's `expected_notes`.
  - a score-floor "honest decline" check over `answerable: false` questions:
    correct when the best hit's distance is at or above `score_floor` (i.e.
    retrieval found nothing confidently relevant for a question that has no
    answer in the vault).

LLM-scored cite-or-decline (see `vaultmind.bench.llm_score`) is layered on
top separately and is never required for `run_bench`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from vaultmind.indexer.ranking import apply_authority

if TYPE_CHECKING:
    from vaultmind.bench.golden import GoldenQuestion

DEFAULT_SCORE_FLOOR = 0.5


class RetrievalStore(Protocol):
    """Structural shape of the store methods `/recall` calls.

    Matches `VaultStore.search`/`VaultStore.hybrid_search` — anything with
    this shape (including a test double or a fixture-backed store) can be
    scored by the runner.
    """

    def search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def hybrid_search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Per-question scoring outcome."""

    question_id: str
    question: str
    answerable: bool
    retrieved_paths: tuple[str, ...]
    hit: bool
    rank: int | None
    retrieval_decline_correct: bool | None = None
    llm_decline_correct: bool | None = None


@dataclass(frozen=True, slots=True)
class BenchReport:
    """Aggregate scores over one bench run."""

    k: int
    n_answerable: int
    n_unanswerable: int
    recall_at_k: float
    mrr: float
    retrieval_decline_accuracy: float | None
    llm_cite_or_decline_accuracy: float | None
    results: tuple[QuestionResult, ...]


def run_query(
    store: RetrievalStore,
    query: str,
    k: int,
    hybrid_enabled: bool,
    ranking_config: Any = None,
) -> list[dict[str, Any]]:
    """Run the exact retrieval call `handle_recall` makes for `/recall`.

    Mirrors `bot/handlers/recall.py::handle_recall`'s branch on
    `settings.search.hybrid_enabled` and its `apply_authority()` re-ranking
    step — the bench exercises the identical live path, not a parallel one.
    """
    if hybrid_enabled:
        hits = store.hybrid_search(query, n_results=k)
    else:
        hits = store.search(query, n_results=k)
    return apply_authority(hits, ranking_config)


def _note_path(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata", {})
    return str(metadata.get("note_path", ""))


def _best_distance(hits: list[dict[str, Any]]) -> float | None:
    if not hits:
        return None
    return min(float(h.get("distance", 1.0)) for h in hits)


def score_question(
    question: GoldenQuestion, hits: list[dict[str, Any]], score_floor: float
) -> QuestionResult:
    """Score one golden question against its retrieved hits."""
    paths = tuple(_note_path(h) for h in hits)

    if not question.answerable:
        best = _best_distance(hits)
        declined = best is None or best >= score_floor
        return QuestionResult(
            question_id=question.id,
            question=question.question,
            answerable=False,
            retrieved_paths=paths,
            hit=False,
            rank=None,
            retrieval_decline_correct=declined,
        )

    rank: int | None = None
    for i, path in enumerate(paths, start=1):
        if path in question.expected_notes:
            rank = i
            break

    return QuestionResult(
        question_id=question.id,
        question=question.question,
        answerable=True,
        retrieved_paths=paths,
        hit=rank is not None,
        rank=rank,
    )


def aggregate(results: list[QuestionResult], k: int) -> BenchReport:
    """Compute recall@k/MRR and decline accuracies from per-question results."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    n_answerable = len(answerable)
    recall_at_k = (sum(1 for r in answerable if r.hit) / n_answerable) if n_answerable else 0.0
    mrr = (
        (sum((1.0 / r.rank) if r.rank else 0.0 for r in answerable) / n_answerable)
        if n_answerable
        else 0.0
    )

    retrieval_decline_scored = [r for r in unanswerable if r.retrieval_decline_correct is not None]
    retrieval_decline_accuracy = (
        sum(1 for r in retrieval_decline_scored if r.retrieval_decline_correct)
        / len(retrieval_decline_scored)
        if retrieval_decline_scored
        else None
    )

    llm_scored = [r for r in results if r.llm_decline_correct is not None]
    llm_cite_or_decline_accuracy = (
        sum(1 for r in llm_scored if r.llm_decline_correct) / len(llm_scored)
        if llm_scored
        else None
    )

    return BenchReport(
        k=k,
        n_answerable=n_answerable,
        n_unanswerable=len(unanswerable),
        recall_at_k=recall_at_k,
        mrr=mrr,
        retrieval_decline_accuracy=retrieval_decline_accuracy,
        llm_cite_or_decline_accuracy=llm_cite_or_decline_accuracy,
        results=tuple(results),
    )


DeclineScorer = Callable[["GoldenQuestion", "list[dict[str, Any]]"], bool]
"""Callable that scores cite-or-decline correctness for one question."""


def run_bench(
    golden: list[GoldenQuestion],
    store: RetrievalStore,
    k: int,
    hybrid_enabled: bool,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    decline_scorer: DeclineScorer | None = None,
    ranking_config: Any = None,
) -> BenchReport:
    """Run the full golden set through the live retrieval path and score it."""
    results: list[QuestionResult] = []
    for question in golden:
        hits = run_query(store, question.question, k, hybrid_enabled, ranking_config)
        result = score_question(question, hits, score_floor)
        if decline_scorer is not None:
            result = replace(result, llm_decline_correct=decline_scorer(question, hits))
        results.append(result)
    return aggregate(results, k)


def passes_thresholds(
    report: BenchReport,
    recall_at_k_threshold: float,
    mrr_threshold: float,
    retrieval_decline_threshold: float,
    llm_decline_threshold: float | None,
) -> bool:
    """True iff every threshold that applies to this report is met.

    A threshold only applies when the report has data for it (e.g. the
    recall@k/MRR thresholds are skipped when the golden set has no
    answerable questions).
    """
    if report.n_answerable > 0:
        if report.recall_at_k < recall_at_k_threshold:
            return False
        if report.mrr < mrr_threshold:
            return False

    retrieval_decline = report.retrieval_decline_accuracy
    if retrieval_decline is not None and retrieval_decline < retrieval_decline_threshold:
        return False

    llm_decline = report.llm_cite_or_decline_accuracy
    if llm_decline_threshold is None or llm_decline is None:
        return True
    return llm_decline >= llm_decline_threshold
