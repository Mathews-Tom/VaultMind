"""Score source discovery, not approved-answer quality.

The runner and Telegram `/recall` share `knowledge.sources.search_sources`.
Candidates are hydrated from the actual authorized current vault files before
source-retrieval recall@k/MRR and raw-distance decline evidence are scored.
Missing vector evidence is unknown; source-only matches are never approved wiki
answers. Optional LLM cite-or-decline scoring is a separate source experiment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from vaultmind.indexer.ranking import RetrievalConfidence, retrieval_confidence
from vaultmind.knowledge.sources import search_sources

if TYPE_CHECKING:
    from vaultmind.bench.golden import GoldenQuestion
    from vaultmind.config import Settings

DEFAULT_SCORE_FLOOR = 0.5


class RetrievalStore(Protocol):
    """Optional source-ranking backend, always scoped before its result limit."""

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
    confidence: RetrievalConfidence
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
    settings: Settings,
    query: str,
    k: int,
    *,
    store: RetrievalStore | None = None,
    allows_path: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Exercise the same current-file, source-only path as Telegram `/recall`."""
    return search_sources(settings, query, store=store, n_results=k, allows_path=allows_path)


def _note_path(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata", {})
    return str(metadata.get("note_path", ""))


def score_question(
    question: GoldenQuestion, hits: list[dict[str, Any]], score_floor: float
) -> QuestionResult:
    """Score one golden question against its retrieved hits."""
    paths = tuple(_note_path(h) for h in hits)
    confidence = retrieval_confidence(hits, score_floor)

    if not question.answerable:
        declined = confidence is RetrievalConfidence.WEAK
        return QuestionResult(
            question_id=question.id,
            question=question.question,
            answerable=False,
            retrieved_paths=paths,
            hit=False,
            rank=None,
            retrieval_decline_correct=declined,
            confidence=confidence,
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
        confidence=confidence,
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
    settings: Settings,
    k: int,
    *,
    store: RetrievalStore | None = None,
    allows_path: Callable[[str], bool] | None = None,
    score_floor: float | None = None,
    decline_scorer: DeclineScorer | None = None,
) -> BenchReport:
    """Score source retrieval over the real current-file path and scope."""
    floor = settings.bench.score_floor if score_floor is None else score_floor
    results: list[QuestionResult] = []
    for question in golden:
        hits = run_query(settings, question.question, k, store=store, allows_path=allows_path)
        result = score_question(question, hits, floor)
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
