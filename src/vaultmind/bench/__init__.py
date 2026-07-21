"""Retrieval self-benchmark — deterministic recall@k/MRR scoring of the live
`/recall` retrieval path, plus opt-in LLM cite-or-decline scoring.
"""

from __future__ import annotations

from vaultmind.bench.fixture_store import Bundle, BundleError, FixtureStore, load_bundle
from vaultmind.bench.golden import GoldenQuestion, GoldenSetError, load_golden_set
from vaultmind.bench.llm_score import LLMAnswer, make_decline_scorer, parse_llm_answer
from vaultmind.bench.runner import (
    BenchReport,
    QuestionResult,
    RetrievalStore,
    passes_thresholds,
    run_bench,
)
from vaultmind.bench.trend import (
    TrendRecord,
    append_trend_record,
    build_trend_record,
    default_trend_path,
)

__all__ = [
    "BenchReport",
    "Bundle",
    "BundleError",
    "FixtureStore",
    "GoldenQuestion",
    "GoldenSetError",
    "LLMAnswer",
    "QuestionResult",
    "RetrievalStore",
    "TrendRecord",
    "append_trend_record",
    "build_trend_record",
    "default_trend_path",
    "load_bundle",
    "load_golden_set",
    "make_decline_scorer",
    "parse_llm_answer",
    "passes_thresholds",
    "run_bench",
]
