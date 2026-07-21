"""Retrieval self-benchmark — deterministic recall@k/MRR scoring of the live
`/recall` retrieval path, plus opt-in LLM cite-or-decline scoring.
"""

from __future__ import annotations

from vaultmind.bench.golden import GoldenQuestion, GoldenSetError, load_golden_set
from vaultmind.bench.runner import (
    BenchReport,
    QuestionResult,
    RetrievalStore,
    passes_thresholds,
    run_bench,
)

__all__ = [
    "BenchReport",
    "GoldenQuestion",
    "GoldenSetError",
    "QuestionResult",
    "RetrievalStore",
    "load_golden_set",
    "passes_thresholds",
    "run_bench",
]
