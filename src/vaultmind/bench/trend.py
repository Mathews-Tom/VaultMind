"""JSONL trend storage for `vaultmind bench` runs.

Each invocation appends exactly one record so `recall@k`/MRR drift can be
tracked across ranking changes over time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.bench.runner import BenchReport


@dataclass(frozen=True, slots=True)
class TrendRecord:
    """One JSONL row recording a single `vaultmind bench` run."""

    timestamp: str
    k: int
    n_answerable: int
    n_unanswerable: int
    recall_at_k: float
    mrr: float
    retrieval_decline_accuracy: float | None
    llm_cite_or_decline_accuracy: float | None
    passed: bool


def build_trend_record(report: BenchReport, passed: bool) -> TrendRecord:
    """Build the trend record for one completed bench run."""
    return TrendRecord(
        timestamp=datetime.now(UTC).isoformat(),
        k=report.k,
        n_answerable=report.n_answerable,
        n_unanswerable=report.n_unanswerable,
        recall_at_k=report.recall_at_k,
        mrr=report.mrr,
        retrieval_decline_accuracy=report.retrieval_decline_accuracy,
        llm_cite_or_decline_accuracy=report.llm_cite_or_decline_accuracy,
        passed=passed,
    )


def default_trend_path() -> Path:
    """Default JSONL trend path: `~/.vaultmind/data/bench/trend.jsonl`."""
    from vaultmind.config import VAULTMIND_HOME

    return VAULTMIND_HOME / "data" / "bench" / "trend.jsonl"


def append_trend_record(record: TrendRecord, path: Path) -> None:
    """Append one JSONL record, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
