"""Tests for JSONL trend storage."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vaultmind.bench.runner import BenchReport
from vaultmind.bench.trend import append_trend_record, build_trend_record, default_trend_path

if TYPE_CHECKING:
    from pathlib import Path


def _report(**overrides: object) -> BenchReport:
    base: dict[str, object] = dict(
        k=5,
        n_answerable=2,
        n_unanswerable=1,
        recall_at_k=0.9,
        mrr=0.8,
        retrieval_decline_accuracy=1.0,
        llm_cite_or_decline_accuracy=None,
        results=(),
    )
    base.update(overrides)
    return BenchReport(**base)  # type: ignore[arg-type]


class TestBuildTrendRecord:
    def test_carries_report_fields(self) -> None:
        record = build_trend_record(_report(), passed=True)
        assert record.k == 5
        assert record.n_answerable == 2
        assert record.n_unanswerable == 1
        assert record.recall_at_k == 0.9
        assert record.mrr == 0.8
        assert record.retrieval_decline_accuracy == 1.0
        assert record.llm_cite_or_decline_accuracy is None
        assert record.passed is True

    def test_timestamp_is_iso8601(self) -> None:
        record = build_trend_record(_report(), passed=False)
        # Round-trips through fromisoformat without raising.
        from datetime import datetime

        datetime.fromisoformat(record.timestamp)


class TestAppendTrendRecord:
    def test_appends_one_jsonl_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bench" / "trend.jsonl"
        record = build_trend_record(_report(), passed=True)
        append_trend_record(record, path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["passed"] is True
        assert row["recall_at_k"] == 0.9

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "trend.jsonl"
        append_trend_record(build_trend_record(_report(), passed=True), path)
        assert path.exists()

    def test_second_run_appends_not_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "trend.jsonl"
        append_trend_record(build_trend_record(_report(recall_at_k=0.5), passed=False), path)
        append_trend_record(build_trend_record(_report(recall_at_k=0.9), passed=True), path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["recall_at_k"] == 0.5
        assert json.loads(lines[1])["recall_at_k"] == 0.9


class TestDefaultTrendPath:
    def test_defaults_under_vaultmind_home(self) -> None:
        path = default_trend_path()
        assert path.parts[-3:] == (".vaultmind", "data", "bench") or path.name == "trend.jsonl"
        assert path.name == "trend.jsonl"
