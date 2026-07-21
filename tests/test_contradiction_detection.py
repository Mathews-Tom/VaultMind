"""Tests for the contradiction conflict-detection surface + eval scoring (M6 PR-1)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaultmind.contradiction.detection import (
    CONFLICT_DETECTION_SYSTEM_PROMPT,
    ConflictVerdict,
    build_prompt,
    detect_conflict,
)
from vaultmind.contradiction.eval import (
    ContradictEvalError,
    ContradictEvalPair,
    load_eval_set,
    run_eval,
)
from vaultmind.llm.client import LLMError, LLMResponse


def _make_llm(response_obj: object) -> MagicMock:
    client = MagicMock()
    if isinstance(response_obj, Exception):
        client.complete.side_effect = response_obj
    else:
        client.complete.return_value = LLMResponse(
            text=json.dumps(response_obj) if isinstance(response_obj, dict) else response_obj,
            model="test-model",
            usage={},
        )
    return client


class TestBuildPrompt:
    def test_includes_both_note_titles_and_bodies(self) -> None:
        prompt = build_prompt("Note A", "Body A", "Note B", "Body B")
        assert "Note A" in prompt
        assert "Body A" in prompt
        assert "Note B" in prompt
        assert "Body B" in prompt

    def test_truncates_long_bodies(self) -> None:
        prompt = build_prompt("A", "x" * 5000, "B", "y" * 5000)
        assert len(prompt) < 6000


class TestDetectConflictSuccess:
    def test_parses_true_verdict(self) -> None:
        client = _make_llm({"materially_conflicts": True, "reasoning": "different DBs"})
        verdict = detect_conflict("A", "uses postgres", "B", "uses mongo", client, "test-model")
        assert verdict.materially_conflicts is True
        assert verdict.reasoning == "different DBs"
        assert verdict.error == ""

    def test_parses_false_verdict(self) -> None:
        client = _make_llm({"materially_conflicts": False, "reasoning": "complementary"})
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.materially_conflicts is False

    def test_strips_markdown_code_fence(self) -> None:
        raw = '```json\n{"materially_conflicts": true, "reasoning": "conflict"}\n```'
        client = _make_llm(raw)
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.materially_conflicts is True

    def test_missing_reasoning_defaults_empty(self) -> None:
        client = _make_llm({"materially_conflicts": True})
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.reasoning == ""


class TestDetectConflictFailure:
    def test_llm_error_returns_non_conflicting_with_error(self) -> None:
        client = _make_llm(LLMError("boom", provider="test"))
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.materially_conflicts is False
        assert verdict.error == "LLM call failed"

    def test_invalid_json_returns_non_conflicting_with_error(self) -> None:
        client = _make_llm("not json at all")
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.materially_conflicts is False
        assert verdict.error == "Invalid JSON from LLM"

    def test_non_dict_json_returns_non_conflicting_with_error(self) -> None:
        client = _make_llm("[1, 2, 3]")
        verdict = detect_conflict("A", "x", "B", "y", client, "test-model")
        assert verdict.materially_conflicts is False
        assert verdict.error == "Invalid JSON from LLM"


class TestConflictVerdictDataclass:
    def test_defaults(self) -> None:
        verdict = ConflictVerdict(materially_conflicts=True)
        assert verdict.reasoning == ""
        assert verdict.error == ""


class TestSystemPromptDefined:
    def test_prompt_requests_json(self) -> None:
        assert "JSON" in CONFLICT_DETECTION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Eval set loading + scoring
# ---------------------------------------------------------------------------


class TestLoadEvalSetSuccess:
    def test_loads_shipped_starter_set(self) -> None:
        pairs = load_eval_set(Path("benchmarks/contradict_eval.yaml"))
        assert len(pairs) >= 25
        assert all(isinstance(p, ContradictEvalPair) for p in pairs)
        assert any(p.materially_conflicts for p in pairs)
        assert any(not p.materially_conflicts for p in pairs)


class TestLoadEvalSetFailure:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ContradictEvalError, match="not found"):
            load_eval_set(tmp_path / "missing.yaml")

    def test_empty_pairs_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.yaml"
        path.write_text("version: 1\npairs: []\n")
        with pytest.raises(ContradictEvalError, match="non-empty list"):
            load_eval_set(path)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.yaml"
        path.write_text(
            "version: 1\n"
            "pairs:\n"
            "  - id: dup\n"
            "    note_a_title: A\n"
            "    note_a_body: a\n"
            "    note_b_title: B\n"
            "    note_b_body: b\n"
            "    materially_conflicts: true\n"
            "  - id: dup\n"
            "    note_a_title: C\n"
            "    note_a_body: c\n"
            "    note_b_title: D\n"
            "    note_b_body: d\n"
            "    materially_conflicts: false\n"
        )
        with pytest.raises(ContradictEvalError, match="duplicate id"):
            load_eval_set(path)

    def test_missing_label_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.yaml"
        path.write_text(
            "version: 1\n"
            "pairs:\n"
            "  - id: p1\n"
            "    note_a_title: A\n"
            "    note_a_body: a\n"
            "    note_b_title: B\n"
            "    note_b_body: b\n"
        )
        with pytest.raises(ContradictEvalError, match="materially_conflicts"):
            load_eval_set(path)


class TestRunEval:
    def _pairs(self) -> list[ContradictEvalPair]:
        return [
            ContradictEvalPair("p1", "A", "postgres", "B", "mongo", True),
            ContradictEvalPair("p2", "A", "x", "B", "y", False),
            ContradictEvalPair("p3", "A", "x", "C", "y again", False),
            ContradictEvalPair("p4", "A", "budget 50k", "D", "budget 30k", True),
        ]

    def test_perfect_detector_beats_baseline(self) -> None:
        pairs = self._pairs()
        client = MagicMock()
        responses = [
            LLMResponse(
                text=json.dumps({"materially_conflicts": p.materially_conflicts, "reasoning": ""}),
                model="m",
                usage={},
            )
            for p in pairs
        ]
        client.complete.side_effect = responses

        report = run_eval(pairs, client, "test-model")

        assert report.n_pairs == 4
        assert report.detector.precision == 1.0
        assert report.detector.recall == 1.0
        assert report.detector.f1 == 1.0
        # Baseline always predicts True: 2 correct (the 2 true positives), 2 false positives
        assert report.baseline.recall == 1.0
        assert report.baseline.precision == pytest.approx(0.5)
        assert report.beats_baseline is True

    def test_always_true_detector_matches_baseline_not_beats(self) -> None:
        pairs = self._pairs()
        client = MagicMock()
        client.complete.return_value = LLMResponse(
            text=json.dumps({"materially_conflicts": True, "reasoning": ""}),
            model="m",
            usage={},
        )

        report = run_eval(pairs, client, "test-model")

        assert report.detector.f1 == pytest.approx(report.baseline.f1)
        assert report.beats_baseline is False
