"""Tests for the golden question set loader and schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaultmind.bench.golden import GoldenQuestion, GoldenSetError, load_golden_set


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadGoldenSet:
    def test_loads_valid_answerable_and_unanswerable_questions(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: q1
                question: "Where is note A?"
                answerable: true
                expected_notes: ["a.md"]
              - id: q2
                question: "Unanswerable question"
                answerable: false
                expected_notes: []
            """,
        )
        questions = load_golden_set(path)
        assert questions == [
            GoldenQuestion(
                id="q1", question="Where is note A?", answerable=True, expected_notes=("a.md",)
            ),
            GoldenQuestion(
                id="q2", question="Unanswerable question", answerable=False, expected_notes=()
            ),
        ]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenSetError, match="not found"):
            load_golden_set(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "questions: [unterminated")
        with pytest.raises(GoldenSetError, match="not valid YAML"):
            load_golden_set(path)

    def test_missing_questions_key_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "version: 1")
        with pytest.raises(GoldenSetError, match="'questions' key"):
            load_golden_set(path)

    def test_empty_questions_list_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "questions: []")
        with pytest.raises(GoldenSetError, match="non-empty list"):
            load_golden_set(path)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: dup
                question: "Q1"
                answerable: true
                expected_notes: ["a.md"]
              - id: dup
                question: "Q2"
                answerable: true
                expected_notes: ["b.md"]
            """,
        )
        with pytest.raises(GoldenSetError, match="duplicate id"):
            load_golden_set(path)

    def test_answerable_without_expected_notes_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: q1
                question: "Q1"
                answerable: true
                expected_notes: []
            """,
        )
        with pytest.raises(GoldenSetError, match="require non-empty"):
            load_golden_set(path)

    def test_unanswerable_with_expected_notes_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: q1
                question: "Q1"
                answerable: false
                expected_notes: ["a.md"]
            """,
        )
        with pytest.raises(GoldenSetError, match="must have empty"):
            load_golden_set(path)

    def test_missing_answerable_field_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: q1
                question: "Q1"
                expected_notes: ["a.md"]
            """,
        )
        with pytest.raises(GoldenSetError, match="'answerable' must be a bool"):
            load_golden_set(path)

    def test_non_mapping_question_entry_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "questions: [1, 2]")
        with pytest.raises(GoldenSetError, match="expected a mapping"):
            load_golden_set(path)

    def test_empty_question_text_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            questions:
              - id: q1
                question: "  "
                answerable: true
                expected_notes: ["a.md"]
            """,
        )
        with pytest.raises(GoldenSetError, match="'question'"):
            load_golden_set(path)

    def test_real_starter_template_loads(self) -> None:
        """The checked-in `benchmarks/golden.yaml` starter must itself be valid."""
        repo_root = Path(__file__).resolve().parents[2]
        questions = load_golden_set(repo_root / "benchmarks" / "golden.yaml")
        assert len(questions) >= 1
        assert any(q.answerable for q in questions)
        assert any(not q.answerable for q in questions)
