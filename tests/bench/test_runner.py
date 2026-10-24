"""Tests for the retrieval scoring runner — recall@k/MRR and score-floor
honest-decline scoring, with zero LLM calls and zero live embeddings."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import pytest

from vaultmind.bench.fixture_store import FixtureStore
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
from vaultmind.config import RankingConfig, Settings, VaultConfig
from vaultmind.indexer.ranking import RetrievalConfidence, apply_authority
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


class FakeStore(FixtureStore):
    """A deterministic, scope-respecting backend recording requested candidates."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__(results_by_query)
        self.search_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.hybrid_calls: list[tuple[str, dict[str, Any] | None]] = []

    def search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.search_calls.append((query, where))
        return super().search(query, n_results=n_results, where=where)

    def hybrid_search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.hybrid_calls.append((query, where))
        return super().search(query, n_results=n_results, where=where)


def _settings(tmp_path: Path) -> Settings:
    root = tmp_path / "vault"
    root.mkdir()
    return Settings.model_construct(vault=VaultConfig(path=root))


def _note(settings: Settings, path: str, text: str) -> None:
    destination = settings.vault.path / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _indexed(settings: Settings, path: str, distance: float | None = 0.1) -> dict[str, Any]:
    parser = VaultParser(settings.vault)
    note = parser.parse_file(settings.vault.path / path)
    note.tags.sort()
    chunk = parser.chunk_note(note)[0]
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": chunk.to_chroma_metadata(),
        "distance": distance,
    }


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
    @pytest.mark.parametrize("hybrid", [True, False])
    def test_uses_current_source_path_and_scoped_backend(
        self, tmp_path: Path, hybrid: bool
    ) -> None:
        settings = _settings(tmp_path)
        settings.search.hybrid_enabled = hybrid
        _note(settings, "allowed/current.md", "Current evidence")
        _note(settings, "denied/secret.md", "Current secret")
        store = FakeStore(
            {"q": [_hit("denied/secret.md"), _indexed(settings, "allowed/current.md")]}
        )
        hits = run_query(
            settings,
            "q",
            k=1,
            store=store,
            allows_path=lambda path: path == "allowed" or path.startswith("allowed/"),
        )
        assert [hit["path"] for hit in hits] == ["allowed/current.md"]
        assert hits[0]["content"] == "Current evidence"
        assert hits[0]["status"] == "source-only" and hits[0]["approved"] is False
        assert hits[0]["distance"] == 0.1
        expected = [("q", {"note_path": {"$in": ["allowed/current.md"]}})]
        assert (store.hybrid_calls if hybrid else store.search_calls) == expected
        assert (store.search_calls if hybrid else store.hybrid_calls) == []

    def test_stale_index_excerpt_is_not_current_evidence(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _note(settings, "a.md", "historical body")
        store = FakeStore({"q": [_indexed(settings, "a.md")]})
        current = "---\ntitle: Current\n---\nCurrent body\n"
        _note(settings, "a.md", current)
        hits = run_query(settings, "q", 5, store=store)
        assert hits[0]["content"] == "Current body"
        assert hits[0]["current_digest"] == hashlib.sha256(current.encode()).hexdigest()
        assert hits[0]["distance"] is None
        assert "historical" not in str(hits)

    def test_missing_history_and_approved_pages_never_score_source_hits(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(tmp_path)
        _note(settings, "a.md", "Current")
        _note(settings, "wiki/answer.md", "answer secret")
        _note(settings, "_wiki/history/retired.md", "history secret")
        store = FakeStore(
            {
                "q": [
                    _hit("_wiki/history/retired.md"),
                    _hit("wiki/answer.md"),
                    _hit("gone.md"),
                    _indexed(settings, "a.md"),
                ]
            }
        )
        hits = run_query(settings, "q", 1, store=store)
        assert [hit["path"] for hit in hits] == ["a.md"]
        assert "secret" not in str(hits)

    def test_preserved_sources_keep_provenance_without_query_side_effects(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(tmp_path)
        with SourceIntake(settings) as intake:
            capture = intake.preserve_text(
                "Retained source evidence",
                original_name="message.txt",
                origin="telegram:actor:42",
                adapter="telegram",
                event_id="42:9",
            )
        before = {
            path.relative_to(settings.vault.path).as_posix(): path.read_bytes()
            for path in settings.vault.path.rglob("*")
            if path.is_file()
        }
        hits = run_query(settings, "Retained", 5)
        assert len(hits) == 1
        assert hits[0]["source_id"] == capture.source_id
        assert hits[0]["provenance"]["origin"] == "telegram:actor:42"
        assert hits[0]["distance"] is None and hits[0]["approved"] is False
        after = {
            path.relative_to(settings.vault.path).as_posix(): path.read_bytes()
            for path in settings.vault.path.rglob("*")
            if path.is_file()
        }
        assert after == before


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

    @pytest.mark.parametrize(
        ("distances", "confidence", "decline"),
        [
            ([], RetrievalConfidence.WEAK, True),
            ([None], RetrievalConfidence.UNKNOWN, False),
            ([0.5], RetrievalConfidence.WEAK, True),
            ([0.499999], RetrievalConfidence.STRONG, False),
            ([0.9, None], RetrievalConfidence.UNKNOWN, False),
            ([None, 0.1], RetrievalConfidence.STRONG, False),
        ],
    )
    def test_fusion_and_authority_cannot_change_decline_evidence(
        self,
        distances: list[float | None],
        confidence: RetrievalConfidence,
        decline: bool,
    ) -> None:
        hits = [
            {
                "chunk_id": f"{i}::0",
                "content": "candidate",
                "metadata": {"note_path": f"{i}.md", "authority": 5 if i else 1},
                "distance": distance,
                "rrf_score": 0.01 + i * 0.001,
            }
            for i, distance in enumerate(distances)
        ]
        question = _q("unknown", answerable=False)
        before = score_question(question, hits, 0.5)
        after = score_question(question, apply_authority(hits, RankingConfig()), 0.5)
        assert before.confidence is after.confidence is confidence
        assert before.retrieval_decline_correct is after.retrieval_decline_correct is decline


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
    def test_zero_llm_calls_by_default(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _note(settings, "a.md", "Evidence")
        golden = [_q("q1", "q1", True, ("a.md",))]
        store = FakeStore({"q1": [_indexed(settings, "a.md")]})
        report = run_bench(golden, settings, k=5, store=store)
        assert report.llm_cite_or_decline_accuracy is None
        assert report.recall_at_k == 1.0 and report.mrr == 1.0

    def test_decline_scorer_receives_only_current_source_evidence(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _note(settings, "a.md", "old evidence")
        store = FakeStore({"q1": [_indexed(settings, "a.md")]})
        _note(settings, "a.md", "current evidence")
        golden = [_q("q1", "q1", True, ("a.md",))]
        seen: list[list[dict[str, Any]]] = []

        def scorer(question: GoldenQuestion, hits: list[dict[str, Any]]) -> bool:
            assert question.id == "q1"
            seen.append(hits)
            return True

        report = run_bench(golden, settings, k=5, store=store, decline_scorer=scorer)
        assert seen[0][0]["content"] == "current evidence"
        assert seen[0][0]["approved"] is False
        assert seen[0][0]["distance"] is None
        assert report.llm_cite_or_decline_accuracy == pytest.approx(1.0)

    def test_settings_scope_and_weak_unknown_declines_reach_report(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        settings.bench.score_floor = 0.8
        _note(settings, "allowed/a.md", "evidence")
        _note(settings, "denied/secret.md", "secret")
        golden = [
            _q("weak", "weak", False),
            _q("unknown", "unknown", False),
            _q("hit", "hit", True, ("allowed/a.md",)),
        ]
        store = FakeStore(
            {
                "weak": [_indexed(settings, "allowed/a.md", 0.9)],
                "unknown": [_indexed(settings, "allowed/a.md", None)],
                "hit": [_hit("denied/secret.md"), _indexed(settings, "allowed/a.md")],
            }
        )
        report = run_bench(
            golden,
            settings,
            1,
            store=store,
            allows_path=lambda path: path == "allowed" or path.startswith("allowed/"),
        )
        assert report.recall_at_k == 1.0
        assert report.results[0].confidence is RetrievalConfidence.WEAK
        assert report.results[1].confidence is RetrievalConfidence.UNKNOWN
        assert report.retrieval_decline_accuracy == 0.5
        assert all("denied/secret.md" not in result.retrieved_paths for result in report.results)

    def test_current_file_authority_changes_source_retrieval_mrr(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _note(settings, "draft.md", "---\nauthority: 1\n---\nEvidence")
        _note(settings, "authored.md", "---\nauthority: 5\n---\nEvidence")
        golden = [_q("q1", "q1", True, ("authored.md",))]
        store = FakeStore(
            {
                "q1": [
                    _indexed(settings, "draft.md", 0.2),
                    _indexed(settings, "authored.md", 0.3),
                ]
            }
        )
        settings.ranking.enabled = False
        before = run_bench(golden, settings, 5, store=store)
        settings.ranking.enabled = True
        after = run_bench(golden, settings, 5, store=store)
        assert before.mrr == 0.5 and after.mrr == 1.0
        hits = run_query(settings, "q1", 5, store=store)
        assert [hit["distance"] for hit in hits] == [0.3, 0.2]
        assert all(hit["approved"] is False for hit in hits)


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
