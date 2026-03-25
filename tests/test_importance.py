"""Tests for content-based importance scoring."""

from __future__ import annotations

import pytest

from vaultmind.indexer.importance import compute_importance
from vaultmind.indexer.ranking import RankedResult


class TestComputeImportance:
    def test_empty_content_returns_low_score(self) -> None:
        score = compute_importance("")
        assert score == pytest.approx(0.0)

    def test_rich_content_returns_high_score(self) -> None:
        content = (
            " ".join(["word"] * 600)
            + " [[Link1]] [[Link2]] [[Link3]] [[Link4]] [[Link5]]"
            + " [[Link6]] [[Link7]] [[Link8]] [[Link9]] [[Link10]]"
        )
        score = compute_importance(
            content=content,
            tags=["t1", "t2", "t3", "t4", "t5"],
            entities=["e1", "e2", "e3", "e4", "e5"],
        )
        assert score == pytest.approx(1.0)

    def test_entity_density_capped_at_one(self) -> None:
        score = compute_importance(
            content="short",
            entities=[f"e{i}" for i in range(10)],
        )
        # entity factor = 1.0, others low
        assert score >= 0.25

    def test_wikilink_density_counted(self) -> None:
        content = "See [[Alpha]] and [[Beta]] and [[Gamma]]."
        score = compute_importance(content=content)
        # 3 links / 10 = 0.3, plus some word count
        assert score > 0.0

    def test_tag_count_contributes(self) -> None:
        score_no_tags = compute_importance("hello world")
        score_with_tags = compute_importance("hello world", tags=["a", "b", "c", "d", "e"])
        assert score_with_tags > score_no_tags

    def test_word_count_contributes(self) -> None:
        short = compute_importance("hello")
        long_content = " ".join(["word"] * 600)
        long_score = compute_importance(long_content)
        assert long_score > short

    def test_no_tags_no_entities_still_scores_content(self) -> None:
        content = " ".join(["word"] * 250) + " [[SomeLink]]"
        score = compute_importance(content)
        assert score > 0.0

    def test_score_always_between_zero_and_one(self) -> None:
        cases = [
            ("", [], []),
            ("short", ["t"], ["e"]),
            (" ".join(["w"] * 1000), ["t"] * 20, ["e"] * 20),
            ("[[a]] [[b]] [[c]]" * 50, [], []),
        ]
        for content, tags, entities in cases:
            score = compute_importance(content, tags, entities)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_none_tags_treated_as_empty(self) -> None:
        score = compute_importance("hello", tags=None)
        assert score >= 0.0

    def test_none_entities_treated_as_empty(self) -> None:
        score = compute_importance("hello", entities=None)
        assert score >= 0.0


class TestNoteChunkImportance:
    def test_importance_score_in_chroma_metadata(self) -> None:
        from vaultmind.vault.models import NoteChunk

        chunk = NoteChunk(
            note_path="test.md",
            note_title="Test",
            chunk_idx=0,
            content="test content",
            importance_score=0.75,
        )
        meta = chunk.to_chroma_metadata()
        assert meta["importance_score"] == 0.75

    def test_importance_score_default_zero(self) -> None:
        from vaultmind.vault.models import NoteChunk

        chunk = NoteChunk(
            note_path="test.md",
            note_title="Test",
            chunk_idx=0,
            content="test content",
        )
        assert chunk.importance_score == 0.0


class TestRankedResultImportance:
    def test_ranked_result_has_importance_field(self) -> None:
        r = RankedResult(
            chunk_id="t::0",
            raw_score=0.5,
            final_score=0.5,
            note_type="fleeting",
            metadata={},
            content="test",
        )
        assert r.importance_score == 0.0

    def test_ranked_result_importance_populated(self) -> None:
        r = RankedResult(
            chunk_id="t::0",
            raw_score=0.5,
            final_score=0.5,
            note_type="fleeting",
            metadata={},
            content="test",
            importance_score=0.8,
        )
        assert r.importance_score == 0.8


class TestRankingConfigImportance:
    def test_importance_scoring_enabled_default(self) -> None:
        from vaultmind.config import RankingConfig

        cfg = RankingConfig()
        assert cfg.importance_scoring_enabled is True
