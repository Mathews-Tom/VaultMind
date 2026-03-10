"""Tests for BM25Index (SQLite FTS5-backed keyword search)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaultmind.indexer.bm25 import BM25Index


@pytest.fixture()
def bm25(tmp_path: Path) -> BM25Index:
    return BM25Index(tmp_path / "test_bm25.db")


class TestUpsertAndSearch:
    def test_upsert_and_search(self, bm25: BM25Index) -> None:
        bm25.upsert(
            "note1::0", "notes/note1.md", "Machine Learning", "neural networks deep learning"
        )
        bm25.upsert("note2::0", "notes/note2.md", "Cooking", "pasta sauce recipe tomato")

        results = bm25.search("neural networks")
        assert len(results) >= 1
        chunk_ids = [r["chunk_id"] for r in results]
        assert "note1::0" in chunk_ids

    def test_search_returns_correct_fields(self, bm25: BM25Index) -> None:
        bm25.upsert("note1::0", "notes/note1.md", "Test Note", "some content here")
        results = bm25.search("content")
        assert len(results) == 1
        r = results[0]
        assert r["chunk_id"] == "note1::0"
        assert r["note_path"] == "notes/note1.md"
        assert r["note_title"] == "Test Note"
        assert isinstance(r["bm25_score"], float)
        assert r["bm25_score"] > 0

    def test_count_reflects_inserts(self, bm25: BM25Index) -> None:
        assert bm25.count == 0
        bm25.upsert("a::0", "a.md", "A", "alpha content")
        assert bm25.count == 1
        bm25.upsert("b::0", "b.md", "B", "beta content")
        assert bm25.count == 2

    def test_upsert_replaces_existing(self, bm25: BM25Index) -> None:
        bm25.upsert("note1::0", "notes/note1.md", "Old Title", "old content")
        bm25.upsert("note1::0", "notes/note1.md", "New Title", "new content")
        assert bm25.count == 1
        results = bm25.search("new content")
        assert len(results) == 1
        assert results[0]["note_title"] == "New Title"

    def test_upsert_batch(self, bm25: BM25Index) -> None:
        rows = [
            ("c1::0", "c1.md", "C1", "first document"),
            ("c2::0", "c2.md", "C2", "second document"),
            ("c3::0", "c3.md", "C3", "third document"),
        ]
        bm25.upsert_batch(rows)
        assert bm25.count == 3


class TestDeleteNote:
    def test_delete_note(self, bm25: BM25Index) -> None:
        bm25.upsert("note1::0", "notes/note1.md", "Title", "unique keyword xyzzy")
        bm25.upsert("note1::1", "notes/note1.md", "Title", "another chunk xyzzy")
        assert bm25.count == 2

        bm25.delete_note("notes/note1.md")
        assert bm25.count == 0

        results = bm25.search("xyzzy")
        assert results == []

    def test_delete_note_only_removes_target(self, bm25: BM25Index) -> None:
        bm25.upsert("note1::0", "notes/note1.md", "A", "quux content")
        bm25.upsert("note2::0", "notes/note2.md", "B", "quux content")

        bm25.delete_note("notes/note1.md")
        assert bm25.count == 1

        results = bm25.search("quux")
        assert len(results) == 1
        assert results[0]["note_path"] == "notes/note2.md"

    def test_delete_nonexistent_note_is_safe(self, bm25: BM25Index) -> None:
        bm25.delete_note("does/not/exist.md")  # must not raise


class TestPorterStemming:
    def test_porter_stemming(self, bm25: BM25Index) -> None:
        """Searching 'running' should match document containing 'run'."""
        bm25.upsert("note1::0", "notes/note1.md", "Exercise", "I go for a run every morning")
        results = bm25.search("running")
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "note1::0"

    def test_plural_stemming(self, bm25: BM25Index) -> None:
        """'dogs' should match 'dog'."""
        bm25.upsert("n::0", "n.md", "Animals", "the dog barked loudly")
        results = bm25.search("dogs")
        assert len(results) >= 1


class TestEmptyQuery:
    def test_empty_string_returns_empty(self, bm25: BM25Index) -> None:
        bm25.upsert("n::0", "n.md", "N", "some content")
        results = bm25.search("")
        assert results == []

    def test_whitespace_only_returns_empty(self, bm25: BM25Index) -> None:
        bm25.upsert("n::0", "n.md", "N", "some content")
        results = bm25.search("   ")
        assert results == []

    def test_search_on_empty_index(self, bm25: BM25Index) -> None:
        results = bm25.search("anything")
        assert results == []
