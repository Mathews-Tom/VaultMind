"""Tests for BM25Index (SQLite FTS5-backed keyword search)."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from vaultmind.indexer.bm25 import BM25Index

if TYPE_CHECKING:
    from pathlib import Path


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


class TestScopedSearchTransactions:
    @pytest.mark.parametrize(
        ("query", "note_paths", "expected_ids"),
        [
            ("scopedtoken", ["allowed.md"], ["allowed::0"]),
            ("missingtoken", ["allowed.md"], []),
            ("scopedtoken", [], []),
        ],
    )
    def test_releases_read_lock_and_observes_other_connection(
        self,
        tmp_path: Path,
        query: str,
        note_paths: list[str],
        expected_ids: list[str],
    ) -> None:
        db_path = tmp_path / "shared_bm25.db"
        reader = BM25Index(db_path)
        writer = BM25Index(db_path)
        try:
            writer._con.execute("PRAGMA busy_timeout = 100")
            writer.upsert("allowed::0", "allowed.md", "Original", "scopedtoken")

            assert [
                row["chunk_id"] for row in reader.search(query, note_paths=note_paths)
            ] == expected_ids

            # A real write must commit without the first connection retaining
            # its scoped SELECT's read lock or snapshot.
            writer.upsert("allowed::0", "allowed.md", "Updated", "updatedtoken")
            assert not reader._con.in_transaction
            assert reader.search("scopedtoken", note_paths=["allowed.md"]) == []
            assert [
                (row["chunk_id"], row["note_title"])
                for row in reader.search("updatedtoken", note_paths=["allowed.md"])
            ] == [("allowed::0", "Updated")]
            assert not reader._con.in_transaction
            assert (
                reader._con.execute("SELECT COUNT(*) FROM bm25_permitted_paths").fetchone()[0] == 0
            )
        finally:
            writer.close()
            reader.close()

    def test_operational_error_fallback_finishes_transaction(
        self, bm25: BM25Index, caplog: pytest.LogCaptureFixture
    ) -> None:
        bm25.upsert("allowed::0", "allowed.md", "Original", "scopedtoken")
        bm25._con.execute("DROP TABLE fts_chunks")

        with caplog.at_level("WARNING", logger="vaultmind.indexer.bm25"):
            assert bm25.search("private query body", note_paths=["allowed.md"]) == []

        assert not bm25._con.in_transaction
        assert bm25._con.execute("SELECT COUNT(*) FROM bm25_permitted_paths").fetchone()[0] == 0
        assert caplog.messages == ["BM25 search failed"]

    def test_scope_setup_failure_rolls_back_transaction(self, bm25: BM25Index) -> None:
        def deny_scope_insert(
            action: int,
            table: str | None,
            _column: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_INSERT and table == "bm25_permitted_paths":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        bm25._con.set_authorizer(deny_scope_insert)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                bm25.search("scopedtoken", note_paths=["allowed.md"])
        finally:
            bm25._con.set_authorizer(None)

        assert not bm25._con.in_transaction
        assert bm25._con.execute("SELECT COUNT(*) FROM bm25_permitted_paths").fetchone()[0] == 0


class TestLargeBatch:
    """Regression: upsert_batch with more rows than SQLITE_MAX_VARIABLE_NUMBER.

    The delete-before-insert step bound one host param per chunk_id in a
    single IN(...) clause, which overflows SQLite's per-statement variable
    limit (999 on many builds) on a full-vault index.
    """

    def test_upsert_batch_exceeding_sqlite_variable_limit(self, bm25: BM25Index) -> None:
        bm25._con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        bm25.upsert("unrelated::0", "unrelated.md", "Unrelated", "unrelatedkeyword")
        rows = [
            (f"note{i}::0", f"notes/note{i}.md", f"Note {i}", f"originalbatchtoken {i}")
            for i in range(1500)
        ]
        bm25.upsert_batch(rows)
        assert sorted(
            (row["chunk_id"], row["note_path"], row["note_title"])
            for row in bm25.search("originalbatchtoken", n_results=2000)
        ) == sorted((chunk_id, path, title) for chunk_id, path, title, _ in rows)

        replacements = [
            (chunk_id, f"updated/{i}.md", f"Revised {i}", f"replacementbatchtoken {i}")
            for i, (chunk_id, _, _, _) in enumerate(rows)
        ]
        bm25.upsert_batch(replacements)
        bm25.upsert("later::0", "later.md", "Later", "laterkeyword")

        assert sorted(
            (row["chunk_id"], row["note_path"], row["note_title"])
            for row in bm25.search("replacementbatchtoken", n_results=2000)
        ) == sorted((chunk_id, path, title) for chunk_id, path, title, _ in replacements)
        assert bm25.search("originalbatchtoken", n_results=2000) == []
        assert [row["chunk_id"] for row in bm25.search("unrelatedkeyword")] == ["unrelated::0"]
        assert [row["chunk_id"] for row in bm25.search("laterkeyword")] == ["later::0"]

    def test_failed_replacement_rolls_back_and_does_not_leak_ids(self, bm25: BM25Index) -> None:
        bm25.upsert("old::0", "old.md", "Old", "originalkeyword")
        bm25.upsert("survivor::0", "survivor.md", "Survivor", "survivorkeyword")

        def deny_fts_insert(
            action: int,
            table: str | None,
            _column: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_INSERT and table == "fts_chunks":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        bm25._con.set_authorizer(deny_fts_insert)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                bm25.upsert("old::0", "updated.md", "Updated", "replacementkeyword")
        finally:
            bm25._con.set_authorizer(None)

        bm25.upsert("fresh::0", "fresh.md", "Fresh", "freshkeyword")
        assert [row["chunk_id"] for row in bm25.search("originalkeyword")] == ["old::0"]
        assert bm25.search("replacementkeyword") == []
        assert [row["chunk_id"] for row in bm25.search("survivorkeyword")] == ["survivor::0"]
        assert [row["chunk_id"] for row in bm25.search("freshkeyword")] == ["fresh::0"]
