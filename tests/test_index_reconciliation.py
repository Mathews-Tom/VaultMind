"""Index reconciliation against real Chroma and SQLite BM25.

Only the embedding provider boundary is synthetic. Every assertion is about what
the vector, keyword and hybrid retrieval paths return, what failures the store
reports, and what is left behind after those failures.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from vaultmind.indexer.search_cache import SearchResultCache
from vaultmind.indexer.store import IndexStage, IndexUpdateError, VaultStore
from vaultmind.vault.parser import VaultParseError, VaultParser

from .index_support import ALL_SURFACES, IndexEnv, make_env, make_neighbour_store

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from vaultmind.indexer.embedder import Embedder
    from vaultmind.vault.models import Note, NoteChunk

THREE_SECTIONS = (
    "## Tides\nobserved tidewrack along the north pier\n\n"
    "## Birds\nemberquill gulls nest on the second jetty\n\n"
    "## Lamps\ncobaltmoor lamp replaced at the lighthouse"
)
ONE_SECTION = "## Tides\nobserved tidewrack along the north pier"
REVISED_THREE = (
    "## Tides\nobserved ironweft along the south pier\n\n"
    "## Birds\nsaltreed terns nest on the first jetty\n\n"
    "## Lamps\nbrasswick lamp replaced at the harbour"
)
REVISED_ONE = "## Tides\nobserved ironweft along the south pier"
OTHER_NOTE = "## Marsh\nfennelwick survey of the marsh"
ALL_TOKENS_OF_THREE = ("tidewrack", "emberquill", "cobaltmoor")


@pytest.fixture
def env(tmp_path: Path) -> Iterator[IndexEnv]:
    environment = make_env(tmp_path.resolve())
    try:
        yield environment
    finally:
        environment.close()


def _reindex(env: IndexEnv, mode: str) -> int:
    if mode == "incremental":
        return env.index("log.md")
    return env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)


def _counts(env: IndexEnv) -> tuple[int, int]:
    return env.store.count, env.bm25.count


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_shorter_revision_removes_stale_chunks_from_both_indexes(env: IndexEnv, mode: str) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.write("other.md", OTHER_NOTE)
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)
    assert _counts(env) == (4, 4)

    env.write("log.md", ONE_SECTION)
    reindexed = _reindex(env, mode)

    assert reindexed == (1 if mode == "incremental" else 2)
    assert _counts(env) == (2, 2)
    assert env.surfaces("tidewrack") == ALL_SURFACES
    assert env.surfaces("emberquill") == set()
    assert env.surfaces("cobaltmoor") == set()
    assert env.surfaces("fennelwick") == ALL_SURFACES


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_note_without_indexable_content_is_retired_from_both_indexes(
    env: IndexEnv, mode: str
) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.write("other.md", OTHER_NOTE)
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

    env.write("log.md", "")
    reindexed = _reindex(env, mode)

    assert reindexed == (0 if mode == "incremental" else 1)
    assert _counts(env) == (1, 1)
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == set()
    assert env.surfaces("fennelwick") == ALL_SURFACES


def test_full_index_after_file_removal_removes_the_stale_rows(env: IndexEnv) -> None:
    env.write("keep.md", OTHER_NOTE)
    env.write("log.md", THREE_SECTIONS)
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

    (env.vault / "log.md").unlink()
    indexed = env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

    assert indexed == 1
    assert _counts(env) == (1, 1)
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == set()
    assert env.surfaces("fennelwick") == ALL_SURFACES


def test_empty_snapshot_clears_configured_indexes_only(env: IndexEnv) -> None:
    path = env.write("log.md", THREE_SECTIONS)
    original = path.read_text(encoding="utf-8")
    env.index("log.md")
    neighbour, neighbour_bm25 = make_neighbour_store(env, "unrelated_chunks")
    try:
        neighbour.index_notes([env.note("log.md")], env.parser)

        cleared = env.store.index_notes([], env.parser)

        assert cleared == 0
        assert _counts(env) == (0, 0)
        assert env.surfaces("tidewrack") == set()
        assert path.read_text(encoding="utf-8") == original
        assert (neighbour.count, neighbour_bm25.count) == (3, 3)
        assert any("tidewrack" in hit["content"] for hit in neighbour.search("tidewrack", 10))
        assert [hit["chunk_id"] for hit in neighbour_bm25.search("tidewrack")] == ["log.md::0"]
    finally:
        neighbour_bm25.close()


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_embedding_failure_before_any_write_keeps_last_good_content(
    env: IndexEnv, mode: str
) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.write("gone.md", "## Gone\nquillfern dossier")
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)
    (env.vault / "gone.md").unlink()  # a full reconcile would remove its rows
    env.write("log.md", REVISED_THREE)
    env.embedder.failure = RuntimeError("embedding provider unavailable")

    with pytest.raises(IndexUpdateError) as failure:
        _reindex(env, mode)

    assert failure.value.stage is IndexStage.EMBED
    assert failure.value.completed_stages == ()
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert _counts(env) == (4, 4)
    for token in (*ALL_TOKENS_OF_THREE, "quillfern"):
        assert env.surfaces(token) == ALL_SURFACES
    assert env.surfaces("ironweft") == set()

    env.embedder.failure = None
    _reindex(env, mode)

    assert env.surfaces("ironweft") == ALL_SURFACES
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == set()
    assert env.surfaces("quillfern") == (set() if mode == "full" else ALL_SURFACES)


def _drop_last(vectors: list[list[float]]) -> list[list[float]]:
    return vectors[:-1]


def _empty_first(vectors: list[list[float]]) -> list[list[float]]:
    return [[], *vectors[1:]]


def _ragged(vectors: list[list[float]]) -> list[list[float]]:
    return [vectors[0][:-1], *vectors[1:]]


def _not_a_number(vectors: list[list[float]]) -> list[list[float]]:
    return [[float("nan"), *vectors[0][1:]], *vectors[1:]]


@pytest.mark.parametrize(
    "corrupt",
    [_drop_last, _empty_first, _ragged, _not_a_number],
    ids=["missing-vector", "empty-vector", "ragged-dimensions", "non-finite"],
)
def test_invalid_embedding_batch_is_rejected_before_any_write(
    env: IndexEnv, corrupt: Callable[[list[list[float]]], list[list[float]]]
) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    env.write("log.md", REVISED_THREE)
    env.embedder.corrupt = corrupt

    with pytest.raises(IndexUpdateError) as failure:
        env.index("log.md")

    assert failure.value.stage is IndexStage.EMBED
    assert failure.value.completed_stages == ()
    assert isinstance(failure.value.__cause__, ValueError)
    assert _counts(env) == (3, 3)
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == ALL_SURFACES
    assert env.surfaces("ironweft") == set()


def test_vector_index_failure_leaves_the_keyword_index_untouched(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    env.write("log.md", REVISED_THREE)
    env.embedder.dimensions = 8  # the real Chroma collection holds 32-dimensional vectors

    with pytest.raises(IndexUpdateError) as failure:
        env.index("log.md")

    env.embedder.dimensions = 32
    assert failure.value.stage is IndexStage.VECTOR_UPSERT
    assert failure.value.completed_stages == ()
    assert _counts(env) == (3, 3)
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == ALL_SURFACES
    assert env.surfaces("ironweft") == set()


def test_keyword_failure_reports_partial_progress_and_retry_converges(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    env.write("log.md", REVISED_ONE)
    env.bm25.fail_on = {"replace_note"}

    with pytest.raises(IndexUpdateError) as failure:
        env.index("log.md")

    error = failure.value
    assert error.operation == "index_single_note"
    assert error.stage is IndexStage.KEYWORD_REPLACE
    assert error.completed_stages == (IndexStage.VECTOR_UPSERT, IndexStage.VECTOR_DELETE)
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    # The reported progress is real: the vector side moved on, the keyword side did not.
    assert _counts(env) == (1, 3)

    env.bm25.fail_on = set()
    assert env.index("log.md") == 1

    assert _counts(env) == (1, 1)
    assert env.surfaces("ironweft") == ALL_SURFACES
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == set()


def test_full_index_keyword_failure_reports_partial_progress_and_retry_converges(
    env: IndexEnv,
) -> None:
    env.write("keep.md", OTHER_NOTE)
    env.write("log.md", THREE_SECTIONS)
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)
    (env.vault / "log.md").unlink()
    env.bm25.fail_on = {"replace_all"}

    with pytest.raises(IndexUpdateError) as failure:
        env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

    error = failure.value
    assert error.operation == "index_notes"
    assert error.stage is IndexStage.KEYWORD_REPLACE
    assert error.completed_stages == (IndexStage.VECTOR_UPSERT, IndexStage.VECTOR_DELETE)
    assert _counts(env) == (1, 4)

    env.bm25.fail_on = set()
    assert env.store.index_notes(env.parser.iter_notes(strict=True), env.parser) == 1

    assert _counts(env) == (1, 1)
    for token in ALL_TOKENS_OF_THREE:
        assert env.surfaces(token) == set()


def test_retiring_a_note_reports_keyword_failure_and_retry_converges(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    env.write("log.md", "")
    env.bm25.fail_on = {"delete_note"}

    with pytest.raises(IndexUpdateError) as failure:
        env.index("log.md")

    assert failure.value.stage is IndexStage.KEYWORD_DELETE
    assert failure.value.completed_stages == (IndexStage.VECTOR_DELETE,)
    assert _counts(env) == (0, 3)

    env.bm25.fail_on = set()
    assert env.index("log.md") == 0
    assert _counts(env) == (0, 0)


def test_delete_note_reports_keyword_failure_and_retry_converges(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    env.bm25.fail_on = {"delete_note"}

    with pytest.raises(IndexUpdateError) as failure:
        env.store.delete_note("log.md")

    assert failure.value.operation == "delete_note"
    assert failure.value.stage is IndexStage.KEYWORD_DELETE
    assert failure.value.completed_stages == (IndexStage.VECTOR_DELETE,)

    env.bm25.fail_on = set()
    assert env.store.delete_note("log.md") is None
    assert _counts(env) == (0, 0)
    assert env.store.delete_note("log.md") is None  # idempotent


def test_duplicate_notes_in_a_snapshot_are_rejected_before_embedding(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.index("log.md")
    embedded_before = env.embedder.calls
    note = env.note("log.md")

    with pytest.raises(ValueError):
        env.store.index_notes([note, note], env.parser)

    assert env.embedder.calls == embedded_before
    assert _counts(env) == (3, 3)


def test_index_calls_return_chunk_counts(env: IndexEnv) -> None:
    env.write("log.md", THREE_SECTIONS)
    env.write("other.md", OTHER_NOTE)

    assert env.store.index_notes(env.parser.iter_notes(strict=True), env.parser) == 4
    assert env.index("log.md") == 3
    assert env.index("other.md") == 1


def test_supplied_search_cache_never_serves_a_retired_note(env: IndexEnv) -> None:
    cache = SearchResultCache()
    store = VaultStore(env.store.config, cast("Embedder", env.embedder), env.bm25, cache)
    env.write("a.md", "## A\ntidewrack survey")
    env.write("b.md", OTHER_NOTE)
    store.index_notes(env.parser.iter_notes(strict=True), env.parser)
    assert any("tidewrack" in hit["content"] for hit in store.search("tidewrack"))
    assert cache.size == 1

    store.delete_note("a.md")

    assert not any("tidewrack" in hit["content"] for hit in store.search("tidewrack"))


class _RejectingParser(VaultParser):
    """Real parser whose chunking refuses one note, as for a note with ambiguous history."""

    def chunk_note(self, note: Note, max_tokens: int = 500) -> list[NoteChunk]:
        if note.path.name == "bad.md":
            raise ValueError("history boundary is ambiguous")
        return super().chunk_note(note, max_tokens)


def test_unchunkable_note_fails_before_any_write_and_is_named(env: IndexEnv) -> None:
    env.write("good.md", THREE_SECTIONS)
    env.write("bad.md", OTHER_NOTE)
    env.index("good.md")
    env.index("bad.md")
    embedded_before = env.embedder.calls
    env.write("good.md", REVISED_THREE)
    parser = _RejectingParser(env.parser.config)

    with pytest.raises(ValueError, match=r"bad\.md"):
        env.store.index_notes(parser.iter_notes(strict=True), parser)
    with pytest.raises(ValueError):
        env.store.index_single_note(parser.parse_file(env.vault / "bad.md"), parser)

    assert env.embedder.calls == embedded_before
    assert _counts(env) == (4, 4)
    for token in (*ALL_TOKENS_OF_THREE, "fennelwick"):
        assert env.surfaces(token) == ALL_SURFACES
    assert env.surfaces("ironweft") == set()


def test_strict_snapshot_with_unparseable_note_leaves_real_indexes_untouched(
    env: IndexEnv,
) -> None:
    env.write("keep.md", OTHER_NOTE)
    env.write("log.md", THREE_SECTIONS)
    env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)
    embedded_before = env.embedder.calls
    (env.vault / "log.md").write_text("---\ntitle: [unclosed\n---\n\nx\n", encoding="utf-8")

    with pytest.raises(VaultParseError) as failure:
        env.store.index_notes(env.parser.iter_notes(strict=True), env.parser)

    assert {str(path) for path, _ in failure.value.failures} == {"log.md"}
    assert env.embedder.calls == embedded_before
    assert _counts(env) == (4, 4)
    for token in (*ALL_TOKENS_OF_THREE, "fennelwick"):
        assert env.surfaces(token) == ALL_SURFACES


def test_non_reconciling_cli_consumer_still_works_with_an_unparseable_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    import vaultmind.cli as app
    import vaultmind.config as config

    root = tmp_path.resolve()
    vault = root / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("---\ntype: permanent\n---\n\nreadable\n", encoding="utf-8")
    (vault / "bad.md").write_text("---\ntitle: [unclosed\n---\n\nx\n", encoding="utf-8")
    monkeypatch.setattr(config, "VAULTMIND_HOME", root / "home")
    toml = root / "fixture.toml"
    toml.write_text(
        f'[vault]\npath = "{vault}"\n[graph]\npersist_path = "{root / "graph.json"}"\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(app.cli, ["--config", str(toml), "stats"])

    assert result.exit_code == 0, result.output
    assert "Total notes: 1" in result.output
