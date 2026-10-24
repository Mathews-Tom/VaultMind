"""Scope-first source discovery against synthetic files and real scoped FTS."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from vaultmind.config import Settings, load_settings
from vaultmind.indexer.bm25 import BM25Index
from vaultmind.indexer.hybrid import reciprocal_rank_fusion
from vaultmind.indexer.store import VaultStore
from vaultmind.knowledge import sources
from vaultmind.knowledge.intake import SourceIntake
from vaultmind.knowledge.sources import SourceSearchError, revalidate_source_hit, search_sources
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    vault = tmp_path / "vault"
    vault.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "source.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault.resolve()))}\n"
        'excluded_folders = ["wiki", "_wiki", "excluded"]\n'
    )
    return load_settings(config)


def note(settings: Settings, relative: str, body: str) -> Path:
    path = settings.vault.path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


def preserve(settings: Settings, text: str, **kwargs: Any) -> Any:
    with SourceIntake(settings) as intake:
        return intake.preserve_text(
            text, original_name="capture.md", origin="test", adapter="synthetic", **kwargs
        )


class Index:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, n_results: int, where: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append(dict(query=query, n_results=n_results, where=where))
        permitted = set(where["note_path"]["$in"])
        return [hit for hit in self.hits if hit["metadata"]["note_path"] in permitted][:n_results]

    hybrid_search = search


def indexed(settings: Settings, relative: str, distance: float = 0.1) -> list[dict[str, Any]]:
    parser = VaultParser(settings.vault)
    return [
        dict(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.to_chroma_metadata(),
            distance=distance,
        )
        for chunk in parser.chunk_note(parser.parse_file(settings.vault.path / relative))
    ]


def test_ordinary_and_preserved_are_unapproved_readonly_sources(settings: Settings) -> None:
    raw = b"---\ntitle: Current cache\n---\nCache lasts 19 days.\r\n"
    path = note(settings, "notes/cache.md", "unused")
    path.write_bytes(raw)
    capture = preserve(settings, "Cache original says 7 days.")
    before = snapshot(settings.vault.path)
    hits = search_sources(settings, "cache", n_results=10)
    assert {hit["path"] for hit in hits} == {"notes/cache.md", capture.path}
    assert all(hit["status"] == "source-only" and hit["approved"] is False for hit in hits)
    assert all(hit["distance"] is None for hit in hits)
    ordinary = next(hit for hit in hits if hit["source_kind"] == "ordinary")
    assert ordinary["current_digest"] == hashlib.sha256(raw).hexdigest()
    assert ordinary["metadata"]["note_title"] == "Current cache"
    retained = next(hit for hit in hits if hit["source_kind"] == "preserved")
    assert (
        retained["source_id"] == capture.source_id
        and retained["provenance"]["adapter"] == "synthetic"
    )
    assert snapshot(settings.vault.path) == before


def test_reserved_raw_history_pending_and_aliases_are_not_ordinary(settings: Settings) -> None:
    note(settings, "visible.md", "needle")
    for relative in (
        "wiki/current.md",
        "_wiki/history/old.md",
        "_wiki/pending/draft.md",
        "_wiki/sources/raw.md",
        "excluded/private.md",
        "WIKI/case.md",
    ):
        note(settings, relative, "needle " * 100)
    (settings.vault.path / "ordinary-alias").symlink_to(
        settings.vault.path / "_wiki", target_is_directory=True
    )
    (settings.vault.path / "note-alias.md").symlink_to(settings.vault.path / "wiki/current.md")
    assert [hit["path"] for hit in search_sources(settings, "needle")] == ["visible.md"]


def test_authorization_precedes_ordinary_and_preserved_reads(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    note(settings, "allowed/a.md", "needle")
    note(settings, "secret/forbidden.md", "needle")
    preserve(settings, "needle retained")
    original = sources._read_current
    seen: list[Path] = []

    def read(path: Path) -> bytes:
        seen.append(path)
        assert "secret" not in path.parts
        return original(path)

    monkeypatch.setattr(sources, "_read_current", read)
    monkeypatch.setattr(
        SourceIntake, "read", lambda *args: pytest.fail("Forbidden _wiki must not be read")
    )
    hits = search_sources(
        settings,
        "needle",
        allows_path=lambda path: path == "allowed" or path.startswith("allowed/"),
    )
    assert [hit["path"] for hit in hits] == ["allowed/a.md"]
    assert len(seen) == 1


def test_binary_filename_discovery_and_attachment_closure(settings: Settings) -> None:
    with SourceIntake(settings) as intake:
        attachment = intake.preserve_attachment(
            b"\x00\xff\x01",
            original_name="needle.bin",
            origin="test",
            adapter="synthetic",
            media_type="application/octet-stream",
        )
    child = preserve(settings, "needle derived text", attachments=[attachment.source_id])
    hits = search_sources(settings, "needle")
    binary = next(hit for hit in hits if hit.get("source_id") == attachment.source_id)
    assert binary["content"] == "" and binary["distance"] is None
    assert binary["provenance"]["kind"] == "attachment"
    assert revalidate_source_hit(settings, binary)["current_digest"] == binary["current_digest"]
    assert any(hit.get("source_id") == child.source_id for hit in hits)
    assert (
        search_sources(settings, "needle", allows_path=lambda path: path != attachment.path) == []
    )


def test_forbidden_index_hits_cannot_crowd_authorized_current_sources(settings: Settings) -> None:
    note(settings, "allowed.md", "cache authorized")
    note(settings, "secret.md", "cache forbidden")
    store = Index(indexed(settings, "secret.md", 0.0) * 100 + indexed(settings, "allowed.md", 0.3))
    hits = search_sources(
        settings, "cache", store=store, n_results=1, allows_path=lambda path: path == "allowed.md"
    )
    assert [hit["path"] for hit in hits] == ["allowed.md"]
    assert hits[0]["distance"] == 0.3
    assert store.calls[0]["where"] == {"note_path": {"$in": ["allowed.md"]}}
    assert store.calls[0]["n_results"] == sources.MAX_INDEX_CANDIDATES + 1


def test_stale_index_hydrates_current_title_body_and_unknown_distance(settings: Settings) -> None:
    path = note(settings, "cache.md", "---\ntitle: Old title\n---\nOld secret excerpt")
    store = Index(indexed(settings, "cache.md", 0.0))
    path.write_text("---\ntitle: New title\n---\nCurrent cache text")
    hit = search_sources(settings, "cache", store=store)[0]
    assert hit["content"] == "Current cache text"
    assert hit["metadata"]["note_title"] == "New title"
    assert hit["distance"] is None
    assert "Old secret excerpt" not in json.dumps(hit)


def test_missing_index_path_is_not_a_candidate(settings: Settings) -> None:
    path = note(settings, "gone.md", "cache")
    store = Index(indexed(settings, "gone.md"))
    path.unlink()
    assert search_sources(settings, "cache", store=store) == []
    assert store.calls == []


def test_exact_current_chunk_retains_raw_not_fusion_confidence(settings: Settings) -> None:
    note(settings, "cache.md", "cache rules")
    store = Index(indexed(settings, "cache.md", 0.41))
    hit = search_sources(settings, "cache", store=store)[0]
    assert hit["distance"] == 0.41
    assert hit["rrf_score"] != hit["distance"]


def test_current_authority_reorders_complete_sources_before_limit(settings: Settings) -> None:
    settings.ranking.enabled = True
    low = note(settings, "a.md", "---\nauthority: 5\n---\ncache rules")
    high = note(settings, "z.md", "---\nauthority: 1\n---\ncache rules")
    store = Index(indexed(settings, "a.md", 0.2) + indexed(settings, "z.md", 0.21))
    # Old index stamps disagree with current frontmatter; only the latter owns
    # provenance ranking. The exact chunk text remains current in both notes.
    low.write_text("---\nauthority: 1\n---\ncache rules")
    high.write_text("---\nauthority: 5\n---\ncache rules")
    hit = search_sources(settings, "cache", store=store, n_results=1)[0]
    assert hit["path"] == "z.md" and hit["metadata"]["authority"] == 5
    assert hit["distance"] == 0.21
    assert hit["authority_score"] != hit["distance"]


@pytest.mark.parametrize(
    ("authority_yaml", "expected"),
    [
        ("invalid", 0),
        ("null", 0),
        ("[]", 0),
        ("{}", 0),
        (".inf", 0),
        ("-.inf", 0),
        (".nan", 0),
        ("0", 0),
        ("-1", 0),
        ("6", 0),
        ("1", 1),
        ("5", 5),
        ("'4'", 4),
        ("true", 1),
    ],
)
@pytest.mark.parametrize("kind", ["ordinary", "preserved"])
def test_source_authority_normalizes_before_ranking(
    settings: Settings, authority_yaml: str, expected: int, kind: str
) -> None:
    settings.ranking.enabled = True
    text = f"---\nauthority: {authority_yaml}\n---\nneedle retained evidence"
    if kind == "preserved":
        path = preserve(settings, text).path
    else:
        note(settings, "current.md", text)
        path = "current.md"
    before = snapshot(settings.vault.path)

    hit = search_sources(settings, "needle")[0]

    assert hit["path"] == path
    assert hit["metadata"]["authority"] == expected
    level = expected or settings.ranking.authority_default
    assert hit["authority_score"] == pytest.approx(
        hit["rrf_score"] * settings.ranking.authority_weight[level]
    )
    assert hit["distance"] is None
    assert hit["content"] == (text if kind == "preserved" else "needle retained evidence")
    assert hit["current_digest"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert snapshot(settings.vault.path) == before


def test_retained_parent_closure_missing_tampered_or_unauthorized_is_not_discoverable(
    settings: Settings,
) -> None:
    parent = preserve(settings, "parent context")
    child = preserve(settings, "needle child", parents=[parent.source_id])
    assert any(
        hit.get("source_id") == child.source_id for hit in search_sources(settings, "needle")
    )
    denied = {parent.path, f"_wiki/sources/{parent.source_id}.json"}
    assert search_sources(settings, "needle", allows_path=lambda path: path not in denied) == []
    parent_path = settings.vault.path / parent.path
    parent_path.write_text("changed parent")
    assert search_sources(settings, "needle") == []
    parent_path.unlink()
    assert search_sources(settings, "needle") == []


def test_missing_new_manifest_does_not_downgrade_to_legacy(settings: Settings) -> None:
    capture = preserve(settings, "needle")
    manifest = settings.vault.path / f"_wiki/sources/{capture.source_id}.intake.json"
    manifest.unlink()
    assert search_sources(settings, "needle") == []


@pytest.mark.parametrize("kind", ["ordinary", "preserved"])
def test_pagination_refuses_changed_and_missing_snapshot(settings: Settings, kind: str) -> None:
    if kind == "ordinary":
        path = note(settings, "current.md", "needle current")
    else:
        capture = preserve(settings, "needle retained")
        path = settings.vault.path / capture.path
    hit = search_sources(settings, "needle")[0]
    assert revalidate_source_hit(settings, hit)["current_digest"] == hit["current_digest"]
    path.write_text("changed secret body")
    with pytest.raises(SourceSearchError, match="rerun source search") as error:
        revalidate_source_hit(settings, hit)
    assert "secret" not in str(error.value) and str(path) not in str(error.value)
    path.unlink()
    with pytest.raises(SourceSearchError, match="rerun source search"):
        revalidate_source_hit(settings, hit)


def test_pagination_refuses_scope_loss(settings: Settings) -> None:
    note(settings, "current.md", "needle")
    hit = search_sources(settings, "needle")[0]
    with pytest.raises(SourceSearchError, match="rerun source search"):
        revalidate_source_hit(settings, hit, allows_path=lambda _: False)


def test_deterministic_ties_and_authorized_only_bounds(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sources, "MAX_SOURCE_CANDIDATES", 2)
    note(settings, "b.md", "needle")
    note(settings, "a.md", "needle")
    for index in range(8):
        note(settings, f"secret/{index}.md", "needle")

    def allow(path: str) -> bool:
        return not path.startswith("secret")

    assert [hit["path"] for hit in search_sources(settings, "needle", allows_path=allow)] == [
        "a.md",
        "b.md",
    ]
    assert search_sources(settings, "needle", allows_path=allow) == search_sources(
        settings, "needle", allows_path=allow
    )
    note(settings, "c.md", "needle")
    with pytest.raises(SourceSearchError, match="narrow"):
        search_sources(settings, "needle", allows_path=allow)


def test_backend_complete_candidate_overflow_refuses_not_silently_truncates(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    note(settings, "a.md", "needle")
    monkeypatch.setattr(sources, "MAX_INDEX_CANDIDATES", 2)
    rows = indexed(settings, "a.md") * 3
    with pytest.raises(SourceSearchError, match="narrow"):
        search_sources(settings, "needle", store=Index(rows), n_results=1)


def test_real_bm25_permitted_relation_is_before_topk_and_parameter_safe(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25.db")
    try:
        index._con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        index.upsert("forbidden::0", "forbidden.md", "needle", "needle")
        index.upsert("allowed::0", "allowed.md", "A", "needle useful source")
        permitted = [f"nonmatching/{i}.md" for i in range(1500)] + ["allowed.md"]
        hits = index.search("needle", n_results=1, note_paths=permitted)
        assert [hit["note_path"] for hit in hits] == ["allowed.md"]
        assert index.search("needle", note_paths=[]) == []
        assert index._con.execute("SELECT count(*) FROM bm25_permitted_paths").fetchone()[0] == 0
    finally:
        index.close()


def test_real_hybrid_passes_same_scope_to_vector_and_fts(
    settings: Settings, tmp_path: Path
) -> None:
    note(settings, "allowed.md", "needle current")
    allowed = indexed(settings, "allowed.md")[0]
    calls: list[dict[str, Any]] = []

    class Collection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            assert kwargs["where"] == {"note_path": {"$in": ["allowed.md"]}}
            return {"ids": [allowed["chunk_id"]], "metadatas": [allowed["metadata"]]}

        def query(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            assert kwargs["where"] == {"note_path": {"$in": ["allowed.md"]}}
            return dict(
                ids=[[allowed["chunk_id"]]],
                documents=[[allowed["content"]]],
                metadatas=[[allowed["metadata"]]],
                distances=[[0.2]],
            )

    class Embedder:
        def embed_query(self, query: str) -> list[float]:
            return [1.0]

    keyword = BM25Index(tmp_path / "fts.db")
    keyword.upsert("forbidden::0", "forbidden.md", "needle", "needle")
    keyword.upsert(allowed["chunk_id"], "allowed.md", "A", allowed["content"])
    store = VaultStore.__new__(VaultStore)
    store._collection = Collection()  # type: ignore[assignment]
    store.embedder = Embedder()  # type: ignore[assignment]
    store._bm25 = keyword
    store._search_cache = None
    try:
        hits = store.hybrid_search(
            "needle", n_results=1, where={"note_path": {"$in": ["allowed.md"]}}
        )
        assert len(hits) == 1 and hits[0]["metadata"]["note_path"] == "allowed.md"
        assert hits[0]["bm25_rank"] == 1 and hits[0]["vector_rank"] == 1
        assert hits[0]["distance"] == 0.2
        assert len(calls) == 3
    finally:
        keyword.close()


def test_scoped_fts_only_note_is_not_lost_without_vector_chunks(tmp_path: Path) -> None:
    class EmptyCollection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            return {"ids": [], "metadatas": []}

        def query(self, **kwargs: Any) -> Any:
            pytest.fail("Empty vector scope must not query or embed")

    keyword = BM25Index(tmp_path / "fts-only.db")
    keyword.upsert("allowed::0", "allowed.md", "A", "needle")
    keyword.upsert("forbidden::0", "forbidden.md", "needle", "needle")
    store = VaultStore.__new__(VaultStore)
    store._collection = EmptyCollection()  # type: ignore[assignment]
    store._bm25 = keyword
    store._search_cache = None
    try:
        hits = store.hybrid_search(
            "needle", n_results=1, where={"note_path": {"$in": ["allowed.md"]}}
        )
        assert [hit["metadata"]["note_path"] for hit in hits] == ["allowed.md"]
        assert hits[0]["distance"] is None
    finally:
        keyword.close()


def test_fusion_ties_are_stable_by_path() -> None:
    hits = [dict(chunk_id="z::0", content="z", metadata={"note_path": "z.md"}, distance=0.1)]
    lexical = [dict(chunk_id="a::0", note_path="a.md", note_title="A", bm25_score=1.0)]
    assert [hit.metadata["note_path"] for hit in reciprocal_rank_fusion(hits, lexical)] == [
        "a.md",
        "z.md",
    ]
