"""Tests for the `--bundle` fixture store and bundle loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vaultmind.bench.fixture_store import BundleError, FixtureStore, load_bundle

if TYPE_CHECKING:
    from pathlib import Path


def _write_bundle(tmp_path: Path, golden: str, retrieval: str) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "golden.yaml").write_text(golden, encoding="utf-8")
    (bundle_dir / "retrieval.yaml").write_text(retrieval, encoding="utf-8")
    return bundle_dir


_VALID_GOLDEN = """
questions:
  - id: q1
    question: "Where is A?"
    answerable: true
    expected_notes: ["a.md"]
"""

_VALID_RETRIEVAL = """
"Where is A?":
  - note_path: "a.md"
    content: "content of a"
    distance: 0.1
"""


class TestFixtureStore:
    def test_search_returns_canned_hits_for_known_query(self) -> None:
        store = FixtureStore({"q": [{"metadata": {"note_path": "a.md"}}]})
        assert store.search("q", n_results=5) == [{"metadata": {"note_path": "a.md"}}]

    def test_hybrid_search_returns_same_canned_hits(self) -> None:
        store = FixtureStore({"q": [{"metadata": {"note_path": "a.md"}}]})
        assert store.hybrid_search("q", n_results=5) == [{"metadata": {"note_path": "a.md"}}]

    def test_unknown_query_returns_empty(self) -> None:
        store = FixtureStore({"q": [{"metadata": {}}]})
        assert store.search("other", n_results=5) == []

    def test_respects_n_results_limit(self) -> None:
        hits = [{"metadata": {"note_path": f"{i}.md"}} for i in range(10)]
        store = FixtureStore({"q": hits})
        assert len(store.search("q", n_results=3)) == 3


class TestLoadBundle:
    def test_loads_valid_bundle(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
        bundle = load_bundle(bundle_dir)
        assert bundle.golden_path == bundle_dir / "golden.yaml"
        hits = bundle.store.search("Where is A?", n_results=5)
        assert hits[0]["metadata"]["note_path"] == "a.md"
        assert hits[0]["distance"] == pytest.approx(0.1)

    def test_missing_golden_file_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "retrieval.yaml").write_text(_VALID_RETRIEVAL, encoding="utf-8")
        with pytest.raises(BundleError, match="golden.yaml"):
            load_bundle(bundle_dir)

    def test_missing_retrieval_file_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "golden.yaml").write_text(_VALID_GOLDEN, encoding="utf-8")
        with pytest.raises(BundleError, match="retrieval.yaml"):
            load_bundle(bundle_dir)

    def test_invalid_retrieval_yaml_raises(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, "not: [valid")
        with pytest.raises(BundleError, match="not valid YAML"):
            load_bundle(bundle_dir)

    def test_retrieval_not_a_mapping_raises(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, "- a\n- b\n")
        with pytest.raises(BundleError, match="mapping"):
            load_bundle(bundle_dir)

    def test_retrieval_hits_not_a_list_raises(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, '"Where is A?": "not a list"\n')
        with pytest.raises(BundleError, match="must be a list"):
            load_bundle(bundle_dir)

    def test_hit_missing_note_path_raises(self, tmp_path: Path) -> None:
        retrieval = '"Where is A?":\n  - content: "no path field"\n'
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        with pytest.raises(BundleError, match="note_path"):
            load_bundle(bundle_dir)

    def test_hit_defaults_distance_when_omitted(self, tmp_path: Path) -> None:
        retrieval = '"Where is A?":\n  - note_path: "a.md"\n'
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        bundle = load_bundle(bundle_dir)
        hits = bundle.store.search("Where is A?", n_results=5)
        assert hits[0]["distance"] == pytest.approx(0.1)

    def test_hit_authority_field_parsed_into_metadata(self, tmp_path: Path) -> None:
        retrieval = '"Where is A?":\n  - note_path: "a.md"\n    authority: 5\n'
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        bundle = load_bundle(bundle_dir)
        hits = bundle.store.search("Where is A?", n_results=5)
        assert hits[0]["metadata"]["authority"] == 5

    def test_hit_without_authority_field_omits_metadata_key(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
        bundle = load_bundle(bundle_dir)
        hits = bundle.store.search("Where is A?", n_results=5)
        assert "authority" not in hits[0]["metadata"]
