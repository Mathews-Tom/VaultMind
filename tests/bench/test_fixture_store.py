"""Tests for the `--bundle` fixture store and bundle loader."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from vaultmind.bench.fixture_store import BundleError, FixtureStore, load_bundle
from vaultmind.bench.golden import GoldenQuestion
from vaultmind.bench.runner import run_query, score_question
from vaultmind.config import Settings, VaultConfig
from vaultmind.indexer.ranking import RetrievalConfidence
from vaultmind.knowledge.intake import SourceIntake

if TYPE_CHECKING:
    from pathlib import Path


def _write_bundle(
    tmp_path: Path, golden: str, retrieval: str, notes: dict[str, str] | None = None
) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "golden.yaml").write_text(golden, encoding="utf-8")
    (bundle_dir / "retrieval.yaml").write_text(retrieval, encoding="utf-8")
    vault = bundle_dir / "vault"
    vault.mkdir()
    for relative, content in ({"a.md": "content of a"} if notes is None else notes).items():
        path = vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
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

    @pytest.mark.parametrize("operation", ["search", "hybrid_search"])
    def test_authorized_filter_precedes_top_k_and_preserves_order(self, operation: str) -> None:
        hits = [
            {"metadata": {"note_path": "denied/secret.md"}, "distance": 0.01},
            {"metadata": {"note_path": "allowed/b.md"}, "distance": None},
            {"metadata": {"note_path": "allowed/a.md"}, "distance": 0.9},
        ]
        store = FixtureStore({"q": hits})
        result = getattr(store, operation)(
            "q",
            n_results=1,
            where={"note_path": {"$in": ["allowed/a.md", "allowed/b.md"]}},
        )
        assert result == [hits[1]]
        assert result[0]["distance"] is None


class TestLoadBundle:
    def test_loads_valid_bundle(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
        bundle = load_bundle(bundle_dir)
        assert bundle.golden_path == bundle_dir / "golden.yaml"
        assert bundle.vault_path == bundle_dir / "vault"
        hits = bundle.store.search("Where is A?", n_results=5)
        assert hits[0]["metadata"]["note_path"] == "a.md"
        assert hits[0]["distance"] == pytest.approx(0.1)

    @pytest.mark.parametrize("seed", ["0", "1", "2"])
    def test_unchanged_tagged_source_keeps_raw_distance_across_process_seeds(
        self, tmp_path: Path, seed: str
    ) -> None:
        bundle_dir = _write_bundle(
            tmp_path,
            _VALID_GOLDEN,
            _VALID_RETRIEVAL,
            {
                "a.md": "---\ntitle: Tagged\ntags: [zebra, alpha, citrus]\n---\n\n"
                "Current multi-tag evidence.\n"
            },
        )
        home = tmp_path / "home"
        home.mkdir()
        script = (
            "import hashlib, sys\n"
            "from pathlib import Path\n"
            "from vaultmind.bench.fixture_store import load_bundle\n"
            "from vaultmind.bench.runner import run_query\n"
            "from vaultmind.config import Settings, VaultConfig\n"
            "bundle = load_bundle(Path(sys.argv[1]))\n"
            "settings = Settings(_env_file=None, vault=VaultConfig(path=bundle.vault_path))\n"
            "hits = run_query(settings, 'Where is A?', 5, store=bundle.store)\n"
            "assert len(hits) == 1\n"
            "assert hits[0]['distance'] == 0.1, hits\n"
            "assert hits[0]['content'] == 'Current multi-tag evidence.'\n"
            "assert hits[0]['approved'] is False\n"
            "raw = (bundle.vault_path / 'a.md').read_bytes()\n"
            "assert hits[0]['current_digest'] == hashlib.sha256(raw).hexdigest()\n"
        )
        result = subprocess.run(
            [sys.executable, "-s", "-c", script, str(bundle_dir)],
            cwd=tmp_path,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "PYTHONHASHSEED": seed,
                "ANONYMIZED_TELEMETRY": "False",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

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

    def test_metadata_comes_only_from_actual_file_frontmatter(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(
            tmp_path,
            _VALID_GOLDEN,
            _VALID_RETRIEVAL,
            {"a.md": "---\ntitle: Real title\nauthority: 5\n---\nActual body\n"},
        )
        bundle = load_bundle(bundle_dir)
        hits = bundle.store.search("Where is A?", n_results=5)
        assert hits[0]["metadata"]["authority"] == 5
        assert hits[0]["metadata"]["note_title"] == "Real title"
        assert "Actual body" in hits[0]["content"]

    def test_inline_payload_cannot_create_a_current_source(self, tmp_path: Path) -> None:
        retrieval = '"Where is A?":\n  - note_path: "a.md"\n    content: "fabricated"\n'
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        with pytest.raises(BundleError, match="bundle-local vault files"):
            load_bundle(bundle_dir)

    def test_current_note_is_required_even_with_canned_distance(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL, {})
        with pytest.raises(BundleError, match="ordinary current note"):
            load_bundle(bundle_dir)

    def test_missing_actual_vault_is_rejected(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "golden.yaml").write_text(_VALID_GOLDEN)
        (bundle_dir / "retrieval.yaml").write_text(_VALID_RETRIEVAL)
        with pytest.raises(BundleError, match="actual bundle-local vault"):
            load_bundle(bundle_dir)

    @pytest.mark.parametrize("path", ["../outside.md", "wiki/page.md", "_wiki/history/retired.md"])
    def test_fixture_candidates_cannot_import_external_or_reserved_notes(
        self, tmp_path: Path, path: str
    ) -> None:
        retrieval = f'"Where is A?":\n  - note_path: "{path}"\n    distance: 0.1\n'
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        with pytest.raises(BundleError, match="ordinary current note"):
            load_bundle(bundle_dir)

    def test_loaded_index_snapshot_does_not_hide_later_current_mutation(
        self, tmp_path: Path
    ) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
        bundle = load_bundle(bundle_dir)
        settings = Settings.model_construct(vault=VaultConfig(path=bundle.vault_path))
        before = run_query(settings, "Where is A?", 5, store=bundle.store)
        assert before[0]["distance"] == 0.1
        current = "---\ntitle: Changed\n---\nFresh body\n"
        (bundle.vault_path / "a.md").write_text(current, encoding="utf-8")
        after = run_query(settings, "Where is A?", 5, store=bundle.store)
        assert after[0]["content"] == "Fresh body"
        assert after[0]["distance"] is None
        assert after[0]["current_digest"] == hashlib.sha256(current.encode()).hexdigest()
        assert "content of a" not in str(after)

    def test_preserved_snapshot_corpus_uses_the_same_current_source_path(
        self, tmp_path: Path
    ) -> None:
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
        settings = Settings.model_construct(vault=VaultConfig(path=bundle_dir / "vault"))
        with SourceIntake(settings) as intake:
            capture = intake.preserve_text(
                "Retained corpus evidence",
                original_name="message.txt",
                origin="fixture:actor:7",
                adapter="telegram",
                event_id="7:9",
            )
        bundle = load_bundle(bundle_dir)
        hits = run_query(settings, "Retained", 5, store=bundle.store)
        assert len(hits) == 1
        assert hits[0]["source_id"] == capture.source_id
        assert hits[0]["current_digest"] == capture.sha256
        assert hits[0]["provenance"]["event_id"] == "7:9"
        assert hits[0]["distance"] is None and hits[0]["approved"] is False

    def test_fixture_index_cannot_read_through_an_intermediate_symlink(
        self, tmp_path: Path
    ) -> None:
        retrieval = '"Where is A?":\n  - note_path: "alias/current.md"\n    distance: 0.1\n'
        bundle_dir = _write_bundle(
            tmp_path, _VALID_GOLDEN, retrieval, {"notes/current.md": "Current evidence"}
        )
        (bundle_dir / "vault/alias").symlink_to("notes", target_is_directory=True)
        with pytest.raises(BundleError, match="ordinary current note"):
            load_bundle(bundle_dir)

    @pytest.mark.parametrize("distance_field", ["", "    distance: null\n"])
    def test_missing_vector_evidence_cannot_score_a_correct_decline(
        self, tmp_path: Path, distance_field: str
    ) -> None:
        retrieval = '"Where is A?":\n  - note_path: "a.md"\n    rrf_score: 0.02\n' + distance_field
        bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, retrieval)
        bundle = load_bundle(bundle_dir)
        settings = Settings.model_construct(vault=VaultConfig(path=bundle.vault_path))
        hits = run_query(settings, "Where is A?", 5, store=bundle.store)
        question = GoldenQuestion(
            id="unknown", question="Where is A?", answerable=False, expected_notes=()
        )
        result = score_question(question, hits, 0.5)
        assert result.confidence is RetrievalConfidence.UNKNOWN
        assert result.retrieval_decline_correct is False


def test_fixture_cli_uses_bundle_current_vault_without_provider_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.bench.runner as runner
    from vaultmind.cli import cli

    bundle_dir = _write_bundle(tmp_path, _VALID_GOLDEN, _VALID_RETRIEVAL)
    operator_vault = tmp_path / "operator-vault"
    operator_vault.mkdir()
    (operator_vault / "a.md").write_text("Operator-only private evidence", encoding="utf-8")
    settings = Settings.model_construct(vault=VaultConfig(path=operator_vault))
    monkeypatch.setattr("vaultmind.config.load_settings", lambda *args, **kwargs: settings)
    observed: list[Path] = []
    observed_hits: list[list[dict[str, Any]]] = []
    shared_search = runner.search_sources

    def record_search(active: Settings, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        observed.append(active.vault.path)
        hits = shared_search(active, query, **kwargs)
        observed_hits.append(hits)
        return hits

    def forbid_provider(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Fixture bench assembled a live provider or index")

    monkeypatch.setattr(runner, "search_sources", record_search)
    monkeypatch.setattr("vaultmind.cli._create_embedding_cache", forbid_provider)
    monkeypatch.setattr("vaultmind.cli._create_bm25_index", forbid_provider)
    monkeypatch.setattr("vaultmind.cli._create_llm_client", forbid_provider)
    result = CliRunner().invoke(
        cli, ["bench", "--bundle", str(bundle_dir), "--trend-path", str(tmp_path / "trend.jsonl")]
    )
    assert result.exit_code == 0, result.output
    assert observed == [bundle_dir / "vault"]
    assert len(observed_hits) == 1
    assert len(observed_hits[0]) == 1
    hit = observed_hits[0][0]
    assert hit["path"] == hit["metadata"]["note_path"] == "a.md"
    assert hit["content"] == "content of a"
    assert hit["current_digest"] == hashlib.sha256(b"content of a").hexdigest()
    assert hit["distance"] == 0.1
    assert hit["status"] == "source-only" and hit["approved"] is False
    assert "Operator-only private evidence" not in str(observed_hits)
