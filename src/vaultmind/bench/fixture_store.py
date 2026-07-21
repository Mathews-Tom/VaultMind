"""Deterministic canned-retrieval store for `vaultmind bench --bundle`.

A `--bundle` directory pairs a `golden.yaml` (see `vaultmind.bench.golden`)
with a `retrieval.yaml` mapping each golden question's exact query text to
a fixed, ordered list of retrieval hits. This lets the bench CLI's config
loading, threshold gating, exit codes, and trend recording be exercised
end-to-end with zero live embeddings, zero network access, and zero API
keys — the same purpose the repo's test-only `FakeStore` doubles serve for
unit tests, promoted to a real CLI feature because `--bundle` must be
runnable end-to-end for CI-safe validation, not just importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from vaultmind.errors import VaultMindError

if TYPE_CHECKING:
    from pathlib import Path

GOLDEN_FILENAME = "golden.yaml"
RETRIEVAL_FILENAME = "retrieval.yaml"


class BundleError(VaultMindError):
    """Raised when a `--bundle` fixture directory is missing or malformed."""


class FixtureStore:
    """`RetrievalStore` backed by a canned per-query results mapping."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self._results = results_by_query

    def search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._results.get(query, [])[:n_results]

    def hybrid_search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._results.get(query, [])[:n_results]


@dataclass(frozen=True, slots=True)
class Bundle:
    """A loaded `--bundle` fixture: its golden set path + fixture store."""

    golden_path: Path
    store: FixtureStore


def _parse_hit(query: str, raw_hit: Any) -> dict[str, Any]:
    if not isinstance(raw_hit, dict) or "note_path" not in raw_hit:
        msg = f"{RETRIEVAL_FILENAME}['{query}']: each hit must be a mapping with 'note_path'"
        raise BundleError(msg)
    note_path = str(raw_hit["note_path"])
    return {
        "chunk_id": f"{note_path}::0",
        "content": str(raw_hit.get("content", "")),
        "metadata": {"note_path": note_path},
        "distance": float(raw_hit.get("distance", 0.1)),
    }


def load_bundle(bundle_dir: Path) -> Bundle:
    """Load a `--bundle` fixture directory: `golden.yaml` + `retrieval.yaml`.

    Raises:
        BundleError: either file is missing, not valid YAML, or fails
            schema validation.
    """
    golden_path = bundle_dir / GOLDEN_FILENAME
    retrieval_path = bundle_dir / RETRIEVAL_FILENAME

    if not golden_path.exists():
        msg = f"Bundle missing {GOLDEN_FILENAME}: {bundle_dir}"
        raise BundleError(msg)
    if not retrieval_path.exists():
        msg = f"Bundle missing {RETRIEVAL_FILENAME}: {bundle_dir}"
        raise BundleError(msg)

    try:
        raw = yaml.safe_load(retrieval_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{RETRIEVAL_FILENAME} is not valid YAML: {retrieval_path}"
        raise BundleError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"{RETRIEVAL_FILENAME} must be a mapping of query text -> hit list: {retrieval_path}"
        raise BundleError(msg)

    results_by_query: dict[str, list[dict[str, Any]]] = {}
    for query, raw_hits in raw.items():
        if not isinstance(raw_hits, list):
            msg = f"{RETRIEVAL_FILENAME}['{query}'] must be a list of hits"
            raise BundleError(msg)
        results_by_query[str(query)] = [_parse_hit(str(query), h) for h in raw_hits]

    return Bundle(golden_path=golden_path, store=FixtureStore(results_by_query))
