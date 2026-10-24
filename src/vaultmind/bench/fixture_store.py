"""Offline source-retrieval bundles with real, bundle-local current files.

`golden.yaml` defines questions; `retrieval.yaml` fixes candidate order and raw
distances. The `vault/` tree supplies actual ordinary notes and optional retained
source snapshots. Loading snapshots indexed chunks once; query-time discovery
uses the same current-file hydration as every other consumer, never YAML excerpts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from vaultmind.config import VaultConfig
from vaultmind.errors import VaultMindError
from vaultmind.vault.parser import VaultParser
from vaultmind.vault.security import validate_ordinary_write_path

if TYPE_CHECKING:
    from pathlib import Path


GOLDEN_FILENAME = "golden.yaml"
RETRIEVAL_FILENAME = "retrieval.yaml"
VAULT_DIRECTORY = "vault"


class BundleError(VaultMindError):
    """Raised when a `--bundle` fixture directory is missing or malformed."""


class FixtureStore:
    """A deterministic candidate backend, filtering authorized paths before top-k."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self._results = results_by_query

    def search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [
            hit
            for hit in self._results.get(query, [])
            if _matches_where(hit.get("metadata", {}), where)
        ][: max(0, n_results)]

    def hybrid_search(
        self, query: str, n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.search(query, n_results=n_results, where=where)


@dataclass(frozen=True, slots=True)
class Bundle:
    """A golden set and indexed snapshots bound to an actual synthetic vault."""

    golden_path: Path
    store: FixtureStore
    vault_path: Path


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for field, condition in where.items():
        if field in ("$and", "$or"):
            if not isinstance(condition, list) or not condition:
                raise ValueError("Invalid fixture metadata filter")
            matches = [_matches_where(metadata, item) for item in condition]
            if not (all(matches) if field == "$and" else any(matches)):
                return False
        elif isinstance(condition, dict):
            for operator, value in condition.items():
                actual = metadata.get(field)
                if operator == "$in":
                    matched = actual in value
                elif operator == "$nin":
                    matched = actual not in value
                elif operator == "$eq":
                    matched = actual == value
                elif operator == "$ne":
                    matched = actual != value
                else:
                    raise ValueError("Unsupported fixture metadata filter")
                if not matched:
                    return False
        elif metadata.get(field) != condition:
            return False
    return True


def _parse_hit(query: str, raw_hit: Any, parser: VaultParser) -> dict[str, Any]:
    if not isinstance(raw_hit, dict) or not isinstance(raw_hit.get("note_path"), str):
        raise BundleError(f"{RETRIEVAL_FILENAME}['{query}']: each hit needs a string 'note_path'")
    if set(raw_hit) - {"note_path", "chunk_index", "distance", "rrf_score"}:
        raise BundleError(
            "Fixture hit content and metadata must come from bundle-local vault files"
        )
    chunk_index = raw_hit.get("chunk_index", 0)
    if type(chunk_index) is not int or chunk_index < 0:
        raise BundleError("Fixture chunk_index must be a nonnegative integer")
    try:
        requested = parser.vault_root / raw_hit["note_path"]
        path = validate_ordinary_write_path(raw_hit["note_path"], parser.vault_root)
        for parent in (requested, *requested.parents):
            if parent == parser.vault_root:
                break
            if parent.is_symlink():
                raise ValueError
        note = parser.parse_file(path)
        chunk = parser.chunk_note(note)[chunk_index]
    except (OSError, ValueError, IndexError, TypeError, yaml.YAMLError):
        raise BundleError(
            "Fixture candidate needs a readable ordinary current note and chunk in vault/"
        ) from None
    hit: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": chunk.to_chroma_metadata(),
        "distance": None,
    }
    for field in ("distance", "rrf_score"):
        value = raw_hit.get(field)
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise BundleError(f"Fixture {field} must be a finite number or null")
            hit[field] = float(value)
    return hit


def load_bundle(bundle_dir: Path) -> Bundle:
    """Load candidate snapshots from a real synthetic `vault/`, never the operator vault.

    Missing files, unsafe paths, legacy inline excerpts and malformed candidates
    fail closed. Subsequent queries independently re-read current source bytes.
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
    vault_path = bundle_dir / VAULT_DIRECTORY
    if vault_path.is_symlink() or not vault_path.is_dir():
        raise BundleError("Bundle needs an actual bundle-local vault/ directory")
    vault_path = vault_path.resolve()
    if not vault_path.is_relative_to(bundle_dir.resolve()):
        raise BundleError("Bundle vault/ must remain inside the bundle")
    parser = VaultParser(VaultConfig(path=vault_path))

    if not isinstance(raw, dict):
        msg = f"{RETRIEVAL_FILENAME} must be a mapping of query text -> hit list: {retrieval_path}"
        raise BundleError(msg)

    results_by_query: dict[str, list[dict[str, Any]]] = {}
    for query, raw_hits in raw.items():
        if not isinstance(raw_hits, list):
            msg = f"{RETRIEVAL_FILENAME}['{query}'] must be a list of hits"
            raise BundleError(msg)
        results_by_query[str(query)] = [_parse_hit(str(query), hit, parser) for hit in raw_hits]

    return Bundle(
        golden_path=golden_path, store=FixtureStore(results_by_query), vault_path=vault_path
    )
