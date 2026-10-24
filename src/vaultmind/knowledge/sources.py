"""Read-only, scope-first discovery of current ordinary and retained sources.

The index is an optional ranking aid, never the source of displayed evidence.
These results are explicitly not approved wiki answers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.indexer.ranking import apply_authority
from vaultmind.knowledge.intake import SourceIntake, SourceRecord
from vaultmind.vault.models import Note, NoteMode
from vaultmind.vault.parser import TAG_PATTERN, VaultParser
from vaultmind.vault.security import validate_ordinary_write_path
from vaultmind.wiki.workspace import WikiError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from vaultmind.config import Settings

MAX_SOURCE_CANDIDATES = 512
MAX_INDEX_CANDIDATES = 4096
MAX_SOURCE_BYTES = 1024 * 1024
MAX_SEARCH_BYTES = 64 * 1024 * 1024
_SOURCE_METADATA = re.compile(r"([A-Za-z0-9][A-Za-z0-9_-]{0,63})\.json\Z")


class SourceSearchError(ValueError):
    """A body/path-redacted source-discovery failure."""


def _too_broad() -> SourceSearchError:
    return SourceSearchError(
        "Source search scope is too broad; narrow the authorized folders or query scope"
    )


def _allows(allows_path: Callable[[str], bool] | None, path: str) -> bool:
    return allows_path is None or allows_path(path)


def _ordinary_paths(
    settings: Settings,
    allows_path: Callable[[str], bool] | None,
) -> Iterator[tuple[str, Path]]:
    root = settings.vault.path.resolve()
    excluded = {name.casefold() for name in settings.vault.excluded_folders} | {"wiki", "_wiki"}

    def walk(directory: Path) -> Iterator[tuple[str, Path]]:
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
            for entry in ordered:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                # Do not stat, resolve or recurse into a forbidden lexical path.
                if any(part.casefold() in excluded for part in Path(relative).parts):
                    continue
                if not _allows(allows_path, relative):
                    continue
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(path)
                elif entry.name.lower().endswith(".md") and entry.is_file(follow_symlinks=False):
                    try:
                        checked = validate_ordinary_write_path(relative, root)
                    except ValueError:
                        continue
                    resolved_relative = checked.relative_to(root).as_posix()
                    if any(
                        part.casefold() in excluded for part in Path(resolved_relative).parts
                    ) or not _allows(allows_path, resolved_relative):
                        continue
                    yield relative, checked
        except OSError:
            raise SourceSearchError("Cannot enumerate authorized source scope") from None

    yield from walk(root)


def _read_current(path: Path) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SourceSearchError("Current source must be a regular file")
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    except OSError:
        raise SourceSearchError(
            "Current source changed or is missing; rerun source search"
        ) from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise _too_broad()
    return raw


def _authority(value: Any) -> int:
    try:
        authority = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return authority if 1 <= authority <= 5 else 0


def _ordinary_hit(
    settings: Settings,
    relative: str,
    raw: bytes,
    *,
    build_chunks: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        post = frontmatter.loads(raw.decode("utf-8"))
        meta = post.metadata
        parser = VaultParser(settings.vault)
        note_type = parser._infer_type(meta, Path(relative))
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags = list(set(tags + TAG_PATTERN.findall(post.content)))
        try:
            mode = NoteMode(meta.get("mode", "learning"))
        except ValueError:
            mode = NoteMode.LEARNING
        authority = _authority(meta.get("authority", 0))
        note = Note(
            path=Path(relative),
            title=meta.get("title", Path(relative).stem.replace("-", " ").replace("_", " ")),
            content=post.content,
            note_type=note_type,
            mode=mode,
            tags=tags,
            entities=meta.get("entities", []),
            status=meta.get("status", "active"),
            frontmatter=meta,
            authority=authority,
        )
        chunks = (
            {chunk.chunk_id: chunk.content for chunk in parser.chunk_note(note)}
            if build_chunks
            else {}
        )
    except Exception:
        raise SourceSearchError("Cannot parse current authorized source") from None
    hit = _hit(relative, note.title, post.content, hashlib.sha256(raw).hexdigest(), "ordinary")
    hit["metadata"].update(note_type=note_type.value, authority=authority)
    return hit, chunks


def _hit(path: str, title: str, text: str, digest: str, kind: str) -> dict[str, Any]:
    return dict(
        chunk_id=f"source:{path}",
        path=path,
        source_kind=kind,
        status="source-only",
        approved=False,
        current_digest=digest,
        content=text,
        distance=None,
        metadata=dict(note_path=path, note_title=title),
    )


def _preserved_hit(record: SourceRecord) -> dict[str, Any]:
    # Binary originals are discoverable by their retained filename, never by
    # decoding arbitrary bytes or fabricating a textual excerpt.
    hit = _hit(
        record.path,
        record.snapshot["metadata"]["original_name"],
        record.text or "",
        record.snapshot["sha256"],
        "preserved",
    )
    hit.update(
        source_id=record.source_id, provenance=record.provenance, source_snapshot=record.snapshot
    )
    hit["metadata"]["note_type"] = "source"
    if record.text is not None:
        hit["metadata"]["authority"] = _authority(
            frontmatter.loads(record.text).metadata.get("authority", 0)
        )
    return hit


def _preserved_records(
    settings: Settings,
    allows_path: Callable[[str], bool] | None,
) -> Iterator[SourceRecord]:
    if not _allows(allows_path, "_wiki") or not _allows(allows_path, "_wiki/sources"):
        return
    with SourceIntake(settings, allows_path=allows_path) as intake:
        try:
            directory = intake.workspace._path("_wiki/sources")
            names = sorted(path.name for path in directory.iterdir()) if directory.exists() else []
        except (OSError, WikiError):
            raise SourceSearchError("Cannot enumerate preserved source scope") from None
        for name in names:
            match = _SOURCE_METADATA.fullmatch(name)
            if match is None or not _allows(allows_path, f"_wiki/sources/{name}"):
                continue
            try:
                # This validates and authorizes every retained dependency, not
                # just the root text, before returning any searchable evidence.
                record = intake.read(match[1])
            except (WikiError, OSError):
                continue
            yield record


def _lexical_score(query: str, hit: dict[str, Any]) -> float:
    terms = set(re.findall(r"\w+", query.casefold()))
    if not terms:
        return 0.0
    title = set(re.findall(r"\w+", str(hit["metadata"]["note_title"]).casefold()))
    body = set(re.findall(r"\w+", str(hit["content"]).casefold()))
    return float(len(terms & body) + 2 * len(terms & title)) / len(terms)


def search_sources(
    settings: Settings,
    query: str,
    *,
    store: Any = None,
    n_results: int = 5,
    note_type: str | None = None,
    allows_path: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Discover complete authorized current sources without writing evidence.

    All candidates are enumerated before ranking/top-k. The bounded complete
    index request is scoped to those paths in both real backends; exceeding a
    bound refuses the query rather than silently dropping authorized evidence.
    """
    if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results < 1:
        raise SourceSearchError("Source search result limit must be a positive integer")
    if not query.strip():
        return []
    candidates: dict[str, dict[str, Any]] = {}
    current_chunks: dict[str, str] = {}
    total_bytes = 0

    def add(hit: dict[str, Any], size: int) -> None:
        nonlocal total_bytes
        total_bytes += size
        if len(candidates) >= MAX_SOURCE_CANDIDATES or total_bytes > MAX_SEARCH_BYTES:
            raise _too_broad()
        candidates[hit["path"]] = hit

    for relative, file_path in _ordinary_paths(settings, allows_path):
        raw = _read_current(file_path)
        hit, chunks = _ordinary_hit(settings, relative, raw, build_chunks=store is not None)
        if note_type is None or hit["metadata"]["note_type"] == note_type:
            add(hit, len(raw))
            if len(current_chunks) + len(chunks) > MAX_INDEX_CANDIDATES:
                raise _too_broad()
            current_chunks.update(chunks)
    for record in _preserved_records(settings, allows_path):
        hit = _preserved_hit(record)
        if note_type is None or note_type == "source":
            add(hit, len(record.text.encode("utf-8")) if record.text is not None else 0)

    lexical = {path: _lexical_score(query, hit) for path, hit in candidates.items()}
    ranked_paths: list[str] = []
    ordinary = sorted(path for path, hit in candidates.items() if hit["source_kind"] == "ordinary")
    if store is not None and ordinary:
        where = {"note_path": {"$in": ordinary}}
        operation = store.hybrid_search if settings.search.hybrid_enabled else store.search
        try:
            indexed = operation(query, n_results=MAX_INDEX_CANDIDATES + 1, where=where)
        except Exception:
            raise SourceSearchError(
                "Scoped source ranking failed; narrow the authorized folders or rerun source search"
            ) from None
        if len(indexed) > MAX_INDEX_CANDIDATES:
            raise _too_broad()
        for indexed_hit in indexed:
            metadata = indexed_hit.get("metadata") or {}
            path = metadata.get("note_path")
            if not isinstance(path, str) or path not in candidates or path not in ordinary:
                continue
            chunk_id = indexed_hit.get("chunk_id")
            # A changed/missing chunk may suggest a path, but cannot lend its
            # old vector distance to freshly hydrated source content.
            distance = indexed_hit.get("distance")
            valid_chunk = (
                isinstance(chunk_id, str)
                and chunk_id.startswith(f"{path}::")
                and chunk_id in current_chunks
                and indexed_hit.get("content") == current_chunks[chunk_id]
            )
            if (
                valid_chunk
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and math.isfinite(distance)
            ):
                previous = candidates[path]["distance"]
                candidates[path]["distance"] = (
                    float(distance) if previous is None else min(previous, float(distance))
                )
            if path not in ranked_paths:
                ranked_paths.append(path)

    # Fuse document ranks, not confidence. Lexical-only evidence remains unknown.
    lexical_paths = sorted(
        (path for path in candidates if lexical[path] > 0), key=lambda path: (-lexical[path], path)
    )
    scores: dict[str, float] = {}
    for ranking in (ranked_paths, lexical_paths):
        for rank, path in enumerate(ranking, 1):
            scores[path] = scores.get(path, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=lambda path: (-scores[path], path))
    for path in ordered:
        candidates[path]["rrf_score"] = scores[path]
    ranked = apply_authority([candidates[path] for path in ordered], settings.ranking)
    return ranked[:n_results]


def revalidate_source_hit(
    settings: Settings,
    hit: dict[str, Any],
    *,
    allows_path: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Refuse stale paginated snapshots; never silently replace their evidence."""
    try:
        relative = hit["path"]
        if not isinstance(relative, str) or not _allows(allows_path, relative):
            raise ValueError
        if hit["source_kind"] == "preserved":
            if not _allows(allows_path, "_wiki") or not _allows(allows_path, "_wiki/sources"):
                raise ValueError
            with SourceIntake(settings, allows_path=allows_path) as intake:
                record = intake.read(hit["source_id"])
            current = _preserved_hit(record)
            if current["source_snapshot"] != hit["source_snapshot"]:
                raise ValueError
        elif hit["source_kind"] == "ordinary":
            path = validate_ordinary_write_path(relative, settings.vault.path)
            root = settings.vault.path.resolve()
            resolved = path.relative_to(root).as_posix()
            excluded = {name.casefold() for name in settings.vault.excluded_folders}
            if any(
                part.casefold() in excluded for part in Path(relative).parts + Path(resolved).parts
            ):
                raise ValueError
            if not _allows(allows_path, resolved):
                raise ValueError
            for parent in (root / relative, *(root / relative).parents):
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError
            current, _ = _ordinary_hit(settings, relative, _read_current(path))
        else:
            raise ValueError
        if current["path"] != relative or current["current_digest"] != hit["current_digest"]:
            raise ValueError
    except Exception:
        raise SourceSearchError(
            "Source search snapshot changed or is unavailable; rerun source search"
        ) from None
    result = dict(hit)
    result.update(current)
    result["distance"] = hit.get("distance")
    return result
