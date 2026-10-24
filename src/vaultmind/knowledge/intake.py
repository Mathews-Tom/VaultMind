"""Provider-free immutable source preservation shared by adapters and manual wiki.

The six-key legacy metadata and approval snapshots are deliberately unchanged.
New captures bind their adjacent versioned record through metadata.origin. An
adapter event ID must come from its authenticated transport, never message text.
"""

from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vaultmind.wiki.workspace import (
    MAX_INPUT_BYTES,
    WikiError,
    WikiWorkspace,
    _canonical,
    _metadata,
    _regular_file,
    _safe_text,
    _scalar,
    _sha,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vaultmind.config import Settings

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_DEPENDENCIES = 32
MAX_CLOSURE_BYTES = 64 * 1024 * 1024
MAX_CLOSURE_DEPTH = 8
MAX_EVENT_SCAN_FILES = 2048
MARKER = "vaultmind-intake:v1:"
_METADATA_KEYS = {"id", "original_name", "origin", "uri", "captured_at", "sha256"}
_RECORD_KEYS = {
    "version",
    "id",
    "sha256",
    "kind",
    "media_type",
    "origin",
    "adapter",
    "role",
    "event_id",
    "revision",
    "upstream_revision",
    "original_retained",
    "parents",
    "attachments",
}


@dataclass(frozen=True)
class StageDisposition:
    """A downstream operation's outcome, not a preservation/approval claim."""

    stage: str
    status: str
    value: Any = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(stage=self.stage, status=self.status, error=self.error)


@dataclass
class CaptureResult:
    source_id: str
    path: str
    sha256: str
    status: str
    preserved: bool = True
    indexed: bool = False
    _intake: SourceIntake | None = field(default=None, repr=False, compare=False)
    approved: bool = False
    dispositions: list[StageDisposition] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(
            id=self.source_id,
            path=self.path,
            sha256=self.sha256,
            status=self.status,
            preserved=self.preserved,
            indexed=self.indexed,
            approved=self.approved,
            dispositions=[item.as_dict() for item in self.dispositions],
        )

    def as_dict_legacy(self) -> dict[str, str]:
        return dict(id=self.source_id, path=self.path, sha256=self.sha256, status=self.status)

    def run_stage(self, stage: str, operation: Callable[[], Any]) -> StageDisposition:
        """Attempt once on this fresh capture; preserve the receipt on failure.

        Replays never repeat downstream effects. Failed operations are not retried
        implicitly, including repeated calls on the same result.
        """
        if stage not in {"projection", "index", "derivative", "provider"}:
            raise WikiError("Unknown capture stage")
        existing = next((item for item in self.dispositions if item.stage == stage), None)
        if existing is not None:
            return existing
        if self._intake is None:
            raise WikiError("Capture receipt has no processing capability")
        disposition = self._intake._run_stage(self, stage, operation)
        if disposition.status == "succeeded" and stage == "index":
            self.indexed = True
        self.dispositions.append(disposition)
        return disposition


@dataclass(frozen=True)
class SourceRecord:
    snapshot: dict[str, Any]
    path: str
    text: str | None
    provenance: dict[str, Any] | None
    dependencies: tuple[SourceRecord, ...] = ()
    content: bytes = b""

    @property
    def source_id(self) -> str:
        return str(self.snapshot["id"])

    def identity(self) -> dict[str, Any]:
        """Keep legacy query fields stable; add explicit new-record retention."""
        result = dict(
            id=self.source_id,
            path=self.path,
            sha256=self.snapshot["sha256"],
            metadata_sha256=self.snapshot["metadata_sha256"],
            uri=self.snapshot["metadata"]["uri"],
        )
        if self.provenance is not None:
            result["provenance"] = self.provenance
            result["retention"] = dict(
                preserved=True,
                original_retained=self.provenance["original_retained"],
                kind=self.provenance["kind"],
            )
            result["retained_dependencies"] = [source.identity() for source in self.dependencies]
        return result


class SourceIntake:
    """Use the existing workspace path, lock and atomic-write capability."""

    def __init__(
        self, settings: Settings, *, allows_path: Callable[[str], bool] | None = None
    ) -> None:
        self.workspace = WikiWorkspace(settings)
        self.allows_path = allows_path

    @classmethod
    def from_workspace(
        cls, workspace: WikiWorkspace, *, allows_path: Callable[[str], bool] | None = None
    ) -> SourceIntake:
        instance = cls.__new__(cls)
        instance.workspace = workspace
        instance.allows_path = allows_path
        return instance

    def _authorize(self, relative: str) -> None:
        if self.allows_path is not None and not self.allows_path(relative):
            raise WikiError("Preserved source dependency is outside the authorized scope")

    def __enter__(self) -> SourceIntake:
        return self

    def __exit__(self, *_: object) -> None:
        self.workspace.close()

    def _binary(self, relative: str, limit: int) -> bytes:
        self._authorize(relative)
        path = self.workspace._path(relative)
        _regular_file(path, "Preserved attachment")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise WikiError("Preserved attachment must be a regular file")
                content = stream.read(limit + 1)
        except OSError:
            raise WikiError("Cannot read preserved source") from None
        if len(content) > limit:
            raise WikiError("Preserved source exceeds its byte limit")
        return content

    def _preflight_write(self, source_id: str, kind: str) -> None:
        for relative in (
            "_wiki",
            "_wiki/wiki.lock",
            f"_wiki/sources/{source_id}.json",
            f"_wiki/sources/{source_id}.intake.json",
            f"_wiki/sources/{source_id}.{'bin' if kind == 'attachment' else 'md'}",
            f"_wiki/sources/{source_id}.processing.json",
        ):
            self._authorize(relative)
            path = self.workspace._path(relative)
            if relative != "_wiki":
                _regular_file(path, "Source output")

    def _run_stage(
        self, capture: CaptureResult, stage: str, operation: Callable[[], Any]
    ) -> StageDisposition:
        relative = f"_wiki/sources/{capture.source_id}.processing.json"
        self._authorize(relative)
        with self.workspace._locked():
            self.read(capture.source_id)
            path = self.workspace._path(relative)
            state: dict[str, Any] = {}
            if path.exists():
                state, _ = self.workspace._load_json(path, "Capture processing disposition")
                if any(
                    key not in {"projection", "index", "derivative", "provider"}
                    or not isinstance(value, str)
                    or value not in {"attempted", "succeeded", "failed"}
                    for key, value in state.items()
                ):
                    raise WikiError("Invalid capture processing disposition")
            if stage in state:
                status = "incomplete" if state[stage] == "attempted" else state[stage]
                return StageDisposition(
                    stage, status, error="Stage already attempted; original preserved"
                )
            if capture.status == "unchanged":
                return StageDisposition(stage, "skipped")
            state[stage] = "attempted"
            self.workspace._json(path, state)
        try:
            value = operation()
        except Exception:
            disposition = StageDisposition(
                stage, "failed", error=f"{stage} failed; original preserved"
            )
        else:
            disposition = StageDisposition(stage, "succeeded", value=value)
        try:
            with self.workspace._locked():
                state, _ = self.workspace._load_json(path, "Capture processing disposition")
                state[stage] = disposition.status
                self.workspace._json(path, state)
        except WikiError:
            return StageDisposition(
                stage, "incomplete", error="Stage outcome not recorded; original preserved"
            )
        return disposition

    def project_text(self, capture: CaptureResult, folder: str) -> StageDisposition:
        """Optionally project exact retained text at a deterministic ordinary path."""
        from vaultmind.vault.security import validate_ordinary_write_path

        relative = f"{folder.rstrip('/')}/{capture.source_id}.md"
        self._authorize(relative)
        target = validate_ordinary_write_path(relative, self.workspace.root)

        def project() -> str:
            source = self.read(capture.source_id)
            if source.text is None:
                raise WikiError("Binary attachments cannot be projected as Markdown")
            validate_ordinary_write_path(relative, self.workspace.root)
            if target.exists():
                raise WikiError("Projection destination already exists")
            self.workspace._atomic_bytes(target, source.text.encode("utf-8"))
            return relative

        return capture.run_stage("projection", project)

    def read(self, source_id: str) -> SourceRecord:
        """Validate the complete retained closure before exposing any evidence."""
        return self._read(source_id, set(), {}, [0], 0)

    def find_event(
        self, adapter: str, event_id: str, revision: str | None, role: str
    ) -> SourceRecord | None:
        """Find complete retained output before repeating an authenticated event.

        This read-only bounded scan intentionally creates no receipt for events
        (such as voice questions) which did not retain a source.
        """
        for value, label in (
            (adapter, "source adapter"),
            (event_id, "source event"),
            (role, "source role"),
        ):
            _scalar(value, label)
        if revision is not None:
            _scalar(revision, "upstream source revision")
        self._authorize("_wiki")
        self._authorize("_wiki/sources")
        directory = self.workspace._path("_wiki/sources")
        if not directory.exists():
            return None
        matches: list[SourceRecord] = []
        total = [0]
        try:
            for count, path in enumerate(directory.iterdir(), 1):
                if count > MAX_EVENT_SCAN_FILES:
                    raise WikiError("Event lookup exceeds its source scan limit")
                if not path.name.endswith(".intake.json"):
                    continue
                source_id = path.name.removesuffix(".intake.json")
                try:
                    source = self._read(source_id, set(), {}, total, 0)
                except WikiError:
                    if total[0] > MAX_CLOSURE_BYTES:
                        raise WikiError(
                            "Event lookup exceeds its retained byte scan limit"
                        ) from None
                    # Incomplete, unknown or unauthorized evidence is not a
                    # completed source and is never exposed by lookup.
                    continue
                record = source.provenance
                if record is not None and (
                    record["adapter"] == adapter
                    and record["event_id"] == event_id
                    and record["upstream_revision"] == revision
                    and record["role"] == role
                ):
                    matches.append(source)
                    if len(matches) > 1:
                        raise WikiError("Authenticated source event has ambiguous retained outputs")
        except OSError:
            raise WikiError("Cannot inspect retained source events") from None
        return matches[0] if matches else None

    def _read(
        self,
        source_id: str,
        active: set[str],
        seen: dict[str, SourceRecord],
        total: list[int],
        depth: int,
    ) -> SourceRecord:
        workspace = self.workspace
        workspace._source_id(source_id)
        if source_id in active or depth > MAX_CLOSURE_DEPTH:
            raise WikiError("Retained source dependency cycle or depth limit")
        if source_id in seen:
            return seen[source_id]
        if len(seen) + len(active) >= MAX_DEPENDENCIES:
            raise WikiError("Retained source dependency count exceeds its limit")
        active.add(source_id)
        metadata_path = f"_wiki/sources/{source_id}.json"
        record_path = f"_wiki/sources/{source_id}.intake.json"
        self._authorize(metadata_path)
        meta, raw_meta = workspace._load_json(workspace._path(metadata_path), "Source metadata")
        if set(meta) != _METADATA_KEYS or meta["id"] != source_id:
            raise WikiError("Source metadata identity is invalid")
        for key in ("original_name", "origin", "captured_at", "sha256"):
            _scalar(meta[key], "source metadata")
        if meta["uri"] is not None:
            _scalar(meta["uri"], "source URI")
        marked = meta["origin"].startswith("vaultmind-intake:")
        self._authorize(record_path)
        record_exists = workspace._path(record_path).exists()
        provenance: dict[str, Any] | None = None
        kind = "text"
        dependencies: list[SourceRecord] = []
        if marked:
            if not meta["origin"].startswith(MARKER) or not record_exists:
                raise WikiError("Marked intake source has missing or unsupported provenance")
            provenance, raw_record = workspace._load_json(
                workspace._path(record_path), "Intake provenance"
            )
            if meta["origin"] != MARKER + _sha(raw_record):
                raise WikiError("Intake provenance hash changed")
            if (
                set(provenance) != _RECORD_KEYS
                or type(provenance["version"]) is not int
                or provenance["version"] != 1
                or provenance["id"] != source_id
                or provenance["sha256"] != meta["sha256"]
                or provenance["kind"] not in ("text", "attachment")
                or type(provenance["original_retained"]) is not bool
            ):
                raise WikiError("Invalid intake provenance identity")
            for key in ("media_type", "origin", "adapter", "role", "revision"):
                _scalar(provenance[key], "intake provenance")
            if provenance["event_id"] is not None:
                _scalar(provenance["event_id"], "intake event")
            if provenance["upstream_revision"] is not None:
                _scalar(provenance["upstream_revision"], "upstream source revision")
            kind = provenance["kind"]
            for relation in ("parents", "attachments"):
                refs = provenance[relation]
                if not isinstance(refs, list) or len(refs) > MAX_DEPENDENCIES:
                    raise WikiError("Invalid retained dependency list")
                ids: set[str] = set()
                for ref in refs:
                    if not isinstance(ref, dict) or set(ref) != {"id", "sha256", "metadata_sha256"}:
                        raise WikiError("Invalid retained dependency identity")
                    dependency = self._read(ref["id"], active, seen, total, depth + 1)
                    if ref["id"] in ids or any(dependency.snapshot[key] != ref[key] for key in ref):
                        raise WikiError("Retained source dependency changed")
                    ids.add(ref["id"])
                    if relation == "attachments" and (
                        dependency.provenance is None
                        or dependency.provenance["kind"] != "attachment"
                    ):
                        raise WikiError("Retained attachment is not an attachment source")
                    dependencies.append(dependency)
        elif record_exists:
            raise WikiError("Intake provenance marker is missing; legacy downgrade refused")
        relative = f"_wiki/sources/{source_id}.{'bin' if kind == 'attachment' else 'md'}"
        content = self._binary(
            relative, MAX_ATTACHMENT_BYTES if kind == "attachment" else MAX_INPUT_BYTES
        )
        total[0] += len(content)
        if total[0] > MAX_CLOSURE_BYTES:
            raise WikiError("Retained source closure exceeds its byte limit")
        if meta["sha256"] != _sha(content):
            raise WikiError("Source bytes or metadata hash changed")
        text = None
        if kind == "text":
            text = _safe_text(content, "Preserved source")
            _metadata(text)
        snapshot = dict(
            id=source_id, sha256=_sha(content), metadata_sha256=_sha(raw_meta), metadata=meta
        )
        result = SourceRecord(snapshot, relative, text, provenance, tuple(dependencies), content)
        active.remove(source_id)
        seen[source_id] = result
        return result

    def add_local(
        self, content: bytes, *, source_id: str, original_name: str, origin: str, uri: str | None
    ) -> dict[str, str]:
        """Legacy local add, including its original bytes, origins and return shape."""
        workspace = self.workspace
        workspace._source_id(source_id)
        _metadata(_safe_text(content, "Source"))
        _scalar(original_name, "source filename", 255)
        _scalar(origin, "source origin")
        if uri is not None:
            _scalar(uri, "source URI")
        self._preflight_write(source_id, "text")
        with workspace._locked():
            return self._write(
                content, source_id, original_name, origin, uri, None
            ).as_dict_legacy()

    def preserve_text(
        self,
        content: str | bytes,
        *,
        original_name: str,
        origin: str,
        adapter: str,
        role: str = "text",
        event_id: str | None = None,
        revision: str | None = None,
        uri: str | None = None,
        parents: Sequence[str] = (),
        attachments: Sequence[str] = (),
        original_retained: bool = True,
        preflight: Callable[[str], None] | None = None,
    ) -> CaptureResult:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        _metadata(_safe_text(raw, "Source"))
        return self._preserve(
            raw,
            original_name,
            origin,
            adapter,
            role,
            event_id,
            revision,
            uri,
            parents,
            attachments,
            original_retained,
            "text",
            "text/markdown",
            preflight,
        )

    def preserve_attachment(
        self,
        content: bytes,
        *,
        original_name: str,
        origin: str,
        adapter: str,
        role: str = "attachment",
        event_id: str | None = None,
        revision: str | None = None,
        uri: str | None = None,
        parents: Sequence[str] = (),
        attachments: Sequence[str] = (),
        original_retained: bool = True,
        media_type: str = "application/octet-stream",
        preflight: Callable[[str], None] | None = None,
    ) -> CaptureResult:
        if not isinstance(content, bytes) or len(content) > MAX_ATTACHMENT_BYTES:
            raise WikiError("Attachment exceeds its byte limit or is not bytes")
        if not original_retained:
            raise WikiError("A preserved attachment must retain its original bytes")
        return self._preserve(
            content,
            original_name,
            origin,
            adapter,
            role,
            event_id,
            revision,
            uri,
            parents,
            attachments,
            original_retained,
            "attachment",
            media_type,
            preflight,
        )

    def _preserve(
        self,
        content: bytes,
        original_name: str,
        origin: str,
        adapter: str,
        role: str,
        event_id: str | None,
        revision: str | None,
        uri: str | None,
        parents: Sequence[str],
        attachments: Sequence[str],
        original_retained: bool,
        kind: str,
        media_type: str,
        preflight: Callable[[str], None] | None,
    ) -> CaptureResult:
        for value, label in (
            (original_name, "source filename"),
            (origin, "source origin"),
            (adapter, "source adapter"),
            (role, "source role"),
            (media_type, "media type"),
        ):
            _scalar(value, label, 255 if label == "source filename" else 2048)
        if event_id is not None:
            _scalar(event_id, "authenticated source event")
        if revision is not None:
            _scalar(revision, "source revision")
        if uri is not None:
            _scalar(uri, "source URI")
        if type(original_retained) is not bool:
            raise WikiError("Original retention must be explicit boolean")
        if len(parents) + len(attachments) > MAX_DEPENDENCIES - 1:
            raise WikiError("Too many retained source dependencies")
        revision_hash = _sha(_canonical(dict(content_sha256=_sha(content), revision=revision)))
        identity = dict(
            adapter=adapter,
            vault=str(self.workspace.root.resolve()),
            event_id=event_id,
            revision=revision_hash,
            role=role,
        )
        source_id = (
            "i-" + _sha(_canonical(identity) if event_id is not None else uuid.uuid4().bytes)[:60]
        )
        self._preflight_write(source_id, kind)
        if preflight is not None:
            preflight(source_id)
        self._dependencies(source_id, content, parents, attachments)
        with self.workspace._locked():
            dependencies = self._dependencies(source_id, content, parents, attachments)
            record = dict(
                version=1,
                id=source_id,
                sha256=_sha(content),
                kind=kind,
                media_type=media_type,
                origin=origin,
                adapter=adapter,
                role=role,
                event_id=event_id,
                revision=revision_hash,
                upstream_revision=revision,
                original_retained=original_retained,
                **dependencies,
            )
            return self._write(content, source_id, original_name, origin, uri, record)

    def _dependencies(
        self, source_id: str, content: bytes, parents: Sequence[str], attachments: Sequence[str]
    ) -> dict[str, list[dict[str, str]]]:
        dependencies: dict[str, list[dict[str, str]]] = {}
        seen: dict[str, SourceRecord] = {}
        total = [len(content)]
        for relation, ids in (("parents", parents), ("attachments", attachments)):
            if len(set(ids)) != len(ids):
                raise WikiError("Duplicate retained dependency")
            refs: list[dict[str, str]] = []
            for dependency_id in ids:
                source = self._read(dependency_id, {source_id}, seen, total, 1)
                if relation == "attachments" and (
                    source.provenance is None or source.provenance["kind"] != "attachment"
                ):
                    raise WikiError("Retained attachment is not an attachment source")
                refs.append(
                    {key: source.snapshot[key] for key in ("id", "sha256", "metadata_sha256")}
                )
            dependencies[relation] = refs
        return dependencies

    def _write(
        self,
        content: bytes,
        source_id: str,
        original_name: str,
        origin: str,
        uri: str | None,
        record: dict[str, Any] | None,
    ) -> CaptureResult:
        workspace = self.workspace
        kind = "text" if record is None else record["kind"]
        relative = f"_wiki/sources/{source_id}.{'bin' if kind == 'attachment' else 'md'}"
        metadata_relative = f"_wiki/sources/{source_id}.json"
        record_relative = f"_wiki/sources/{source_id}.intake.json"
        for path in (relative, metadata_relative, record_relative):
            self._authorize(path)
        source = workspace._path(relative)
        meta_path = workspace._path(metadata_relative)
        record_path = workspace._path(record_relative)
        if source.exists() or meta_path.exists() or record_path.exists():
            existing = self.read(source_id)
            if existing.snapshot["sha256"] != _sha(content):
                raise WikiError("Source ID already contains different bytes; choose a new ID")
            if record is not None and (
                existing.provenance != record
                or existing.snapshot["metadata"]["original_name"] != original_name
                or existing.snapshot["metadata"]["uri"] != uri
            ):
                raise WikiError("Source event identity has different provenance")
            return CaptureResult(source_id, existing.path, _sha(content), "unchanged", _intake=self)
        raw_record = None if record is None else _canonical(record) + b"\n"
        metadata = dict(
            id=source_id,
            original_name=original_name,
            origin=origin if raw_record is None else MARKER + _sha(raw_record),
            uri=uri,
            captured_at=datetime.now(UTC).isoformat(),
            sha256=_sha(content),
        )
        # A metadata file is the completion point. Orphans fail validation and
        # are never automatically repaired or made discoverable by a replay.
        workspace._atomic_bytes(source, content)
        if raw_record is not None:
            workspace._atomic_bytes(record_path, raw_record)
        workspace._json(meta_path, metadata)
        return CaptureResult(source_id, relative, _sha(content), "added", _intake=self)
