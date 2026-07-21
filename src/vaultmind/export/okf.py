"""Export permanent/concept notes as a conformant OKF bundle for Kosha.

VaultMind stays the messy capture layer; the OKF bundle exported here is the
curated subset Kosha can independently validate, govern, and serve over its
own traversal MCP. Only ``permanent``/``concept`` notes are exported — every
other note type stays VaultMind-only.

The bundle format follows Kosha's OKF v0.1 conventions (`kosha validate`,
`docs/authoring-bundles.md` in the Kosha repo): a directory of concept
Markdown documents (YAML frontmatter + body), a reserved ``index.md`` per
directory (a bare bullet-link directory map; the bundle root's carries only
``okf_version``), and a reserved ``log.md`` with `## YYYY-MM-DD` headings.

Obsidian ``[[wikilinks]]`` are not OKF-conformant (Kosha's own writer rejects
them). Every wikilink whose target resolves to another exported note is
rewritten as a standard bundle-relative link (`[text](/path.md)`); every
other wikilink is flattened to plain display text — the exported body never
contains literal `[[...]]` syntax.

This module is a pure read path: it never mutates the source vault.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from vaultmind.vault.models import NoteType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vaultmind.vault.models import Note

# Note types exported into the OKF bundle — VaultMind's curated tier.
EXPORTED_NOTE_TYPES = frozenset({NoteType.PERMANENT, NoteType.CONCEPT})

# ``[[Title]]`` or ``[[Title|Alias]]`` Obsidian wikilink.
_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@dataclass(frozen=True)
class OkfExportResult:
    """Outcome of one :func:`export_okf_bundle` run."""

    output_dir: Path
    concept_count: int
    concept_paths: tuple[Path, ...]


def export_okf_bundle(
    notes: Iterable[Note],
    output_dir: Path,
    *,
    okf_version: str = "0.1",
) -> OkfExportResult:
    """Export ``permanent``/``concept`` notes from ``notes`` as an OKF bundle.

    Writes concept documents under ``output_dir`` mirroring each note's
    existing vault-relative folder structure, a directory-map ``index.md``
    per level (including the bundle root), and a root ``log.md`` recording
    this export run. ``output_dir`` is created if absent; a re-run overwrites
    the concept/index/log files it produces, in place.
    """
    exported = sorted(
        (note for note in notes if note.note_type in EXPORTED_NOTE_TYPES),
        key=lambda note: note.path.as_posix(),
    )
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    title_to_path = {note.title: note.path for note in exported}

    directories: set[Path] = {Path()}
    for note in exported:
        directories.update(note.path.parents)
        concept_file = output_dir / note.path
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(_render_concept(note, title_to_path), encoding="utf-8")

    for directory in directories:
        _write_index(output_dir, directory, exported, directories, okf_version=okf_version)

    _write_log(output_dir, len(exported))

    return OkfExportResult(
        output_dir=output_dir,
        concept_count=len(exported),
        concept_paths=tuple(note.path for note in exported),
    )


def _render_concept(note: Note, title_to_path: dict[str, Path]) -> str:
    """Render one note as an OKF concept document (frontmatter block + body)."""
    fm: dict[str, object] = {"type": note.note_type.value}
    if note.title:
        fm["title"] = note.title
    description = note.frontmatter.get("description")
    if isinstance(description, str) and description.strip():
        fm["description"] = description
    if note.tags:
        fm["tags"] = sorted(note.tags)
    fm["timestamp"] = note.modified.isoformat()
    # Provenance metadata (M2): 1-5 stamped authority, 0 = unstamped/neutral.
    fm["authority"] = note.authority

    frontmatter_yaml = yaml.safe_dump(
        fm, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    body = _rewrite_links(note, title_to_path)
    return f"---\n{frontmatter_yaml}---\n\n{body}\n"


def _rewrite_links(note: Note, title_to_path: dict[str, Path]) -> str:
    """Rewrite ``[[wikilinks]]`` to standard bundle-relative links, or flatten them."""

    def _replace(match: re.Match[str]) -> str:
        target_title = match.group(1).strip()
        display = (match.group(2) or match.group(1)).strip()
        target_path = title_to_path.get(target_title)
        if target_path is None:
            # Not part of the exported bundle — no valid in-bundle target to
            # link to; keep the visible text, drop the wikilink syntax.
            return display
        return f"[{display}]({_bundle_link(target_path)})"

    return _WIKILINK_PATTERN.sub(_replace, note.content)


def _write_index(
    output_dir: Path,
    directory: Path,
    exported: list[Note],
    all_directories: set[Path],
    *,
    okf_version: str,
) -> None:
    """Write one directory's ``index.md`` — a bare bullet-link map of its direct contents."""
    child_notes = sorted(
        (note for note in exported if note.path.parent == directory),
        key=lambda note: note.path.name,
    )
    child_dirs = sorted(
        (d for d in all_directories if d != directory and d.parent == directory),
        key=lambda d: d.as_posix(),
    )

    lines: list[str] = []
    is_root = directory == Path()
    if is_root:
        lines.extend(["---", f"okf_version: '{okf_version}'", "---", ""])

    heading = output_dir.name if is_root else directory.name
    lines.append(f"# {heading}")
    lines.append("")
    for child_dir in child_dirs:
        lines.append(f"* [{child_dir.name}]({_bundle_link(child_dir / 'index.md')})")
    for note in child_notes:
        description = note.frontmatter.get("description")
        suffix = f" - {description}" if isinstance(description, str) and description.strip() else ""
        lines.append(f"* [{note.title}]({_bundle_link(note.path)}){suffix}")

    index_path = output_dir / directory / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_log(output_dir: Path, concept_count: int) -> None:
    """Write the bundle root ``log.md`` — one dated entry for this export run."""
    today = datetime.now(UTC).date().isoformat()
    noun = "concept" if concept_count == 1 else "concepts"
    content = (
        "# Update Log\n\n"
        f"## {today}\n"
        f"* **Export**: {concept_count} {noun} exported from VaultMind.\n"
    )
    (output_dir / "log.md").write_text(content, encoding="utf-8")


def _bundle_link(rel_path: Path) -> str:
    """Render a bundle-relative link target: leading ``/`` + POSIX path."""
    return "/" + rel_path.as_posix()
