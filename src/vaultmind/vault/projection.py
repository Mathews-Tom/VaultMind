"""Current-content projection for notes carrying bot supersede history.

A bot `/delete` or `/edit` never removes text from a note (M9). The complete
prior body stays on disk beneath a dated `> [!superseded]` callout, so the
raw note body is a *history*, not the current claim. Retrieval must only see
the current claim, so one projection decides which part of the body is
current evidence. `Note.current_content()` is the single consumer-facing
entry point; this module holds the grammar so the writer
(`contradiction/marking.py`) and the reader can never drift apart.

Two generated grammars are recognized, both anchored to bot metadata so an
ordinary authored `[!superseded]` callout is never a retirement instruction:

New (explicit boundary). The writer appends a random-id marker line directly
above the generated callout and pairs it with frontmatter::

    superseded_at: [..., <value>]
    supersede_boundaries: ["<32 hex id>=<value>", ...]

    <current text>

    <!-- vaultmind:supersede id=<32 hex id> -->
    > [!superseded] Edited via bot — YYYY-MM-DD
    > Instruction: ...

    <preserved prior body>

The marker only counts when its id is paired in `supersede_boundaries`;
unpaired marker-looking lines are ordinary text. Every paired record must
match exactly one marker, in newest-first order, or the history is ambiguous.
Arbitrary current text (including callout lookalikes) is therefore safe: the
random 128-bit id keeps authored text from colliding with a boundary by
accident. It is a disambiguator, not authentication — anyone who can edit the
note can forge or remove history, and no integrity beyond reconciliation of
the markers with the metadata is claimed.

Legacy (already generated before explicit boundaries). Recognized only when
the exact generated shape matches the metadata one-to-one:

* delete: `superseded_at` holds `bot:delete` and the body holds exactly one
  `> [!superseded] Deleted via bot — YYYY-MM-DD` callout followed by the
  generated rationale line.
* edit: every generated UTC ISO stamp in `superseded_at` pairs, newest first
  from the top of the body, with one `> [!superseded] Edited via bot —
  <stamp date>` callout whose next line starts `> Instruction: `. The current
  text is everything above the topmost callout.

Anything that does not reconcile (missing, duplicated or extra callouts,
date/order mismatch, malformed or inconsistent records, or `superseded_at`
holding a container or non-string/non-date value) raises
`AmbiguousHistoryError`; nothing is guessed and nothing is rewritten. Plain
authored strings and dates in `superseded_at` are foreign metadata and are
never given bot meaning.

A bot delete is terminal: the projection is empty regardless of later edits,
but every explicit boundary record is still validated first.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

SUPERSEDED_KEY = "superseded_at"
BOUNDARY_KEY = "supersede_boundaries"
SUPERSEDED_CALLOUT = "superseded"

DELETE_MARKER = "bot:delete"
DELETE_TITLE = "Deleted via bot"
DELETE_RATIONALE = "Requested via Telegram `/delete` — content preserved below."
EDIT_TITLE = "Edited via bot"
EDIT_RATIONALE_PREFIX = "Instruction: "

_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?\+00:00")
_RECORD = re.compile(r"([0-9a-f]{32})=(.+)")
_MARKER = re.compile(r"^<!-- vaultmind:supersede id=([0-9a-f]{32}) -->[ \t]*\r?$", re.MULTILINE)
_LEGACY_EDIT = re.compile(
    rf"^> \[!{SUPERSEDED_CALLOUT}\] {EDIT_TITLE} — (\d{{4}}-\d{{2}}-\d{{2}})\r?\n"
    rf"> {re.escape(EDIT_RATIONALE_PREFIX)}",
    re.MULTILINE,
)
_LEGACY_DELETE = re.compile(
    rf"^> \[!{SUPERSEDED_CALLOUT}\] {DELETE_TITLE} — \d{{4}}-\d{{2}}-\d{{2}}\r?\n"
    rf"> {re.escape(DELETE_RATIONALE)}[ \t]*\r?$",
    re.MULTILINE,
)


class AmbiguousHistoryError(ValueError):
    """Bot supersede metadata/callouts do not reconcile; the current text is unknowable."""


class RetiredNoteError(ValueError):
    """The operation is not defined for a note retired by a bot delete."""


@dataclass(frozen=True, slots=True)
class Projection:
    """The current evidence of a note body."""

    current: str
    deleted: bool


def is_generated_stamp(value: object) -> bool:
    """True when `value` has the exact shape of a bot edit stamp (UTC ISO)."""
    return _generated_stamp(value) is not None


def new_boundary_id() -> str:
    """Fresh unguessable boundary id (128 random bits, lowercase hex)."""
    return secrets.token_hex(16)


def marker_line(boundary_id: str) -> str:
    """The body line that opens a generated supersede block."""
    return f"<!-- vaultmind:supersede id={boundary_id} -->"


def boundary_record(boundary_id: str, value: str) -> str:
    """The `supersede_boundaries` entry pairing `boundary_id` with a `superseded_at` value."""
    return f"{boundary_id}={value}"


def project(metadata: dict[str, Any], body: str) -> Projection:
    """Project `body` (frontmatter already removed) to its current evidence.

    Notes without bot supersede metadata project to the body unchanged.
    Raises `AmbiguousHistoryError` when the bot metadata and body disagree.
    """
    values = _bot_values(metadata)
    records = _records(metadata)

    recorded = [value for _, value in records]
    if recorded and values[max(len(values) - len(recorded), 0) :] != recorded:
        raise AmbiguousHistoryError(
            f"{BOUNDARY_KEY} does not match the trailing {SUPERSEDED_KEY} entries"
        )

    positions: dict[str, list[int]] = {}
    for match in _MARKER.finditer(body):
        positions.setdefault(match.group(1), []).append(match.start())

    # Every explicit boundary must reconcile before any outcome (including a delete) is trusted.
    offsets = [_single_marker(positions, rid) for rid, _ in records]
    if any(newer >= older for older, newer in zip(offsets, offsets[1:], strict=False)):
        raise AmbiguousHistoryError("supersede boundaries are out of newest-first order")

    if DELETE_MARKER in recorded:
        return Projection(current="", deleted=True)
    if DELETE_MARKER in values:
        if len(_LEGACY_DELETE.findall(body)) != 1:
            raise AmbiguousHistoryError(
                f"{SUPERSEDED_KEY} marks the note deleted but the body does not contain "
                "exactly one generated delete callout"
            )
        return Projection(current="", deleted=True)

    if records:
        return Projection(current=body[: offsets[-1]].strip(), deleted=False)

    # No delete and no explicit boundary remain: `values` holds only legacy edit stamps.
    if not values:
        return Projection(current=body, deleted=False)

    callouts = list(_LEGACY_EDIT.finditer(body))
    if len(callouts) != len(values):
        raise AmbiguousHistoryError(
            f"{len(values)} bot edit stamp(s) in {SUPERSEDED_KEY} but "
            f"{len(callouts)} generated edit callout(s) in the body"
        )
    for callout, stamp in zip(callouts, reversed(values), strict=True):
        if callout.group(1) != stamp[:10]:
            raise AmbiguousHistoryError(
                f"generated edit callout dated {callout.group(1)} does not match stamp {stamp}"
            )
    return Projection(current=body[: callouts[0].start()].strip(), deleted=False)


def _generated_stamp(value: object) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            return None
        text = value.isoformat()
    elif isinstance(value, str):
        text = value
    else:
        return None
    if _STAMP.fullmatch(text) is None:
        return None
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return None
    return text


def _bot_values(metadata: dict[str, Any]) -> list[str]:
    """Bot-generated `superseded_at` entries in chronological order.

    Strings and dates that are not exactly a bot marker (`bot:delete` or a
    generated UTC ISO stamp) are foreign authored metadata and ignored.
    Anything that is neither a string nor a date (mappings, nested lists,
    numbers, booleans, nulls) cannot be classified and raises, so malformed
    metadata can never silently fall back to the whole retired body.
    """
    raw = metadata.get(SUPERSEDED_KEY)
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    values: list[str] = []
    for item in items:
        if not isinstance(item, str | date):
            raise AmbiguousHistoryError(
                f"{SUPERSEDED_KEY} holds an unsupported {type(item).__name__} value: {item!r}"
            )
        if item == DELETE_MARKER:
            values.append(DELETE_MARKER)
            continue
        stamp = _generated_stamp(item)
        if stamp is not None:
            values.append(stamp)
    return values


def _records(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    raw = metadata.get(BOUNDARY_KEY)
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, list):
        raise AmbiguousHistoryError(f"{BOUNDARY_KEY} must be a list of strings")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        match = _RECORD.fullmatch(item) if isinstance(item, str) else None
        if match is None:
            raise AmbiguousHistoryError(f"malformed {BOUNDARY_KEY} entry: {item!r}")
        boundary_id, value = match.groups()
        if value != DELETE_MARKER and _generated_stamp(value) is None:
            raise AmbiguousHistoryError(f"malformed {BOUNDARY_KEY} value: {item!r}")
        if boundary_id in seen:
            raise AmbiguousHistoryError(f"duplicate boundary id {boundary_id}")
        seen.add(boundary_id)
        records.append((boundary_id, value))
    return records


def _single_marker(positions: dict[str, list[int]], boundary_id: str) -> int:
    found = positions.get(boundary_id, [])
    if len(found) != 1:
        raise AmbiguousHistoryError(
            f"boundary {boundary_id} appears {len(found)} time(s) in the body; expected exactly one"
        )
    return found[0]
