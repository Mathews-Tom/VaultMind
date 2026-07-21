"""Closed, explicit connector registry — no dynamic/plugin-based loading.

`ALLOWED_KINDS` is the closed set of connector-kind strings `config/sources.toml`
instances may reference; it is defined independently of which connectors'
`fetch()` implementations have registered into `REGISTRY`, so config
validation (unknown kind -> hard error) works from PR-1 onward even before
PR-2/PR-3 land the actual `rss`/`youtube-channel`/`github-activity`
implementations.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from vaultmind.sources.models import ConnectorDefinition, SourceInstance

if TYPE_CHECKING:
    from pathlib import Path

ALLOWED_KINDS: frozenset[str] = frozenset({"rss", "youtube-channel", "github-activity"})

REGISTRY: dict[str, ConnectorDefinition] = {}


def register_connector(definition: ConnectorDefinition) -> None:
    """Register a connector implementation into the closed registry.

    Raises `ValueError` for a `kind` outside `ALLOWED_KINDS` — the registry
    only ever holds implementations for the three named connectors, never an
    arbitrary/dynamically-discovered kind.
    """
    if definition.kind not in ALLOWED_KINDS:
        msg = f"Unknown connector kind {definition.kind!r} — must be one of {sorted(ALLOWED_KINDS)}"
        raise ValueError(msg)
    REGISTRY[definition.kind] = definition


def get_connector(kind: str) -> ConnectorDefinition:
    """Look up a registered connector implementation by kind.

    Raises `KeyError` (with the closed set listed) if `kind` has no
    registered implementation — never falls back to a default or a no-op.
    """
    try:
        return REGISTRY[kind]
    except KeyError:
        msg = (
            f"No connector implementation registered for kind {kind!r} "
            f"(registered: {sorted(REGISTRY)})"
        )
        raise KeyError(msg) from None


def load_source_instances(path: Path) -> list[SourceInstance]:
    """Load and validate every `[[instance]]` entry in `config/sources.toml`.

    Fails loud (raises `ValueError`) on an unknown `kind`, a duplicate
    `name`, or a missing required field — never silently skips a malformed
    instance. Returns an empty list if `path` does not exist (no
    `config/sources.toml` configured is a valid, connector-free state).
    """
    if not path.exists():
        return []

    with open(path, "rb") as f:
        data = tomllib.load(f)

    raw_instances = data.get("instance", [])
    instances: list[SourceInstance] = []
    seen_names: set[str] = set()

    for raw in raw_instances:
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        target = str(raw.get("target", "")).strip()

        if not name:
            msg = "config/sources.toml: every [[instance]] requires a non-empty 'name'"
            raise ValueError(msg)
        if name in seen_names:
            msg = f"config/sources.toml: duplicate instance name {name!r}"
            raise ValueError(msg)
        if kind not in ALLOWED_KINDS:
            msg = (
                f"config/sources.toml: instance {name!r} has unknown kind {kind!r} "
                f"— must be one of {sorted(ALLOWED_KINDS)}"
            )
            raise ValueError(msg)
        if not target:
            msg = f"config/sources.toml: instance {name!r} requires a non-empty 'target'"
            raise ValueError(msg)

        seen_names.add(name)
        options = {str(k): str(v) for k, v in dict(raw.get("options", {})).items()}
        instances.append(
            SourceInstance(
                name=name,
                kind=kind,
                target=target,
                enabled=bool(raw.get("enabled", False)),
                interval_hours=int(raw.get("interval_hours", 24)),
                output_folder=str(raw.get("output_folder", "00-inbox/sources")),
                options=options,
            )
        )

    return instances


__all__ = [
    "ALLOWED_KINDS",
    "REGISTRY",
    "get_connector",
    "load_source_instances",
    "register_connector",
]
