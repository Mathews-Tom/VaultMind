"""MCP profile enforcement and audit logging."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.errors import VaultMindError

logger = logging.getLogger(__name__)


class ProfileError(VaultMindError):
    """Raised when an MCP operation violates the active profile policy."""


@dataclass(frozen=True, slots=True)
class ProfilePolicy:
    """Immutable policy definition for an MCP profile."""

    name: str
    description: str
    allowed_tools: frozenset[str]
    folder_scope: tuple[str, ...]
    write_enabled: bool
    max_note_size_bytes: int | None = None
    requires_confirmation: bool = False

    @property
    def allows_all_tools(self) -> bool:
        return "*" in self.allowed_tools

    @property
    def allows_all_folders(self) -> bool:
        return "*" in self.folder_scope


class ProfileEnforcer:
    """Stateless, thread-safe policy enforcement for MCP operations."""

    def __init__(self, policy: ProfilePolicy, vault_root: Path) -> None:
        self._policy = policy
        self._vault_root = vault_root

    @property
    def policy(self) -> ProfilePolicy:
        return self._policy

    def check_tool(self, tool_name: str) -> None:
        """Raise ProfileError if tool not allowed by active profile."""
        if self._policy.allows_all_tools:
            return
        if tool_name not in self._policy.allowed_tools:
            raise ProfileError(
                f"Tool '{tool_name}' not allowed by profile '{self._policy.name}'. "
                f"Allowed: {', '.join(sorted(self._policy.allowed_tools))}"
            )

    def check_write(self) -> None:
        """Raise ProfileError if writes are disabled."""
        if not self._policy.write_enabled:
            raise ProfileError(f"Write operations disabled for profile '{self._policy.name}'")

    def check_path(self, target_path: Path) -> None:
        """Raise ProfileError if path is outside folder scope.

        Validates at call time (not startup) to avoid chicken-and-egg
        with vaultmind init. Resolves symlinks before checking.
        """
        if self._policy.allows_all_folders:
            return

        resolved_root = self._vault_root.resolve()
        resolved_target = (resolved_root / target_path).resolve()

        # Path traversal check
        try:
            resolved_target.relative_to(resolved_root)
        except ValueError:
            raise ProfileError(
                f"Path traversal blocked: '{target_path}' escapes vault root"
            ) from None

        # Folder scope check
        rel = resolved_target.relative_to(resolved_root)
        first_folder = rel.parts[0] if rel.parts else ""

        if first_folder not in self._policy.folder_scope:
            raise ProfileError(
                f"Path '{target_path}' outside folder scope for profile "
                f"'{self._policy.name}'. Allowed: {', '.join(self._policy.folder_scope)}"
            )

    def check_size(self, content: str) -> None:
        """Raise ProfileError if content exceeds size limit."""
        if self._policy.max_note_size_bytes is not None:
            size = len(content.encode("utf-8"))
            if size > self._policy.max_note_size_bytes:
                raise ProfileError(
                    f"Content size {size} bytes exceeds limit "
                    f"{self._policy.max_note_size_bytes} for profile '{self._policy.name}'"
                )


class AuditLogger:
    """Append-only JSONL audit logger for MCP operations."""

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        profile: str,
        tool: str,
        params: dict[str, Any],
        result: str,
        reason: str = "",
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profile": profile,
            "tool": tool,
            "params": {k: str(v)[:200] for k, v in params.items()},
            "result": result,
        }
        if reason:
            entry["reason"] = reason

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.warning("Failed to write audit log entry for %s/%s", profile, tool)
