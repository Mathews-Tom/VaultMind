"""MCP profile enforcement and audit logging."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
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
    """Append-only JSONL + SQLite audit logger for MCP operations."""

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite for queryable audit history
        self._db_path = log_path.with_suffix(".db")
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                profile TEXT NOT NULL,
                tool TEXT NOT NULL,
                result TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER,
                note_path TEXT,
                change_detail TEXT,
                output_summary TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_events(tool)")
        self._conn.commit()

    def log(
        self,
        profile: str,
        tool: str,
        params: dict[str, Any],
        result: str,
        reason: str = "",
        duration_ms: int | None = None,
        change_detail: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "profile": profile,
            "tool": tool,
            "params": {k: str(v)[:200] for k, v in params.items()},
            "result": result,
        }
        if reason:
            entry["reason"] = reason
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        if change_detail is not None:
            entry["change_detail"] = change_detail
        if output_summary is not None:
            entry["output_summary"] = output_summary

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.warning("Failed to write audit log entry for %s/%s", profile, tool)

        try:
            note_path = ""
            if change_detail and isinstance(change_detail, dict):
                note_path = str(change_detail.get("note_path", ""))
            self._conn.execute(
                "INSERT INTO audit_events"
                " (ts, profile, tool, result, reason,"
                "  duration_ms, note_path, change_detail, output_summary)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["ts"],
                    profile,
                    tool,
                    result,
                    reason,
                    duration_ms,
                    note_path,
                    json.dumps(change_detail, default=str) if change_detail else None,
                    json.dumps(output_summary, default=str) if output_summary else None,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.debug("Failed to write audit to SQLite", exc_info=True)

    def query(
        self,
        days: int = 7,
        profile: str | None = None,
        tool: str | None = None,
        result: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events from SQLite."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        sql = (
            "SELECT ts, profile, tool, result, reason, duration_ms, note_path,"
            " change_detail, output_summary FROM audit_events WHERE ts >= ?"
        )
        params_list: list[Any] = [cutoff]

        if profile:
            sql += " AND profile = ?"
            params_list.append(profile)
        if tool:
            sql += " AND tool = ?"
            params_list.append(tool)
        if result:
            sql += " AND result = ?"
            params_list.append(result)

        sql += " ORDER BY ts DESC LIMIT ?"
        params_list.append(limit)

        rows = self._conn.execute(sql, params_list).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event: dict[str, Any] = {
                "ts": row[0],
                "profile": row[1],
                "tool": row[2],
                "result": row[3],
            }
            if row[4]:
                event["reason"] = row[4]
            if row[5] is not None:
                event["duration_ms"] = row[5]
            if row[6]:
                event["note_path"] = row[6]
            if row[7]:
                event["change_detail"] = json.loads(row[7])
            if row[8]:
                event["output_summary"] = json.loads(row[8])
            events.append(event)
        return events

    def purge_old_entries(self, retention_days: int = 90) -> int:
        """Delete audit entries older than retention_days. Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        cursor = self._conn.execute("DELETE FROM audit_events WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return cursor.rowcount
