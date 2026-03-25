"""Tests for enhanced MCP audit logging — SQLite queries, purge, change detail builders, config."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from vaultmind.mcp.auth import AuditLogger
from vaultmind.mcp.server import _build_change_detail, _build_output_summary

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# A. Enhanced AuditLogger
# ---------------------------------------------------------------------------


class TestEnhancedAuditLogger:
    @pytest.fixture()
    def audit_logger(self, tmp_path: Path) -> AuditLogger:
        return AuditLogger(tmp_path / "audit.jsonl")

    def test_log_basic_backward_compat(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        audit_logger.log("researcher", "vault_search", {"query": "test"}, "OK")

        log_path = tmp_path / "audit.jsonl"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["profile"] == "researcher"
        assert entry["tool"] == "vault_search"
        assert entry["result"] == "OK"
        assert "params" in entry

    def test_log_with_duration_ms(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        audit_logger.log("full", "vault_write", {"path": "a.md"}, "OK", duration_ms=42)

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["duration_ms"] == 42

    def test_log_with_change_detail(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        detail = {"type": "vault_write", "note_path": "a.md", "size_bytes": 100}
        audit_logger.log("full", "vault_write", {"path": "a.md"}, "OK", change_detail=detail)

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["change_detail"] == detail

    def test_log_with_output_summary(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        summary = {"count": 5}
        audit_logger.log("researcher", "vault_search", {"query": "x"}, "OK", output_summary=summary)

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["output_summary"] == summary

    def test_log_writes_to_sqlite(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("researcher", "vault_read", {"path": "b.md"}, "OK")

        rows = audit_logger.query(days=1)
        assert len(rows) == 1
        assert rows[0]["tool"] == "vault_read"
        assert rows[0]["profile"] == "researcher"

    def test_log_timestamp_has_milliseconds(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("full", "vault_search", {"query": "q"}, "OK")

        rows = audit_logger.query(days=1)
        assert len(rows) == 1
        assert "." in rows[0]["ts"]

    def test_log_params_truncated_to_200(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        long_value = "x" * 500
        audit_logger.log("full", "vault_write", {"content": long_value}, "OK")

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert len(entry["params"]["content"]) == 200


# ---------------------------------------------------------------------------
# B. SQLite Query
# ---------------------------------------------------------------------------


class TestAuditQuery:
    @pytest.fixture()
    def audit_logger(self, tmp_path: Path) -> AuditLogger:
        return AuditLogger(tmp_path / "audit.jsonl")

    def test_query_returns_recent_entries(self, audit_logger: AuditLogger) -> None:
        for i in range(3):
            audit_logger.log("full", f"tool_{i}", {}, "OK")

        rows = audit_logger.query(days=7)
        assert len(rows) == 3

    def test_query_filters_by_profile(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("researcher", "vault_search", {}, "OK")
        audit_logger.log("planner", "vault_write", {}, "OK")
        audit_logger.log("researcher", "vault_read", {}, "OK")

        rows = audit_logger.query(days=7, profile="researcher")
        assert len(rows) == 2
        assert all(r["profile"] == "researcher" for r in rows)

    def test_query_filters_by_tool(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("full", "vault_search", {}, "OK")
        audit_logger.log("full", "vault_write", {}, "OK")

        rows = audit_logger.query(days=7, tool="vault_search")
        assert len(rows) == 1
        assert rows[0]["tool"] == "vault_search"

    def test_query_filters_by_result(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("full", "vault_search", {}, "OK")
        audit_logger.log("full", "vault_write", {}, "DENIED", reason="no write")
        audit_logger.log("full", "vault_read", {}, "ERROR", reason="crash")

        rows = audit_logger.query(days=7, result="DENIED")
        assert len(rows) == 1
        assert rows[0]["result"] == "DENIED"

    def test_query_respects_limit(self, audit_logger: AuditLogger) -> None:
        for i in range(5):
            audit_logger.log("full", f"tool_{i}", {}, "OK")

        rows = audit_logger.query(days=7, limit=2)
        assert len(rows) == 2

    def test_query_ordered_by_ts_desc(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("full", "first", {}, "OK")
        audit_logger.log("full", "second", {}, "OK")
        audit_logger.log("full", "third", {}, "OK")

        rows = audit_logger.query(days=7)
        timestamps = [r["ts"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_empty_returns_empty_list(self, audit_logger: AuditLogger) -> None:
        rows = audit_logger.query(days=7)
        assert rows == []


# ---------------------------------------------------------------------------
# C. Purge
# ---------------------------------------------------------------------------


class TestAuditPurge:
    @pytest.fixture()
    def audit_logger(self, tmp_path: Path) -> AuditLogger:
        return AuditLogger(tmp_path / "audit.jsonl")

    def test_purge_removes_old_entries(self, audit_logger: AuditLogger) -> None:
        # Insert an old entry directly into SQLite (backdated 100 days)
        old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat(timespec="milliseconds")
        audit_logger._conn.execute(
            "INSERT INTO audit_events (ts, profile, tool, result) VALUES (?, ?, ?, ?)",
            (old_ts, "full", "vault_search", "OK"),
        )
        audit_logger._conn.commit()

        # Verify it exists
        all_rows = audit_logger._conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        assert all_rows[0] == 1

        deleted = audit_logger.purge_old_entries(retention_days=1)
        assert deleted == 1

        remaining = audit_logger._conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        assert remaining[0] == 0

    def test_purge_keeps_recent_entries(self, audit_logger: AuditLogger) -> None:
        # Insert a recent entry (today)
        audit_logger.log("full", "vault_search", {"query": "keep"}, "OK")

        # Insert an old entry (200 days ago)
        old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat(timespec="milliseconds")
        audit_logger._conn.execute(
            "INSERT INTO audit_events (ts, profile, tool, result) VALUES (?, ?, ?, ?)",
            (old_ts, "full", "old_tool", "OK"),
        )
        audit_logger._conn.commit()

        deleted = audit_logger.purge_old_entries(retention_days=90)
        assert deleted == 1

        rows = audit_logger.query(days=7)
        assert len(rows) == 1
        assert rows[0]["tool"] == "vault_search"


# ---------------------------------------------------------------------------
# D. Change Detail Builders
# ---------------------------------------------------------------------------


class TestChangeDetailBuilders:
    def test_build_change_detail_vault_write(self) -> None:
        args = {"path": "notes/test.md", "content": "hello world"}
        result = {"status": "ok", "path": "notes/test.md"}

        detail = _build_change_detail("vault_write", args, result)
        assert detail is not None
        assert detail["note_path"] == "notes/test.md"
        assert "size_bytes" in detail
        assert detail["size_bytes"] == len(b"hello world")

    def test_build_change_detail_capture(self) -> None:
        args = {"content": "capture this"}
        result = {"status": "ok", "path": "00-inbox/note.md", "title": "My Note"}

        detail = _build_change_detail("capture", args, result)
        assert detail is not None
        assert detail["note_path"] == "00-inbox/note.md"
        assert detail["title"] == "My Note"

    def test_build_change_detail_vault_search(self) -> None:
        args = {"query": "find stuff"}
        result = {"results": [], "count": 3}

        detail = _build_change_detail("vault_search", args, result)
        assert detail is not None
        assert detail["result_count"] == 3

    def test_build_change_detail_unknown_tool_returns_none(self) -> None:
        detail = _build_change_detail("nonexistent_tool", {}, {})
        assert detail is None

    def test_build_output_summary_vault_search(self) -> None:
        result = {"results": [{"chunk": "..."}], "count": 7}

        summary = _build_output_summary("vault_search", result)
        assert summary is not None
        assert summary["count"] == 7

    def test_build_output_summary_vault_write(self) -> None:
        result = {"status": "ok", "path": "notes/test.md"}

        summary = _build_output_summary("vault_write", result)
        assert summary is not None
        assert summary["status"] == "ok"
        assert summary["path"] == "notes/test.md"

    def test_build_output_summary_unknown_returns_none(self) -> None:
        summary = _build_output_summary("nonexistent_tool", {})
        assert summary is None


# ---------------------------------------------------------------------------
# E. Config
# ---------------------------------------------------------------------------


class TestMCPAuditConfig:
    def test_default_config_valid(self) -> None:
        from vaultmind.config import MCPAuditConfig

        cfg = MCPAuditConfig()
        assert cfg.enabled is True
        assert cfg.level == "standard"
        assert cfg.log_search_queries is False
        assert cfg.retention_days == 90

    def test_config_in_mcp_settings(self) -> None:
        from vaultmind.config import MCPAuditConfig, MCPConfig

        mcp = MCPConfig()
        assert hasattr(mcp, "audit")
        assert isinstance(mcp.audit, MCPAuditConfig)
