"""Tests for MCP profile enforcement and audit logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError, ProfilePolicy
from vaultmind.mcp.profiles import load_profile


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Create a mock vault structure."""
    for folder in ["00-inbox", "02-projects", "04-resources"]:
        (tmp_path / folder).mkdir()
    return tmp_path


@pytest.fixture
def researcher_policy() -> ProfilePolicy:
    return ProfilePolicy(
        name="researcher",
        description="Read-only",
        allowed_tools=frozenset({"vault_search", "vault_read", "graph_query"}),
        folder_scope=("*",),
        write_enabled=False,
    )


@pytest.fixture
def planner_policy() -> ProfilePolicy:
    return ProfilePolicy(
        name="planner",
        description="Read/write for projects",
        allowed_tools=frozenset({"vault_search", "vault_read", "vault_write", "capture"}),
        folder_scope=("02-projects", "00-inbox"),
        write_enabled=True,
        max_note_size_bytes=50000,
    )


@pytest.fixture
def full_policy() -> ProfilePolicy:
    return ProfilePolicy(
        name="full",
        description="Unrestricted",
        allowed_tools=frozenset({"*"}),
        folder_scope=("*",),
        write_enabled=True,
    )


class TestProfileEnforcer:
    def test_researcher_cannot_write(
        self, researcher_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(researcher_policy, vault_root)
        with pytest.raises(ProfileError, match="Write operations disabled"):
            enforcer.check_write()

    def test_researcher_cannot_use_write_tool(
        self, researcher_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(researcher_policy, vault_root)
        with pytest.raises(ProfileError, match="not allowed"):
            enforcer.check_tool("vault_write")

    def test_researcher_can_search(
        self, researcher_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(researcher_policy, vault_root)
        enforcer.check_tool("vault_search")  # Should not raise

    def test_planner_cannot_write_outside_scope(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        with pytest.raises(ProfileError, match="outside folder scope"):
            enforcer.check_path(Path("04-resources/test.md"))

    def test_planner_can_write_in_scope(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        enforcer.check_path(Path("02-projects/test.md"))  # Should not raise

    def test_full_profile_allows_all(self, full_policy: ProfilePolicy, vault_root: Path) -> None:
        enforcer = ProfileEnforcer(full_policy, vault_root)
        enforcer.check_tool("vault_write")
        enforcer.check_tool("any_tool")
        enforcer.check_path(Path("04-resources/test.md"))
        enforcer.check_write()

    def test_path_traversal_blocked(self, planner_policy: ProfilePolicy, vault_root: Path) -> None:
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        with pytest.raises(ProfileError, match="Path traversal blocked"):
            enforcer.check_path(Path("../../etc/passwd"))

    def test_symlink_resolved_before_scope_check(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        # Create a symlink from 02-projects/link -> 04-resources/
        resources = vault_root / "04-resources"
        link = vault_root / "02-projects" / "sneaky-link"
        link.symlink_to(resources)
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        # The symlink resolves outside of 02-projects scope
        with pytest.raises(ProfileError):
            enforcer.check_path(Path("02-projects/sneaky-link/secret.md"))

    def test_size_check_blocks_oversized(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        with pytest.raises(ProfileError, match="exceeds limit"):
            enforcer.check_size("x" * 60000)

    def test_size_check_allows_within_limit(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, vault_root)
        enforcer.check_size("x" * 1000)

    def test_enforcer_is_stateless(
        self, researcher_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        """Verify ProfilePolicy is frozen (immutable)."""
        assert researcher_policy.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_missing_folder_scope_errors_at_call_time(
        self,
        vault_root: Path,
    ) -> None:
        """Non-existent folder in scope doesn't error at construction time."""
        policy = ProfilePolicy(
            name="test",
            description="test",
            allowed_tools=frozenset({"vault_read"}),
            folder_scope=("nonexistent-folder",),
            write_enabled=False,
        )
        # Construction succeeds
        enforcer = ProfileEnforcer(policy, vault_root)
        # Only errors when path is actually checked
        with pytest.raises(ProfileError, match="outside folder scope"):
            enforcer.check_path(Path("00-inbox/test.md"))

    def test_default_profile_is_readonly(self) -> None:
        policy = load_profile("researcher")
        assert not policy.write_enabled


class TestAuditLogger:
    def test_audit_log_written_on_success(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        audit_logger.log("researcher", "vault_search", {"query": "test"}, "OK")
        assert log_path.exists()
        line = log_path.read_text().strip()
        entry = json.loads(line)
        assert entry["profile"] == "researcher"
        assert entry["tool"] == "vault_search"
        assert entry["result"] == "OK"

    def test_audit_log_written_on_denial(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        audit_logger.log(
            "researcher", "vault_write", {"path": "test.md"}, "DENIED", reason="write disabled"
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["result"] == "DENIED"
        assert entry["reason"] == "write disabled"

    def test_audit_log_is_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        for i in range(5):
            audit_logger.log("test", f"tool_{i}", {"key": f"val_{i}"}, "OK")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # Should not raise

    def test_audit_log_truncates_large_params(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        audit_logger.log("test", "tool", {"content": "x" * 1000}, "OK")
        entry = json.loads(log_path.read_text().strip())
        assert len(entry["params"]["content"]) <= 200


class TestLoadProfile:
    def test_load_builtin_researcher(self) -> None:
        policy = load_profile("researcher")
        assert policy.name == "researcher"
        assert not policy.write_enabled

    def test_load_builtin_planner(self) -> None:
        policy = load_profile("planner")
        assert policy.write_enabled
        assert "02-projects" in policy.folder_scope

    def test_load_builtin_full(self) -> None:
        policy = load_profile("full")
        assert policy.allows_all_tools
        assert policy.allows_all_folders

    def test_load_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            load_profile("nonexistent")

    def test_config_profiles_override_defaults(self) -> None:
        custom = {
            "researcher": {
                "description": "Custom researcher",
                "allowed_tools": ["vault_search"],
                "folder_scope": ["04-resources"],
                "write_enabled": False,
            }
        }
        policy = load_profile("researcher", config_profiles=custom)
        assert policy.description == "Custom researcher"
        assert policy.folder_scope == ("04-resources",)
