"""Tests for MCP profile enforcement and audit logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaultmind.config import VaultConfig
from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer, ProfileError, ProfilePolicy
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool
from vaultmind.vault.parser import VaultParser


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
        with pytest.raises(ProfileError):
            enforcer.check_write()

    def test_researcher_cannot_use_write_tool(
        self, researcher_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(researcher_policy, vault_root)
        with pytest.raises(ProfileError):
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
        with pytest.raises(ProfileError):
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
        with pytest.raises(ProfileError):
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
        with pytest.raises(ProfileError):
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
        with pytest.raises(ProfileError):
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
        with pytest.raises(ValueError):
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


class TestWildcardContainment:
    def test_full_profile_blocks_path_escape(
        self, full_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        enforcer = ProfileEnforcer(full_policy, vault_root)
        with pytest.raises(ProfileError):
            enforcer.check_path(Path("../../etc/passwd"))

    def test_full_profile_blocks_symlink_escape(
        self, full_policy: ProfilePolicy, vault_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.md").write_text("secret", encoding="utf-8")
        (vault_root / "04-resources" / "escape.md").symlink_to(outside / "secret.md")
        enforcer = ProfileEnforcer(full_policy, vault_root)
        with pytest.raises(ProfileError):
            enforcer.check_path(Path("04-resources/escape.md"))

    def test_full_profile_allows_vault_root(
        self, full_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        ProfileEnforcer(full_policy, vault_root).check_path(Path(""))

    def test_restricted_profile_rejects_vault_root_path(
        self, planner_policy: ProfilePolicy, vault_root: Path
    ) -> None:
        with pytest.raises(ProfileError):
            ProfileEnforcer(planner_policy, vault_root).check_path(Path(""))


class TestConfiguredProfileUsesSameEnforcer:
    def test_config_override_scope_is_enforced(self, vault_root: Path) -> None:
        custom = {
            "planner": {
                "description": "Narrowed planner",
                "allowed_tools": ["read_frontmatter"],
                "folder_scope": ["04-resources"],
                "write_enabled": False,
            }
        }
        enforcer = ProfileEnforcer(load_profile("planner", config_profiles=custom), vault_root)
        enforcer.check_path(Path("04-resources/a.md"))
        with pytest.raises(ProfileError):
            enforcer.check_path(Path("02-projects/a.md"))


def _write_note(path: Path, title: str, body: str, description: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = f"description: {description}\n" if description else ""
    path.write_text(f"---\ntitle: {title}\n{desc}---\n\n{body}\n", encoding="utf-8")


class _Unused:
    pass


def _traversal(name: str, args: dict, vault_root: Path, enforcer: ProfileEnforcer | None) -> dict:
    return _dispatch_tool(
        name,
        args,
        vault_root,
        _Unused(),  # type: ignore[arg-type]
        _Unused(),  # type: ignore[arg-type]
        VaultParser(VaultConfig(path=vault_root)),  # type: ignore[arg-type]
        enforcer=enforcer,
    )


@pytest.fixture
def linked_vault(vault_root: Path) -> Path:
    _write_note(
        vault_root / "02-projects" / "hub.md",
        "Hub",
        "Links [[Open Note]] [[Private Secret]] [[Ghost Note]].",
        "Hub description",
    )
    _write_note(vault_root / "02-projects" / "open-note.md", "Open Note", "Open body.")
    _write_note(vault_root / "04-resources" / "open-note.md", "Open Note", "Shadow body.")
    _write_note(
        vault_root / "04-resources" / "private-secret.md",
        "Private Secret",
        "Back to [[Hub]].",
        "PRIVATE-DESCRIPTION",
    )
    return vault_root


class TestTraversalResultScopes:
    def test_missing_enforcer_fails_closed(self, linked_vault: Path) -> None:
        for name, args in (
            ("read_frontmatter", {"path": "02-projects/hub.md"}),
            ("list_folder_index", {}),
            ("follow_links", {"path": "02-projects/hub.md"}),
        ):
            with pytest.raises(ProfileError):
                _traversal(name, args, linked_vault, None)

    def test_planner_root_aggregate_excludes_private(
        self, planner_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        for args in ({}, {"folder": ""}, {"folder": "."}):
            result = _traversal("list_folder_index", args, linked_vault, enforcer)
            paths = [n["path"] for n in result["notes"]]
            assert paths == ["02-projects/hub.md", "02-projects/open-note.md"]
            assert result["count"] == 2
            assert "PRIVATE-DESCRIPTION" not in str(result)

    def test_planner_explicit_forbidden_folder_denied_before_existence(
        self, planner_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        for folder in ("04-resources", "09-missing"):
            with pytest.raises(ProfileError):
                _traversal("list_folder_index", {"folder": folder}, linked_vault, enforcer)

    def test_planner_follow_links_hides_private_titles_paths_and_counts(
        self, planner_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        result = _traversal("follow_links", {"path": "02-projects/hub.md"}, linked_vault, enforcer)
        # Allowed duplicate title is not shadowed by the private same-title note.
        assert result["out_links"] == [{"title": "Open Note", "path": "02-projects/open-note.md"}]
        assert result["out_link_count"] == 1
        assert result["backlinks"] == []
        assert result["backlink_count"] == 0
        serialized = str(result)
        assert "Private Secret" not in serialized
        assert "private-secret" not in serialized
        assert "Ghost Note" not in serialized

    def test_planner_forbidden_note_denied(
        self, planner_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        for name in ("read_frontmatter", "follow_links"):
            with pytest.raises(ProfileError):
                _traversal(name, {"path": "04-resources/private-secret.md"}, linked_vault, enforcer)

    def test_planner_symlink_to_private_note_excluded(
        self, planner_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        (linked_vault / "02-projects" / "alias.md").symlink_to(
            linked_vault / "04-resources" / "private-secret.md"
        )
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        listing = _traversal("list_folder_index", {}, linked_vault, enforcer)
        assert "02-projects/alias.md" not in [n["path"] for n in listing["notes"]]
        assert "PRIVATE-DESCRIPTION" not in str(listing)
        hub = _traversal("follow_links", {"path": "02-projects/hub.md"}, linked_vault, enforcer)
        assert hub["backlinks"] == []
        with pytest.raises(ProfileError):
            _traversal("read_frontmatter", {"path": "02-projects/alias.md"}, linked_vault, enforcer)

    @pytest.mark.parametrize(
        ("tool", "parameter", "alias", "target"),
        [
            ("list_folder_index", "folder", "alias-dir", "04-resources"),
            ("list_folder_index", "folder", "alias.md", "04-resources/private-secret.md"),
            ("read_frontmatter", "path", "alias.md", "04-resources/private-secret.md"),
            ("follow_links", "path", "alias.md", "04-resources/private-secret.md"),
        ],
    )
    def test_symlink_denial_does_not_disclose_unknown_target(
        self,
        planner_policy: ProfilePolicy,
        linked_vault: Path,
        tool: str,
        parameter: str,
        alias: str,
        target: str,
    ) -> None:
        (linked_vault / "02-projects" / alias).symlink_to(linked_vault / target)
        enforcer = ProfileEnforcer(planner_policy, linked_vault)
        with pytest.raises(ProfileError) as excinfo:
            _traversal(tool, {parameter: f"02-projects/{alias}"}, linked_vault, enforcer)
        assert all(
            token not in str(excinfo.value)
            for token in (
                "04-resources",
                "private-secret.md",
                "Private Secret",
                "PRIVATE-DESCRIPTION",
            )
        )

    def test_wildcard_profile_semantics_unchanged(
        self, researcher_policy: ProfilePolicy, linked_vault: Path
    ) -> None:
        enforcer = ProfileEnforcer(researcher_policy, linked_vault)
        listing = _traversal("list_folder_index", {}, linked_vault, enforcer)
        assert listing["count"] == 4
        hub = _traversal("follow_links", {"path": "02-projects/hub.md"}, linked_vault, enforcer)
        titles = {link["title"]: link["path"] for link in hub["out_links"]}
        assert titles["Private Secret"] == "04-resources/private-secret.md"
        assert titles["Ghost Note"] is None
        assert hub["out_link_count"] == 3
        assert [b["title"] for b in hub["backlinks"]] == ["Private Secret"]
        assert hub["backlink_count"] == 1
