"""Tests for MCP profile enforcement and audit logging."""

from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import pytest

from vaultmind.config import MCPProfileConfig, Settings, VaultConfig
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
        assert entry["reason"] == "DENIED"

    def test_audit_log_is_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        for i in range(5):
            audit_logger.log("test", f"tool_{i}", {"key": f"val_{i}"}, "OK")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # Should not raise

    def test_audit_log_redacts_body_params(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit_logger = AuditLogger(log_path)
        audit_logger.log("test", "tool", {"content": "x" * 1000}, "OK")
        entry = json.loads(log_path.read_text().strip())
        assert "content" not in entry["params"]


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
            "researcher": MCPProfileConfig(
                description="Custom researcher",
                allowed_tools=["vault_search"],
                folder_scope=["04-resources"],
                write_enabled=False,
            )
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
            "planner": MCPProfileConfig(
                description="Narrowed planner",
                allowed_tools=["read_frontmatter"],
                folder_scope=["04-resources"],
                write_enabled=False,
            )
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


def _matches_where(metadata: dict, where: dict | None) -> bool:
    """Implement the store's metadata predicate before applying its result limit."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


class _RankedStore:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, dict | None]] = []

    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        self.calls.append((query, n_results, where))
        matches = [hit for hit in self.hits if _matches_where(hit.get("metadata", {}), where)]
        return matches[:n_results]


def _hit(path: object, *, distance: float = 0.01, note_type: str = "permanent") -> dict:
    return {
        "content": f"Chunk for {path}",
        "distance": distance,
        "metadata": {"note_path": path, "note_title": str(path), "note_type": note_type},
    }


def _note_dispatch(
    name: str,
    args: dict,
    root: Path,
    enforcer: ProfileEnforcer | None,
    store: object | None = None,
    **kwargs: object,
) -> dict:
    return _dispatch_tool(
        name,
        args,
        root,
        store if store is not None else _Unused(),  # type: ignore[arg-type]
        _Unused(),  # type: ignore[arg-type]
        VaultParser(VaultConfig(path=root)),  # type: ignore[arg-type]
        enforcer=enforcer,
        settings=Settings(
            _env_file=None, vault=VaultConfig(path=root), search={"hybrid_enabled": False}
        ),
        **kwargs,  # type: ignore[arg-type]
    )


class TestNoteResultBoundary:
    @pytest.mark.parametrize(
        "provenance",
        [
            None,
            7,
            True,
            {},
            [],
            Path("02-projects/open-note.md"),
            "",
            "02-projects/missing.md",
            "02-projects",
            "04-resources/private-secret.md",
            "../outside.md",
            "02-projects/bad\x00.md",
        ],
    )
    def test_unknown_or_forbidden_provenance_is_not_a_note(
        self, linked_vault: Path, planner_policy: ProfilePolicy, provenance: object
    ) -> None:
        scope = ProfileEnforcer(planner_policy, linked_vault)
        assert scope.note_path(provenance) is None

    def test_allowed_provenance_and_inventory(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        (linked_vault / "02-projects" / "not-a-note.txt").write_text("text")
        scope = ProfileEnforcer(planner_policy, linked_vault)
        assert scope.note_path("02-projects/open-note.md") == Path("02-projects/open-note.md")
        assert scope.note_path("02-projects/not-a-note.txt") is None
        assert sorted(scope.note_paths()) == ["02-projects/hub.md", "02-projects/open-note.md"]

    @pytest.mark.parametrize("profile", ["planner", "full"])
    def test_symlink_escape_and_unresolvable_provenance_are_omitted(
        self, linked_vault: Path, tmp_path: Path, profile: str
    ) -> None:
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        _write_note(outside / "external.md", "External", "Outside content")
        (linked_vault / "02-projects" / "escape.md").symlink_to(outside / "external.md")
        (linked_vault / "02-projects" / "loop.md").symlink_to("loop.md")
        scope = ProfileEnforcer(load_profile(profile), linked_vault)
        assert scope.note_path("02-projects/escape.md") is None
        assert scope.note_path("02-projects/loop.md") is None
        assert "02-projects/escape.md" not in scope.note_paths()
        assert "02-projects/loop.md" not in scope.note_paths()

    def test_search_filters_before_top_k_and_preserves_note_type(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        (linked_vault / "02-projects/open-note.md").write_text(
            "---\ntype: permanent\n---\n\nCurrent allowed body\n", encoding="utf-8"
        )
        store = _RankedStore(
            [
                *[_hit("04-resources/private-secret.md") for _ in range(30)],
                _hit("02-projects/hub.md", note_type="project"),
                _hit("02-projects/open-note.md"),
            ]
        )
        result = _note_dispatch(
            "vault_search",
            {"query": "query", "n_results": 1, "note_type": "permanent"},
            linked_vault,
            ProfileEnforcer(planner_policy, linked_vault),
            store,
        )
        assert result["count"] == 1
        assert result["approved"] is False
        assert result["results"][0]["path"] == "02-projects/open-note.md"
        assert "Current allowed body" in result["results"][0]["content"]
        assert result["results"][0]["distance"] is None
        assert "private-secret" not in str(result)

    @pytest.mark.parametrize("profile", ["planner", "full"])
    def test_search_defensively_omits_invalid_store_provenance(
        self, linked_vault: Path, profile: str
    ) -> None:
        class IgnoringFilterStore(_RankedStore):
            def search(
                self, query: str, n_results: int = 5, where: dict | None = None
            ) -> list[dict]:
                return self.hits

        valid = _hit("02-projects/open-note.md")
        store = IgnoringFilterStore(
            [
                _hit(None),
                _hit(5),
                _hit({}),
                _hit("02-projects/missing.md"),
                _hit("../outside.md"),
                {"content": "Unknown source"},
                valid,
            ]
        )
        result = _note_dispatch(
            "vault_search",
            {"query": "query"},
            linked_vault,
            ProfileEnforcer(load_profile(profile), linked_vault),
            store,
        )
        assert result["count"] == 1
        hit = result["results"][0]
        assert hit["path"] == "02-projects/open-note.md"
        assert hit["content"] == frontmatter.loads((linked_vault / hit["path"]).read_text()).content
        assert hit["status"] == "source-only" and hit["approved"] is False
        assert hit["distance"] is None

    def test_full_search_hydrates_private_hits_without_stale_scores(
        self, linked_vault: Path
    ) -> None:
        hits = [_hit("04-resources/private-secret.md"), _hit("02-projects/hub.md", distance=0.12)]
        result = _note_dispatch(
            "vault_search",
            {"query": "query"},
            linked_vault,
            ProfileEnforcer(load_profile("full"), linked_vault),
            _RankedStore(hits),
        )
        assert result["count"] == 2
        assert {hit["path"] for hit in result["results"]} == {
            "04-resources/private-secret.md",
            "02-projects/hub.md",
        }
        for hit in result["results"]:
            current = frontmatter.loads((linked_vault / hit["path"]).read_text())
            assert hit["content"] == current.content
            assert hit["status"] == "source-only" and hit["approved"] is False
            assert hit["distance"] is None

    @pytest.mark.parametrize("folder", [None, "", "."])
    def test_list_is_virtual_allowed_root_aggregate(
        self, linked_vault: Path, planner_policy: ProfilePolicy, folder: str | None
    ) -> None:
        result = _note_dispatch(
            "vault_list",
            {} if folder is None else {"folder": folder},
            linked_vault,
            ProfileEnforcer(planner_policy, linked_vault),
        )
        assert result == {
            "notes": ["02-projects/hub.md", "02-projects/open-note.md"],
            "count": 2,
        }

    @pytest.mark.parametrize("folder", ["04-resources", "09-missing"])
    def test_explicit_forbidden_list_is_denied(
        self, linked_vault: Path, planner_policy: ProfilePolicy, folder: str
    ) -> None:
        with pytest.raises(ProfileError):
            _note_dispatch(
                "vault_list",
                {"folder": folder},
                linked_vault,
                ProfileEnforcer(planner_policy, linked_vault),
            )

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("vault_search", {"query": "query"}),
            ("vault_list", {}),
            ("vault_read", {"path": "02-projects/hub.md"}),
            ("vault_write", {"path": "02-projects/new.md", "content": "body"}),
            ("find_duplicates", {"note_path": "02-projects/hub.md"}),
            ("suggest_links", {"note_path": "02-projects/hub.md"}),
            ("capture", {"content": "body"}),
            ("capture_note", {"content": "body"}),
        ],
    )
    def test_note_dispatch_requires_enforcer(
        self, linked_vault: Path, name: str, args: dict
    ) -> None:
        with pytest.raises(ProfileError):
            _note_dispatch(name, args, linked_vault, None)

    def test_direct_read_scope_and_full_control(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        scope = ProfileEnforcer(planner_policy, linked_vault)
        allowed = _note_dispatch(
            "vault_read", {"path": "02-projects/open-note.md"}, linked_vault, scope
        )
        assert "Open body." in allowed["content"]
        with pytest.raises(ProfileError):
            _note_dispatch(
                "vault_read", {"path": "04-resources/private-secret.md"}, linked_vault, scope
            )
        full = _note_dispatch(
            "vault_read",
            {"path": "04-resources/private-secret.md"},
            linked_vault,
            ProfileEnforcer(load_profile("full"), linked_vault),
        )
        assert "Back to [[Hub]]." in full["content"]

    def test_restricted_duplicates_find_allowed_hit_below_private_top_k(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        from vaultmind.config import DuplicateDetectionConfig
        from vaultmind.indexer.duplicate_detector import DuplicateDetector

        store = _RankedStore(
            [
                *[_hit("04-resources/private-secret.md") for _ in range(30)],
                _hit("02-projects/open-note.md"),
            ]
        )
        detector = DuplicateDetector(
            DuplicateDetectionConfig(min_content_length=1),
            store,  # type: ignore[arg-type]
        )
        result = _note_dispatch(
            "find_duplicates",
            {"note_path": "02-projects/hub.md", "max_results": 1},
            linked_vault,
            ProfileEnforcer(planner_policy, linked_vault),
            store,
            duplicate_detector=detector,
        )
        assert result["count"] == 1
        assert [match["path"] for match in result["matches"]] == ["02-projects/open-note.md"]
        assert "private-secret" not in str(result)

    def test_restricted_suggestions_are_denied_before_producer_access(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        with pytest.raises(ProfileError):
            _note_dispatch(
                "suggest_links",
                {"note_path": "02-projects/hub.md"},
                linked_vault,
                ProfileEnforcer(planner_policy, linked_vault),
                note_suggester=_Unused(),
            )

    def test_full_duplicate_and_suggestion_controls(self, linked_vault: Path) -> None:
        from vaultmind.config import DuplicateDetectionConfig, NoteSuggestionsConfig
        from vaultmind.indexer.duplicate_detector import DuplicateDetector
        from vaultmind.indexer.note_suggester import NoteSuggester

        scope = ProfileEnforcer(load_profile("full"), linked_vault)
        duplicate_store = _RankedStore([_hit("04-resources/private-secret.md")])
        duplicates = _note_dispatch(
            "find_duplicates",
            {"note_path": "02-projects/hub.md"},
            linked_vault,
            scope,
            duplicate_store,
            duplicate_detector=DuplicateDetector(
                DuplicateDetectionConfig(min_content_length=1),
                duplicate_store,  # type: ignore[arg-type]
            ),
        )
        assert duplicates["count"] == 1
        assert duplicates["matches"][0]["path"] == "04-resources/private-secret.md"
        suggestion_store = _RankedStore([_hit("04-resources/private-secret.md", distance=0.25)])
        suggestions = _note_dispatch(
            "suggest_links",
            {"note_path": "02-projects/hub.md"},
            linked_vault,
            scope,
            suggestion_store,
            note_suggester=NoteSuggester(
                NoteSuggestionsConfig(min_content_length=1),
                suggestion_store,  # type: ignore[arg-type]
            ),
        )
        assert suggestions["count"] == 1
        assert suggestions["suggestions"][0] == {
            "path": "04-resources/private-secret.md",
            "title": "04-resources/private-secret.md",
            "similarity": 0.75,
            "shared_entities": [],
            "graph_distance": None,
            "composite_score": 0.75,
        }

    def test_explicit_vault_root_list_is_scoped(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        result = _note_dispatch(
            "vault_list",
            {"folder": str(linked_vault.resolve())},
            linked_vault,
            ProfileEnforcer(planner_policy, linked_vault),
        )
        assert result == {
            "notes": ["02-projects/hub.md", "02-projects/open-note.md"],
            "count": 2,
        }

    def test_full_list_retains_private_notes(self, linked_vault: Path) -> None:
        result = _note_dispatch(
            "vault_list",
            {},
            linked_vault,
            ProfileEnforcer(load_profile("full"), linked_vault),
        )
        assert result == {
            "notes": [
                "02-projects/hub.md",
                "02-projects/open-note.md",
                "04-resources/open-note.md",
                "04-resources/private-secret.md",
            ],
            "count": 4,
        }

    def test_direct_write_denial_preserves_existing_note(
        self, linked_vault: Path, planner_policy: ProfilePolicy
    ) -> None:
        path = linked_vault / "04-resources/private-secret.md"
        before = path.read_bytes()
        with pytest.raises(ProfileError):
            _note_dispatch(
                "vault_write",
                {"path": "04-resources/private-secret.md", "content": "replacement"},
                linked_vault,
                ProfileEnforcer(planner_policy, linked_vault),
            )
        assert path.read_bytes() == before

    @pytest.mark.parametrize("write,limit", [(False, None), (True, 1)])
    def test_direct_write_permissions_and_size_precede_directory_creation(
        self, linked_vault: Path, write: bool, limit: int | None
    ) -> None:
        scope = ProfileEnforcer(
            ProfilePolicy(
                name="write-test",
                description="Write limits",
                allowed_tools=frozenset({"*"}),
                folder_scope=("*",),
                write_enabled=write,
                max_note_size_bytes=limit,
            ),
            linked_vault,
        )
        with pytest.raises(ProfileError):
            _note_dispatch(
                "vault_write",
                {"path": "02-projects/new/note.md", "content": "é"},
                linked_vault,
                scope,
            )
        assert not (linked_vault / "02-projects/new").exists()

    @pytest.mark.parametrize("profile", ["planner", "full"])
    def test_direct_write_allowed_and_indexed(self, linked_vault: Path, profile: str) -> None:
        class IndexStore:
            def __init__(self) -> None:
                self.paths: list[str] = []

            def index_single_note(self, note: object, parser: VaultParser) -> None:
                self.paths.append(str(note.path))  # type: ignore[attr-defined]

        store = IndexStore()
        result = _note_dispatch(
            "vault_write",
            {"path": "02-projects/new/note.md", "content": "# Written\n\nbody"},
            linked_vault,
            ProfileEnforcer(load_profile(profile), linked_vault),
            store,
        )
        assert result == {"status": "ok", "path": "02-projects/new/note.md"}
        assert (linked_vault / result["path"]).read_text() == "# Written\n\nbody"
        assert store.paths == ["02-projects/new/note.md"]

    def test_restricted_search_defensively_rejects_private_hits(self, linked_vault: Path) -> None:
        class IgnoringFilterStore(_RankedStore):
            def search(
                self, query: str, n_results: int = 5, where: dict | None = None
            ) -> list[dict]:
                return self.hits

        valid = _hit("02-projects/open-note.md")
        result = _note_dispatch(
            "vault_search",
            {"query": "query"},
            linked_vault,
            ProfileEnforcer(load_profile("planner"), linked_vault),
            IgnoringFilterStore([_hit("04-resources/private-secret.md"), valid]),
        )
        assert result["status"] == "source-only" and result["approved"] is False
        assert result["count"] == 1
        hit = result["results"][0]
        assert hit["path"] == "02-projects/open-note.md"
        assert hit["content"] == "Open body."
        assert hit["distance"] is None
        assert "private-secret" not in str(result)

    def test_full_suggestion_omits_unverified_provenance(self, linked_vault: Path) -> None:
        from vaultmind.config import NoteSuggestionsConfig
        from vaultmind.indexer.note_suggester import NoteSuggester

        store = _RankedStore(
            [
                _hit("02-projects/missing.md", distance=0.25),
                _hit("../outside.md", distance=0.25),
                _hit("02-projects/open-note.md", distance=0.25),
            ]
        )
        result = _note_dispatch(
            "suggest_links",
            {"note_path": "02-projects/hub.md"},
            linked_vault,
            ProfileEnforcer(load_profile("full"), linked_vault),
            store,
            note_suggester=NoteSuggester(
                NoteSuggestionsConfig(min_content_length=1),
                store,  # type: ignore[arg-type]
            ),
        )
        assert result["count"] == 1
        assert result["suggestions"][0]["path"] == "02-projects/open-note.md"


class TestSourceDiscoveryBoundary:
    def test_scope_before_top_k_current_hydration_and_no_control_history(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        current = "---\ntitle: Current\n---\n\nCurrent text, never retired text.\n"
        for relative, text in {
            "02-projects/current.md": current,
            "04-resources/private.md": "PRIVATE-UNAUTHORIZED",
            "wiki/approved.md": "APPROVED-WIKI-NOT-A-SOURCE",
            "_wiki/history/retired.md": "RETIRED-CONTROL-HISTORY",
        }.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        store = _RankedStore(
            [
                *[_hit("04-resources/private.md") for _ in range(20)],
                _hit("wiki/approved.md"),
                _hit("_wiki/history/retired.md"),
                {**_hit("02-projects/current.md"), "content": "STALE-INDEX-EXCERPT"},
            ]
        )
        before = {
            path.relative_to(tmp_path).as_posix(): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        result = _note_dispatch(
            "vault_search",
            {"query": "Current", "n_results": 1},
            tmp_path,
            ProfileEnforcer(load_profile("planner"), tmp_path),
            store,
        )
        assert result["count"] == 1 and result["approved"] is False
        hit = result["results"][0]
        assert hit["path"] == "02-projects/current.md"
        assert hit["content"] == frontmatter.loads(current).content
        assert hit["current_digest"] == hashlib.sha256(current.encode()).hexdigest()
        assert hit["distance"] is None
        assert hit["source_kind"] == "ordinary"
        assert hit["status"] == "source-only" and hit["approved"] is False
        assert "PRIVATE-UNAUTHORIZED" not in str(result)
        assert "APPROVED-WIKI-NOT-A-SOURCE" not in str(result)
        assert "RETIRED-CONTROL-HISTORY" not in str(result)
        assert "STALE-INDEX-EXCERPT" not in str(result)
        assert store.calls[0][2] == {"note_path": {"$in": ["02-projects/current.md"]}}
        assert before == {
            path.relative_to(tmp_path).as_posix(): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    def test_preserved_dependency_closure_authorized_before_search_output(
        self, tmp_path: Path
    ) -> None:
        from vaultmind.knowledge.intake import SourceIntake

        settings = Settings(_env_file=None, vault=VaultConfig(path=tmp_path))
        with SourceIntake(settings) as intake:
            parent = intake.preserve_text(
                "PRIVATE-PARENT", original_name="parent.md", origin="fixture", adapter="fixture"
            )
            child = intake.preserve_text(
                "searchable child",
                original_name="child.md",
                origin="fixture",
                adapter="fixture",
                parents=(parent.source_id,),
            )
        all_scope = ProfileEnforcer(load_profile("full"), tmp_path)
        allowed = _note_dispatch("vault_search", {"query": "searchable"}, tmp_path, all_scope)
        assert [hit["source_id"] for hit in allowed["results"]] == [child.source_id]
        assert allowed["results"][0]["approved"] is False

        class DenyParent(ProfileEnforcer):
            def allows_path(self, relative: str) -> bool:
                return parent.source_id not in relative and super().allows_path(relative)

        denied = _note_dispatch(
            "vault_search",
            {"query": "searchable"},
            tmp_path,
            DenyParent(load_profile("full"), tmp_path),
        )
        assert denied["results"] == [] and denied["count"] == 0
        assert "searchable child" not in str(denied)
        assert "PRIVATE-PARENT" not in str(denied)

    @pytest.mark.parametrize(
        "relative",
        [
            "wiki/new.md",
            "_wiki/new.md",
            "Wiki/new.md",
            "_WIKI/new.md",
        ],
    )
    def test_ordinary_write_never_enters_reserved_namespace(
        self, tmp_path: Path, relative: str
    ) -> None:
        from vaultmind.vault.security import ReservedNamespaceError

        with pytest.raises(ReservedNamespaceError):
            _note_dispatch(
                "vault_write",
                {"path": relative, "content": "REJECTED"},
                tmp_path,
                ProfileEnforcer(load_profile("full"), tmp_path),
            )
        assert list(tmp_path.iterdir()) == []

    def test_ordinary_symlink_write_cannot_mutate_retained_source(self, tmp_path: Path) -> None:
        from vaultmind.vault.security import ReservedNamespaceError

        path = tmp_path / "_wiki/sources/kept.md"
        path.parent.mkdir(parents=True)
        path.write_text("exact preserved bytes", encoding="utf-8")
        (tmp_path / "02-projects").mkdir()
        (tmp_path / "02-projects/alias.md").symlink_to(path)
        with pytest.raises(ReservedNamespaceError):
            _note_dispatch(
                "vault_write",
                {"path": "02-projects/alias.md", "content": "REJECTED"},
                tmp_path,
                ProfileEnforcer(load_profile("full"), tmp_path),
            )
        assert path.read_text() == "exact preserved bytes"
