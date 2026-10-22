"""Tests for traversal-first MCP tools: read_frontmatter, list_folder_index, follow_links."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vaultmind.config import VaultConfig
from vaultmind.mcp.auth import ProfileEnforcer
from vaultmind.mcp.profiles import load_profile
from vaultmind.mcp.server import _dispatch_tool
from vaultmind.vault.parser import VaultParser

if TYPE_CHECKING:
    from types import ModuleType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Create a fixture vault with frontmatter variety and a nested folder."""
    inbox = tmp_path / "00-inbox"
    inbox.mkdir()
    projects = tmp_path / "02-projects"
    projects.mkdir()
    sub = projects / "sub"
    sub.mkdir()

    (inbox / "fleeting-note.md").write_text(
        "---\n"
        "type: fleeting\n"
        "tags: [test, capture]\n"
        "authority: 2\n"
        "status: active\n"
        "source: manual\n"
        "created: 2026-01-01\n"
        "---\n\n"
        "# Fleeting Note\n\n"
        "First body line as the description fallback.\n\nMore content below.\n",
        encoding="utf-8",
    )
    (projects / "project-alpha.md").write_text(
        "---\n"
        "title: Project Alpha\n"
        "type: project\n"
        "tags: [proj]\n"
        "authority: 4\n"
        "description: Explicit frontmatter description.\n"
        "created: 2026-01-02\n"
        "---\n\n"
        "# Project Alpha\n\n"
        "Body text that must never leak into read_frontmatter or list_folder_index output.\n",
        encoding="utf-8",
    )
    (sub / "nested-note.md").write_text(
        "---\ntype: literature\ntags: [nested]\ncreated: 2026-01-03\n---\n\nNested content.\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def parser(vault_root: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=vault_root))  # type: ignore[arg-type]


class _StubStore:
    pass


class _StubGraph:
    pass


def _full_enforcer(vault_root: Path) -> ProfileEnforcer:
    """Explicit unrestricted policy; the dispatcher has no implicit default enforcer."""
    return ProfileEnforcer(load_profile("full"), vault_root)


def _dispatch(name: str, args: dict, vault_root: Path, parser: VaultParser) -> dict:
    return _dispatch_tool(
        name,
        args,
        vault_root,
        _StubStore(),  # type: ignore[arg-type]
        _StubGraph(),  # type: ignore[arg-type]
        parser,
        enforcer=_full_enforcer(vault_root),
    )


# ---------------------------------------------------------------------------
# read_frontmatter
# ---------------------------------------------------------------------------


class TestReadFrontmatter:
    def test_returns_parsed_frontmatter_fields(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch(
            "read_frontmatter", {"path": "02-projects/project-alpha.md"}, vault_root, parser
        )
        assert result["path"] == "02-projects/project-alpha.md"
        assert result["title"] == "Project Alpha"
        assert result["note_type"] == "project"
        assert result["tags"] == ["proj"]
        assert result["authority"] == 4
        assert result["frontmatter"]["description"] == "Explicit frontmatter description."
        assert "created" in result and "modified" in result

    def test_never_returns_body_text(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch(
            "read_frontmatter", {"path": "02-projects/project-alpha.md"}, vault_root, parser
        )
        assert "content" not in result
        assert "body" not in result
        serialized = str(result)
        assert "Body text that must never leak" not in serialized

    def test_missing_note_returns_error(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch("read_frontmatter", {"path": "00-inbox/nope.md"}, vault_root, parser)
        assert "error" in result

    def test_path_traversal_blocked(self, vault_root: Path, parser: VaultParser) -> None:
        result = _dispatch("read_frontmatter", {"path": "../../etc/passwd"}, vault_root, parser)
        assert "error" in result


# ---------------------------------------------------------------------------
# list_folder_index
# ---------------------------------------------------------------------------


class TestListFolderIndex:
    def test_folder_index_lists_titles_and_descriptions(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "02-projects"}, vault_root, parser)
        by_path = {entry["path"]: entry for entry in result["notes"]}
        alpha = by_path["02-projects/project-alpha.md"]
        assert alpha["title"] == "Project Alpha"
        assert alpha["description"] == "Explicit frontmatter description."
        assert alpha["note_type"] == "project"
        assert "content" not in alpha
        assert "Body text that must never leak" not in str(alpha)

    def test_folder_index_description_falls_back_to_first_body_line(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "00-inbox"}, vault_root, parser)
        entry = result["notes"][0]
        assert entry["description"] == "First body line as the description fallback."

    def test_folder_index_recursive_listing_includes_subfolders(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "02-projects"}, vault_root, parser)
        paths = {entry["path"] for entry in result["notes"]}
        assert "02-projects/sub/nested-note.md" in paths
        assert result["count"] == len(paths)

    def test_folder_index_default_folder_lists_whole_vault(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {}, vault_root, parser)
        assert result["count"] >= 3

    def test_folder_index_not_found_returns_error(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "99-missing"}, vault_root, parser)
        assert "error" in result

    def test_folder_index_path_traversal_blocked(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {"folder": "../../etc"}, vault_root, parser)
        assert "error" in result


# ---------------------------------------------------------------------------
# follow_links
# ---------------------------------------------------------------------------


@pytest.fixture
def linked_vault_root(tmp_path: Path) -> Path:
    """Create a fixture vault with wikilinked notes for follow_links tests."""
    notes_dir = tmp_path / "02-projects"
    notes_dir.mkdir(parents=True)

    (notes_dir / "hub.md").write_text(
        "---\ntitle: Hub Note\ntype: permanent\ncreated: 2026-01-01\n---\n\n"
        "# Hub Note\n\n"
        "Links to [[Spoke A]], [[Spoke B]], and [[Missing Target]].\n",
        encoding="utf-8",
    )
    (notes_dir / "spoke-a.md").write_text(
        "---\ntitle: Spoke A\ntype: permanent\ncreated: 2026-01-02\n---\n\n"
        "# Spoke A\n\nLinks back to [[Hub Note]].\n",
        encoding="utf-8",
    )
    (notes_dir / "spoke-b.md").write_text(
        "---\ntitle: Spoke B\ntype: permanent\ncreated: 2026-01-03\n---\n\n"
        "# Spoke B\n\nNo backlink to the hub.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def linked_parser(linked_vault_root: Path) -> VaultParser:
    return VaultParser(VaultConfig(path=linked_vault_root))  # type: ignore[arg-type]


class TestFollowLinks:
    def test_follow_links_returns_out_links(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        assert result["title"] == "Hub Note"
        out_titles = {link["title"] for link in result["out_links"]}
        assert out_titles == {"Spoke A", "Spoke B", "Missing Target"}
        assert result["out_link_count"] == 3

    def test_follow_links_resolves_out_link_paths(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        by_title = {link["title"]: link["path"] for link in result["out_links"]}
        assert by_title["Spoke A"] == "02-projects/spoke-a.md"
        assert by_title["Missing Target"] is None

    def test_follow_links_returns_backlinks(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        backlink_titles = {link["title"] for link in result["backlinks"]}
        assert backlink_titles == {"Spoke A"}
        assert result["backlink_count"] == 1
        backlink = result["backlinks"][0]
        assert backlink["path"] == "02-projects/spoke-a.md"

    def test_follow_links_never_returns_body_text(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/hub.md"}, linked_vault_root, linked_parser
        )
        assert "content" not in result
        assert "No backlink to the hub" not in str(result)

    def test_follow_links_missing_note_returns_error(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "02-projects/nope.md"}, linked_vault_root, linked_parser
        )
        assert "error" in result

    def test_follow_links_path_traversal_blocked(
        self, linked_vault_root: Path, linked_parser: VaultParser
    ) -> None:
        result = _dispatch(
            "follow_links", {"path": "../../etc/passwd"}, linked_vault_root, linked_parser
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Dispatcher enforcement is mandatory (no implicit default)
# ---------------------------------------------------------------------------


class TestDispatcherRequiresEnforcer:
    def test_traversal_tools_fail_closed_through_the_real_factory_handler(
        self, acceptance: ModuleType, tmp_path: Path
    ) -> None:
        checks: list[dict] = []
        receipt = asyncio.run(
            acceptance.mcp_factory_none_handlers(
                tmp_path.resolve() / "factory", acceptance.make_checker(checks)
            )
        )
        assert set(receipt) == {"read_frontmatter", "list_folder_index", "follow_links"}
        for row in receipt.values():
            assert set(row["enforcer_none_response"]) == {"error"}
            assert row["real_enforcer_positive"][1] is False
        assert receipt["list_folder_index"]["real_enforcer_positive"][0] == sorted(
            acceptance.MCP_NOTES
        )

    def test_note_listing_requires_an_explicit_enforcer(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        from vaultmind.mcp.auth import ProfileError

        with pytest.raises(ProfileError):
            _dispatch_tool(
                "vault_list",
                {"folder": "00-inbox"},
                vault_root,
                _StubStore(),  # type: ignore[arg-type]
                _StubGraph(),  # type: ignore[arg-type]
                parser,
                enforcer=None,
            )

    def test_full_policy_enforcer_preserves_unrestricted_semantics(
        self, vault_root: Path, parser: VaultParser
    ) -> None:
        result = _dispatch("list_folder_index", {}, vault_root, parser)
        assert sorted(entry["path"] for entry in result["notes"]) == [
            "00-inbox/fleeting-note.md",
            "02-projects/project-alpha.md",
            "02-projects/sub/nested-note.md",
        ]
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# Actual subprocess stdio: registered CLI + installed MCP SDK client/server
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_acceptance.py"
SCENARIOS = [
    "planner-scoped",
    "planner-configured-override",
    "researcher-default-control",
    "full-control",
]
ALLOWED_AGGREGATE = [
    "00-inbox/inbox-note.md",
    "02-projects/hub.md",
    "02-projects/peer.md",
    "02-projects/sub/nested.md",
]


@pytest.fixture(scope="module")
def acceptance() -> ModuleType:
    """Load the shared fixture runner from scripts/verify_acceptance.py (mcp case)."""
    spec = importlib.util.spec_from_file_location("vm_verify_acceptance", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stdio_run(acceptance: ModuleType, tmp_path_factory: pytest.TempPathFactory):
    """Run each scenario once as a real subprocess; the runner raises on any mismatch."""
    cache: dict[str, tuple[dict, list[dict], Path]] = {}

    def run(name: str) -> tuple[dict, list[dict], Path]:
        if name not in cache:
            root = tmp_path_factory.mktemp("stdio").resolve() / name
            checks: list[dict] = []
            receipt = asyncio.run(
                acceptance.run_mcp_scenario(name, root, acceptance.make_checker(checks))
            )
            cache[name] = (receipt, checks, root)
        return cache[name]

    return run


def _actual(receipt: dict, tool: str, **arguments: str) -> dict:
    matches = [
        call["actual"]
        for call in receipt["calls"]
        if call["tool"] == tool and call["arguments"] == arguments
    ]
    assert len(matches) == 1, (tool, arguments)
    return matches[0]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stdio_lifecycle_emits_only_valid_frames_offline(scenario: str, stdio_run) -> None:
    receipt, checks, _root = stdio_run(scenario)
    assert checks, "runner recorded no expected/actual assertions"
    lines = receipt["stdout_lines"]
    # initialize + tools/list + one response per tools/call, nothing else on stdout
    assert len(lines) == 2 + len(receipt["calls"])
    for line in lines:
        frame = json.loads(line)
        assert frame["jsonrpc"] == "2.0"
        assert "result" in frame
        assert "error" not in frame
    assert receipt["child_network_attempts"] == []
    exits = [e for e in receipt["launcher_events"] if e["kind"] == "exit"]
    assert [(e["code"], e["tee_joined"]) for e in exits] == [(0, True)]


def test_planner_hub_leaks_no_private_metadata_or_counts(acceptance: ModuleType, stdio_run) -> None:
    receipt, _checks, root = stdio_run("planner-scoped")
    hub = _actual(receipt, "follow_links", path="02-projects/hub.md")
    assert hub["out_links"] == [{"title": "Planner Peer", "path": "02-projects/peer.md"}]
    assert hub["out_link_count"] == 1
    assert sorted(link["path"] for link in hub["backlinks"]) == [
        "00-inbox/inbox-note.md",
        "02-projects/peer.md",
    ]
    assert hub["backlink_count"] == 2
    spec = acceptance.mcp_scenario("planner-scoped", root)
    for call, line in zip(receipt["calls"], receipt["stdout_lines"][2:], strict=True):
        for text in (json.dumps(call["actual"]), line):
            assert acceptance.mcp_leaks(text, spec["forbidden"], call["arguments"]) == []
    assert "HUB-BODY-SENTINEL" not in "\n".join(receipt["stdout_lines"])


def test_planner_root_forms_are_the_allowed_virtual_aggregate(stdio_run) -> None:
    receipt, _checks, root = stdio_run("planner-scoped")
    results = [
        _actual(receipt, "list_folder_index", folder=folder)
        for folder in ("", ".", str(root / "vault"))
    ]
    results.append(_actual(receipt, "list_folder_index"))
    for result in results:
        assert sorted(entry["path"] for entry in result["notes"]) == ALLOWED_AGGREGATE
        assert result["count"] == 4
    nested = _actual(receipt, "list_folder_index", folder="02-projects/sub")
    assert [entry["path"] for entry in nested["notes"]] == ["02-projects/sub/nested.md"]


DIRECTORY_AND_FILE_ALIASES = [
    "02-projects/leak-private.md",
    "02-projects/leak-ext.md",
    "02-projects/linkdir-private",
    "02-projects/linkdir-ext",
    "02-projects/loop-private.md",
    "02-projects/loop-ext.md",
    "02-projects/loopdir-private",
    "02-projects/loopdir-ext",
    "02-projects/self.md",
]


def test_every_private_outside_loop_alias_is_denied_on_payload_and_raw_frames(
    acceptance: ModuleType, stdio_run
) -> None:
    receipt, _checks, root = stdio_run("planner-scoped")
    spec = acceptance.mcp_scenario("planner-scoped", root)
    outside = root / "outside"
    private = root / "vault" / "03-private"
    targets = [
        *DIRECTORY_AND_FILE_ALIASES,
        "03-private",
        "03-private/secret.md",
        "03-private/missing.md",
        "../outside",
        "../outside/outside.md",
        str(outside),
        str(outside / "outside.md"),
        str(private),
        str(private / "secret.md"),
    ]
    frames = receipt["stdout_lines"][2:]
    seen: set[tuple[str, str]] = set()
    for call, frame in zip(receipt["calls"], frames, strict=True):
        arguments = call["arguments"]
        target = arguments.get("path", arguments.get("folder"))
        if target not in targets:
            continue
        seen.add((call["tool"], target))
        assert set(call["actual"]) == {"error"}
        assert isinstance(call["actual"]["error"], str) and call["actual"]["error"]
        for text in (json.dumps(call["actual"]), frame):
            assert acceptance.mcp_leaks(text, spec["forbidden"], arguments) == []
    # Every alias is requested through all three named traversal tools.
    for tool in ("list_folder_index", "read_frontmatter", "follow_links"):
        for alias in DIRECTORY_AND_FILE_ALIASES:
            assert (tool, alias) in seen
    for tool in ("read_frontmatter", "follow_links"):
        for target in targets:
            assert (tool, target) in seen


def test_loop_candidates_do_not_change_the_allowed_aggregate_or_links(stdio_run) -> None:
    receipt, _checks, _root = stdio_run("planner-scoped")
    projects = _actual(receipt, "list_folder_index", folder="02-projects")
    assert sorted(entry["path"] for entry in projects["notes"]) == [
        "02-projects/hub.md",
        "02-projects/peer.md",
        "02-projects/sub/nested.md",
    ]
    assert projects["count"] == 3
    aggregate = _actual(receipt, "list_folder_index")
    assert sorted(entry["path"] for entry in aggregate["notes"]) == ALLOWED_AGGREGATE
    assert aggregate["count"] == 4
    hub = _actual(receipt, "follow_links", path="02-projects/hub.md")
    assert (hub["out_link_count"], hub["backlink_count"]) == (1, 2)
    # An allowed-but-missing note keeps ordinary not-found semantics (error, no data).
    for tool in ("read_frontmatter", "follow_links"):
        assert set(_actual(receipt, tool, path="02-projects/missing.md")) == {"error"}


@pytest.mark.parametrize("scenario", ["researcher-default-control", "full-control"])
def test_wildcard_profiles_survive_loop_candidates_with_exact_counts(
    scenario: str, stdio_run
) -> None:
    receipt, _checks, _root = stdio_run(scenario)
    projects = _actual(receipt, "list_folder_index", folder="02-projects")
    assert projects["count"] == 3
    assert set(_actual(receipt, "read_frontmatter", path="02-projects/self.md")) == {"error"}


def test_read_frontmatter_never_returns_body_text(stdio_run) -> None:
    receipt, _checks, _root = stdio_run("planner-scoped")
    result = _actual(receipt, "read_frontmatter", path="02-projects/hub.md")
    assert sorted(result) == [
        "authority",
        "created",
        "frontmatter",
        "modified",
        "note_type",
        "path",
        "source",
        "status",
        "tags",
        "title",
    ]
    assert result["title"] == "Hub Note"
    assert result["authority"] == 4
    assert "HUB-BODY-SENTINEL" not in json.dumps(result)


def test_configured_planner_override_beats_the_builtin_profile(stdio_run) -> None:
    receipt, _checks, _root = stdio_run("planner-configured-override")
    listing = _actual(receipt, "list_folder_index")
    assert [entry["path"] for entry in listing["notes"]] == ["00-inbox/inbox-note.md"]
    inbox = _actual(receipt, "follow_links", path="00-inbox/inbox-note.md")
    assert (inbox["out_links"], inbox["backlinks"]) == ([], [])
    assert set(_actual(receipt, "read_frontmatter", path="02-projects/hub.md")) == {"error"}
    assert set(_actual(receipt, "vault_search", query="override probe")) == {"error"}
    assert "Profile: planner (Fixture override)" in receipt["stderr"]


@pytest.mark.parametrize("scenario", ["researcher-default-control", "full-control"])
def test_researcher_and_full_access_are_unchanged(scenario: str, stdio_run) -> None:
    receipt, _checks, _root = stdio_run(scenario)
    listing = _actual(receipt, "list_folder_index")
    assert listing["count"] == 5
    assert "03-private/secret.md" in {entry["path"] for entry in listing["notes"]}
    hub = _actual(receipt, "follow_links", path="02-projects/hub.md")
    by_title = {link["title"]: link["path"] for link in hub["out_links"]}
    assert by_title == {
        "Planner Peer": "02-projects/peer.md",
        "Private Secret": "03-private/secret.md",
        "Nowhere Target": None,
    }
    assert hub["out_link_count"] == 3
    assert hub["backlink_count"] == 3
    private = _actual(receipt, "read_frontmatter", path="03-private/secret.md")
    assert private["title"] == "Private Secret"


def test_direct_dispatcher_matrix_with_explicit_enforcers(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    checks: list[dict] = []
    acceptance.mcp_direct_dispatcher(tmp_path.resolve() / "direct", acceptance.make_checker(checks))


def test_leak_scanner_masks_only_complete_exact_supplied_values(acceptance: ModuleType) -> None:
    leaks = acceptance.mcp_leaks
    oracle = ["secret.md", "03-private/secret.md", "/x/outside", "/x/outside/outside.md"]
    # The caller's exact full input may be echoed.
    assert leaks("Path '03-private/secret.md' x", oracle, {"path": "03-private/secret.md"}) == []
    # A known parent never masks the segment inside an unknown longer target.
    assert "03-private/secret.md" in leaks(
        "Path '03-private/secret.md' x", oracle, {"folder": "03-private"}
    )
    assert "secret.md" in leaks("a secret.md b", oracle, {"folder": "."})
    # A supplied relative path never masks a longer absolute target containing it.
    assert "secret.md" in leaks(
        "/abs/vault/03-private/secret.md", oracle, {"path": "03-private/secret.md"}
    )
    # An unknown resolved outside target stays visible next to a supplied alias.
    assert "/x/outside/outside.md" in leaks(
        "resolved /x/outside/outside.md", oracle, {"path": "02-projects/leak-ext.md"}
    )
    # Supplying the absolute file masks only that exact token, not the bare root.
    assert leaks(
        "'/x/outside/outside.md' and /x/outside", oracle, {"path": "/x/outside/outside.md"}
    ) == ["/x/outside"]


# ---------------------------------------------------------------------------
# Ordinary strict-resolution failures through the registered CLI + installed SDK stdio
# ---------------------------------------------------------------------------

LONG_COMPONENT = "z" * 300
TRAVERSAL_TOOLS = (
    ("read_frontmatter", "path"),
    ("follow_links", "path"),
    ("list_folder_index", "folder"),
)
WILDCARD_SCENARIOS = ["researcher-default-control", "full-control"]
PRIVATE_SUFFIX_ALIASES = [
    "02-projects/leak-private.md/x",
    "02-projects/linkdir-private/secret.md/x",
    "02-projects/linkdir-private/" + LONG_COMPONENT,
]
OUTSIDE_SUFFIX_ALIASES = [
    "02-projects/leak-ext.md/x",
    "02-projects/linkdir-ext/outside.md/x",
]
SEALED_PRIVATE_ALIASES = [
    "02-projects/lockdir-private",
    "02-projects/lockdir-private/hidden-priv.md",
    "02-projects/lockfile-private.md",
]
SEALED_OUTSIDE_ALIASES = [
    "02-projects/lockdir-ext",
    "02-projects/lockdir-ext/hidden-out.md",
    "02-projects/lockfile-ext.md",
]
# Basenames/sentinels of hidden targets; a caller can only echo them inside its own exact path.
HIDDEN_BASENAMES = [
    "outside.md",
    "03-private",
    "sealed-priv",
    "sealed-out",
    "hidden-priv",
    "hidden-out",
    "PRIVATE-SEALED-SENTINEL",
    "OUTSIDE-SEALED-SENTINEL",
    "OUTSIDE-SENTINEL",
]


def _record(receipt: dict, tool: str, **arguments: str) -> dict:
    matches = [
        call for call in receipt["calls"] if call["tool"] == tool and call["arguments"] == arguments
    ]
    assert len(matches) == 1, (tool, arguments)
    return matches[0]


def _surfaces(receipt: dict, record: dict) -> list[str]:
    """Payload, raw stdout frame and audit row of one call."""
    raw = receipt["stdout_lines"][2 + receipt["calls"].index(record)]
    return [json.dumps(record["actual"]), raw, json.dumps(record["audit"])]


def _error_only(record: dict) -> str:
    assert set(record["actual"]) == {"error"}
    error = record["actual"]["error"]
    assert isinstance(error, str) and error
    return error


def _require_permission_denial(acceptance: ModuleType) -> None:
    if not acceptance.mcp_permission_denial_enforced():
        pytest.skip("mode-000 directories do not deny lookups here (root or permissionless mount)")


def test_suffix_overlong_and_sealed_aliases_fail_as_policy_not_as_physical_errors(
    acceptance: ModuleType, stdio_run
) -> None:
    receipt, _checks, root = stdio_run("planner-scoped")
    private = list(PRIVATE_SUFFIX_ALIASES)
    outside = list(OUTSIDE_SUFFIX_ALIASES)
    if acceptance.mcp_permission_denial_enforced():
        private += SEALED_PRIVATE_ALIASES
        outside += SEALED_OUTSIDE_ALIASES
    cases = [(target, "DENIED") for target in private] + [(target, "OK") for target in outside]
    for target, index_audit in cases:
        for tool, key in TRAVERSAL_TOOLS:
            arguments = {key: target}
            record = _record(receipt, tool, **arguments)
            error = _error_only(record)
            expected_audit = index_audit if tool == "list_folder_index" else "DENIED"
            assert record["audit"]["result"] == expected_audit, (tool, target)
            # The error names only the caller's own relative path ...
            assert acceptance.mcp_path_residue(error, arguments) == [], (tool, target)
            for text in _surfaces(receipt, record):
                # ... so no absolute location of any kind appears on any surface.
                assert str(root) not in text, (tool, target)
                assert acceptance.mcp_leaks(text, HIDDEN_BASENAMES, arguments) == [], (tool, target)


@pytest.mark.parametrize("scenario", ["planner-scoped", *WILDCARD_SCENARIOS])
def test_allowed_file_children_and_missing_directories_stay_caller_path_not_found(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    receipt, _checks, root = stdio_run(scenario)
    targets = ["02-projects/hub.md/child", "02-projects/missing-dir/child.md"]
    if scenario != "planner-scoped":
        # Wildcard profiles may read the private note, so a file used as a directory behind a
        # directory alias is an ordinary not-found there (the planner denies it by policy).
        targets.append("02-projects/linkdir-private/secret.md/x")
    for target in targets:
        for tool, key in TRAVERSAL_TOOLS:
            arguments = {key: target}
            record = _record(receipt, tool, **arguments)
            error = _error_only(record)
            # An ordinary lookup failure is neither a typed denial nor a server error.
            assert record["audit"]["result"] == "OK", (tool, target)
            assert record["audit"]["reason"] == ""
            assert acceptance.mcp_path_residue(error, arguments) == [], (tool, target)
            for text in _surfaces(receipt, record):
                assert str(root) not in text, (tool, target)


@pytest.mark.parametrize("scenario", WILDCARD_SCENARIOS)
def test_wildcard_outside_directory_alias_with_suffix_is_contained(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    receipt, _checks, root = stdio_run(scenario)
    target = "02-projects/linkdir-ext/outside.md/x"
    for tool, key in TRAVERSAL_TOOLS:
        arguments = {key: target}
        record = _record(receipt, tool, **arguments)
        error = _error_only(record)
        assert record["audit"]["result"] == ("OK" if tool == "list_folder_index" else "DENIED")
        assert acceptance.mcp_path_residue(error, arguments) == []
        for text in _surfaces(receipt, record):
            assert str(root / "outside") not in text


@pytest.mark.parametrize("scenario", ["planner-scoped", *WILDCARD_SCENARIOS])
def test_overlong_component_returns_an_ordinary_error_without_unknown_targets(
    scenario: str, stdio_run
) -> None:
    receipt, _checks, root = stdio_run(scenario)
    hidden = [str(root / "outside")]
    if scenario == "planner-scoped":
        hidden.append(str(root / "vault" / "03-private"))
    for tool, key in TRAVERSAL_TOOLS:
        record = _record(receipt, tool, **{key: "02-projects/" + LONG_COMPONENT + ".md"})
        _error_only(record)
        # The real syscall error may surface, but never as a typed denial of an allowed path.
        assert record["audit"]["result"] in ("OK", "ERROR")
        for text in _surfaces(receipt, record):
            for token in hidden:
                assert token not in text


@pytest.mark.parametrize("scenario", ["planner-scoped", *WILDCARD_SCENARIOS])
def test_permission_denied_public_directory_is_empty_and_its_children_are_ordinary_errors(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    _require_permission_denial(acceptance)
    receipt, _checks, _root = stdio_run(scenario)
    listing = _actual(receipt, "list_folder_index", folder="02-projects/sealed-pub")
    assert listing == {"folder": "02-projects/sealed-pub", "notes": [], "count": 0}
    for tool, key in TRAVERSAL_TOOLS:
        record = _record(receipt, tool, **{key: "02-projects/sealed-pub/hidden-pub.md"})
        _error_only(record)
        assert record["audit"]["result"] in ("OK", "ERROR")


@pytest.mark.parametrize("scenario", WILDCARD_SCENARIOS)
def test_unreadable_private_candidates_do_not_abort_wildcard_reads_or_counts(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    _require_permission_denial(acceptance)
    receipt, _checks, root = stdio_run(scenario)
    for target in (
        "02-projects/lockdir-private/hidden-priv.md",
        "02-projects/lockfile-private.md",
    ):
        for tool, key in TRAVERSAL_TOOLS:
            record = _record(receipt, tool, **{key: target})
            _error_only(record)
            assert record["audit"]["result"] in ("OK", "ERROR")
    assert _actual(receipt, "list_folder_index", folder="02-projects/lockdir-private")["count"] == 0
    # Unreadable candidates sit in the vault, yet every valid count is unchanged.
    assert _actual(receipt, "list_folder_index")["count"] == 5
    assert _actual(receipt, "list_folder_index", folder=str(root / "vault"))["count"] == 5
    assert _actual(receipt, "list_folder_index", folder="02-projects")["count"] == 3
    hub = _actual(receipt, "follow_links", path="02-projects/hub.md")
    assert (hub["out_link_count"], hub["backlink_count"]) == (3, 3)


def test_unreadable_candidates_do_not_abort_planner_root_list_or_links(
    acceptance: ModuleType, stdio_run
) -> None:
    _require_permission_denial(acceptance)
    receipt, _checks, _root = stdio_run("planner-scoped")
    assert receipt["permissions"]["denial_enforced"] is True
    aggregate = _actual(receipt, "list_folder_index")
    assert sorted(entry["path"] for entry in aggregate["notes"]) == ALLOWED_AGGREGATE
    assert aggregate["count"] == 4
    assert _actual(receipt, "list_folder_index", folder="02-projects")["count"] == 3
    hub = _actual(receipt, "follow_links", path="02-projects/hub.md")
    assert (hub["out_link_count"], hub["backlink_count"]) == (1, 2)
    assert _actual(receipt, "read_frontmatter", path="02-projects/hub.md")["title"] == "Hub Note"


def test_sealed_permission_fixtures_are_restored_for_deletion(
    acceptance: ModuleType, stdio_run
) -> None:
    receipt, _checks, root = stdio_run("planner-scoped")
    sealed = receipt["permissions"]["sealed_directories"]
    if not acceptance.mcp_permission_denial_enforced():
        assert sealed == []
        return
    assert sorted(sealed) == [
        "outside/sealed-out",
        "vault/02-projects/sealed-pub",
        "vault/03-private/sealed-priv",
    ]
    for relative in sealed:
        assert (root / relative).stat().st_mode & 0o700 == 0o700


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_call_carries_independent_oracle_evidence_on_all_surfaces(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    receipt, _checks, root = stdio_run(scenario)
    spec = acceptance.mcp_scenario(scenario, root)
    assert len(receipt["audit"]) == len(receipt["calls"]) > 0
    outside = root / "outside"
    private = root / "vault" / "03-private"
    assert str(outside / "outside.md") in spec["forbidden"]
    assert "outside.md" in spec["forbidden"]
    if scenario in ("planner-scoped", "planner-configured-override"):
        assert str(private / "secret.md") in spec["forbidden"]
        assert "03-private/secret.md" in spec["forbidden"]
    for record in receipt["calls"]:
        oracle = record["oracle"]
        assert oracle["forbidden_tokens"] == len(spec["forbidden"])
        assert oracle["payload_leaks"] == []
        assert oracle["raw_frame_leaks"] == []
        assert record["audit"]["private_fields_leaks"] == []
    stderr = receipt["stderr_oracle"]
    assert stderr["forbidden"], "stderr scan must not run with an empty oracle"
    assert stderr["leaks"] == []
    assert stderr["caller_supplied_overlap"] == []
    # Independent of the scenario's own token list: unknown physical targets, no masking.
    unknown = [
        str(outside / "loop-a.md"),
        str(outside / "outside.md") + "/x",
        str(outside / "sealed-out"),
    ]
    if scenario in ("planner-scoped", "planner-configured-override"):
        unknown += [
            str(private / "secret.md") + "/x",
            str(private / "loop-a.md"),
            str(private / "sealed-priv"),
        ]
    flat = acceptance.mcp_flat(receipt["stderr"])
    for token in unknown:
        assert acceptance.mcp_flat(token) not in flat, token


def test_path_residue_and_flat_helpers_apply_no_cross_call_exemption(
    acceptance: ModuleType,
) -> None:
    residue = acceptance.mcp_path_residue
    own = {"path": "02-projects/hub.md/child"}
    assert residue("Note not found: 02-projects/hub.md/child", own) == []
    # A physical location the caller never supplied leaves a separator behind ...
    assert residue("Not a directory: '/abs/vault/03-private/secret.md/x'", own) == ["/"]
    # ... and an argument of ANOTHER call never exempts it; only this call's exact path does.
    assert residue("Not a directory: '/abs/vault/03-private/x'", {"path": "/other"}) == ["/"]
    unreadable = "Not a directory: '/abs/vault/03-private/x'"
    assert residue(unreadable, {"path": "/abs/vault/03-private/x"}) == []
    wrapped = "/tmp/a/03-pri\n    vate/x.md   server.py:523"
    assert acceptance.mcp_flat(wrapped) == "/tmp/a/03-private/x.md"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_stderr_chunk_is_scanned_with_all_absolute_targets_and_only_its_own_arguments(
    acceptance: ModuleType, scenario: str, stdio_run
) -> None:
    receipt, _checks, root = stdio_run(scenario)
    private = root / "vault" / "03-private"
    outside = root / "outside"
    roots = [str(outside), str(outside / "outside.md"), str(outside / "sealed-out")]
    if scenario in ("planner-scoped", "planner-configured-override"):
        roots += [str(private), str(private / "secret.md"), str(private / "sealed-priv")]
    assert receipt["stderr_oracle"]["startup_leaks"] == []
    for record in receipt["calls"]:
        chunk = record["stderr"]["chunk"]
        # Every absolute root/file target is forbidden in every chunk, even when ANOTHER
        # call supplied it; only this call's exact arguments are masked first.
        flat = acceptance.mcp_flat(acceptance.mcp_mask(chunk, record["arguments"]))
        assert [token for token in roots if acceptance.mcp_flat(token) in flat] == []
        assert record["stderr"]["private_fields_leaks"] == []
        assert record["stderr"]["absolute_target_leaks"] == []
