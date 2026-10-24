"""Credential hygiene at real offline consumers, including SDK baseline merge-back."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_acceptance.py"
CONTROLS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "VOYAGE_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "VAULTMIND_OPENAI_API_KEY",
    "VAULTMIND_TELEGRAM__BOT_TOKEN",
    "VAULTMIND_HOME",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_ASKPASS",
    "GIT_SSH_COMMAND",
    "SSH_ASKPASS",
    "SSH_AUTH_SOCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SHARED_CREDENTIALS_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_SECRET",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "LOGNAME",
    "SHELL",
    "TERM",
    "USER",
)
SENTINELS = {key: f"M28-HARMLESS-{key}-4cb0f1" for key in CONTROLS}

# This runs the reviewed script, not a replacement consumer. Observe real Python
# calls and real child spawns; retain only names/booleans, never credential values.
OBSERVER = r"""
import asyncio
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

script, directory, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
controls = json.loads(sys.argv[4])
injected = {key: os.environ[key] for key in controls if key in os.environ}
original_environment = dict(os.environ)
observations = []
scans = []
artifacts = directory / "artifacts"
artifacts.mkdir()

def observe(kind, environment):
    observations.append({
        "kind": kind,
        "keys": sorted(environment),
        "home": environment.get("HOME"),
        "forbidden": sorted(key for key, value in injected.items()
                            if environment.get(key) == value),
    })

def audit(event, args):
    if event == "subprocess.Popen":
        observe("subprocess.Popen", os.environ if args[3] is None else args[3])

sys.addaudithook(audit)

def scan(root):
    files, leaks = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        relative = str(path.relative_to(root))
        files.append(relative)
        for key, value in injected.items():
            if value.encode() in data:
                leaks.append([relative, key])
    scans.append({"files": files, "leaks": leaks})

original_temporary_directory = tempfile.TemporaryDirectory
class ScannedTemporaryDirectory(original_temporary_directory):
    def __init__(self, *args, **kwargs):
        if kwargs.get("prefix", "").startswith("vm-"):
            kwargs["dir"] = str(artifacts)
        super().__init__(*args, **kwargs)
    def __exit__(self, *args):
        try:
            scan(Path(self.name))
        except BaseException:
            super().__exit__(*sys.exc_info())
            raise
        return super().__exit__(*args)

def profile(frame, event, arg):
    if event != "call":
        return
    name = frame.f_code.co_name
    if name not in {"load_settings", "handle_recall", "run_bench", "think",
                    "_dispatch_tool", "approve", "_embed_texts_uncached"}:
        return
    module = frame.f_globals.get("__name__", "")
    if module.startswith("vaultmind."):
        observe(module + "." + name, os.environ)

# Observe the *actual fresh Chroma reader*, keeping its real -I process, persisted
# collection and full output. The added prelude records only control names.
original_run = subprocess.run
reader_report = directory / "reader-environments.jsonl"
def observed_run(command, *args, **kwargs):
    if (isinstance(command, list) and len(command) > 3 and command[1:3] == ["-I", "-c"]
            and "collection.get(include=" in command[3]):
        prelude = (
            "import json, os\n"
            + "with open(" + repr(str(reader_report)) + ", 'a') as observed:\n"
            + "    observed.write(json.dumps({'keys': sorted(os.environ), "
            + "'home': os.environ.get('HOME'), 'forbidden': sorted(set(os.environ) & "
            + repr(set(controls)) + ")}) + '\\n')\n"
        )
        command = [*command[:3], prelude + command[3], *command[4:]]
    return original_run(command, *args, **kwargs)

subprocess.run = observed_run
tempfile.TemporaryDirectory = ScannedTemporaryDirectory
if mode in {"retrieval", "mcp", "approval", "config"}:
    sys.setprofile(profile)
    threading.setprofile(profile)
try:
    if mode in {"retrieval", "mcp", "approval", "config"}:
        sys.argv = [str(script), "--case", mode, "--offline", "--receipt-out",
                    str(directory / "receipt.json")]
        runpy.run_path(str(script), run_name="__main__")
    else:
        spec = importlib.util.spec_from_file_location("credential_acceptance", script)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        checks = []
        check = verifier.make_checker(checks)
        root = artifacts / "direct"
        if mode == "direct":
            result = asyncio.run(verifier.run_mcp_scenario("root-scoped-small", root, check))
            # A real post-SDK writer followed by a fresh persisted reader must see
            # vectors, documents and metadata, not a stale in-process HNSW view.
            _, fixture, _ = verifier.mcp_root_fixture(artifacts / "snapshot", "root-scoped-small")
            before = verifier.mcp_root_snapshot(fixture)
            fixture["store"]._collection.upsert(
                ids=["credential-boundary-write"], documents=["Persisted boundary document"],
                metadatas=[{"note_path": "02-projects/boundary.md", "authority": 5}],
                embeddings=[[0.25, 0.75]],
            )
            after = verifier.mcp_root_snapshot(fixture)
            check("snapshot.persisted-write-full-record", sorted(before["index"] + [[
                "credential-boundary-write", "Persisted boundary document",
                {"note_path": "02-projects/boundary.md", "authority": 5}, [0.25, 0.75],
            ]]), after["index"])
            check("snapshot.unrelated-files-preserved", before["vault"], after["vault"])
            (directory / "receipt.json").write_text(
                json.dumps({"checks": checks, "scenario": result})
            )
        elif mode == "sdk-failure":
            root.mkdir()
            try:
                asyncio.run(
                    verifier.mcp_stdio_session(root, ["--not-a-real-option"], [], deadline=15)
                )
            except Exception:
                events = [json.loads(line) for line in
                          (root / "launcher-state/launcher.jsonl").read_text().splitlines()]
                assert any(event["kind"] == "console-script" for event in events)
                (directory / "receipt.json").write_text(json.dumps({"events": events}))
            else:
                raise AssertionError("Invalid actual CLI invocation unexpectedly succeeded")
        elif mode == "sdk-missing-path":
            root.mkdir()
            asyncio.run(verifier.mcp_stdio_session(root, ["--help"], []))
        elif mode == "snapshot-missing-path":
            root.mkdir()
            (root / "vault").mkdir()
            (root / "outside").mkdir()
            from types import SimpleNamespace
            verifier.mcp_root_snapshot({
                "vault": root / "vault", "outside": root / "outside",
                "store": SimpleNamespace(config=SimpleNamespace(
                    persist_dir=root / "chroma", collection_name="missing-path")),
            })
        elif mode == "binding-missing-path":
            verifier.approval_git_binding(root)
        scan(artifacts)
finally:
    sys.setprofile(None)
    threading.setprofile(None)
    subprocess.run = original_run
    tempfile.TemporaryDirectory = original_temporary_directory
    (directory / "observations.json").write_text(json.dumps({
        "consumers": observations, "scans": scans,
        "environment_restored": dict(os.environ) == original_environment,
    }))
"""


def run_observed(
    directory: Path, mode: str, *, path: str | None = "runtime"
) -> tuple[subprocess.CompletedProcess[str], dict, list[dict]]:
    """Run real consumers with synthetic ingress and a forwarding real-Git observer."""
    home = directory / "caller-home"
    home.mkdir()
    binary = directory / "bin"
    binary.mkdir()
    real_git = shutil.which("git")
    assert real_git is not None
    git_report = directory / "git-environments.jsonl"
    # Observe the effective child environment before execing the real Git binary.
    # No fake Git responses or credential helper execution.
    wrapper = binary / "git"
    wrapper.write_text(
        f"#!{sys.executable} -I\n"
        "import json, os, sys\n"
        f"with open({str(git_report)!r}, 'a') as report:\n"
        "    report.write(json.dumps({'argv': sys.argv[1:], 'keys': sorted(os.environ), "
        "'home': os.environ.get('HOME'), 'forbidden': sorted(set(os.environ) & "
        f"{set(CONTROLS)!r})}}) + '\\n')\n"
        f"os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    environment = {"HOME": str(home), **SENTINELS}
    if path is not None:
        environment["PATH"] = (
            str(binary) + os.pathsep + os.environ["PATH"] if path == "runtime" else path
        )
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            OBSERVER,
            str(SCRIPT),
            str(directory),
            mode,
            json.dumps(CONTROLS),
        ],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    observations = json.loads((directory / "observations.json").read_text())
    git = (
        [json.loads(line) for line in git_report.read_text().splitlines()]
        if git_report.exists()
        else []
    )
    return done, observations, git


def assert_no_leaks(
    directory: Path, done: subprocess.CompletedProcess[str], observations: dict
) -> None:
    """Scan retained output and all surviving binary/text files without redacting leaks."""
    for key, sentinel in SENTINELS.items():
        if sentinel in done.stdout or sentinel in done.stderr:
            pytest.fail(f"Consumer output leaked control: {key}")
    for file in directory.rglob("*"):
        if file.is_symlink() or not file.is_file():
            continue
        data = file.read_bytes()
        for key, sentinel in SENTINELS.items():
            if sentinel.encode() in data:
                pytest.fail(f"Consumer artifact leaked control: {key} in {file.name}")
    assert all(scan["leaks"] == [] for scan in observations["scans"])
    assert all(consumer["forbidden"] == [] for consumer in observations["consumers"])
    assert observations["environment_restored"] is True


@pytest.mark.parametrize("case", ["retrieval", "mcp", "approval", "config"])
def test_actual_entrypoints_exclude_ambient_credentials(tmp_path: Path, case: str) -> None:
    done, observations, git = run_observed(tmp_path, case)
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode == 0, done.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["result"] == "pass"
    assert receipt["artifact"]["runtime_matches_reviewed"] is True
    assert receipt["artifact"]["runtime"]["interpreter"]["executable"] == sys.executable
    assert receipt["network_attempts"] == []
    outcomes = {row["id"]: row["actual"] for row in receipt["checks"]}
    assert all(row["expected"] == row["actual"] for row in receipt["checks"])
    required_consumer = (
        "vaultmind.mcp.server._dispatch_tool" if case == "mcp" else "vaultmind.config.load_settings"
    )
    assert any(row["kind"] == required_consumer for row in observations["consumers"])
    assert all(row["home"] != str(tmp_path / "caller-home") for row in observations["consumers"])
    commands = [row["argv"][2:] for row in git]
    assert ["rev-parse", "HEAD"] in commands
    assert ["status", "--porcelain"] in commands
    assert all(row["forbidden"] == [] for row in git)
    files = {name for scan in observations["scans"] for name in scan["files"]}
    assert any(name.endswith("chroma.sqlite3") for name in files)
    if case == "retrieval":
        assert outcomes["vector.authority-reorder"] == ["alpha.md", "beta.md"]
        assert outcomes["hybrid.authority-reorder"] == ["alpha.md", "beta.md"]
        assert outcomes["vector.empty.gap"] == ["weak_retrieval"]
        assert outcomes["hybrid.lexical.confidence"] == "unknown"
    elif case == "approval":
        assert outcomes["migration.legacy-schema-from-git"] is True
        assert outcomes["migration.explicit-retry-of-known-failure-applies"] == [
            "applied",
            ["lf"],
        ]
        assert outcomes["migration.legacy-pending-approves-with-real-applier"] == [
            "applied",
            ["lp"],
        ]
        assert any(command[0] == "log" for command in commands)
        assert any(command[0] == "show" for command in commands)
        assert any(command[0] == "rev-list" for command in commands)
    elif case == "config":
        assert outcomes["config.consumer.chroma-document-count"] == 3
        assert outcomes["config.cli.invalid-env-rejected"] is True
        assert outcomes["config.isolation.sentinel-unchanged"] is True
        assert outcomes["config.mcp.false.no-persisted-write"] is False
    else:
        for scenario in receipt["scenarios"].values():
            environment = next(
                event for event in scenario["launcher_events"] if event["kind"] == "environment"
            )
            assert environment["home"] != str(tmp_path / "caller-home")
        events = [
            event
            for scenario in receipt["scenarios"].values()
            for event in scenario["launcher_events"]
            if event["kind"] == "environment"
        ]
        assert all(event["forbidden"] == [] for event in events)
        assert all(not set(CONTROLS).intersection(event["keys"]) for event in events)
        assert any(name.endswith("launcher.jsonl") for name in files)
        assert any(name.endswith("stdout.raw") for name in files)
        assert any("audit" in name for name in files)
        readers = [
            json.loads(line)
            for line in (tmp_path / "reader-environments.jsonl").read_text().splitlines()
        ]
        assert all(reader["forbidden"] == [] for reader in readers)


def test_direct_sdk_and_snapshot_keep_real_state_without_baseline_merge(tmp_path: Path) -> None:
    done, observations, _ = run_observed(tmp_path, "direct")
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode == 0, done.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert all(row["expected"] == row["actual"] for row in receipt["checks"])
    scenario = receipt["scenario"]
    environment = next(
        event for event in scenario["launcher_events"] if event["kind"] == "environment"
    )
    assert environment["forbidden"] == []
    assert not set(CONTROLS).intersection(environment["keys"])
    readers = [
        json.loads(line)
        for line in (tmp_path / "reader-environments.jsonl").read_text().splitlines()
    ]
    assert len(readers) == 4  # scenario before/after plus independent persisted mutation
    assert all(reader["forbidden"] == [] for reader in readers)
    assert all(reader["home"] != str(tmp_path / "caller-home") for reader in readers)
    assert any(row["id"] == "snapshot.persisted-write-full-record" for row in receipt["checks"])


def test_sdk_failure_restores_parent_after_real_console_execution(tmp_path: Path) -> None:
    done, observations, _ = run_observed(tmp_path, "sdk-failure")
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode == 0, done.stderr
    events = json.loads((tmp_path / "receipt.json").read_text())["events"]
    environment = next(event for event in events if event["kind"] == "environment")
    assert environment["forbidden"] == []
    assert any(event["kind"] == "console-script" for event in events)


@pytest.mark.parametrize("path", [None, ""])
def test_entrypoint_missing_runtime_path_fails_explicitly(tmp_path: Path, path: str | None) -> None:
    done, observations, git = run_observed(tmp_path, "retrieval", path=path)
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode != 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["result"] == "fail"
    assert "PATH" in receipt["failure"]
    assert git == []


def test_nonempty_path_without_git_fails_the_actual_entrypoint(tmp_path: Path) -> None:
    done, observations, git = run_observed(tmp_path, "retrieval", path=str(tmp_path / "absent-bin"))
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode != 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["result"] == "fail"
    assert "FileNotFoundError" in receipt["failure"]
    assert "git" in receipt["failure"]
    assert git == []


@pytest.mark.parametrize(
    "mode", ["sdk-missing-path", "snapshot-missing-path", "binding-missing-path"]
)
def test_direct_spawns_missing_path_do_not_succeed_or_become_optional_refs(
    tmp_path: Path, mode: str
) -> None:
    done, observations, git = run_observed(tmp_path, mode, path=None)
    assert_no_leaks(tmp_path, done, observations)
    assert done.returncode != 0
    assert "Missing required offline runtime input: PATH" in done.stderr
    assert git == []
