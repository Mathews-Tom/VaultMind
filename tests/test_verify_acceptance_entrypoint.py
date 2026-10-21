"""Exercise the reviewed verifier from foreign cwd and isolated CLI environments."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_acceptance.py"


@pytest.fixture
def reviewed_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "reviewed"
    subprocess.run(
        ["git", "clone", "--quiet", "--local", "--no-hardlinks", str(ROOT), str(clone)],
        check=True,
        capture_output=True,
    )
    # Include the candidate script before its commit, without copying ignored state.
    shutil.copyfile(SCRIPT, clone / "scripts" / SCRIPT.name)
    return clone


def run_case(script: Path, directory: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    receipt = directory / "receipt.json"
    env = dict(os.environ, HOME=str(directory), VAULTMIND_HOME=str(directory / "unused"))
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--case",
            "retrieval",
            "--offline",
            "--receipt-out",
            str(receipt),
        ],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return done, json.loads(receipt.read_text())


@pytest.mark.parametrize("missing", ["DEVELOPMENT_PLAN.md", "EXECUTION_PROMPTS.md"])
def test_missing_required_contract_fails(
    reviewed_clone: Path, tmp_path: Path, missing: str
) -> None:
    (reviewed_clone / ".docs" / missing).unlink()
    done, receipt = run_case(reviewed_clone / "scripts" / SCRIPT.name, tmp_path)
    assert done.returncode != 0
    assert receipt["result"] == "fail"
    assert "FileNotFoundError" in receipt["failure"]
    assert str(reviewed_clone / ".docs" / missing) in receipt["failure"]
    assert receipt["checks"] == []


def test_disagreeing_contracts_fail(reviewed_clone: Path, tmp_path: Path) -> None:
    contract = reviewed_clone / ".docs" / "EXECUTION_PROMPTS.md"
    contract.write_text(contract.read_text().replace("Plan ID: `", "Plan ID: `different-", 1))
    done, receipt = run_case(reviewed_clone / "scripts" / SCRIPT.name, tmp_path)
    assert done.returncode != 0
    assert receipt["result"] == "fail"
    assert "Contract Plan IDs disagree" in receipt["failure"]
    assert receipt["checks"] == []


def test_different_imported_code_is_not_accepted_as_reviewed(
    reviewed_clone: Path, tmp_path: Path
) -> None:
    source = reviewed_clone / "src" / "vaultmind" / "__init__.py"
    source.write_text(source.read_text() + "\n# Different reviewed package.\n")
    done, receipt = run_case(reviewed_clone / "scripts" / SCRIPT.name, tmp_path)
    assert done.returncode != 0
    assert receipt["result"] == "fail"
    assert "does not match the reviewed source" in receipt["failure"]
    assert receipt["artifact"]["runtime_matches_reviewed"] is False
    assert (
        receipt["artifact"]["runtime"]["code_sha256"]
        != receipt["artifact"]["reviewed_package_sha256"]
    )
    assert receipt["checks"] == []


def test_retrieval_from_foreign_cwd_keeps_state_isolated(
    reviewed_clone: Path, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[vault]\npath = "must-not-use-cwd-config"\n')
    before = config.read_bytes()
    done, receipt = run_case(reviewed_clone / "scripts" / SCRIPT.name, tmp_path)
    assert done.returncode == 0, done.stderr
    assert receipt["result"] == "pass"
    outcomes = {check["id"]: check["actual"] for check in receipt["checks"]}
    assert outcomes["vector.authority-reorder"] == ["alpha.md", "beta.md"]
    assert outcomes["hybrid.authority-reorder"] == ["alpha.md", "beta.md"]
    assert outcomes["vector.empty.gap"] == ["weak_retrieval"]
    assert outcomes["hybrid.lexical.confidence"] == "unknown"
    assert receipt["network_attempts"] == []
    assert config.read_bytes() == before
    assert not (tmp_path / "must-not-use-cwd-config").exists()
    assert not (tmp_path / "unused").exists()
    assert receipt["artifact"]["runtime_matches_reviewed"] is True
    assert receipt["artifact"]["runtime"]["interpreter"]["executable"] == sys.executable


def test_mcp_launcher_consumes_sibling_script_without_resolving_interpreter(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("acceptance_entrypoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    verifier: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    # A synthetic venv reuses installed dependencies, not an injected checkout path.
    venv = tmp_path / "venv"
    binary = venv / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.symlink_to(sys.executable)
    shutil.copyfile(Path(sys.prefix) / "pyvenv.cfg", venv / "pyvenv.cfg")
    library = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    library.mkdir(parents=True)
    (library / "site-packages").symlink_to(sysconfig.get_path("purelib"), target_is_directory=True)
    sibling = binary / "vaultmind"
    original = Path(sys.executable).parent / "vaultmind"
    # Change the *actual consumed script*: Click must see --version, not requested --help.
    sibling.write_text('import sys\nsys.argv[1:] = ["--version"]\n' + original.read_text())
    state = tmp_path / "state"
    state.mkdir()
    done = subprocess.run(
        [str(python), "-I", "-c", verifier.MCP_LAUNCHER, str(state), "--help"],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "ANONYMIZED_TELEMETRY": "False"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    from click.testing import CliRunner

    from vaultmind.cli import cli

    assert done.returncode == 0, done.stderr
    assert done.stdout == CliRunner().invoke(cli, ["--version"], prog_name=sibling.name).output
    events = [json.loads(line) for line in (state / "launcher.jsonl").read_text().splitlines()]
    consumed = [event for event in events if event["kind"] == "console-script"]
    assert [event["path"] for event in consumed] == [str(sibling)]
    assert [event["executable"] for event in events if event["kind"] == "entrypoint"] == [
        str(python)
    ]
    assert not (tmp_path / ".docs").exists()
