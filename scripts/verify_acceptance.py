#!/usr/bin/env python3
"""Offline acceptance: retrieval, MCP, approval, configuration and manual wiki consumers."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from aiogram.types import Message as TelegramMessage

    from vaultmind.indexer.embedder import Embedder
    from vaultmind.llm.client import LLMResponse, Message, MultimodalMessage

ROOT = Path(__file__).resolve().parent.parent
FLOOR = 0.5
ALPHA = "alpha.md"
BETA = "beta.md"
TEXT = "harbor lantern inspections record copper valve readings before winter gales"
NOTES = {ALPHA: (1, TEXT), BETA: (5, TEXT + " addendum")}
BOUNDARIES = [
    ("below", [FLOOR - 1e-9], "strong"),
    ("equal", [FLOOR], "weak"),
    ("above", [FLOOR + 1e-9], "weak"),
    ("unknown", [None], "unknown"),
    ("mixed-weak", [0.9, None], "unknown"),
    ("mixed-strong", [None, 0.1], "strong"),
    ("empty", [], "weak"),
]


def offline_environment(home: Path) -> dict[str, str]:
    """Build fixture-only runtime inputs; absent PATH is an error, not optional history."""
    path = os.environ.get("PATH")
    if not path:
        raise RuntimeError("Missing required offline runtime input: PATH")
    return {"PATH": path, "HOME": str(home.absolute()), "ANONYMIZED_TELEMETRY": "False"}


@contextmanager
def offline_environment_scope(environment: dict[str, str]) -> Iterator[None]:
    """Install one sequential consumer's environment and restore it even on failure."""
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environment)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


class DeterministicEmbedder:
    """Synthetic token-count vectors; not evidence of hosted embedding quality."""

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 1021
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vector[bucket % len(vector)] += 1
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class ScriptedLLM:
    """Scripted llm.client boundary; no provider or native-model qualification."""

    provider_name = "scripted-acceptance"

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        from vaultmind.llm.client import LLMResponse

        return LLMResponse(text="Scripted acceptance reply.", model=model, usage={})

    def complete_multimodal(
        self,
        messages: list[MultimodalMessage],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        from vaultmind.llm.client import LLMResponse

        return LLMResponse(text="Scripted acceptance reply.", model=model, usage={})


class FakeMessage:
    """Record the actual handler output instead of delivering Telegram messages."""

    def __init__(self) -> None:
        self.from_user = SimpleNamespace(id=42)
        self.answers: list[str] = []

    async def answer(self, text: str, **kwargs: Any) -> None:
        self.answers.append(text)


class BoundaryStore:
    """Labeled raw-hit substitution ONLY for exact threshold/unknown boundaries."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def search(self, query: str, n_results: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        return self.hits[:n_results]

    def hybrid_search(self, query: str, n_results: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        return self.search(query, n_results)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_identity() -> dict[str, Any]:
    """Identify the imported package, not the checkout containing this verifier."""
    import vaultmind

    package = Path(vaultmind.__file__).resolve().parent
    hashes = {
        str(path.relative_to(package)): digest(path.read_bytes())
        for path in sorted(package.rglob("*.py"))
    }
    if "__init__.py" not in hashes:
        raise ValueError(f"Missing required package source: {package / '__init__.py'}")
    dist = importlib.metadata.distribution("vaultmind")
    return {
        "module_file": str(Path(vaultmind.__file__).resolve()),
        "package_root": str(package),
        "file_sha256": hashes,
        "code_sha256": digest(json.dumps(hashes, sort_keys=True).encode()),
        "distribution": {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "root": str(dist.locate_file("").resolve()),
            "metadata_files": [
                str(dist.locate_file(path).resolve())
                for path in dist.files or ()
                if str(path).endswith(".dist-info/METADATA")
            ],
            "direct_url": dist.read_text("direct_url.json"),
        },
        # Resolving the executable would discard the virtual environment's bin directory.
        "interpreter": {"executable": sys.executable, "version": sys.version},
    }


def artifact_identity(home: Path) -> dict[str, Any]:
    runtime = package_identity()
    reviewed = {
        str(path.relative_to(ROOT / "src" / "vaultmind")): digest(path.read_bytes())
        for path in sorted((ROOT / "src" / "vaultmind").rglob("*.py"))
    }
    if "__init__.py" not in reviewed:
        raise ValueError(f"Missing required reviewed package source: {ROOT / 'src' / 'vaultmind'}")
    hashes = {f"src/vaultmind/{name}": value for name, value in runtime["file_sha256"].items()}
    hashes["scripts/verify_acceptance.py"] = digest(Path(__file__).read_bytes())
    environment = offline_environment(home)
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, env=environment
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True, env=environment
    ).splitlines()
    return {
        "head": head,
        "working_tree_dirty": bool(changes),
        "working_tree_changes": changes,
        "code_sha256": digest(json.dumps(hashes, sort_keys=True).encode()),
        "file_sha256": hashes,
        "runtime": runtime,
        "reviewed_root": str(ROOT),
        "reviewed_package_sha256": digest(json.dumps(reviewed, sort_keys=True).encode()),
        "runtime_matches_reviewed": runtime["file_sha256"] == reviewed,
        "dependency_sha256": {
            name: digest((ROOT / name).read_bytes()) for name in ("pyproject.toml", "uv.lock")
        },
    }


def fixture_config(home: Path, name: str, hybrid: bool) -> Any:
    from vaultmind.config import load_settings

    root = home / name
    (root / "vault").mkdir(parents=True)
    (root / "data").mkdir()
    config = root / "config.toml"
    config.write_text(
        f'[vault]\npath = "{root / "vault"}"\n'
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        f'[chroma]\npersist_dir = "{root / "data/chroma"}"\n'
        f'[graph]\npersist_path = "{root / "data/graph.json"}"\n'
        f"[search]\nhybrid_enabled = {str(hybrid).lower()}\nmax_results = 10\npage_size = 10\n"
        f'bm25_db_path = "{root / "data/bm25.db"}"\ncache_enabled = false\n'
        f"[bench]\nk = 10\nscore_floor = {FLOOR}\n"
        f'trend_path = "{root / "data/trend.jsonl"}"\n'
        f'[gaps]\nenabled = true\ndb_path = "{root / "data/gaps.db"}"\n'
        "[embedding]\ncache_enabled = false\n",
        encoding="utf-8",
    )
    return load_settings(config)


async def retrieval(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    import vaultmind
    from vaultmind.bench.golden import GoldenQuestion
    from vaultmind.bench.runner import run_bench, run_query
    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.handlers.recall import handle_recall
    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import RankingConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.ranking import retrieval_confidence
    from vaultmind.indexer.store import VaultStore
    from vaultmind.knowledge.sources import search_sources
    from vaultmind.memory.gaps import GapStore
    from vaultmind.vault.parser import VaultParser

    def check(name: str, expected: Any, observed: Any) -> None:
        checks.append({"id": name, "expected": expected, "actual": observed})
        if expected != observed:
            raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")

    corpus_hashes: dict[str, str] = {}
    settings = fixture_config(home, "live", False)
    for name, (authority, body) in NOTES.items():
        content = f"---\ntitle: Harbor\ntype: project\nauthority: {authority}\n---\n\n{body}\n"
        (settings.vault.path / name).write_text(content, encoding="utf-8")
        corpus_hashes[name] = digest(content.encode())
    parser = VaultParser(settings.vault)
    bm25 = BM25Index(Path(settings.search.bm25_db_path))
    embedder = cast("Embedder", DeterministicEmbedder())
    store = VaultStore(settings.chroma, embedder, bm25=bm25)
    notes = parser.iter_notes()
    store.index_notes(notes, parser)
    exact = parser.chunk_note(parser.parse_file(settings.vault.path / ALPHA))[0].content
    bm25.upsert("lexical::0", "lexical.md", "Lexical", "kestrelquartz field memo")
    (settings.vault.path / "lexical.md").write_text(
        "---\ntitle: Lexical\n---\n\nkestrelquartz field memo\n", encoding="utf-8"
    )
    corpus_hashes["lexical"] = digest(b"lexical::0 lexical.md Lexical kestrelquartz field memo")
    corpus_hashes["exact-query"] = digest(exact.encode())
    gaps: list[Any] = []

    def context(cfg: Any, target: Any) -> Any:
        gap = GapStore(Path(cfg.gaps.db_path), stale_after_days=cfg.gaps.stale_after_days)
        gaps.append(gap)
        llm = ScriptedLLM()
        thinking = ThinkingPartner(cfg, llm_factory=lambda: llm)
        return HandlerContext(
            cfg,
            target,
            KnowledgeGraph(cfg.graph),
            parser,
            thinking,
            llm,
            cfg.vault.path,
            gap_store=gap,
        )

    async def observe(label: str, ctx: Any, query: str, expected: str) -> list[dict[str, Any]]:
        before = {g.gap_id for g in ctx.gap_store.list_open()}
        message = FakeMessage()
        sessions: dict[int, Any] = {}
        await handle_recall(ctx, cast("TelegramMessage", message), query, sessions)
        hits = sessions[42].results if 42 in sessions else []
        answerable = label.endswith(".self-match")
        question = GoldenQuestion(
            id=label,
            question=query,
            answerable=answerable,
            expected_notes=(ALPHA,) if answerable else (),
        )
        report = run_bench(
            [question],
            ctx.settings,
            k=ctx.settings.search.max_results,
            store=ctx.store,
            score_floor=ctx.settings.bench.score_floor,
        )
        outcome = report.results[0]
        confidence = retrieval_confidence(hits, ctx.settings.bench.score_floor)
        new = [g for g in ctx.gap_store.list_open() if g.gap_id not in before]
        check(label + ".confidence", expected, str(confidence))
        check(label + ".bench-confidence", expected, str(outcome.confidence))
        check(
            label + ".decline",
            None if answerable else expected == "weak",
            outcome.retrieval_decline_correct,
        )
        if answerable:
            check(label + ".retrieval-hit", True, outcome.hit)
            check(label + ".recall-at-k", 1.0, report.recall_at_k)
        check(
            label + ".gap",
            ["weak_retrieval"] if expected == "weak" else [],
            [str(g.kind) for g in new],
        )
        check(
            label + ".same-paths",
            list(outcome.retrieved_paths),
            [h["metadata"]["note_path"] for h in hits],
        )
        check(
            label + ".same-hits",
            hits,
            run_query(
                ctx.settings,
                query,
                ctx.settings.search.max_results,
                store=ctx.store,
            ),
        )
        if expected == "unknown":
            check(label + ".unknown-rendered", True, "relevance unknown" in message.answers[-1])
        return hits

    try:
        for hybrid in (False, True):
            mode = "hybrid" if hybrid else "vector"
            cfg = fixture_config(home, mode, hybrid)
            cfg = cfg.model_copy(update={"vault": settings.vault})
            ctx = context(cfg, store)
            hits = await observe(mode + ".self-match", ctx, exact, "strong")
            alpha = next(h for h in hits if h["metadata"]["note_path"] == ALPHA)
            check(mode + ".raw-self-distance", True, abs(alpha["distance"]) < 0.01)
            raw = run_query(
                cfg.model_copy(update={"ranking": RankingConfig(enabled=False)}),
                exact,
                10,
                store=store,
            )
            check(
                mode + ".authority-reorder",
                [ALPHA, BETA],
                [raw[0]["metadata"]["note_path"], hits[0]["metadata"]["note_path"]],
            )
            check(
                mode + ".distance-invariant",
                {h["chunk_id"]: h["distance"] for h in raw},
                {h["chunk_id"]: h["distance"] for h in hits},
            )
            check(
                mode + ".confidence-invariant",
                str(retrieval_confidence(raw, FLOOR)),
                str(retrieval_confidence(hits, FLOOR)),
            )
            await observe(mode + ".lexical", ctx, "kestrelquartz", "unknown")
            empty_cfg = fixture_config(home, mode + "-empty", hybrid)
            empty_bm25 = BM25Index(Path(empty_cfg.search.bm25_db_path))
            try:
                empty = VaultStore(empty_cfg.chroma, embedder, bm25=empty_bm25)
                await observe(mode + ".empty", context(empty_cfg, empty), "unanswered", "weak")
            finally:
                empty_bm25.close()

        cfg = fixture_config(home, "boundary", True)
        boundary_parser = VaultParser(cfg.vault)
        boundary_chunks = {}
        for i in range(2):
            path = cfg.vault.path / f"{i}.md"
            path.write_text("boundary input\n", encoding="utf-8")
            boundary_chunks[i] = boundary_parser.chunk_note(boundary_parser.parse_file(path))[0]
        for name, distances, expected in BOUNDARIES + [("missing", [None], "unknown")]:
            hits = [
                {
                    "chunk_id": boundary_chunks[i].chunk_id,
                    "content": boundary_chunks[i].content,
                    "metadata": {"note_path": f"{i}.md"},
                    "rrf_score": 0.02,
                    **({"distance": distance} if name != "missing" else {}),
                }
                for i, distance in enumerate(distances)
            ]
            ctx = context(cfg, BoundaryStore(hits))
            await observe("boundary." + name, ctx, name, expected)
        # A larger, truly closer forbidden population must not consume top-k.
        for i in range(20):
            private_path = settings.vault.path / "private" / f"closer-{i:02d}.md"
            private_path.parent.mkdir(exist_ok=True)
            private_path.write_text(
                f"---\ntitle: Harbor\ntype: project\nauthority: 5\n---\n\n{NOTES[BETA][1]}\n",
                encoding="utf-8",
            )
            store.index_single_note(parser.parse_file(private_path), parser)
        closer_query = parser.chunk_note(parser.parse_file(settings.vault.path / BETA))[0].content
        unscoped = store.search(closer_query, n_results=10)
        check(
            "source.scope.has-real-closer-forbidden-population",
            True,
            len(unscoped) == 10 and all(hit["metadata"]["note_path"] != ALPHA for hit in unscoped),
        )
        scoped = search_sources(
            settings,
            closer_query,
            store=store,
            n_results=1,
            allows_path=lambda path: path == ALPHA,
        )
        check("source.scope.filters-before-topk", [ALPHA], [hit["path"] for hit in scoped])
        check(
            "source.scope.bench-same-boundary",
            scoped,
            run_query(
                settings, closer_query, 1, store=store, allows_path=lambda path: path == ALPHA
            ),
        )
        check(
            "source.scope.explicitly-unapproved",
            [("source-only", False)],
            [(hit["status"], hit["approved"]) for hit in scoped],
        )
        current_bytes = (
            b"---\ntitle: Harbor\n---\n\nnewcurrenttoken replaces the old indexed excerpt\n"
        )
        (settings.vault.path / ALPHA).write_bytes(current_bytes)
        current = search_sources(
            settings,
            "newcurrenttoken",
            store=store,
            n_results=1,
            allows_path=lambda path: path == ALPHA,
        )[0]
        check("source.current-byte-digest", digest(current_bytes), current["current_digest"])
        check(
            "source.current-body-not-stale",
            True,
            "newcurrenttoken" in current["content"] and TEXT not in current["content"],
        )
        check("source.stale-vector-confidence-unknown", None, current["distance"])
        return {
            "corpus_sha256": corpus_hashes,
            "configuration_sha256": {
                str(p.relative_to(home)): digest(p.read_bytes()) for p in home.rglob("*.toml")
            },
            "import_origin": str(Path(vaultmind.__file__).resolve()),
            "chromadb": importlib.metadata.version("chromadb"),
        }
    finally:
        for gap in gaps:
            gap.close()
        bm25.close()


# ---------------------------------------------------------------------------
# mcp case: the actual registered vaultmind entry point, served as a subprocess over the
# installed MCP SDK stdio transport. Fixture seams: temp HOME/vault/Chroma/graph/audit/TOML,
# an empty OpenAI constructor key and a launcher-side network guard. Original traversal
# fixtures never embed; root-note fixtures replace only provider embedding I/O with fixed
# vectors. No fake transport, package shadowing, canned tool responses or PYTHONPATH.
# ---------------------------------------------------------------------------

MCP_SCENARIOS = (
    "planner-scoped",
    "planner-configured-override",
    "researcher-default-control",
    "full-control",
)
MCP_ROOT_SCENARIOS = (
    "root-scoped",
    "root-scoped-small",
    "root-full",
    "root-capture-scope",
    "root-capture-symlink",
    "root-capture-size",
)
MCP_DERIVED_SCENARIOS = ("derived-scoped", "derived-full", "derived-writes")
MCP_QUERY_SCENARIOS = ("knowledge-researcher-default",)
MCP_TRAVERSAL_TOOLS = ("read_frontmatter", "list_folder_index", "follow_links")
MCP_HUB_SENTINEL = "HUB-BODY-SENTINEL-7f3a91"
MCP_ISO = "<iso-datetime>"
MCP_ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
MCP_PRIVATE_TOKENS = (
    "Private Secret",
    "03-private",
    "PRIVATE-DESCRIPTION-SENTINEL",
    "PRIVATE-SHADOW-DESCRIPTION",
    "secret.md",
    "shadow.md",
    "leak-private",
    "leak-ext",
    "linkdir",
    "loop-a",
    "loop-b",
    "loopdir-a",
    "loopdir-b",
    "Outside Secret",
    "OUTSIDE-SENTINEL",
    "outside.md",
    "lockdir",
    "lockfile",
    "sealed-priv",
    "sealed-out",
    "hidden-priv",
    "hidden-out",
    "PRIVATE-SEALED-SENTINEL",
    "OUTSIDE-SEALED-SENTINEL",
)
# Permission-denied fixtures: mode-000 directories holding these notes (private, outside and
# public). A caller can only name them through the aliases created by mcp_build_fixture.
MCP_SEALED_PRIVATE = "PRIVATE-SEALED-SENTINEL"
MCP_SEALED_OUTSIDE = "OUTSIDE-SEALED-SENTINEL"
MCP_SEALED_PUBLIC = "PUBLIC-SEALED-NOTE"
# A single path component well beyond NAME_MAX: the real syscall (ENAMETOOLONG) is the oracle.
MCP_LONG_COMPONENT = "z" * 300
# Errors of these kinds may echo only the caller's own path string.
MCP_CALLER_ONLY_KINDS = ("denied", "not-found", "path-not-allowed")
MCP_NOTES: dict[str, dict[str, Any]] = {
    "00-inbox/inbox-note.md": {
        "title": "Inbox Note",
        "type": "fleeting",
        "tags": [],
        "authority": None,
        "description": None,
        "summary": "Inbox first line.",
        "created": "2026-01-01",
        "body": "Inbox first line.\n\nSee [[Hub Note]] for the plan.",
    },
    "02-projects/hub.md": {
        "title": "Hub Note",
        "type": "project",
        "tags": ["hub"],
        "authority": 4,
        "description": "Hub description.",
        "summary": "Hub description.",
        "created": "2026-01-04",
        "body": (
            "Links to [[Planner Peer]], [[Private Secret]] and [[Nowhere Target]].\n\n"
            + MCP_HUB_SENTINEL
        ),
    },
    "02-projects/peer.md": {
        "title": "Planner Peer",
        "type": "project",
        "tags": ["peer"],
        "authority": 3,
        "description": "Allowed peer description.",
        "summary": "Allowed peer description.",
        "created": "2026-01-02",
        "body": "Back to [[Hub Note]].",
    },
    "02-projects/sub/nested.md": {
        "title": "Nested Allowed",
        "type": "literature",
        "tags": ["nested"],
        "authority": None,
        "description": None,
        "summary": "Nested allowed first line.",
        "created": "2026-01-03",
        "body": "Nested allowed first line.",
    },
    "03-private/secret.md": {
        "title": "Private Secret",
        "type": "permanent",
        "tags": ["private"],
        "authority": 5,
        "description": "PRIVATE-DESCRIPTION-SENTINEL",
        "summary": "PRIVATE-DESCRIPTION-SENTINEL",
        "created": "2026-01-05",
        "body": "Links to [[Hub Note]] and [[Planner Peer]].",
    },
}
# Present only in the scoped fixture: a private note that duplicates an allowed title.
MCP_EXTRA_NOTES: dict[str, dict[str, Any]] = {
    "03-private/shadow.md": {
        "title": "Planner Peer",
        "type": "permanent",
        "tags": ["private-shadow"],
        "authority": None,
        "description": "PRIVATE-SHADOW-DESCRIPTION",
        "summary": "PRIVATE-SHADOW-DESCRIPTION",
        "created": "2026-01-06",
        "body": "Shadow of [[Hub Note]].",
    },
}
MCP_OUTSIDE_NOTE: dict[str, Any] = {
    "title": "Outside Secret",
    "type": "permanent",
    "tags": [],
    "authority": None,
    "description": "OUTSIDE-SENTINEL",
    "summary": "OUTSIDE-SENTINEL",
    "created": "2026-01-07",
    "body": "Outside note linking [[Hub Note]].",
}

# Runs under `python -I -c`: installs the network guard, resolves the registered `vaultmind`
# console script from installed metadata, tees fd 1 byte-for-byte, then runs the Click group.
MCP_LAUNCHER = r"""
import importlib.metadata
import hashlib
import json
import os
import runpy
import socket
import sys
import threading
from pathlib import Path

state = sys.argv[1]
cli_args = sys.argv[2:]
log = open(os.path.join(state, "launcher.jsonl"), "a", buffering=1, encoding="utf-8")
console_script = Path(sys.executable).parent / "vaultmind"


def emit(**fields):
    log.write(json.dumps(fields, sort_keys=True) + "\n")
    log.flush()


probing = [False]
HOST_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
PEER_EVENTS = {"socket.connect", "socket.sendto"}


def guard(event, args):
    if event == "exec" and args[0].co_filename == str(console_script):
        emit(
            kind="console-script",
            path=args[0].co_filename,
            sha256=hashlib.sha256(console_script.read_bytes()).hexdigest(),
        )
    if event in HOST_EVENTS or (event in PEER_EVENTS and args[0].family != socket.AF_UNIX):
        emit(kind="network", event=event, probe=probing[0])
        raise RuntimeError("network forbidden by offline MCP acceptance")


sys.addaudithook(guard)
emit(kind="start", pid=os.getpid())
probing[0] = True
try:
    socket.getaddrinfo("guard-probe.invalid", 443)
    emit(kind="probe", result="not-blocked")
except RuntimeError:
    emit(kind="probe", result="blocked")
finally:
    probing[0] = False

allowed_environment = {"HOME", "PATH", "ANONYMIZED_TELEMETRY", "COLUMNS", "LC_CTYPE"}
# Runtime locale fields can be created after exec, absent from the supplied map.
if sys.platform == "darwin":
    allowed_environment.add("__CF_USER_TEXT_ENCODING")
emit(
    kind="environment",
    home=os.environ.get("HOME"),
    forbidden=sorted(set(os.environ) - allowed_environment),
    keys=sorted(os.environ),
)
real_stdout = os.dup(1)
read_fd, write_fd = os.pipe()
os.dup2(write_fd, 1)
os.close(write_fd)
raw_path = os.path.join(state, "stdout.raw")


def pump():
    with os.fdopen(read_fd, "rb", buffering=0) as src, open(raw_path, "ab", buffering=0) as raw:
        while True:
            chunk = src.read(65536)
            if not chunk:
                return
            raw.write(chunk)
            view = memoryview(chunk)
            while view:
                view = view[os.write(real_stdout, view) :]


tee = threading.Thread(target=pump, name="stdout-tee")
tee.start()
matches = [
    ep.value
    for ep in importlib.metadata.entry_points(group="console_scripts")
    if ep.name == "vaultmind"
]
emit(kind="entrypoint-lookup", matches=matches)
if len(matches) != 1:
    sys.exit(97)
entry = [
    ep for ep in importlib.metadata.entry_points(group="console_scripts") if ep.name == "vaultmind"
][0]
import vaultmind

dist = importlib.metadata.distribution("vaultmind")
emit(
    kind="entrypoint",
    name=entry.name,
    group=entry.group,
    value=entry.value,
    dist_version=dist.version,
    direct_url=dist.read_text("direct_url.json"),
    module_file=os.path.realpath(vaultmind.__file__),
    package_sha256=hashlib.sha256(json.dumps({
        str(path.relative_to(Path(vaultmind.__file__).resolve().parent)):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(vaultmind.__file__).resolve().parent.rglob("*.py"))
    }, sort_keys=True).encode()).hexdigest(),
    executable=sys.executable,
    python=sys.version.split()[0],
    sys_path=sys.path,
)

# Only the root-note fixtures opt in. Keep the real Embedder constructor, VaultStore,
# Chroma filtering/ranking, registered tools and SDK transport; replace provider I/O only.
embedding_fixture = Path(state) / "root-embedding.json"
if embedding_fixture.exists():
    from vaultmind.indexer.embedder import Embedder

    vector = json.loads(embedding_fixture.read_text(encoding="utf-8"))["vector"]

    def offline_embeddings(self, texts):
        emit(kind="embedding-boundary", count=len(texts))
        return [list(vector) for _ in texts]

    Embedder._embed_texts_uncached = offline_embeddings
    emit(kind="embedding-fixture", dimensions=len(vector))

# Query fixtures replace provider I/O only. The selected CLI assembly, registered
# handler, scope checks and approved-current query service remain real.
query_fixture = Path(state) / "query-provider.json"
if query_fixture.exists():
    import importlib
    from vaultmind.llm.client import LLMResponse

    response = json.loads(query_fixture.read_text(encoding="utf-8"))

    class OfflineQueryProvider:
        provider_name = "offline-acceptance"

        def complete(self, messages, model, max_tokens=4096, system=None):
            emit(kind="query-provider", model=model, thread=threading.get_ident(),
                 main_thread=threading.main_thread().ident)
            return LLMResponse(text=json.dumps(response), model=model, usage={})

        def complete_multimodal(self, *args, **kwargs):
            raise AssertionError("Knowledge queries must not invoke multimodal generation")

    def query_factory(settings):
        emit(kind="query-factory", model=settings.llm.thinking_model)
        return OfflineQueryProvider()

    importlib.import_module("vaultmind.cli")._create_llm_client = query_factory

# Read-only observation, not dependency injection: bind real producer entry and buffer
# state to each dispatch. No arguments, return values or producer state are modified.
if embedding_fixture.exists():
    producer_methods = {
        "vaultmind.memory.store": {"query_pending", "query_resolved", "search_by_entity"},
        "vaultmind.memory.procedural": {"suggest_workflow"},
        "vaultmind.graph.evolution": {"scan"},
    }

    def dispatch_state(frame):
        result = {}
        for name, attribute in (
            ("duplicate_detector", "_results"),
            ("note_suggester", "_results"),
            ("evolution_detector", "_dismissed"),
        ):
            obj = frame.f_locals.get(name)
            if obj is not None:
                value = getattr(obj, attribute)
                encoded = json.dumps(
                    sorted(value) if isinstance(value, set) else value,
                    sort_keys=True, default=str,
                )
                result[name] = {
                    "count": len(value),
                    "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                }
        graph = frame.f_locals["graph"]._graph
        import networkx as nx
        result["graph"] = hashlib.sha256(
            json.dumps(nx.node_link_data(graph, edges="edges"), sort_keys=True).encode()
        ).hexdigest()
        return result

    def observe(frame, event, arg):
        module = frame.f_globals.get("__name__")
        method = frame.f_code.co_name
        if module == "vaultmind.mcp.server" and method == "_dispatch_tool":
            if event in ("call", "return"):
                emit(kind="dispatch-state", event=event,
                     tool=frame.f_locals["name"], state=dispatch_state(frame))
        elif event == "call" and method in producer_methods.get(module, ()):
            emit(kind="producer-entry", module=module, method=method)

    sys.setprofile(observe)
    threading.setprofile(observe)

code = 0
try:
    sys.argv = [str(console_script), *cli_args]
    runpy.run_path(str(console_script), run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    code = 70
    raise
finally:
    # The real SDK closes its stdout wrapper when the stdio context exits.
    if not sys.stdout.closed:
        sys.stdout.flush()
    os.dup2(real_stdout, 1)
    tee.join(10)
    emit(kind="exit", code=code, tee_joined=not tee.is_alive())
sys.exit(code)
"""


def make_checker(checks: list[dict[str, Any]]) -> Any:
    def check(name: str, expected: Any, observed: Any) -> None:
        checks.append({"id": name, "expected": expected, "actual": observed})
        if expected != observed:
            raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")

    return check


def mcp_mask(value: str, arguments: dict[str, Any]) -> str:
    """``value`` with the exact string arguments of ONE call masked.

    Only a complete string argument of this call, delimited as a path token, is masked
    (also in its JSON-escaped form for raw frames). A known ``03-private`` or ``.`` never
    masks the segment inside an unknown ``03-private/secret.md``, and a supplied relative
    path never masks a longer absolute target that merely contains it.
    """
    masked = value
    supplied = {s for s in arguments.values() if isinstance(s, str) and s}
    for text in sorted(supplied, key=len, reverse=True):
        for form in {text, json.dumps(text)[1:-1]}:
            pattern = rf"(?<![\w./\\-]){re.escape(form)}(?![\w/-]|\.\w|\\(?![\"nrt]))"
            masked = re.sub(pattern, "\x00", masked)
    return masked


def mcp_leaks(value: str, forbidden: Any, arguments: dict[str, Any]) -> list[str]:
    """Forbidden values present in ``value`` after masking exact caller-supplied strings."""
    masked = mcp_mask(value, arguments)
    return [token for token in forbidden if token in masked]


def mcp_path_residue(text: str, arguments: dict[str, Any]) -> list[str]:
    """Path separators left once the caller's own strings are masked.

    An error that names only the caller's path leaves none; any separator that remains is a
    physical location the caller did not supply.
    """
    masked = mcp_mask(text, arguments)
    return [separator for separator in ("/", "\\") if separator in masked]


def mcp_trim(text: str) -> str:
    """Drop the trailing padding Rich adds to each log line (COLUMNS is deliberately wide)."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def mcp_flat(text: str) -> str:
    """Undo log wrapping: drop RichHandler's ``file.py:123`` column and non-path characters."""
    return re.sub(r"[^A-Za-z0-9._/-]", "", re.sub(r"\w+\.py:\d+", "", text))


def mcp_audit_options(expected_call: dict[str, Any]) -> tuple[str, ...]:
    """Accepted audit results of one call (a string, or several for real syscall errors)."""
    audit = expected_call["audit"]
    return (audit,) if isinstance(audit, str) else tuple(audit)


def mcp_render(spec: dict[str, Any]) -> str:
    front = [f"title: {spec['title']}", f"type: {spec['type']}"]
    if spec["tags"]:
        front.append(f"tags: [{', '.join(spec['tags'])}]")
    if spec["authority"] is not None:
        front.append(f"authority: {spec['authority']}")
    if spec["description"] is not None:
        front.append(f"description: {spec['description']}")
    front.append(f'created: "{spec["created"]}"')
    return "---\n" + "\n".join(front) + "\n---\n\n" + spec["body"] + "\n"


@functools.cache
def mcp_permission_denial_enforced() -> bool:
    """True when a mode-000 directory really denies lookups (false for root or such mounts)."""
    with tempfile.TemporaryDirectory(prefix="vm-m12-permission-probe-") as directory:
        locked = Path(directory) / "locked"
        locked.mkdir()
        (locked / "probe.md").write_text("probe\n", encoding="utf-8")
        locked.chmod(0)
        try:
            (locked / "probe.md").stat()
            return False
        except PermissionError:
            return True
        finally:
            locked.chmod(0o700)


def mcp_restore_permissions(fixture: dict[str, Any]) -> None:
    """Give mode-000 fixture directories their search bit back so cleanup can delete them."""
    for directory in fixture["sealed"]:
        if directory.exists():
            directory.chmod(0o700)


def mcp_build_fixture(root: Path, scoped: bool, consumer: bool = False) -> dict[str, Any]:
    """Write the fixture vault; ``scoped`` adds the duplicate-title note and symlinks.

    ``consumer`` adds the directory aliases every real-consumer profile can reach and, when
    the platform honours mode-000 directories, the permission-denied targets (private,
    outside and public) with the aliases that reach them. Those directories are chmod'ed
    last and listed in ``sealed`` so the caller restores them before cleanup.
    """
    vault = root / "vault"
    outside = root / "outside"
    outside.mkdir(parents=True)
    (root / "data").mkdir()
    specs = {**MCP_NOTES, **(MCP_EXTRA_NOTES if scoped else {})}
    corpus: dict[str, str] = {}
    for rel, spec in specs.items():
        text = mcp_render(spec)
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        corpus[rel] = digest(text.encode())
    outside_text = mcp_render(MCP_OUTSIDE_NOTE)
    (outside / "outside.md").write_text(outside_text, encoding="utf-8")
    corpus["outside/outside.md"] = digest(outside_text.encode())
    symlinks: dict[str, str] = {}
    projects = vault / "02-projects"
    private = vault / "03-private"
    links: list[tuple[Path, str, bool]] = [
        # symlink loops: private, outside and an in-scope self-loop (files and directories)
        (private / "loop-a.md", "loop-b.md", False),
        (private / "loop-b.md", "loop-a.md", False),
        (private / "loopdir-a", "loopdir-b", True),
        (private / "loopdir-b", "loopdir-a", True),
        (outside / "loop-a.md", "loop-b.md", False),
        (outside / "loop-b.md", "loop-a.md", False),
        (outside / "loopdir-a", "loopdir-b", True),
        (outside / "loopdir-b", "loopdir-a", True),
        (projects / "self.md", "self.md", False),
        (projects / "loop-private.md", "../03-private/loop-a.md", False),
        (projects / "loopdir-private", "../03-private/loopdir-a", True),
        (projects / "loop-ext.md", str(outside / "loop-a.md"), False),
        (projects / "loopdir-ext", str(outside / "loopdir-a"), True),
    ]
    if scoped:
        links += [
            (projects / "leak-private.md", "../03-private/secret.md", False),
            (projects / "leak-ext.md", str(outside / "outside.md"), False),
            (projects / "linkdir-private", "../03-private", True),
            (projects / "linkdir-ext", str(outside), True),
        ]
    sealed: list[Path] = []
    if consumer:
        if not scoped:
            # Directory aliases never match ``*.md``, so no listing count changes.
            links += [
                (projects / "linkdir-private", "../03-private", True),
                (projects / "linkdir-ext", str(outside), True),
            ]
        if mcp_permission_denial_enforced():
            for directory, name, sentinel in (
                (private / "sealed-priv", "hidden-priv.md", MCP_SEALED_PRIVATE),
                (outside / "sealed-out", "hidden-out.md", MCP_SEALED_OUTSIDE),
                (projects / "sealed-pub", "hidden-pub.md", MCP_SEALED_PUBLIC),
            ):
                directory.mkdir()
                (directory / name).write_text(sentinel + "\n", encoding="utf-8")
                corpus[str((directory / name).relative_to(root))] = digest(
                    (sentinel + "\n").encode()
                )
                sealed.append(directory)
            links += [
                (projects / "lockdir-private", "../03-private/sealed-priv", True),
                (projects / "lockdir-ext", str(outside / "sealed-out"), True),
                (
                    projects / "lockfile-private.md",
                    "../03-private/sealed-priv/hidden-priv.md",
                    False,
                ),
                (
                    projects / "lockfile-ext.md",
                    str(outside / "sealed-out" / "hidden-out.md"),
                    False,
                ),
            ]
    for link, target, is_dir in links:
        link.symlink_to(target, target_is_directory=is_dir)
        symlinks[str(link.relative_to(root))] = target
    for directory in sealed:
        directory.chmod(0)
    return {
        "vault": vault,
        "outside": outside,
        "corpus": corpus,
        "symlinks": symlinks,
        "sealed": sealed,
    }


def mcp_write_config(root: Path, override: bool) -> Path:
    sections = [
        f"[vault]\npath = {json.dumps(str(root / 'vault'))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n',
        f"[chroma]\npersist_dir = {json.dumps(str(root / 'data' / 'chroma'))}\n",
        f"[graph]\npersist_path = {json.dumps(str(root / 'data' / 'graph.json'))}\n",
        "[embedding]\ncache_enabled = false\n",
    ]
    if override:
        sections.append(
            "[mcp.profiles.planner]\n"
            'description = "Fixture override"\n'
            'allowed_tools = ["read_frontmatter", "list_folder_index", "follow_links"]\n'
            'folder_scope = ["00-inbox"]\n'
            "write_enabled = false\n"
        )
    config = root / "config.toml"
    config.write_text("\n".join(sections), encoding="utf-8")
    return config


class _RecordingReceive:
    """Tap on the SDK read stream: records every inbound frame and every parser error."""

    def __init__(self, inner: Any, frames: list[dict[str, Any]], errors: list[str]) -> None:
        self._inner = inner
        self._frames = frames
        self._errors = errors

    def _record(self, item: Any) -> None:
        if isinstance(item, Exception):
            self._errors.append(f"{type(item).__name__}: {item}")
        else:
            self._frames.append(
                item.message.model_dump(by_alias=True, mode="json", exclude_none=True)
            )

    async def __aenter__(self) -> _RecordingReceive:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._inner.__aexit__(*exc_info)

    def __aiter__(self) -> _RecordingReceive:
        return self

    async def __anext__(self) -> Any:
        item = await self._inner.__anext__()
        self._record(item)
        return item

    async def receive(self) -> Any:
        item = await self._inner.receive()
        self._record(item)
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()


class _RecordingSend:
    """Tap on the SDK write stream: records every outbound frame."""

    def __init__(self, inner: Any, frames: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._frames = frames

    async def __aenter__(self) -> _RecordingSend:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._inner.__aexit__(*exc_info)

    async def send(self, item: Any) -> None:
        self._frames.append(item.message.model_dump(by_alias=True, mode="json", exclude_none=True))
        await self._inner.send(item)

    async def aclose(self) -> None:
        await self._inner.aclose()


async def mcp_stdio_session(
    root: Path,
    cli_args: list[str],
    calls: list[tuple[str, dict[str, Any]]],
    deadline: float = 180.0,
    *,
    root_embeddings: bool = False,
    query_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn the registered CLI and drive initialize/list_tools/call_tool over SDK stdio."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client import stdio as sdk_stdio

    state = root / "launcher-state"
    state.mkdir()
    if query_response is not None:
        (state / "query-provider.json").write_text(json.dumps(query_response), encoding="utf-8")
    if root_embeddings:
        (state / "root-embedding.json").write_text(
            json.dumps({"vector": [1.0, 0.0]}), encoding="utf-8"
        )
    # COLUMNS keeps Rich from wrapping a caller argument across log lines, so the per-call
    # scan can mask exactly that call's own strings without any substring exemption.
    environment = offline_environment(root)
    environment["COLUMNS"] = "2000"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-c", MCP_LAUNCHER, str(state), *cli_args],
        env=environment,
        cwd=str(root),
    )
    frames_in: list[dict[str, Any]] = []
    frames_out: list[dict[str, Any]] = []
    errors: list[str] = []
    results: list[Any] = []
    embedding_boundary_tools: list[str] = []
    embedding_boundary_counts: list[int] = []
    stderr_chunks: list[bytes] = []
    stderr_path = state / "stderr.txt"

    def stderr_since(offset: int) -> tuple[bytes, int]:
        data = stderr_path.read_bytes()
        return data[offset:], len(data)

    # SDK stdio merges POSIX baseline variables from its parent before params.env.
    # Direct callers bypass main, so bound that merge here for the entire lifecycle.
    with offline_environment_scope(environment), stderr_path.open("w", encoding="utf-8") as errlog:
        with anyio.fail_after(deadline):
            async with sdk_stdio.stdio_client(params, errlog=errlog) as (read, write):
                session = ClientSession(
                    _RecordingReceive(read, frames_in, errors),
                    _RecordingSend(write, frames_out),
                    read_timeout_seconds=timedelta(seconds=deadline),
                )
                async with session:
                    init = await session.initialize()
                    listed = await session.list_tools()
                    # The server writes (and flushes) its log records for a request before
                    # it sends the response, so the bytes between two awaited calls are
                    # that call's own stderr (plus only post-response bookkeeping lines of
                    # the previous call, which never carry arguments).
                    startup, offset = stderr_since(0)
                    with (state / "launcher.jsonl").open(encoding="utf-8") as boundary_log:
                        boundary_log.seek(0, os.SEEK_END)
                        for tool, arguments in calls:
                            results.append(await session.call_tool(tool, arguments))
                            chunk, offset = stderr_since(offset)
                            stderr_chunks.append(chunk)
                            count = 0
                            for line in boundary_log:
                                if json.loads(line)["kind"] == "embedding-boundary":
                                    embedding_boundary_tools.append(tool)
                                    count += 1
                            embedding_boundary_counts.append(count)
                closing = time.monotonic()
            shutdown_seconds = time.monotonic() - closing
    return {
        "init": init,
        "tools": listed.tools,
        "results": results,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "errors": errors,
        "shutdown_seconds": shutdown_seconds,
        "termination_timeout": getattr(sdk_stdio, "PROCESS_TERMINATION_TIMEOUT", 2.0),
        "state": state,
        "stderr_startup": startup.decode("utf-8", errors="replace"),
        "stderr_chunks": [chunk.decode("utf-8", errors="replace") for chunk in stderr_chunks],
        "embedding_boundary_tools": embedding_boundary_tools,
        "embedding_boundary_counts": embedding_boundary_counts,
    }


def mcp_root_fixture(root: Path, name: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Seed real Chroma with synthetic provenance, not replacement tool responses."""
    import frontmatter

    from vaultmind.config import ChromaConfig
    from vaultmind.indexer.store import VaultStore

    vault, outside = root / "vault", root / "outside"
    source = "02-projects/source.md"
    peer = "02-projects/peer.md"
    suggestion = "02-projects/suggestion.md"
    private = [f"03-private/closer-{i:02d}.md" for i in range(16)]
    small_index = name == "root-scoped-small"
    contents: dict[str, str] = {
        source: "---\ntitle: Source\ntype: project\n---\n\n# Source\n\n" + TEXT * 4 + "\n",
        peer: "---\ntitle: Allowed Peer\ntype: permanent\n---\n\n# Allowed Peer\n\n"
        + TEXT * 4
        + "\n",
        suggestion: (
            "---\ntitle: Allowed Suggestion\ntype: permanent\n---\n\nA neighboring subject.\n"
        ),
        **{
            path: (
                f"---\ntitle: ROOT-PRIVATE-{i:02d}\ntype: permanent\n---\n\n"
                f"ROOT-PRIVATE-BODY-{i:02d}\n"
            )
            for i, path in enumerate(private)
        },
    }
    for rel, content in contents.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    outside.mkdir()
    (outside / "outside.md").write_text("# ROOT-OUTSIDE\n\nROOT-OUTSIDE-BODY\n", encoding="utf-8")
    (vault / "02-projects" / "not-markdown.txt").write_text("ROOT-NONMARKDOWN", encoding="utf-8")
    links = {
        "02-projects/allowed-alias.md": "peer.md",
        "02-projects/private-alias.md": "../03-private/closer-00.md",
        "02-projects/outside-alias.md": str(outside / "outside.md"),
        "02-projects/broken-alias.md": "missing.md",
        "02-projects/loop.md": "loop.md",
    }
    for rel, target in links.items():
        (vault / rel).symlink_to(target)
    (vault / "02-projects" / "private-dir").symlink_to("../03-private", target_is_directory=True)
    if name == "root-capture-symlink":
        (vault / "00-inbox").symlink_to(outside, target_is_directory=True)

    # Large and small scoped scenarios have identical files and authorized chunks.
    # Only forbidden/unknown indexed rows differ. The large private population
    # exceeds find_duplicates(max=1)'s candidate window of 11; the small TOTAL
    # population is below 11, so that candidate query also exercises oversizing.
    rows: list[dict[str, Any]] = []

    def row(
        key: str,
        provenance: object,
        title: str,
        distance: float,
        *,
        valid: bool = True,
        scoped: bool = False,
        note_type: str = "permanent",
    ) -> None:
        metadata: dict[str, Any] = {"note_title": title, "note_type": note_type}
        if provenance is not None:
            metadata["note_path"] = provenance
        rows.append(
            {
                "chunk_id": key,
                "content": title + "-CONTENT",
                "metadata": metadata,
                "distance": distance,
                "valid": valid,
                "scoped": scoped,
            }
        )

    for i, path in enumerate([] if small_index else private):
        row(f"private-{i}", path, f"ROOT-PRIVATE-{i:02d}", 0.0001 * (i + 1))
    row("allowed-peer", peer, "Allowed Peer", 0.01, scoped=True)
    row("allowed-source", source, "Source", 0.02, scoped=True, note_type="project")
    row("source-second", source, "Source Second", 0.021, scoped=True, note_type="project")
    row("source-third", source, "Source Third", 0.022, scoped=True, note_type="project")
    row("allowed-suggestion", suggestion, "Allowed Suggestion", 0.25, scoped=True)
    row("safe-alias", "02-projects/allowed-alias.md", "Allowed Alias", 0.03, scoped=True)
    row("dot-path", "./" + peer, "Allowed Dot Path", 0.04)
    row("absolute-path", str(vault / peer), "Allowed Absolute Path", 0.05)
    row("private-alias", "02-projects/private-alias.md", "ROOT-PRIVATE-ALIAS", 0.006)
    invalid = [
        None,
        42,
        False,
        1.5,
        "",
        "02-projects/missing.md",
        "02-projects/not-markdown.txt",
        "02-projects",
        "../outside/outside.md",
        str(outside / "outside.md"),
        "02-projects/outside-alias.md",
        "02-projects/broken-alias.md",
        "02-projects/loop.md",
    ]
    for i, provenance in enumerate(invalid[:1] if small_index else invalid):
        row(f"invalid-{i}", provenance, f"ROOT-INVALID-{i:02d}", 0.007 + i * 0.00001, valid=False)

    # Cold Chroma/PostHog OS discovery can spawn Linux helpers before SDK startup.
    with offline_environment_scope(offline_environment(root)):
        store = VaultStore(
            ChromaConfig(persist_dir=root / "data" / "chroma"),
            cast("Embedder", DeterministicEmbedder()),
        )
        store._collection.upsert(
            ids=[r["chunk_id"] for r in rows],
            documents=[r["content"] for r in rows],
            metadatas=[r["metadata"] for r in rows],
            embeddings=[[1 - r["distance"], math.sqrt(1 - (1 - r["distance"]) ** 2)] for r in rows],
        )
    config = root / "config.toml"
    scopes = ["02-projects"]
    if name in ("root-capture-size", "root-capture-symlink"):
        scopes.append("00-inbox")
    if name == "root-capture-size":
        # Explicit synthetic permission reaches serialized-size enforcement;
        # the planner/default-scope cases still deny preservation first.
        scopes.append("_wiki")
    maximum = 256 if name == "root-capture-size" else 50000
    profile = "full" if name == "root-full" else "root-scope"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        f"[chroma]\npersist_dir = {json.dumps(str(root / 'data' / 'chroma'))}\n"
        f"[graph]\npersist_path = {json.dumps(str(root / 'data' / 'graph.json'))}\n"
        "[embedding]\ncache_enabled = false\ndimensions = 2\n"
        '[mcp.profiles.root-scope]\ndescription = "Root note fixture"\n'
        'allowed_tools = ["vault_search", "vault_read", "vault_write", "vault_list", '
        '"find_duplicates", "suggest_links", "capture", "capture_note", '
        '"read_frontmatter", "list_folder_index", "follow_links"]\n'
        f"folder_scope = {json.dumps(scopes)}\nwrite_enabled = true\n"
        f"max_note_size_bytes = {maximum}\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def call(
        tool: str, arguments: dict[str, Any], expected: dict[str, Any], audit: str = "OK"
    ) -> None:
        calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "expected": expected,
                "audit": audit,
                "kind": "denied" if audit == "DENIED" else None,
            }
        )

    full = profile == "full"
    visible_paths = {path for path in contents if full or path.startswith("02-projects/")}

    def search_expected(limit: int, note_type: str | None = None) -> dict[str, Any]:
        # Independent current-file oracle: no shared-search call, no synthetic
        # current bytes per query and no trust in the adversarial indexed body.
        posts = {path: frontmatter.loads(contents[path]) for path in visible_paths}
        posts = {
            path: post
            for path, post in posts.items()
            if note_type is None or post.metadata["type"] == note_type
        }
        indexed_paths = list(
            dict.fromkeys(
                row["metadata"].get("note_path")
                for row in sorted(rows, key=lambda row: row["distance"])
                if row["metadata"].get("note_path") in posts
            )
        )
        lexical_paths = sorted(path for path in posts if path in (peer, source))
        scores: dict[str, float] = {}
        for ranking in (indexed_paths, lexical_paths):
            for rank, path in enumerate(ranking, 1):
                scores[path] = scores.get(path, 0.0) + 1.0 / (60 + rank)
        selected = sorted(scores, key=lambda path: (-scores[path], path))[:limit]
        results = [
            {
                "chunk_id": f"source:{path}",
                "path": path,
                "source_kind": "ordinary",
                "status": "source-only",
                "approved": False,
                "current_digest": digest(contents[path].encode()),
                "content": posts[path].content,
                "distance": None,
                "metadata": {
                    "note_path": path,
                    "note_title": posts[path].metadata["title"],
                    "note_type": posts[path].metadata["type"],
                    "authority": 0,
                },
                "rrf_score": scores[path],
                "authority_score": scores[path],
            }
            for path in selected
        ]
        return {
            "status": "source-only",
            "approved": False,
            "results": results,
            "count": len(results),
        }

    if name in ("root-scoped", "root-scoped-small", "root-full"):
        call("vault_search", {"query": TEXT, "n_results": 100}, search_expected(100))
        call("vault_search", {"query": TEXT, "n_results": 1}, search_expected(1))
        call(
            "vault_search",
            {"query": TEXT, "note_type": "project", "n_results": 1},
            search_expected(1, "project"),
        )
        call(
            "vault_search",
            {"query": TEXT, "note_type": "project", "n_results": 100},
            search_expected(100, "project"),
        )
        call(
            "vault_search",
            {"query": TEXT, "note_type": "literature", "n_results": 100},
            search_expected(100, "literature"),
        )
        allowed_notes = [source, peer, suggestion, "02-projects/allowed-alias.md"]
        listed = sorted(
            allowed_notes + (private + ["02-projects/private-alias.md"] if full else [])
        )
        for folder in (None, "", ".", str(vault)):
            call(
                "vault_list",
                {} if folder is None else {"folder": folder},
                {"notes": listed, "count": len(listed)},
            )
        call(
            "vault_list",
            {"folder": "02-projects"},
            {
                "notes": sorted(allowed_notes + (["02-projects/private-alias.md"] if full else [])),
                "count": len(allowed_notes) + int(full),
            },
        )
        for path in (source, "./" + source, str(vault / source), "02-projects/allowed-alias.md"):
            content_path = peer if path.endswith("allowed-alias.md") else source
            call("vault_read", {"path": path}, {"path": path, "content": contents[content_path]})
        if not full:
            for path in (
                private[0],
                "02-projects/private-alias.md",
                "02-projects/private-dir/closer-00.md",
            ):
                call("vault_read", {"path": path}, {"error": True}, "DENIED")
            for folder in ("03-private", "02-projects/private-dir"):
                call("vault_list", {"folder": folder}, {"error": True}, "DENIED")
        for path in (
            "02-projects/outside-alias.md",
            str(outside / "outside.md"),
            "../outside/outside.md",
        ):
            call("vault_read", {"path": path}, {"error": True}, "DENIED")
        match_path, title, similarity = (
            (private[0], "ROOT-PRIVATE-00", 0.9999) if full else (peer, "Allowed Peer", 0.99)
        )
        call(
            "find_duplicates",
            {"note_path": source, "max_results": 1},
            {
                "note_path": source,
                "matches": [
                    {
                        "path": match_path,
                        "title": title,
                        "similarity": similarity,
                        "type": "duplicate",
                    }
                ],
                "count": 1,
            },
        )
        call(
            "suggest_links",
            {"note_path": source, "max_results": 50},
            {
                "note_path": source,
                "suggestions": [
                    {
                        "path": suggestion,
                        "title": "Allowed Suggestion",
                        "similarity": 0.75,
                        "shared_entities": [],
                        "graph_distance": None,
                        "composite_score": 0.75,
                    }
                ],
                "count": 1,
            }
            if full
            else {"error": True},
            "OK" if full else "DENIED",
        )
    elif name == "root-capture-size":
        for tool in ("capture", "capture_note"):
            # The raw content fits; the generated title/frontmatter pushes serialized bytes
            # over policy. Inbox is absent so even mkdir is an observable forbidden mutation.
            call(tool, {"content": "x", "title": "Long title " * 40}, {"error": True}, "DENIED")
            call(
                tool,
                {"content": "x", "title": "small", "tags": ["界" * 120]},
                {"error": True},
                "DENIED",
            )
    else:
        call(
            "capture",
            {"content": "ROOT-CAPTURE", "title": "Denied implicit destination"},
            {"error": True},
            "DENIED",
        )
        call(
            "capture_note",
            {"content": "ROOT-CAPTURE", "title": "Denied default"},
            {"error": True},
            "DENIED",
        )
        if name == "root-capture-scope":
            call(
                "capture",
                {"content": "ROOT-CAPTURE", "title": "x/../../02-projects/y"},
                {"error": True},
                "DENIED",
            )
            for folder in (
                "03-private/new-directory",
                "02-projects/private-dir/new-directory",
                "../outside/new-directory",
            ):
                call(
                    "capture_note",
                    {"content": "ROOT-CAPTURE", "folder": folder},
                    {"error": True},
                    "DENIED",
                )
            call(
                "vault_write",
                {"path": "03-private/new-directory/new.md", "content": "denied"},
                {"error": True},
                "DENIED",
            )
        else:
            call(
                "capture_note",
                {"content": "ROOT-CAPTURE", "folder": "00-inbox/new-directory"},
                {"error": True},
                "DENIED",
            )

    forbidden = ["ROOT-INVALID", "ROOT-OUTSIDE", "ROOT-NONMARKDOWN"]
    if not full:
        forbidden += ["ROOT-PRIVATE", "closer-", "03-private"]
    fixture = {
        "vault": vault,
        "outside": outside,
        "store": store,
        "index_document_count": len(rows),
        "sealed": [],
        "corpus": {path: digest(content.encode()) for path, content in contents.items()},
        "symlinks": links,
    }
    spec = {
        "name": name,
        "profile": profile,
        "scoped": not full,
        "cli_args": ["-c", str(config), "mcp-serve", "--profile", profile],
        "calls": calls,
        "forbidden": forbidden,
        "stderr_forbidden": ["ROOT-INVALID", "ROOT-OUTSIDE"],
    }
    if name in ("root-scoped", "root-scoped-small"):
        spec["cli_args"].insert(0, "--verbose")
    return spec, fixture, config


def mcp_root_snapshot(fixture: dict[str, Any]) -> dict[str, Any]:
    """Logical fixture state: include empty directories, symlinks and all indexed metadata."""
    import subprocess

    trees: dict[str, Any] = {}
    for key in ("vault", "outside"):
        root = fixture[key]
        trees[key] = {
            str(p.relative_to(root)): (
                ["symlink", os.readlink(p)]
                if p.is_symlink()
                else ["directory"]
                if p.is_dir()
                else ["file", digest(p.read_bytes())]
            )
            for p in sorted(root.rglob("*"))
        }
    # A cached Chroma HNSW reader misses vectors committed by the stdio process.
    # Read persisted records in a fresh process; never truncate or omit embeddings.
    reader = r"""
import json
import sys

attempts = []
def guard(event, arguments):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
        attempts.append(event)
        raise RuntimeError("Network forbidden in index snapshot: " + event)
sys.addaudithook(guard)

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path=sys.argv[1], settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(sys.argv[2])
records = collection.get(include=["documents", "metadatas", "embeddings"])
rows = sorted(
    [key, document, metadata, [float(value) for value in embedding]]
    for key, document, metadata, embedding in zip(
        records["ids"], records["documents"], records["metadatas"],
        records["embeddings"], strict=True
    )
)
if attempts:
    raise RuntimeError("Index snapshot attempted network access: " + repr(attempts))
print(json.dumps(rows))
"""
    config = fixture["store"].config
    snapshot = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            reader,
            str(config.persist_dir),
            config.collection_name,
        ],
        cwd=fixture["vault"].parent,
        env=offline_environment(fixture["vault"].parent),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    trees["index"] = json.loads(snapshot.stdout)
    return trees


def mcp_derived_fixture(root: Path, name: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Persist adversarial provenance and substantive real optional producer data."""
    from collections import Counter

    import frontmatter
    import networkx as nx

    from vaultmind.config import GraphConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.memory.models import OutcomeStatus
    from vaultmind.memory.procedural import ProceduralMemory
    from vaultmind.memory.store import EpisodeStore
    from vaultmind.tracking.preferences import Interaction, InteractionType, PreferenceStore

    full = name == "derived-full"
    spec, fixture, config = mcp_root_fixture(root, "root-full" if full else "root-scoped")
    vault = fixture["vault"]
    source = next(path for path in fixture["corpus"] if path.endswith("/source.md"))
    peer, private = "02-projects/peer.md", "03-private/closer-00.md"
    for path in ("02-projects/typed.md", ".hidden/hidden.md", "02-projects/.hidden.md"):
        content = "---\ntype: project\n---\n# Typed\n"
        target = vault / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        fixture["corpus"][path] = digest(content.encode())
    graph_path = root / "data" / "graph.json"
    graph = KnowledgeGraph(GraphConfig(persist_path=graph_path))
    # The real node-link loader supports multigraphs. Two parallel edges engage
    # scan's confidence drift and relationship shift rather than an empty control.
    graph._graph = nx.MultiDiGraph()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, dict[str, Any]]] = []
    missing = object()

    def node(label: str, provenance: Any = missing) -> None:
        attrs: dict[str, Any] = {"label": label, "type": "concept", "confidence": 1.0}
        if provenance is not missing:
            attrs["source_notes"] = provenance
        nodes[label] = attrs
        graph._graph.add_node(label.lower(), **attrs)

    def edge(
        a: str,
        b: str,
        provenance: Any = missing,
        *,
        relation: str = "related_to",
        confidence: float = 1.0,
    ) -> None:
        attrs: dict[str, Any] = {"relation": relation, "confidence": confidence}
        if provenance is not missing:
            attrs["source_notes"] = provenance
        edges.append((a, b, attrs))
        graph._graph.add_edge(a.lower(), b.lower(), **attrs)

    node("Start", [source])
    node("Middle", ["./" + peer, str(vault / peer), "02-projects/allowed-alias.md"])
    node("Finish", ["02-projects/typed.md"])
    node("EdgeOnly", [peer])
    edge("Start", "Middle", ["./" + source, source])
    edge("Middle", "Finish", [peer])
    bad_sources = [
        missing,
        [],
        peer,
        42,
        None,
        {"note": source},
        [42],
        [False],
        [""],
        [source, 42],
        ["02-projects/missing.md"],
        [source, "02-projects/missing.md"],
        [source, private],
        [private],
        ["02-projects/private-alias.md"],
        ["../outside/outside.md"],
        [str(fixture["outside"] / "outside.md")],
        ["02-projects/outside-alias.md"],
        ["02-projects/broken-alias.md"],
        ["02-projects/loop.md"],
        ["02-projects/not-markdown.txt"],
        ["02-projects"],
    ]
    for i, provenance in enumerate(bad_sources):
        label = f"DERIVED-REJECTED-{i:02d}"
        node(label, provenance)
        edge("Start", label, [source])
        # Two valid endpoints make edge authorization independently necessary.
        edge("Start", "EdgeOnly", provenance)
    edge("Start", "Finish", [source, private])
    node("DriftA", [private])
    node("DriftB", [private])
    edge("DriftA", "DriftB", [source], relation="depends_on", confidence=0.1)
    edge("DriftA", "DriftB", [private], relation="competes_with", confidence=0.9)
    # Real producer merges omit unknown contributions from otherwise allowed source
    # lists. Transfer their complete records into the persisted multigraph without
    # changing attributes; retain the independent parallel-edge evolution controls.
    producer = KnowledgeGraph(GraphConfig(persist_path=root / "merge-producer.json"))
    for order in ("UnknownFirst", "AllowedFirst"):
        a, b = order + "A", order + "B"
        unknown_first = order == "UnknownFirst"
        marker = "UNPROVENANCED-FIRST" if unknown_first else "UNPROVENANCED-SECOND"
        for origin in ("", source) if unknown_first else (source, ""):
            for label in (a, b):
                producer.add_entity(
                    label,
                    "concept",
                    source_note=origin,
                    confidence=0.7 if origin else 0.95,
                    **({} if origin else {"private_context": marker}),
                )
            producer.add_relationship(
                a,
                b,
                relation="related_to" if origin else "depends_on",
                source_note=origin,
                confidence=0.7 if origin else 0.95,
            )
        for label in (a, b):
            nodes[label] = {
                "label": label,
                "type": "concept",
                "confidence": 0.95,
                "source_notes": [source],
                "private_context": marker,
            }
        edges.append(
            (
                a,
                b,
                {
                    "relation": "depends_on" if unknown_first else "related_to",
                    "confidence": 0.95,
                    "source_notes": [source],
                },
            )
        )
    graph._graph.update(producer._graph)
    graph.save()
    dismissed = root / ".vaultmind" / "data" / "evolution_dismissed.json"
    dismissed.parent.mkdir(parents=True, exist_ok=True)
    dismissed.write_text('{"dismissed": ["fixture-unrelated-dismissal"]}\n', encoding="utf-8")
    episodes_path, workflows_path, preferences_path = (
        root / "data" / filename for filename in ("episodes.db", "procedural.db", "preferences.db")
    )
    episodes = EpisodeStore(episodes_path)
    episode = episodes.create(
        "DERIVED-PRIVATE-DECISION",
        context="DERIVED-PRIVATE-CONTEXT",
        entities=["DriftA"],
        source_notes=[source, private],
    )
    episodes.resolve(
        episode.episode_id,
        "DERIVED-PRIVATE-OUTCOME",
        OutcomeStatus.SUCCESS,
        ["DERIVED-PRIVATE-LESSON"],
    )
    resolved = episodes.get(episode.episode_id)
    assert resolved is not None
    pending = episodes.create("DERIVED-PENDING", entities=["DriftA"])
    workflows = ProceduralMemory(workflows_path)
    workflow = workflows.create_workflow(
        "DERIVED-PRIVATE-WORKFLOW",
        "DERIVED-PRIVATE-DESCRIPTION",
        ["DERIVED-PRIVATE-STEP"],
        "harbor winter",
        [episode.episode_id],
    )
    preferences = PreferenceStore(preferences_path)
    preferences.record(Interaction(InteractionType.SEARCH, "DERIVED-PRIVATE-INTERACTION"))
    for producer in (episodes, workflows, preferences):
        producer.close()
    fixture["business"] = [episodes_path, workflows_path, preferences_path]
    fixture["immutable_files"] = [graph_path, dismissed]
    derived_tools = [
        "graph_query",
        "graph_path",
        "vault_stats",
        "episode_query",
        "workflow_suggest",
        "graph_evolution",
        "recent_activity",
    ]
    text = config.read_text(encoding="utf-8").replace(
        '"read_frontmatter", "list_folder_index", "follow_links"]',
        '"read_frontmatter", "list_folder_index", "follow_links", '
        + ", ".join(json.dumps(tool) for tool in derived_tools)
        + "]",
    )
    text += (
        f"\n[episodic]\nenabled = true\ndb_path = {json.dumps(str(episodes_path))}\n"
        f"[procedural]\nenabled = true\ndb_path = {json.dumps(str(workflows_path))}\n"
        f"[tracking]\nenabled = true\ndb_path = {json.dumps(str(preferences_path))}\n"
        "[evolution]\nenabled = true\nconfidence_drift_threshold = 0.3\n"
    )
    config.write_text(text, encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def call(
        tool: str, arguments: dict[str, Any], expected: dict[str, Any], policy: str = "filter"
    ) -> None:
        if not full and tool in {"graph_query", "graph_path", "vault_stats"}:
            expected, policy = {"error": True}, "deny"
        denied = policy == "deny"
        calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "expected": expected,
                "audit": "DENIED" if denied else "OK",
                "kind": "denied" if denied else None,
                "policy": "allow" if full and not denied else policy,
            }
        )

    safe_labels = {"Start", "Middle", "Finish", "EdgeOnly"}

    def neighborhood(label: str, depth: int = 1) -> dict[str, Any]:
        if label not in nodes:
            return {"entity": None, "outgoing": [], "incoming": [], "neighbors": []}
        near = {label}
        for _ in range(depth):
            previous = set(near)
            near |= {b for a, b, _ in edges if a in previous}
            near |= {a for a, b, _ in edges if b in previous}
        return {
            "entity": {"id": label.lower(), **nodes[label]},
            "outgoing": [
                {"target": b, "relation": attrs["relation"], "confidence": attrs["confidence"]}
                for a, b, attrs in edges
                if a == label
            ],
            "incoming": [
                {"source": a, "relation": attrs["relation"], "confidence": attrs["confidence"]}
                for a, b, attrs in edges
                if b == label
            ],
            "neighbors": [{"id": other.lower(), **nodes[other]} for other in sorted(near - {label})]
            if depth > 1
            else [],
        }

    call("graph_query", {"entity": "Start", "depth": 2}, neighborhood("Start", 2))
    call("graph_query", {"entity": "Middle"}, neighborhood("Middle"))
    for label in nodes:
        if label not in safe_labels:
            call("graph_query", {"entity": label}, neighborhood(label))
    for order in ("UnknownFirst", "AllowedFirst"):
        call(
            "graph_path",
            {"source": order + "A", "target": order + "B"},
            {"path": [order + "A", order + "B"], "length": 1},
        )
    call(
        "graph_path",
        {"source": "Start", "target": "Finish"},
        {"path": ["Start", "Finish"], "length": 1},
    )
    call(
        "graph_path",
        {"source": "Start", "target": "EdgeOnly"},
        {"path": ["Start", "EdgeOnly"], "length": 1},
    )
    canonical_notes = sorted(
        p
        for p in fixture["corpus"]
        if not any(part.startswith(".") for part in Path(p).parts)
        and (full or p.startswith("02-projects/"))
    )
    note_types = [frontmatter.load(vault / path).get("type") for path in canonical_notes]
    call(
        "vault_stats",
        {},
        {
            "total_notes": len(canonical_notes),
            "by_type": dict(Counter(kind for kind in note_types if kind)),
            "by_folder": dict(Counter(Path(path).parts[0] for path in canonical_notes)),
            "graph": {"entities": len(nodes), "edges": len(edges)},
        },
    )
    for days in (7, 0):
        created = canonical_notes if days else []
        call(
            "recent_activity",
            {"days": days},
            {
                "days": days,
                "created": created,
                "modified": [],
                "created_count": len(created),
                "modified_count": 0,
            },
        )

    def episode_payload(ep: Any) -> dict[str, Any]:
        return {
            "episode_id": ep.episode_id,
            "decision": ep.decision,
            "context": ep.context,
            "outcome": ep.outcome,
            "status": ep.outcome_status.value,
            "lessons": ep.lessons,
            "entities": ep.entities,
            "created": ep.created.isoformat(),
        }

    for arguments, selected in (
        ({"status": "resolved", "limit": 1}, [resolved]),
        ({"status": "pending", "limit": 1}, [pending]),
        ({"entity": "DriftA", "limit": 1}, [pending]),
        ({"entity": "nonexistent", "limit": 1}, []),
    ):
        call(
            "episode_query",
            arguments,
            {"episodes": [episode_payload(ep) for ep in selected], "count": len(selected)}
            if full
            else {"error": True},
            "allow" if full else "deny",
        )
    call(
        "workflow_suggest",
        {"context": "harbor winter"},
        {
            "workflow": {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": workflow.steps,
                "trigger_pattern": workflow.trigger_pattern,
                "success_rate": 0.0,
                "usage_count": 0,
            },
        }
        if full
        else {"error": True},
        "allow" if full else "deny",
    )
    call(
        "workflow_suggest",
        {"context": "no matching terms"},
        {"workflow": None, "message": "No matching workflow found"} if full else {"error": True},
        "allow" if full else "deny",
    )
    signals = []
    for kind, detail, severity in (
        ("confidence_drift", "Confidence ranges from 0.1 to 0.9 across 2 source(s)", 1.0),
        ("relationship_shift", "Relationship changed: competes_with vs depends_on", 0.8),
    ):
        raw = f"DriftA:DriftB:{kind}:{':'.join(sorted([source, private]))}"
        signals.append(
            {
                "evolution_id": digest(raw.encode())[:16],
                "entity_a": "DriftA",
                "entity_b": "DriftB",
                "signal_type": kind,
                "detail": detail,
                "severity": severity,
                "source_notes": [source, private],
            }
        )
    for threshold in (0.0, 0.9):
        selected_signals = [s for s in signals if s["severity"] >= threshold]
        call(
            "graph_evolution",
            {"min_severity": threshold},
            {"signals": selected_signals, "count": len(selected_signals)}
            if full
            else {"error": True},
            "allow" if full else "deny",
        )
    spec.update(
        {
            "name": name,
            "calls": calls,
            "forbidden": []
            if full
            else [
                "DERIVED-REJECTED",
                "DERIVED-PRIVATE",
                "UNPROVENANCED",
                "DERIVED-PENDING",
                "DriftA",
                "DriftB",
                "ROOT-PRIVATE",
                "03-private",
                "closer-",
            ],
        }
    )
    if name == "derived-writes":
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                'folder_scope = ["02-projects"]',
                'folder_scope = ["02-projects", "00-inbox", "_wiki"]',
            ),
            encoding="utf-8",
        )
        spec["calls"] = []
        for tool, arguments, expected in (
            (
                "vault_write",
                {
                    "path": "02-projects/bounded-write.md",
                    "content": "# Bounded Write\n\nAcceptance bounded write.\n",
                },
                {"status": "ok", "path": "02-projects/bounded-write.md"},
            ),
            (
                "capture",
                {"content": "Acceptance bounded capture.", "title": "Bounded Capture"},
                {"status": "ok", "path": "<generated>"},
            ),
            (
                "capture_note",
                {
                    "content": "Acceptance bounded structured capture.",
                    "title": "Bounded Structured",
                    "folder": "02-projects",
                },
                {"status": "ok", "path": "<generated>", "title": "Bounded Structured"},
            ),
        ):
            spec["calls"].append(
                {
                    "tool": tool,
                    "arguments": arguments,
                    "expected": expected,
                    "audit": "OK",
                    "kind": None,
                    "policy": "allow",
                }
            )
    return spec, fixture, config


def mcp_derived_snapshot(fixture: dict[str, Any]) -> dict[str, Any]:
    """Canonical bytes + SQL business rows; audit appends alone are exempt."""
    import sqlite3
    from contextlib import closing

    result = mcp_root_snapshot(fixture)
    result["note_mtimes"] = {
        str(path.relative_to(fixture["vault"])): path.stat().st_mtime_ns
        for path in fixture["vault"].rglob("*")
        if not path.is_symlink() and path.is_file()
    }
    result["immutable_files"] = {
        str(path): [digest(path.read_bytes()), path.stat().st_mtime_ns]
        for path in fixture["immutable_files"]
    }
    result["business"] = {}
    for path in fixture["business"]:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
            result["business"][str(path)] = list(connection.iterdump())
    # Reopening an unchanged SQLite WAL can touch its clock without changing bytes.
    # Keep journal bytes, full SQL state, and main-database clocks under the guard.
    result["business_files"] = {
        str(candidate): digest(candidate.read_bytes())
        for path in fixture["business"]
        for candidate in (path, Path(str(path) + "-wal"))
        if candidate.exists()
    }
    result["business_mtimes"] = {str(path): path.stat().st_mtime_ns for path in fixture["business"]}
    return result


def mcp_scenario(name: str, root: Path) -> dict[str, Any]:
    """Return the CLI arguments and the independently written expectation for each call."""
    vault = str(root / "vault")
    outside_file = str(root / "outside" / "outside.md")
    scoped = name in ("planner-scoped", "planner-configured-override")
    override = name == "planner-configured-override"
    config = str(root / "config.toml")

    def idx(rel: str) -> dict[str, Any]:
        spec = MCP_NOTES[rel]
        return {
            "path": rel,
            "title": spec["title"],
            "description": spec["summary"],
            "note_type": spec["type"],
            "tags": spec["tags"],
        }

    def listing(folder: str, rels: list[str]) -> dict[str, Any]:
        return {"folder": folder, "notes": [idx(r) for r in rels], "count": len(rels)}

    def front(rel: str) -> dict[str, Any]:
        spec = MCP_NOTES[rel]
        frontmatter: dict[str, Any] = {
            "title": spec["title"],
            "type": spec["type"],
            "created": spec["created"],
        }
        for key in ("tags", "authority", "description"):
            if spec[key] not in (None, []):
                frontmatter[key] = spec[key]
        return {
            "path": rel,
            "title": spec["title"],
            "note_type": spec["type"],
            "tags": spec["tags"],
            "authority": spec["authority"] or 0,
            "status": "active",
            "source": "manual",
            "created": f"{spec['created']}T00:00:00",
            "modified": MCP_ISO,
            "frontmatter": frontmatter,
        }

    def follow(
        rel: str, out: list[tuple[str, str | None]], back: list[tuple[str, str]]
    ) -> dict[str, Any]:
        return {
            "path": rel,
            "title": MCP_NOTES[rel]["title"],
            "out_links": [{"title": t, "path": p} for t, p in out],
            "backlinks": [{"title": t, "path": p} for t, p in back],
            "out_link_count": len(out),
            "backlink_count": len(back),
        }

    def call(
        tool: str,
        arguments: dict[str, Any],
        expected: dict[str, Any],
        audit: str | tuple[str, ...] = "OK",
        kind: str | None = None,
    ) -> dict[str, Any]:
        return {
            "tool": tool,
            "arguments": arguments,
            "expected": expected,
            "audit": audit,
            "kind": kind,
        }

    # Ordinary strict-resolution outcomes. ``denied``/``not-found``/``path-not-allowed`` errors
    # may name only the caller's own path; ``syscall`` is the real OS error of a path the
    # profile allows (never a typed denial), so only the unknown-target oracle applies.
    denied = ("denied", "DENIED")
    not_found = ("not-found", "OK")
    path_not_allowed = ("path-not-allowed", "OK")
    syscall = ("syscall", ("OK", "ERROR"))

    def probe(
        target: str, *, read: tuple[str, Any], index: tuple[str, Any]
    ) -> list[dict[str, Any]]:
        """The same caller path through all three traversal tools."""
        out = []
        for tool, (kind, audit) in (
            ("read_frontmatter", read),
            ("follow_links", read),
            ("list_folder_index", index),
        ):
            key = "folder" if tool == "list_folder_index" else "path"
            out.append(call(tool, {key: target}, {"error": True}, audit, kind))
        return out

    enforced = mcp_permission_denial_enforced()

    inbox = "00-inbox/inbox-note.md"
    hub = "02-projects/hub.md"
    peer = "02-projects/peer.md"
    nested = "02-projects/sub/nested.md"
    secret = "03-private/secret.md"
    outside_root = root / "outside"
    private_root = root / "vault" / "03-private"
    # Independent fixture target oracle: exact relative/absolute private and outside
    # targets (including loop targets) plus basenames, never derived from the calls.
    outside_targets = [
        str(outside_root),
        outside_file,
        *(str(outside_root / n) for n in ("loop-a.md", "loop-b.md", "loopdir-a", "loopdir-b")),
        str(outside_root / "sealed-out"),
        str(outside_root / "sealed-out" / "hidden-out.md"),
        "outside.md",
        "Outside Secret",
        "OUTSIDE-SENTINEL",
        "sealed-out",
        "hidden-out",
        MCP_SEALED_OUTSIDE,
    ]
    private_targets = [
        secret,
        str(private_root),
        *(
            str(private_root / n)
            for n in ("secret.md", "shadow.md", "loop-a.md", "loop-b.md", "loopdir-a", "loopdir-b")
        ),
        str(private_root / "sealed-priv"),
        str(private_root / "sealed-priv" / "hidden-priv.md"),
    ]
    # Absolute unknown physical targets no call ever supplies verbatim (an OS error filename
    # for a suffix/overlong alias path would carry exactly these). The stderr scan applies
    # them with NO masking, so no call's arguments can exempt another call.
    unknown_private = [
        *(
            str(private_root / n)
            for n in ("shadow.md", "loop-a.md", "loop-b.md", "loopdir-a", "loopdir-b")
        ),
        str(private_root / "sealed-priv"),
        str(private_root / "sealed-priv" / "hidden-priv.md"),
        str(private_root / "secret.md") + "/x",
        str(private_root) + "/" + MCP_LONG_COMPONENT,
    ]
    unknown_outside = [
        *(
            str(outside_root / n)
            for n in ("loop-a.md", "loop-b.md", "loopdir-a", "loopdir-b", "sealed-out")
        ),
        str(outside_root / "sealed-out" / "hidden-out.md"),
        outside_file + "/x",
    ]
    calls: list[dict[str, Any]] = []
    if name == "planner-scoped":
        profile = "planner"
        cli_args = ["-c", config, "-v", "mcp-serve", "--profile", "planner"]
        aggregate = [inbox, hub, peer, nested]
        forbidden = [*MCP_PRIVATE_TOKENS, MCP_HUB_SENTINEL, *outside_targets, *private_targets]
        calls += [
            call("list_folder_index", {}, listing("", aggregate)),
            call("list_folder_index", {"folder": ""}, listing("", aggregate)),
            call("list_folder_index", {"folder": "."}, listing(".", aggregate)),
            call("list_folder_index", {"folder": vault}, listing(vault, aggregate)),
            call(
                "list_folder_index",
                {"folder": "02-projects"},
                listing("02-projects", [hub, peer, nested]),
            ),
            call(
                "list_folder_index",
                {"folder": "02-projects/sub"},
                listing("02-projects/sub", [nested]),
            ),
            call("list_folder_index", {"folder": "00-inbox"}, listing("00-inbox", [inbox])),
            call("read_frontmatter", {"path": hub}, front(hub)),
            call("read_frontmatter", {"path": nested}, front(nested)),
            call("read_frontmatter", {"path": inbox}, front(inbox)),
            call(
                "follow_links",
                {"path": hub},
                follow(
                    hub,
                    [("Planner Peer", peer)],
                    [("Inbox Note", inbox), ("Planner Peer", peer)],
                ),
            ),
            call(
                "follow_links",
                {"path": peer},
                follow(peer, [("Hub Note", hub)], [("Hub Note", hub)]),
            ),
            call("follow_links", {"path": inbox}, follow(inbox, [("Hub Note", hub)], [])),
        ]
        for tool in ("capture", "capture_note"):
            calls.append(
                call(
                    tool,
                    {"content": "PLANNER-CAPTURE-BODY-NOT-FOR-AUDIT"},
                    {"error": True},
                    "DENIED",
                )
            )
        # Resolved-forbidden or private targets are denied by the policy; unresolvable
        # (loop) or outside-resolving folders are rejected by the path validator.
        for folder in (
            secret.rsplit("/", 1)[0],
            "03-private/missing",
            str(private_root),
            "02-projects/linkdir-private",
            "02-projects/leak-private.md",
        ):
            calls.append(call("list_folder_index", {"folder": folder}, {"error": True}, "DENIED"))
        for folder in (
            "../outside",
            str(outside_root),
            "02-projects/linkdir-ext",
            "02-projects/leak-ext.md",
            "02-projects/loop-private.md",
            "02-projects/loop-ext.md",
            "02-projects/loopdir-private",
            "02-projects/loopdir-ext",
            "02-projects/self.md",
            "02-projects/missing",
        ):
            calls.append(call("list_folder_index", {"folder": folder}, {"error": True}))
        for tool in ("read_frontmatter", "follow_links"):
            for target in (
                secret,
                "03-private/missing.md",
                "03-private",
                str(private_root / "secret.md"),
                str(private_root),
                "02-projects/leak-private.md",
                "02-projects/leak-ext.md",
                "02-projects/loop-private.md",
                "02-projects/loop-ext.md",
                "02-projects/linkdir-private",
                "02-projects/linkdir-ext",
                "02-projects/loopdir-private",
                "02-projects/loopdir-ext",
                "02-projects/self.md",
                outside_file,
                str(outside_root),
                "../outside/outside.md",
                "../outside",
            ):
                calls.append(call(tool, {"path": target}, {"error": True}, "DENIED"))
            calls.append(call(tool, {"path": "02-projects/missing.md"}, {"error": True}))
        # Ordinary strict-resolution failures (ENOTDIR, EACCES, ENAMETOOLONG, ENOENT) keep their
        # lookup semantics: real policy/containment/not-found outcomes that name only the
        # caller's own path, never the unknown physical target behind an alias.
        private_like = [
            "02-projects/leak-private.md/x",
            "02-projects/linkdir-private/secret.md/x",
            "02-projects/linkdir-private/" + MCP_LONG_COMPONENT,
        ]
        outside_like = [
            "02-projects/leak-ext.md/x",
            "02-projects/linkdir-ext/outside.md/x",
        ]
        if enforced:
            private_like += [
                "02-projects/lockdir-private",
                "02-projects/lockdir-private/hidden-priv.md",
                "02-projects/lockfile-private.md",
            ]
            outside_like += [
                "02-projects/lockdir-ext",
                "02-projects/lockdir-ext/hidden-out.md",
                "02-projects/lockfile-ext.md",
            ]
        for target in private_like:
            calls += probe(target, read=denied, index=denied)
        for target in outside_like:
            calls += probe(target, read=denied, index=path_not_allowed)
        # Allowed controls: a file used as a directory and a missing directory are not found.
        for target in ("02-projects/hub.md/child", "02-projects/missing-dir/child.md"):
            calls += probe(target, read=not_found, index=not_found)
        calls += probe("02-projects/" + MCP_LONG_COMPONENT + ".md", read=syscall, index=syscall)
        if enforced:
            calls.append(
                call(
                    "list_folder_index",
                    {"folder": "02-projects/sealed-pub"},
                    listing("02-projects/sealed-pub", []),
                )
            )
            calls += probe("02-projects/sealed-pub/hidden-pub.md", read=syscall, index=syscall)
        stderr_forbidden = [*unknown_private, *unknown_outside]
    elif name == "planner-configured-override":
        profile = "planner"
        cli_args = ["-c", config, "mcp-serve", "--profile", "planner"]
        forbidden = [
            "Hub Note",
            "Planner Peer",
            "Private Secret",
            "Hub description.",
            "Allowed peer description.",
            "PRIVATE-DESCRIPTION-SENTINEL",
            MCP_HUB_SENTINEL,
            "hub.md",
            "peer.md",
            "secret.md",
            *outside_targets,
            *private_targets,
        ]
        calls += [
            call("list_folder_index", {}, listing("", [inbox])),
            call("list_folder_index", {"folder": "00-inbox"}, listing("00-inbox", [inbox])),
            call(
                "list_folder_index",
                {"folder": "02-projects"},
                {"error": True},
                "DENIED",
            ),
            call("read_frontmatter", {"path": inbox}, front(inbox)),
            call("read_frontmatter", {"path": hub}, {"error": True}, "DENIED"),
            call("follow_links", {"path": inbox}, follow(inbox, [], [])),
            call("follow_links", {"path": hub}, {"error": True}, "DENIED"),
            call(
                "vault_search",
                {"query": "override probe"},
                {"error": True},
                "DENIED",
            ),
        ]
        stderr_forbidden = [*unknown_private, *unknown_outside]
    else:
        full = name == "full-control"
        profile = "full" if full else "researcher"
        cli_args = ["-c", config, "mcp-serve"] + (["--profile", "full"] if full else [])
        everything = [inbox, hub, peer, nested, secret]
        forbidden = [MCP_HUB_SENTINEL, *outside_targets]
        calls += [
            call("list_folder_index", {}, listing("", everything)),
            call("list_folder_index", {"folder": vault}, listing(vault, everything)),
            call("list_folder_index", {"folder": "03-private"}, listing("03-private", [secret])),
            call("read_frontmatter", {"path": secret}, front(secret)),
            call("read_frontmatter", {"path": hub}, front(hub)),
            call(
                "follow_links",
                {"path": hub},
                follow(
                    hub,
                    [
                        ("Planner Peer", peer),
                        ("Private Secret", secret),
                        ("Nowhere Target", None),
                    ],
                    [
                        ("Inbox Note", inbox),
                        ("Planner Peer", peer),
                        ("Private Secret", secret),
                    ],
                ),
            ),
            call(
                "read_frontmatter",
                {"path": "../outside/outside.md"},
                {"error": True},
                "DENIED",
            ),
            call(
                "list_folder_index",
                {"folder": "../outside"},
                {"error": True},
            ),
            call(
                "list_folder_index",
                {"folder": "02-projects"},
                listing("02-projects", [hub, peer, nested]),
            ),
            call("read_frontmatter", {"path": "02-projects/self.md"}, {"error": True}, "DENIED"),
            call("follow_links", {"path": "02-projects/self.md"}, {"error": True}, "DENIED"),
            call(
                "read_frontmatter", {"path": "02-projects/loop-ext.md"}, {"error": True}, "DENIED"
            ),
            call("read_frontmatter", {"path": outside_file}, {"error": True}, "DENIED"),
            call("list_folder_index", {"folder": "02-projects/loopdir-ext"}, {"error": True}),
            call("list_folder_index", {"folder": str(outside_root)}, {"error": True}),
            call("read_frontmatter", {"path": "02-projects/missing.md"}, {"error": True}),
        ]
        # The same ordinary failures for wildcard profiles, including unreadable candidates
        # that must not change the valid counts asserted above and below.
        for target in (
            "02-projects/hub.md/child",
            "02-projects/linkdir-private/secret.md/x",
            "02-projects/missing-dir/child.md",
        ):
            calls += probe(target, read=not_found, index=not_found)
        calls += probe("02-projects/linkdir-ext/outside.md/x", read=denied, index=path_not_allowed)
        calls += probe("02-projects/" + MCP_LONG_COMPONENT + ".md", read=syscall, index=syscall)
        if enforced:
            for folder in ("02-projects/lockdir-private", "02-projects/sealed-pub"):
                calls.append(call("list_folder_index", {"folder": folder}, listing(folder, [])))
            for target in (
                "02-projects/lockdir-private/hidden-priv.md",
                "02-projects/lockfile-private.md",
                "02-projects/sealed-pub/hidden-pub.md",
            ):
                calls += probe(target, read=syscall, index=syscall)
            for target in (
                "02-projects/lockdir-ext",
                "02-projects/lockdir-ext/hidden-out.md",
                "02-projects/lockfile-ext.md",
            ):
                calls += probe(target, read=denied, index=path_not_allowed)
        stderr_forbidden = list(unknown_outside)
    return {
        "name": name,
        "profile": profile,
        "scoped": scoped,
        "override": override,
        "cli_args": cli_args,
        "calls": calls,
        "forbidden": forbidden,
        "stderr_forbidden": stderr_forbidden,
    }


def mcp_normalize(tool: str, payload: Any) -> Any:
    """Make order-free lists comparable and mask the wall-clock ``modified`` value."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if (
        tool == "read_frontmatter"
        and isinstance(out.get("modified"), str)
        and MCP_ISO_PATTERN.match(out["modified"])
    ):
        out["modified"] = MCP_ISO
    if tool == "list_folder_index" and isinstance(out.get("notes"), list):
        out["notes"] = sorted(out["notes"], key=lambda e: str(e.get("path")))
    if tool == "vault_search" and isinstance(out.get("results"), list):
        out["results"] = sorted(
            [
                {**hit, "distance": round(hit["distance"], 4)}
                if isinstance(hit.get("distance"), (int, float))
                else hit
                for hit in out["results"]
            ],
            key=lambda hit: hit["chunk_id"],
        )
    if tool == "follow_links":
        for key in ("out_links", "backlinks"):
            if isinstance(out.get(key), list):
                out[key] = sorted(
                    out[key], key=lambda link: (str(link.get("path")), str(link.get("title")))
                )
    if tool == "graph_query":
        for key in ("outgoing", "incoming", "neighbors"):
            if isinstance(out.get(key), list):
                out[key] = sorted(out[key], key=lambda row: json.dumps(row, sort_keys=True))
    return out


def mcp_knowledge_fixture(root: Path, config: Path, spec: dict[str, Any]) -> None:
    """Publish real conflicting evidence and exercise the shipped selected default."""
    import tomllib

    from vaultmind.config import load_settings
    from vaultmind.knowledge.query import query_knowledge
    from vaultmind.wiki.workspace import WikiWorkspace, proposal_digest

    public_defaults = Path(__file__).resolve().parents[1] / "config" / "default.toml"
    researcher = tomllib.loads(public_defaults.read_text(encoding="utf-8"))["mcp"]["profiles"][
        "researcher"
    ]
    config.write_text(
        config.read_text(encoding="utf-8")
        + f"\n[autonomy]\ndb_path = {json.dumps(str(root / 'data' / 'review.db'))}\n"
        + "\n[mcp.profiles.researcher]\n"
        + "".join(f"{key} = {json.dumps(value)}\n" for key, value in researcher.items()),
        encoding="utf-8",
    )
    settings = load_settings(config)
    source_ids = ["sdk-cache-30", "sdk-cache-7"]
    draft = root / "reviewed-cache.md"
    draft.write_text(
        "# Cache\n\n## Current understanding\n\nCache retention is 30 days. "
        "[[_wiki/sources/sdk-cache-30]]\n\n## Conflicting information\n\n"
        "The second policy says 7 days; this is unresolved. [[_wiki/sources/sdk-cache-7]]\n",
        encoding="utf-8",
    )
    with WikiWorkspace(settings) as workspace:
        for source_id, days in zip(source_ids, (30, 7), strict=True):
            original = root / f"{source_id}.md"
            original.write_text(f"Cache retention is {days} days.\n", encoding="utf-8")
            workspace.add_source(original, source_id, uri=f"urn:offline:{source_id}")
        proposal = workspace.propose("wiki/cache.md", source_ids, draft_file=draft)
        workspace.review(proposal.proposal_id)
        workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))
    question = "What is the cache retention policy?"
    spec["calls"] = []
    approved = query_knowledge(settings, question, evidence_only=True)
    response = {
        "status": "conflicting",
        "answer": "The retained policies disagree: 30 days versus 7 days; no winner is approved.",
        "citations": [
            row["id"]
            for row in approved["evidence"]
            if "30 days" in row["text"] or "7 days" in row["text"]
        ],
    }
    spec["query_response"] = response
    for arguments in (
        {"question": question},
        {"question": question, "pages": ["wiki/cache.md"]},
        {"question": question, "evidence_only": True},
        {"question": "unmatchedxylophone"},
    ):
        expected = query_knowledge(
            settings,
            arguments["question"],
            arguments.get("pages"),
            evidence_only=True,
        )
        if arguments["question"] == question and not arguments.get("evidence_only"):
            expected = {**expected, **response}
        spec["calls"].append(
            dict(
                tool="knowledge_query",
                arguments=arguments,
                expected=expected,
                audit="OK",
                kind=None,
            )
        )
    spec["calls"].append(
        dict(
            tool="knowledge_query",
            arguments={"question": "QUERY-INPUT-BODY-SENTINEL", "pages": "PAGE-INPUT-SENTINEL"},
            expected={"error": True},
            audit="ERROR",
            kind=None,
        )
    )


async def run_mcp_scenario(name: str, root: Path, check: Any) -> dict[str, Any]:
    """One actual subprocess lifecycle with per-call, per-frame and per-origin assertions."""
    from mcp.types import JSONRPCMessage

    derived = name in MCP_DERIVED_SCENARIOS
    root_notes = name in MCP_ROOT_SCENARIOS or derived
    if derived:
        spec, fixture, config = mcp_derived_fixture(root, name)
        before = mcp_derived_snapshot(fixture)
    elif root_notes:
        spec, fixture, config = mcp_root_fixture(root, name)
        before = mcp_root_snapshot(fixture)
    else:
        spec = mcp_scenario(name, root)
        fixture = mcp_build_fixture(root, spec["scoped"], consumer=True)
        config = mcp_write_config(root, spec["override"])
        if name in MCP_QUERY_SCENARIOS:
            mcp_knowledge_fixture(root, config, spec)
            knowledge_before = {
                path.relative_to(fixture["vault"]).as_posix(): digest(path.read_bytes())
                for folder in ("wiki", "_wiki")
                for path in (fixture["vault"] / folder).rglob("*")
                if path.is_file()
            }
        capture_before = {
            path: digest((fixture["vault"] / path).read_bytes())
            for path in MCP_NOTES
            if not path.startswith("03-private/")
        }
    calls = spec["calls"]
    try:
        run = await mcp_stdio_session(
            root,
            spec["cli_args"],
            [(c["tool"], c["arguments"]) for c in calls],
            root_embeddings=root_notes,
            query_response=spec.get("query_response"),
        )
    finally:
        mcp_restore_permissions(fixture)
    if name in MCP_QUERY_SCENARIOS:
        check(
            name + ".query-no-knowledge-side-effects",
            knowledge_before,
            {
                path.relative_to(fixture["vault"]).as_posix(): digest(path.read_bytes())
                for folder in ("wiki", "_wiki")
                for path in (fixture["vault"] / folder).rglob("*")
                if path.is_file()
            },
        )
    if name == "planner-scoped":
        check(name + ".capture-no-source-storage", False, (fixture["vault"] / "_wiki").exists())
        check(
            name + ".capture-no-projection-or-note-changes",
            capture_before,
            {path: digest((fixture["vault"] / path).read_bytes()) for path in capture_before},
        )
        check(
            name + ".capture-no-new-inbox-notes",
            sorted(path for path in MCP_NOTES if path.startswith("00-inbox/")),
            sorted(
                path.relative_to(fixture["vault"]).as_posix()
                for path in (fixture["vault"] / "00-inbox").rglob("*.md")
            ),
        )
    if root_notes:
        after = mcp_derived_snapshot(fixture) if derived else mcp_root_snapshot(fixture)
        if name == "derived-writes":
            new_paths = []
            for call_spec, result in zip(calls, run["results"], strict=True):
                payload = json.loads(result.content[0].text)
                path = payload.get("path", "")
                destination = fixture["vault"] / path
                check(
                    name + "." + call_spec["tool"] + ".bounded-existing-file",
                    True,
                    destination.is_file()
                    and destination.is_relative_to(fixture["vault"])
                    and path not in before["vault"],
                )
                content = destination.read_text(encoding="utf-8")
                check(
                    name + "." + call_spec["tool"] + ".exact-body-persisted",
                    True,
                    call_spec["arguments"]["content"] in content,
                )
                new_paths.append(path)
                if call_spec["tool"] in ("capture", "capture_note"):
                    original = payload["original"]
                    source_path = fixture["vault"] / original["path"]
                    check(
                        name + "." + call_spec["tool"] + ".exact-source-bytes",
                        True,
                        call_spec["arguments"]["content"].encode() == source_path.read_bytes(),
                    )
                    check(
                        name + "." + call_spec["tool"] + ".source-identity",
                        [True, False, True, digest(source_path.read_bytes())],
                        [
                            payload["preserved"],
                            payload["approved"],
                            original["indexed"],
                            original["sha256"],
                        ],
                    )
                    projection = destination.read_text(encoding="utf-8")
                    check(
                        name + "." + call_spec["tool"] + ".projection-links-source",
                        True,
                        payload["source_id"] in projection,
                    )
            check(name + ".three-distinct-writes", 3, len(set(new_paths)))
            new_records = [r for r in after["index"] if r[0] not in {r[0] for r in before["index"]}]
            check(
                name + ".writes-real-indexed-paths",
                sorted(new_paths),
                sorted({r[2]["note_path"] for r in new_records}),
            )
            after["index"] = [r for r in after["index"] if r not in new_records]
            for path in new_paths:
                after["vault"].pop(path)
                after["note_mtimes"].pop(path)
            after["vault"].pop("00-inbox", None)
            expected_control = {"_wiki", "_wiki/sources", "_wiki/wiki.lock"}
            for call_spec, result in zip(calls, run["results"], strict=True):
                payload = json.loads(result.content[0].text)
                if "original" not in payload:
                    continue
                source_id = payload["source_id"]
                expected_control.update(
                    f"_wiki/sources/{source_id}{suffix}"
                    for suffix in (".md", ".json", ".intake.json", ".processing.json")
                )
                manifest = json.loads(
                    (fixture["vault"] / "_wiki/sources" / f"{source_id}.intake.json").read_text()
                )
                check(name + ".capture-origin", "mcp", manifest["origin"])
                check(
                    name + ".capture-adapter",
                    f"mcp:{spec['profile']}:{call_spec['tool']}",
                    manifest["adapter"],
                )
                metadata = json.loads(
                    (fixture["vault"] / "_wiki/sources" / f"{source_id}.json").read_text()
                )
                check(
                    name + ".capture-manifest-hashbound",
                    "vaultmind-intake:v1:"
                    + digest(
                        (
                            fixture["vault"] / "_wiki/sources" / f"{source_id}.intake.json"
                        ).read_bytes()
                    ),
                    metadata["origin"],
                )
                check(name + ".capture-original-retained", True, manifest["original_retained"])
            actual_control = {path for path in after["vault"] if path.startswith("_wiki")}
            check(
                name + ".only-declared-source-control-artifacts",
                sorted(expected_control),
                sorted(actual_control),
            )
            for path in expected_control:
                after["vault"].pop(path, None)
                after["note_mtimes"].pop(path, None)
        check(
            name + ".read-only-and-denied-mutations-state-unchanged",
            before,
            after,
        )
    state: Path = run["state"]
    events = [
        json.loads(line)
        for line in (state / "launcher.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if root_notes:
        check(
            name + ".embedding.provider-boundary-only",
            [{"kind": "embedding-fixture", "dimensions": 2}],
            [event for event in events if event["kind"] == "embedding-fixture"],
        )
        check(
            name + ".embedding.root-search-exercised",
            any(c["tool"] == "vault_search" and c["expected"].get("results") for c in calls),
            "vault_search" in run["embedding_boundary_tools"],
        )
        states = [event for event in events if event["kind"] == "dispatch-state"]
        check(name + ".dispatch-state.paired", 0, len(states) % 2)
        if derived:
            check(name + ".dispatch-state.complete", len(calls) * 2, len(states))
        for position in range(len(states) // 2):
            entry, exit_event = states[position * 2 : position * 2 + 2]
            check(
                f"{name}.{position}.dispatch-state.lifecycle",
                [("call", entry["tool"]), ("return", entry["tool"])],
                [(entry["event"], entry["tool"]), (exit_event["event"], exit_event["tool"])],
            )
            check(
                f"{name}.{position}.producer-buffers-and-graph-unchanged",
                entry["state"],
                exit_event["state"],
            )
        entries = [event for event in events if event["kind"] == "producer-entry"]
        if derived:
            expected_entries = (
                [
                    ("vaultmind.memory.store", "query_resolved"),
                    ("vaultmind.memory.store", "query_pending"),
                    ("vaultmind.memory.store", "search_by_entity"),
                    ("vaultmind.memory.store", "search_by_entity"),
                    ("vaultmind.memory.procedural", "suggest_workflow"),
                    ("vaultmind.memory.procedural", "suggest_workflow"),
                    ("vaultmind.graph.evolution", "scan"),
                    ("vaultmind.graph.evolution", "scan"),
                ]
                if name == "derived-full"
                else []
            )
            check(
                name + ".real-producer-entry-or-pre-producer-denial",
                expected_entries,
                [(event["module"], event["method"]) for event in entries],
            )
    raw = (state / "stdout.raw").read_bytes().decode("utf-8")
    stderr = mcp_trim((state / "stderr.txt").read_text(encoding="utf-8"))
    stderr_startup = mcp_trim(run["stderr_startup"])
    stderr_chunks = [mcp_trim(chunk) for chunk in run["stderr_chunks"]]
    prefix = f"{name}."
    # Inspect complete stderr, including warnings emitted after startup. Rich's
    # source-location column can interrupt a wrapped diagnostic phrase.
    diagnostics = re.sub(r"\w+\.py:\d+", "", stderr)
    diagnostic_aggregates = re.findall(
        r"\b(?:\d+\s+(?:documents|nodes|edges)|elements\s+in\s+index\s+\d+|"
        r"(?:updating\s+)?n_results\s*=\s*\d+)\b",
        diagnostics,
        flags=re.IGNORECASE,
    )
    if spec["scoped"]:
        check(prefix + "stderr.global-aggregates-absent", [], diagnostic_aggregates)
    elif name == "root-full":
        check(
            prefix + "stderr.unrestricted-index-count-retained",
            [f"{fixture['index_document_count']} documents"],
            [value for value in diagnostic_aggregates if value.endswith(" documents")],
        )
        for diagnostic, pattern in (
            ("elements-in-index", r"\belements\s+in\s+index\s+(\d+)\b"),
            ("bounded-n-results", r"\bn_results\s*=\s*(\d+)\b"),
        ):
            check(
                prefix + f"stderr.unrestricted-{diagnostic}-retained",
                [fixture["index_document_count"]],
                sorted({int(value) for value in re.findall(pattern, diagnostics)}),
            )

    # --- protocol: initialize / list_tools / every stdout line is a valid frame --------------
    check(prefix + "initialize.server-name", "vaultmind", run["init"].serverInfo.name)
    check(prefix + "initialize.tools-capability", True, run["init"].capabilities.tools is not None)
    names = [tool.name for tool in run["tools"]]
    check(prefix + "list_tools.unique", len(names), len(set(names)))
    check(
        prefix + "list_tools.traversal-registered",
        sorted(MCP_TRAVERSAL_TOOLS),
        sorted(set(names) & set(MCP_TRAVERSAL_TOOLS)),
    )
    schemas = {tool.name: tool.inputSchema for tool in run["tools"]}
    check(
        prefix + "list_tools.traversal-schemas",
        {
            "read_frontmatter": (["path"], ["path"]),
            "list_folder_index": (None, ["folder"]),
            "follow_links": (["path"], ["path"]),
        },
        {
            tool: (schemas[tool].get("required"), sorted(schemas[tool]["properties"]))
            for tool in MCP_TRAVERSAL_TOOLS
        },
    )
    check(prefix + "stdout.terminated-by-newline", True, raw.endswith("\n"))
    lines = raw.removesuffix("\n").split("\n") if raw else []
    parse_failures = []
    for line in lines:
        try:
            JSONRPCMessage.model_validate_json(line)
        except Exception as exc:
            parse_failures.append({"line": line[:200], "error": type(exc).__name__})
    check(prefix + "stdout.every-line-valid-mcp-frame", [], parse_failures)
    check(prefix + "stdout.client-parser-errors", [], run["errors"])
    check(prefix + "stdout.line-count-equals-client-frames", len(lines), len(run["frames_in"]))
    sent = run["frames_out"]
    check(
        prefix + "protocol.request-methods",
        ["initialize", "notifications/initialized", "tools/list"] + ["tools/call"] * len(calls),
        [frame.get("method") for frame in sent],
    )
    check(
        prefix + "protocol.response-ids",
        [frame["id"] for frame in sent if "id" in frame],
        [frame.get("id") for frame in run["frames_in"]],
    )
    check(
        prefix + "protocol.responses-only-no-rpc-errors",
        [],
        [frame for frame in run["frames_in"] if "result" not in frame or "error" in frame],
    )

    # --- every call: exact payload, absence of private fields, audit receipt ---------------
    observed_calls = []
    check(prefix + "calls.count", len(calls), len(run["results"]))
    for position, (expected_call, result) in enumerate(zip(calls, run["results"], strict=True)):
        tool = expected_call["tool"]
        arguments = expected_call["arguments"]
        kind = expected_call.get("kind")
        label = f"{prefix}{position:02d}.{tool}"
        check(
            label + ".content-shape",
            [["text", False]],
            [[part.type, bool(result.isError)] for part in result.content],
        )
        text = result.content[0].text
        payload = json.loads(text)
        actual = mcp_normalize(tool, payload)
        if "error" in expected_call["expected"]:
            check(
                label + ".denied-without-result-data",
                True,
                isinstance(payload, dict)
                and set(payload) == {"error"}
                and isinstance(payload["error"], str)
                and bool(payload["error"]),
            )
        else:
            expected_payload = expected_call["expected"]
            if expected_payload.get("path") == "<generated>":
                path = payload.get("path", "")
                source_id = payload.get("source_id")
                folder = "00-inbox" if tool == "capture" else "02-projects"
                check(
                    label + ".generated-path-bounded",
                    True,
                    isinstance(source_id, str)
                    and path == f"{folder}/{source_id}.md"
                    and re.fullmatch(r"[A-Za-z0-9_-]+", source_id) is not None,
                )
                original = payload["original"]
                check(
                    label + ".capture-dispositions",
                    [
                        "ok",
                        False,
                        True,
                        source_id,
                        f"_wiki/sources/{source_id}.md",
                        "added",
                        True,
                        True,
                        False,
                    ],
                    [
                        payload["status"],
                        payload["approved"],
                        payload["preserved"],
                        original["id"],
                        original["path"],
                        original["status"],
                        original["preserved"],
                        original["indexed"],
                        original["approved"],
                    ],
                )
                for stage in ("projection", "index"):
                    check(
                        label + "." + stage + ".succeeded",
                        {"stage": stage, "status": "succeeded", "error": None},
                        payload[stage],
                    )
                expected_payload = {
                    "status": "ok",
                    "path": path,
                    "title": arguments["title"],
                    "source_id": source_id,
                    "approved": False,
                    "preserved": True,
                    "original": original,
                    "projection": payload["projection"],
                    "index": payload["index"],
                }
            check(label + ".payload", mcp_normalize(tool, expected_payload), actual)
            if tool == "vault_search":
                check(
                    label + ".source-only-current-evidence",
                    True,
                    payload["status"] == "source-only"
                    and payload["approved"] is False
                    and all(
                        hit["status"] == "source-only"
                        and hit["approved"] is False
                        and hit["distance"] is None
                        and "-CONTENT" not in hit["content"]
                        for hit in payload["results"]
                    ),
                )
        if root_notes and tool in ("vault_search", "find_duplicates"):
            expected = expected_call["expected"]
            # Empty filtered results legitimately avoid embedding; another tool's
            # provider event must not make this call appear to have embedded.
            embeds = bool(
                expected.get("results") if tool == "vault_search" else expected.get("matches")
            )
            check(
                label + ".embedding.provider-boundary",
                int(embeds),
                run["embedding_boundary_counts"][position],
            )
        # Independent fixture-target oracle on both surfaces; only complete, boundary-delimited
        # caller-supplied argument strings of THIS call are masked (see mcp_leaks).
        oracle: dict[str, Any] = {"forbidden_tokens": len(spec["forbidden"])}
        for surface, value in (("payload", text), ("raw-frame", lines[position + 2])):
            leaks = mcp_leaks(value, spec["forbidden"], arguments)
            check(label + f".{surface}.private-fields-absent", [], leaks)
            oracle[surface.replace("-", "_") + "_leaks"] = leaks
        if kind in MCP_CALLER_ONLY_KINDS:
            residue = mcp_path_residue(payload["error"], arguments)
            check(label + ".error-names-only-caller-path", [], residue)
            oracle["error_path_residue"] = residue
        observed_calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "policy": expected_call.get(
                    "policy",
                    "deny"
                    if "error" in payload
                    else "allow"
                    if spec["profile"] == "full"
                    else "filter",
                ),
                "kind": kind,
                "actual": actual,
                "embedding_boundary_count": run["embedding_boundary_counts"][position],
                "oracle": oracle,
            }
        )
    audit_path = root / ".vaultmind" / "data" / "mcp_audit.jsonl"
    audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    check(prefix + "audit.row-count", len(calls), len(audit))
    expected_rows = []
    for position, c in enumerate(calls):
        options = mcp_audit_options(c)
        result_seen = audit[position]["result"]
        expected_result = result_seen if result_seen in options else options[0]
        expected_rows.append((spec["profile"], c["tool"], expected_result))
    check(
        prefix + "audit.rows",
        expected_rows,
        [(row["profile"], row["tool"], row["result"]) for row in audit],
    )
    # Independent per-row audit oracle: each row is bound to ITS call's arguments (never the
    # union of other calls') and scanned for unknown physical targets and path residue.
    for position, (c, row) in enumerate(zip(calls, audit, strict=True)):
        arguments = c["arguments"]
        label = f"{prefix}{position:02d}.{c['tool']}"
        reason = row.get("reason", "")
        check(
            label + ".audit.params-are-this-call",
            {}
            if c["tool"] in ("capture", "capture_note", "vault_write", "knowledge_query")
            else {key: str(value)[:200] for key, value in arguments.items()},
            row["params"],
        )
        if c["tool"] in ("capture", "capture_note", "vault_write", "knowledge_query"):
            body_key = "question" if c["tool"] == "knowledge_query" else "content"
            check(
                label + ".audit.body-absent",
                False,
                arguments[body_key] in json.dumps(row),
            )
        masked_for = {
            **arguments,
            **{f"{key}@audit": str(value)[:200] for key, value in arguments.items()},
        }
        row_text = json.dumps({"params": row["params"], "reason": reason}, sort_keys=True)
        audit_leaks = mcp_leaks(row_text, spec["forbidden"], masked_for)
        check(label + ".audit.private-fields-absent", [], audit_leaks)
        reason_residue = (
            mcp_path_residue(reason, arguments) if c.get("kind") in MCP_CALLER_ONLY_KINDS else []
        )
        check(label + ".audit.reason-names-only-caller-path", [], reason_residue)
        if c.get("kind") == "syscall":
            check(label + ".audit.real-error-not-typed-denial", True, row["result"] != "DENIED")
        observed_calls[position]["audit"] = {
            "result": row["result"],
            "reason": reason,
            "params": row["params"],
            "private_fields_leaks": audit_leaks,
            "reason_path_residue": reason_residue,
        }

    # --- stderr diagnostics, origin, environment, network guard, lifecycle -------------------
    stderr_flat = mcp_flat(stderr)
    supplied_overlap = [
        token
        for token in spec["stderr_forbidden"]
        if any(token in str(value) for c in calls for value in c["arguments"].values())
    ]
    check(prefix + "stderr.oracle-targets-never-caller-supplied", [], supplied_overlap)
    stderr_leaks = [
        token
        for token in spec["stderr_forbidden"]
        if token in stderr or mcp_flat(token) in stderr_flat
    ]
    check(prefix + "stderr.unknown-physical-targets-absent", [], stderr_leaks)
    # Per-call stderr: the chunk written while THIS call was in flight is scanned with the
    # complete independent oracle (every absolute private/outside root and file target, even
    # ones another call supplied, plus basenames and sentinels), masking only the exact
    # string arguments of THAT call. No later or earlier call can exempt anything.
    startup_flat = mcp_flat(stderr_startup)
    startup_leaks = [
        token
        for token in spec["stderr_forbidden"]
        if token in stderr_startup or mcp_flat(token) in startup_flat
    ]
    check(prefix + "stderr.startup-unknown-targets-absent", [], startup_leaks)
    check(prefix + "stderr.chunk-count", len(calls), len(stderr_chunks))
    absolute_targets = [token for token in spec["forbidden"] if token.startswith("/")]
    for position, (c, chunk) in enumerate(zip(calls, stderr_chunks, strict=True)):
        arguments = c["arguments"]
        label = f"{prefix}{position:02d}.{c['tool']}"
        chunk_leaks = mcp_leaks(chunk, spec["forbidden"], arguments)
        chunk_flat = mcp_flat(mcp_mask(chunk, arguments))
        absolute_leaks = [
            token
            for token in [*absolute_targets, *spec["stderr_forbidden"]]
            if mcp_flat(token) in chunk_flat
        ]
        check(label + ".stderr.private-fields-absent", [], chunk_leaks)
        check(label + ".stderr.absolute-targets-absent", [], absolute_leaks)
        observed_calls[position]["stderr"] = {
            "chunk": chunk,
            "private_fields_leaks": chunk_leaks,
            "absolute_target_leaks": absolute_leaks,
        }
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(event["kind"], []).append(event)
    entry = by_kind["entrypoint"][0]
    check(
        prefix + "origin.single-console-script",
        [["vaultmind.cli:cli"]],
        [e["matches"] for e in by_kind["entrypoint-lookup"]],
    )
    check(
        prefix + "origin.registered-entry-point",
        ["vaultmind", "console_scripts", "vaultmind.cli:cli"],
        [entry["name"], entry["group"], entry["value"]],
    )
    sibling = Path(sys.executable).parent / "vaultmind"
    check(
        prefix + "origin.consumed-sibling-console-script",
        [{"kind": "console-script", "path": str(sibling), "sha256": digest(sibling.read_bytes())}],
        by_kind.get("console-script", []),
    )
    runtime = package_identity()
    check(prefix + "origin.same-package", runtime["module_file"], entry["module_file"])
    check(prefix + "origin.same-code", runtime["code_sha256"], entry["package_sha256"])
    check(prefix + "origin.running-interpreter", sys.executable, entry["executable"])
    check(
        prefix + "origin.not-fixture-or-cwd-shadowed",
        [False, False],
        [
            Path(entry["module_file"]).is_relative_to(root),
            str(root) in entry["sys_path"] or "" in entry["sys_path"],
        ],
    )
    environment = by_kind["environment"][0]
    check(prefix + "environment.home", str(root), environment["home"])
    check(prefix + "environment.no-inherited-config-or-credentials", [], environment["forbidden"])
    check(
        prefix + "network.guard-self-test",
        {"blocked": ["socket.getaddrinfo"], "probe": ["blocked"]},
        {
            "blocked": [e["event"] for e in by_kind.get("network", []) if e["probe"]],
            "probe": [e["result"] for e in by_kind["probe"]],
        },
    )
    child_attempts = [e for e in by_kind.get("network", []) if not e["probe"]]
    check(prefix + "network.outbound-attempts", [], child_attempts)
    if "query_response" in spec:
        check(prefix + "query.one-factory-per-generated-answer", 2, len(by_kind["query-factory"]))
        completions = by_kind["query-provider"]
        check(prefix + "query.no-evidence-only-empty-invalid-retry-completion", 2, len(completions))
        check(
            prefix + "query.blocking-provider-off-main-thread",
            True,
            all(event["thread"] != event["main_thread"] for event in completions),
        )
    pid = by_kind["start"][0]["pid"]
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    check(
        prefix + "lifecycle.clean-exit",
        {"exit": [[0, True]], "alive": False, "forced": False},
        {
            "exit": [[e["code"], e["tee_joined"]] for e in by_kind.get("exit", [])],
            "alive": alive,
            "forced": run["shutdown_seconds"] >= run["termination_timeout"],
        },
    )
    return {
        "profile": spec["profile"],
        "cli_args": spec["cli_args"],
        "configuration_sha256": digest(config.read_bytes()),
        "corpus_sha256": fixture["corpus"],
        "symlinks": fixture["symlinks"],
        "import_origin": entry["module_file"],
        "distribution": {"version": entry["dist_version"], "direct_url": entry["direct_url"]},
        "launcher_events": events,
        "embedding_boundary_tools": run["embedding_boundary_tools"],
        "embedding_boundary_counts": run["embedding_boundary_counts"],
        "stdout_lines": lines,
        "stderr": stderr,
        "audit": audit,
        "calls": observed_calls,
        "listed_tools": names,
        "schemas": schemas,
        "stderr_oracle": {
            "forbidden": spec["stderr_forbidden"],
            "startup_leaks": startup_leaks,
            "leaks": stderr_leaks,
            "caller_supplied_overlap": supplied_overlap,
        },
        "permissions": {
            "denial_enforced": mcp_permission_denial_enforced(),
            "sealed_directories": [str(path.relative_to(root)) for path in fixture["sealed"]],
        },
        "shutdown_seconds": run["shutdown_seconds"],
        "child_network_attempts": child_attempts,
    }


def mcp_direct_dispatcher(root: Path, check: Any) -> None:
    """Explicit full/planner enforcers keep exact dispatcher-level semantics."""
    from vaultmind.config import Settings, VaultConfig
    from vaultmind.mcp.auth import ProfileEnforcer, ProfileError
    from vaultmind.mcp.profiles import load_profile
    from vaultmind.mcp.server import _dispatch_tool
    from vaultmind.vault.parser import VaultParser

    vault = mcp_build_fixture(root, scoped=False)["vault"]
    parser = VaultParser(VaultConfig(path=vault))
    settings = Settings(vault={"path": vault, "excluded_folders": ["wiki", "_wiki"]})
    hub = "02-projects/hub.md"
    secret = "03-private/secret.md"

    def dispatch(tool: str, args: dict[str, Any], **context: Any) -> Any:
        try:
            return _dispatch_tool(
                tool, args, vault, object(), object(), parser, settings=settings, **context
            )
        except ProfileError as exc:
            return {"error": str(exc)}

    full = ProfileEnforcer(load_profile("full"), vault)
    planner = ProfileEnforcer(load_profile("planner"), vault)
    everything = sorted(MCP_NOTES)
    check(
        "direct.full-enforcer-unrestricted-index",
        everything,
        sorted(e["path"] for e in dispatch("list_folder_index", {}, enforcer=full)["notes"]),
    )
    check(
        "direct.planner-enforcer-aggregate-index",
        [rel for rel in everything if not rel.startswith("03-private/")],
        sorted(e["path"] for e in dispatch("list_folder_index", {}, enforcer=planner)["notes"]),
    )
    denied = dispatch("read_frontmatter", {"path": secret}, enforcer=planner)
    check(
        "direct.planner-enforcer-denies-private-read",
        True,
        set(denied) == {"error"} and isinstance(denied["error"], str) and bool(denied["error"]),
    )
    check(
        "direct.planner-denial-private-fields-absent",
        [],
        mcp_leaks(json.dumps(denied), MCP_PRIVATE_TOKENS, {"path": secret}),
    )
    check(
        "direct.planner-enforcer-filters-links",
        mcp_normalize(
            "follow_links",
            {
                "path": hub,
                "title": "Hub Note",
                "out_links": [{"title": "Planner Peer", "path": "02-projects/peer.md"}],
                "backlinks": [
                    {"title": "Inbox Note", "path": "00-inbox/inbox-note.md"},
                    {"title": "Planner Peer", "path": "02-projects/peer.md"},
                ],
                "out_link_count": 1,
                "backlink_count": 2,
            },
        ),
        mcp_normalize("follow_links", dispatch("follow_links", {"path": hub}, enforcer=planner)),
    )
    before_capture = {
        str(path.relative_to(vault)): digest(path.read_bytes())
        for path in vault.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    for tool in ("capture", "capture_note"):
        denied_capture = dispatch(
            tool, {"content": "PLANNER-CAPTURE-PRIVATE-BODY"}, enforcer=planner
        )
        check("direct.planner." + tool + ".denied", True, set(denied_capture) == {"error"})
        check(
            "direct.planner." + tool + ".body-redacted",
            False,
            "PLANNER-CAPTURE-PRIVATE-BODY" in json.dumps(denied_capture),
        )
    check(
        "direct.planner.capture-no-source-or-projection",
        before_capture,
        {
            str(path.relative_to(vault)): digest(path.read_bytes())
            for path in vault.rglob("*")
            if path.is_file() and not path.is_symlink()
        },
    )


async def mcp_factory_none_handlers(
    root: Path, check: Any, all_calls: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Real ``create_mcp_server(enforcer=None)`` SDK request handler: named tools fail closed.

    Each request goes through the SDK-registered ``CallToolRequest`` handler (the same
    handler the stdio server runs). The identical arguments are replayed against a server
    built with a real full-profile enforcer, so a failure to start/validate cannot read
    as a green denial.
    """
    from mcp.types import CallToolRequest, CallToolRequestParams

    from vaultmind.config import Settings, VaultConfig
    from vaultmind.mcp.auth import ProfileEnforcer
    from vaultmind.mcp.profiles import load_profile
    from vaultmind.mcp.server import create_mcp_server
    from vaultmind.vault.parser import VaultParser

    vault = mcp_build_fixture(root, scoped=False)["vault"]
    parser = VaultParser(VaultConfig(path=vault))
    settings = Settings(vault={"path": vault, "excluded_folders": ["wiki", "_wiki"]})
    none_server = create_mcp_server(
        vault, object(), object(), parser, settings=settings, enforcer=None
    )
    full_server = create_mcp_server(
        vault,
        object(),
        object(),
        parser,
        settings=settings,
        enforcer=ProfileEnforcer(load_profile("full"), vault),
    )

    async def invoke(server: Any, tool: str, arguments: dict[str, Any]) -> tuple[Any, Any, str]:
        handler = server.request_handlers[CallToolRequest]
        request = CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name=tool, arguments=arguments)
        )
        result = (await handler(request)).root
        text = result.content[0].text
        return result, json.loads(text), text

    hub = "02-projects/hub.md"
    protected = [
        *mcp_scenario("planner-scoped", root)["forbidden"],
        *MCP_NOTES,
        "Hub Note",
        "Planner Peer",
        "Inbox Note",
        "Nested Allowed",
        "Private Secret",
        "Hub description.",
        "Allowed peer description.",
        "PRIVATE-DESCRIPTION-SENTINEL",
        MCP_HUB_SENTINEL,
    ]
    receipt: dict[str, Any] = {}
    for tool, arguments, positive in (
        ("read_frontmatter", {"path": hub}, lambda p: [p.get("title"), "error" in p]),
        (
            "list_folder_index",
            {},
            lambda p: [sorted(e["path"] for e in p.get("notes", [])), "error" in p],
        ),
        ("follow_links", {"path": hub}, lambda p: [p.get("out_link_count"), "error" in p]),
    ):
        label = f"factory.{tool}"
        denied, denied_payload, denied_text = await invoke(none_server, tool, arguments)
        check(
            label + ".none-handler-text-content",
            ["text"],
            [part.type for part in denied.content],
        )
        check(
            label + ".none-handler-error-only",
            True,
            isinstance(denied_payload, dict)
            and set(denied_payload) == {"error"}
            and isinstance(denied_payload["error"], str)
            and bool(denied_payload["error"]),
        )
        check(
            label + ".none-handler-private-fields-absent",
            [],
            mcp_leaks(denied_text, protected, arguments),
        )
        allowed, allowed_payload, allowed_text = await invoke(full_server, tool, arguments)
        check(label + ".same-args-real-enforcer-not-error", False, bool(allowed.isError))
        expected = {
            "read_frontmatter": ["Hub Note", False],
            "list_folder_index": [sorted(MCP_NOTES), False],
            "follow_links": [3, False],
        }[tool]
        check(label + ".same-args-real-enforcer-positive", expected, positive(allowed_payload))
        receipt[tool] = {
            "arguments": arguments,
            "enforcer_none_response": denied_payload,
            "real_enforcer_response_bytes": len(allowed_text),
            "real_enforcer_positive": positive(allowed_payload),
        }
    _unused, root_payload, _text = await invoke(none_server, "vault_list", {"folder": "00-inbox"})
    check(
        "factory.root-note-tool-without-enforcer-denied",
        True,
        isinstance(root_payload, dict)
        and set(root_payload) == {"error"}
        and bool(root_payload["error"]),
    )
    for tool, arguments in sorted(all_calls.items()):
        _, payload, text = await invoke(none_server, tool, arguments)
        check(
            "factory.all-tools." + tool + ".error-only",
            True,
            isinstance(payload, dict)
            and set(payload) == {"error"}
            and isinstance(payload["error"], str)
            and bool(payload["error"]),
        )
        check(
            "factory.all-tools." + tool + ".private-fields-absent",
            [],
            mcp_leaks(text, protected, arguments),
        )
        receipt.setdefault(tool, {})["all_inventory_none_response"] = payload
    return receipt


async def mcp_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    home = home.resolve()
    check = make_checker(checks)
    scenarios: dict[str, Any] = {}
    configuration_refusals: dict[str, Any] = {}
    for name in (*MCP_SCENARIOS, *MCP_ROOT_SCENARIOS, *MCP_DERIVED_SCENARIOS, *MCP_QUERY_SCENARIOS):
        if name == "root-capture-symlink":
            # Configured ordinary output aliases now fail before SDK startup.
            # Exercise the real CLI refusal rather than treating a closed SDK
            # connection as either successful capture or a transport defect.
            from click.testing import CliRunner

            from vaultmind.cli import cli

            root = home / name
            spec, fixture, config = mcp_root_fixture(root, name)
            before = mcp_root_snapshot(fixture)
            result = CliRunner().invoke(
                cli, ["-c", str(config), "mcp-serve", "--profile", spec["profile"]]
            )
            check(name + ".configuration-refused", True, result.exit_code != 0)
            check(name + ".no-side-effects", before, mcp_root_snapshot(fixture))
            diagnostic = result.output + str(result.exception)
            check(name + ".refused-destination-identified", True, "00-inbox" in diagnostic)
            check(
                name + ".unknown-target-not-disclosed", False, str(root / "outside") in diagnostic
            )
            configuration_refusals[name] = {
                "exit_code": result.exit_code,
                "diagnostic": diagnostic,
                "unchanged": True,
            }
            continue
        scenarios[name] = await run_mcp_scenario(name, home / name, check)
    check(
        "root-scoped.forbidden-chunk-population-does-not-change-authorized-results",
        [
            (c["tool"], c["arguments"], c["actual"])
            for c in scenarios["root-scoped"]["calls"]
            if c["tool"] in ("vault_search", "find_duplicates")
        ],
        [
            (c["tool"], c["arguments"], c["actual"])
            for c in scenarios["root-scoped-small"]["calls"]
            if c["tool"] in ("vault_search", "find_duplicates")
        ],
    )
    mcp_direct_dispatcher(home / "direct-dispatcher", check)
    listed = set(scenarios["derived-full"]["listed_tools"])
    observations: dict[str, list[dict[str, Any]]] = {tool: [] for tool in sorted(listed)}
    all_calls: dict[str, dict[str, Any]] = {}
    for name, scenario in scenarios.items():
        check(
            "inventory." + name + ".actual-list-tools-consistent",
            sorted(listed),
            sorted(scenario["listed_tools"]),
        )
        for position, call in enumerate(scenario["calls"]):
            check(
                "inventory." + name + f".{position}.executed-tool-listed",
                True,
                call["tool"] in listed,
            )
            all_calls.setdefault(call["tool"], call["arguments"])
            observations[call["tool"]].append(
                {
                    "scenario": name,
                    "call_index": position,
                    "profile": scenario["profile"],
                    "policy": call["policy"],
                    "arguments": call["arguments"],
                    "actual": call["actual"],
                    "audit_result": call["audit"]["result"],
                }
            )
    check("inventory.all-listed-tools-executed", sorted(listed), sorted(all_calls))
    for tool, rows in observations.items():
        check(
            "inventory." + tool + ".substantive-allow-or-filter-control",
            True,
            any("error" not in row["actual"] and row["audit_result"] == "OK" for row in rows),
        )
        if tool in {
            "graph_query",
            "graph_path",
            "vault_stats",
            "episode_query",
            "workflow_suggest",
            "graph_evolution",
            "suggest_links",
        }:
            check(
                "inventory." + tool + ".restricted-denial-and-full-positive",
                True,
                any(
                    row["policy"] == "deny"
                    and row["profile"] != "full"
                    and row["audit_result"] == "DENIED"
                    for row in rows
                )
                and any(
                    row["profile"] == "full"
                    and "error" not in row["actual"]
                    and row["audit_result"] == "OK"
                    for row in rows
                ),
            )
        if tool == "recent_activity":
            check(
                "inventory." + tool + ".scoped-filter-and-global-positive",
                True,
                any(
                    row["scenario"] == "derived-scoped"
                    and row["policy"] == "filter"
                    and row["audit_result"] == "OK"
                    for row in rows
                )
                and any(
                    row["scenario"] == "derived-full"
                    and "error" not in row["actual"]
                    and row["audit_result"] == "OK"
                    for row in rows
                ),
            )
    factory = await mcp_factory_none_handlers(home / "factory-none", check, all_calls)
    origins = {s["import_origin"] for s in scenarios.values()}
    check("origin.consistent-across-launches", 1, len(origins))
    return {
        "configuration_refusals": configuration_refusals,
        "mcp_sdk": importlib.metadata.version("mcp"),
        "import_origin": origins.pop(),
        "factory_none_handlers": factory,
        "scenarios": scenarios,
        "policy_inventory": {
            "source": "actual SDK list_tools plus executed, asserted call observations",
            "tools": {
                tool: {
                    "folder_restricted_policy": (
                        "deny"
                        if tool
                        in {
                            "graph_query",
                            "graph_path",
                            "vault_stats",
                            "episode_query",
                            "workflow_suggest",
                            "graph_evolution",
                            "suggest_links",
                        }
                        else "filter"
                        if tool
                        in {
                            "vault_search",
                            "vault_list",
                            "list_folder_index",
                            "follow_links",
                            "find_duplicates",
                            "recent_activity",
                        }
                        else "allow"
                    ),
                    "full_policy": "allow",
                    "path_and_write_guards": (
                        "Root confinement, authorized canonical note paths, write_enabled, "
                        "serialized size and capture destination guards remain applicable."
                    ),
                    "executed_observations": rows,
                }
                for tool, rows in observations.items()
            },
            "schemas": scenarios["derived-full"]["schemas"],
            "read_only_audit_exception": "mcp_audit.jsonl append receipts",
        },
        "child_network_attempts": [
            attempt for s in scenarios.values() for attempt in s["child_network_attempts"]
        ],
    }


# ---------------------------------------------------------------------------
# M13 approval case: real CLI producers -> real queue -> real CommandHandlers and
# the real registered aiogram router. Offline boundaries are labelled in the receipt.
# ---------------------------------------------------------------------------
APPROVAL_INSIDER, APPROVAL_PEER, APPROVAL_OUTSIDER = 7, 8, 99
APPROVAL_CHAT = 555
APPROVAL_WORDS = [
    "harbor",
    "lantern",
    "copper",
    "valve",
    "winter",
    "gale",
    "anchor",
    "ledger",
    "signal",
    "buoy",
    "keeper",
    "tide",
    "chart",
    "beacon",
    "rigging",
    "mast",
    "cargo",
    "pilot",
    "fog",
    "quay",
]


class ApprovalLLM(ScriptedLLM):
    """Scripted llm.client boundary (provider boundary only); `reply` is set per phase."""

    provider_name = "scripted-approval-acceptance"

    def __init__(self) -> None:
        self.reply = "Scripted acceptance reply."
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        from vaultmind.llm.client import LLMResponse

        self.calls.append(
            {
                "messages_sha256": digest(
                    json.dumps([{"role": m.role, "content": m.content} for m in messages]).encode()
                ),
                "system_sha256": digest(system.encode()) if system is not None else None,
                "response_sha256": digest(self.reply.encode()),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        return LLMResponse(text=self.reply, model=model, usage={})


def approval_session() -> Any:
    """aiogram BaseSession that records every request and never touches a network."""
    from datetime import UTC, datetime

    from aiogram.client.session.base import BaseSession
    from aiogram.types import Chat
    from aiogram.types import Message as TgMessage

    class RecordingSession(BaseSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []
            self.fail_sends = False
            self.counter = 1000

        async def close(self) -> None:
            return None

        async def make_request(self, bot: Any, method: Any, timeout: int | None = None) -> Any:
            name = type(method).__name__
            data = method.model_dump(exclude_none=True)
            self.calls.append({"method": name, "data": data})
            if name == "SendMessage":
                if self.fail_sends:
                    raise RuntimeError("fake transport: send refused")
                self.counter += 1
                return TgMessage(
                    message_id=self.counter,
                    date=datetime.now(UTC),
                    chat=Chat(id=data["chat_id"], type="private"),
                    text=data["text"],
                )
            return True

        async def stream_content(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("fake transport has no file streams")
            yield b""

    return RecordingSession()


def tg_texts(calls: list[dict[str, Any]]) -> list[str]:
    names = ("EditMessageText", "AnswerCallbackQuery")
    return [c["data"].get("text", "") for c in calls if c["method"] in names]


def tg_keyboards(calls: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [b["callback_data"] for row in c["data"]["reply_markup"]["inline_keyboard"] for b in row]
        for c in calls
        if c["method"] == "SendMessage" and "reply_markup" in c["data"]
    ]


class TelegramHarness:
    """Feed real aiogram Updates through the router registered by `register_handlers`."""

    def __init__(self, handlers: Any, session: Any, bot: Any) -> None:
        from aiogram import Dispatcher, Router

        import vaultmind.bot.telegram as telegram

        self.session, self.bot, self.n = session, bot, 0
        telegram.router = Router()  # register_handlers decorates the module-level `router`
        telegram.register_handlers(handlers)
        self.dp = Dispatcher()
        self.dp.include_router(telegram.router)

    async def _feed(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        from aiogram.types import Update

        self.n += 1
        mark = len(self.session.calls)
        update = Update.model_validate({"update_id": self.n, **raw}, context={"bot": self.bot})
        await self.dp.feed_update(self.bot, update)
        return self.session.calls[mark:]

    async def callback(self, data: str, actor: int) -> list[dict[str, Any]]:
        return await self._feed(
            {
                "callback_query": {
                    "id": str(self.n + 1),
                    "from": {"id": actor, "is_bot": False, "first_name": "t"},
                    "chat_instance": "acceptance",
                    "data": data,
                    "message": {
                        "message_id": 5,
                        "date": 1700000000,
                        "chat": {"id": actor, "type": "private"},
                        "text": "prompt",
                    },
                }
            }
        )

    async def command(self, text: str, actor: int) -> list[dict[str, Any]]:
        return await self._feed(
            {
                "message": {
                    "message_id": self.n + 10,
                    "date": 1700000000,
                    "chat": {"id": actor, "type": "private"},
                    "from": {"id": actor, "is_bot": False, "first_name": "t"},
                    "text": text,
                    "entities": [
                        {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
                    ],
                }
            }
        )


APPROVAL_MERGE_COSINE = 0.85


class ApprovalEmbedder(DeterministicEmbedder):
    """Provider boundary: fixed synthetic vectors for the dupA/dupB/dupC fixtures (cosine 0.85)."""

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 1021
        if "vectora" in text.split():
            vector[0] = 1.0
        elif "vectorb" in text.split():
            vector[0] = APPROVAL_MERGE_COSINE
            vector[1] = (1 - APPROVAL_MERGE_COSINE**2) ** 0.5
        else:
            return super().embed_query(text)
        return vector


def approval_world(
    root: Path,
    *,
    force_block: bool,
    allowed: list[int],
    notes: dict[str, str],
    llm: ApprovalLLM,
    seed_queue: Any = None,
    embedder: Any = None,
    closers: list[Any],
) -> SimpleNamespace:
    """Real settings/queue (cli._build_review_queue)/store/parser/handlers over fixture state."""
    from aiogram import Bot

    import vaultmind.cli as app
    from vaultmind.bot.commands import CommandHandlers
    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import load_settings
    from vaultmind.graph.evolution import LineageStore
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

    vault, data = root / "vault", root / "data"
    vault.mkdir(parents=True)
    data.mkdir()
    for rel, text in notes.items():
        (vault / rel).write_text(text, encoding="utf-8")
    config = root / "config.toml"
    config.write_text(
        'anthropic_api_key = "OFFLINE-SENTINEL-NOT-CREDENTIAL"\n'
        f'[vault]\npath = "{vault}"\n'
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        f'[autonomy]\nforce_block = {str(force_block).lower()}\ndb_path = "{data / "review.db"}"\n'
        f"[telegram]\nallowed_user_ids = [{', '.join(map(str, allowed))}]\n"
        f'[chroma]\npersist_dir = "{data / "chroma"}"\n'
        f'[graph]\npersist_path = "{data / "graph.json"}"\n'
        f'[search]\nbm25_db_path = "{data / "fts.db"}"\n'
        "[embedding]\ncache_enabled = false\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    if seed_queue is not None:
        seed_queue(Path(settings.autonomy.db_path))
    initial_files = {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in [config, *sorted(vault.rglob("*.md"))]
    }
    seed_db = Path(settings.autonomy.db_path)
    input_identity = {
        "files_sha256": initial_files,
        "effective_config_sha256": approval_json_digest(settings.model_dump(mode="json")),
        "seed_queue_db_sha256": digest(seed_db.read_bytes()) if seed_db.exists() else None,
    }
    queue = app._build_review_queue(settings)
    bm25 = BM25Index(data / "fts.db")
    store = VaultStore(settings.chroma, embedder or DeterministicEmbedder(), bm25=bm25)
    parser = VaultParser(settings.vault)
    graph = KnowledgeGraph(settings.graph)
    lineage = LineageStore(data / "lineage.json")
    session = approval_session()
    bot = Bot("123456:OFFLINE-ACCEPTANCE", session=session)
    thinking = ThinkingPartner(settings, llm_factory=lambda: llm)
    handlers = CommandHandlers(
        settings,
        store,
        graph,
        parser,
        thinking,
        llm,
        review_queue=queue,
        lineage_store=lineage,
    )
    closers += [queue.close, bm25.close]
    return SimpleNamespace(
        root=root,
        vault=vault,
        data=data,
        config=config,
        settings=settings,
        input_identity=input_identity,
        queue=queue,
        store=store,
        parser=parser,
        graph=graph,
        lineage=lineage,
        llm=llm,
        session=session,
        bot=bot,
        handlers=handlers,
        harness=TelegramHarness(handlers, session, bot),
    )


def approval_json_digest(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, default=str).encode())


def approval_observe(world: SimpleNamespace, home_data: Path) -> dict[str, Any]:
    """Queue rows (raw SQLite), every vault/vocabulary/lineage byte+mtime, pending edits."""
    import sqlite3

    files: dict[str, Any] = {}
    paths = sorted(world.vault.rglob("*.md"))
    paths += [home_data / "tag_vocabulary.json", world.data / "lineage.json"]
    for path in paths:
        if path.exists():
            files[str(path)] = (digest(path.read_bytes()), path.stat().st_mtime_ns)
    with sqlite3.connect(f"file:{world.settings.autonomy.db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT * FROM review_proposals ORDER BY proposal_id").fetchall()
    pending = {k: dict(v) for k, v in world.handlers._pending_edits.items()}
    return {"queue": rows, "files": files, "pending_edits": pending}


def approval_git_binding(home: Path) -> dict[str, Any]:
    environment = offline_environment(home)

    def git(*argv: str, required: bool = False) -> str | None:
        done = subprocess.run(
            ["git", "-C", str(ROOT), *argv], capture_output=True, text=True, env=environment
        )
        if required:
            done.check_returncode()
        return done.stdout.strip() if done.returncode == 0 else None

    refs: dict[str, Any] = {}
    for ref in ("fix/vm-m13-review-appliers", "origin/main", "main"):
        commit = git("rev-parse", "--verify", "--quiet", ref)
        if commit:
            refs[ref] = {"commit": commit, "merge_base_with_head": git("merge-base", "HEAD", ref)}
    return {
        "head": git("rev-parse", "HEAD", required=True),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", required=True),
        "parents": cast("str", git("rev-list", "--parents", "-n1", "HEAD", required=True)).split()[
            1:
        ],
        "tree": git("rev-parse", "HEAD^{tree}", required=True),
        "reference_commits": refs,
    }


def approval_legacy_schema(home: Path) -> tuple[str, str]:
    """The newest committed review_proposals schema that predates the legacy_* columns."""
    path = "src/vaultmind/services/review_queue.py"
    environment = offline_environment(home)
    log = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--", path], text=True, env=environment
    ).split()
    for commit in log:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"], text=True, env=environment
        )
        match = re.search(r'_SCHEMA = """(.*?)"""', source, re.S)
        if match and "legacy_status" not in match.group(1):
            return commit, match.group(1)
    raise RuntimeError("no committed legacy review_proposals schema found in git history")


def approval_note(title: str, body: str, **meta: str) -> str:
    front = "".join(f"{k}: {v}\n" for k, v in meta.items())
    return f"---\ntitle: {title}\n{front}---\n{body}\n"


async def approval_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    import logging
    import stat

    home = home.resolve()
    failures: list[str] = []

    def json_value(value: Any) -> Any:
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, set):
            return [json_value(item) for item in sorted(value, key=repr)]
        return value

    def ck(name: str, expected: Any, actual: Any) -> None:
        checks.append({"id": name, "expected": json_value(expected), "actual": json_value(actual)})
        if expected != actual:
            failures.append(name)

    config_preloaded = "vaultmind.config" in sys.modules
    ck("sanitization.config-not-imported-before-temp-home", False, config_preloaded)
    if config_preloaded:
        raise RuntimeError("Refusing approval acceptance: configuration predates temporary HOME")
    import frontmatter
    from click.testing import CliRunner

    import vaultmind
    import vaultmind.cli as app
    import vaultmind.sources.connectors  # noqa: F401  (registers the closed rss/youtube/github set)
    from vaultmind.bot.handlers.autonomy import build_block_notifier, callback_data
    from vaultmind.bot.notifier import Notifier
    from vaultmind.config import VAULTMIND_HOME
    from vaultmind.contradiction.detector import ContradictionDetector
    from vaultmind.contradiction.policy import resolve_conflict
    from vaultmind.indexer.duplicate_detector import DuplicateDetector
    from vaultmind.memory.gaps import GapStore
    from vaultmind.pipeline.maturation import MaturationPipeline
    from vaultmind.services.review_delivery import DeliveryStatus, deliver_block
    from vaultmind.services.review_queue import (
        AutonomyThresholds,
        Impact,
        Lane,
        ProposalKind,
        ProposalStatus,
        ReviewQueue,
    )
    from vaultmind.sources.models import SourceInstance
    from vaultmind.sources.pipeline import make_applier, run_connector_once
    from vaultmind.sources.store import SourceStore
    from vaultmind.vault.events import NoteCreatedEvent, VaultEventBus

    vm_data = VAULTMIND_HOME / "data"
    configured_home = VAULTMIND_HOME.resolve()
    ck("sanitization.vaultmind-home-is-temp", str(home / ".vaultmind"), str(configured_home))
    if configured_home != home / ".vaultmind":
        raise RuntimeError(
            "Refusing approval acceptance: application state is outside fixture HOME"
        )
    log_records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(f"{record.levelname} {record.name}: {record.getMessage()}")

    capture = Capture()
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.addHandler(capture)
    root_logger.setLevel(logging.INFO)
    closers: list[Any] = []
    probes: list[str] = []
    llm = ApprovalLLM()
    delivery: dict[str, str] = {}
    try:
        # ---- 1. CLI-created BLOCK proposals (tags) ---------------------------------
        body = "Synthetic harbor inspection text preserved through approval. " * 4
        vm_data.mkdir(parents=True)
        (vm_data / "tag_vocabulary.json").write_text(json.dumps({"approved": ["known"]}))
        seed_vocabulary_sha256 = digest((vm_data / "tag_vocabulary.json").read_bytes())
        w = approval_world(
            home / "w",
            force_block=True,
            allowed=[APPROVAL_INSIDER, APPROVAL_PEER],
            notes={"tagged.md": approval_note("Tagged", body, tags="[original]")},
            llm=llm,
            closers=closers,
        )
        queue, session, harness = w.queue, w.session, w.harness
        llm.reply = json.dumps({"tags": ["known"], "new_tags": ["approved-new"]})
        from unittest.mock import patch

        with patch.object(app, "_create_llm_client", return_value=llm):  # offline provider boundary
            cli = await asyncio.to_thread(
                CliRunner().invoke, app.cli, ["--config", str(w.config), "auto-tag", "--apply"]
            )
        ck("cli.auto-tag-exit-code", 0, cli.exit_code)
        ck("cli.auto-tag-no-exception", None, repr(cli.exception) if cli.exception else None)
        pending = {p.kind: p for p in queue.list_pending()}
        ck(
            "cli.created-kinds",
            sorted([ProposalKind.TAG_APPLICATION.value, ProposalKind.TAG_VOCABULARY.value]),
            sorted(k.value for k in pending),
        )
        ck(
            "cli.created-lanes-forced-block",
            ["BLOCK", "BLOCK"],
            sorted(p.lane.name for p in pending.values()),
        )
        ck(
            "cli.delivery-reported-unavailable-not-silent",
            True,
            "unavailable" in (cli.output + "\n".join(log_records)).lower(),
        )
        p_app, p_voc = pending[ProposalKind.TAG_APPLICATION], pending[ProposalKind.TAG_VOCABULARY]
        note = w.vault / "tagged.md"
        ck("cli.nothing-applied-while-pending", ["original"], frontmatter.load(note)["tags"])
        approve = lambda p: callback_data("approve", p.proposal_id)  # noqa: E731
        reject = lambda p: callback_data("reject", p.proposal_id)  # noqa: E731
        retry = lambda p: callback_data("retry", p.proposal_id)  # noqa: E731

        # ---- 2. unauthorized actor: no read/state/byte mutation ---------------------
        pre = approval_observe(w, vm_data)
        for label, data in (
            ("approve", approve(p_app)),
            ("reject", reject(p_app)),
            ("retry", retry(p_app)),
            ("approve-all", "autonomy_approve_all_skim"),
        ):
            calls = await harness.callback(data, APPROVAL_OUTSIDER)
            ck(
                f"auth.outsider-{label}-answered-unauthorized",
                True,
                any("Unauthorized" in t for t in tg_texts(calls)),
            )
            ck(
                f"auth.outsider-{label}-no-message-edit",
                [],
                [c for c in calls if c["method"] == "EditMessageText"],
            )
        ck(
            "auth.outsider-autonomy-no-state-or-byte-change",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )

        # ---- 3. generic keyboard delivery + approval from the keyboard --------------
        block_notifier = build_block_notifier(Notifier(w.bot, APPROVAL_CHAT))
        mark = len(session.calls)
        for p in (p_app, p_voc):
            status = await deliver_block(queue.get(p.proposal_id), block_notifier)
            delivery[f"cli-created:{p.kind.value}"] = status.value
            ck(f"delivery.cli-created-{p.kind.value}-sent", DeliveryStatus.SENT.value, status.value)
        keyboards = tg_keyboards(session.calls[mark:])
        ck("delivery.one-keyboard-per-proposal", 2, len(keyboards))
        ck(
            "delivery.keyboard-buttons-are-bounded-approve-reject",
            sorted([[approve(p), reject(p)] for p in (p_app, p_voc)]),
            sorted(keyboards),
        )
        ck(
            "delivery.callback-data-within-telegram-limit",
            True,
            all(len(c.encode()) <= 64 for k in keyboards for c in k),
        )
        ck(
            "delivery.sending-leaves-pending",
            ["pending", "pending"],
            [queue.get(p.proposal_id).status.value for p in (p_app, p_voc)],
        )
        mark = len(session.calls)
        for p in (p_app, p_voc):
            replay = queue.propose(p.kind, p.confidence, p.impact, p.summary, p.payload)
            ck(
                f"delivery.dedup-{p.kind.value}-not-required",
                DeliveryStatus.NOT_REQUIRED.value,
                (await deliver_block(replay, block_notifier)).value,
            )
            ck(f"delivery.dedup-{p.kind.value}-occurrence", 2, replay.occurrence_count)
        ck("delivery.dedup-sent-nothing", 0, len(session.calls) - mark)

        for p in (p_app, p_voc):
            calls = await harness.callback(approve(p), APPROVAL_INSIDER)
            ck(f"approve.{p.kind.value}-applied", "applied", queue.get(p.proposal_id).status.value)
            ck(
                f"approve.{p.kind.value}-reported-applied",
                True,
                any("Applied" in t for t in tg_texts(calls)),
            )
        post = frontmatter.load(note)
        ck("approve.frontmatter-tags-merged", ["original", "known"], post["tags"])
        ck("approve.note-body-preserved", body.strip(), post.content)
        ck("approve.parser-sees-tags", True, "known" in w.parser.parse_file(note).tags)
        ck(
            "approve.vocabulary-file",
            {"approved": ["approved-new", "known"]},
            json.loads((vm_data / "tag_vocabulary.json").read_text()),
        )
        pre = approval_observe(w, vm_data)
        for label, fn in (("approve", approve), ("reject", reject), ("retry", retry)):
            for p in (p_app, p_voc):
                calls = await harness.callback(fn(p), APPROVAL_INSIDER)
                ck(
                    f"replay.{label}-{p.kind.value}-terminal-text",
                    True,
                    any("Already resolved" in t or "Not failed" in t for t in tg_texts(calls)),
                )
        ck(
            "replay.terminal-no-file-write-or-state-change",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )

        # ---- 4. human reject; disabled/unavailable/failed delivery -------------------
        def block_proposal(tag: str) -> Any:
            return queue.propose(
                ProposalKind.TAG_VOCABULARY, 0.1, Impact.HIGH, f"vocab {tag}", {"tag": tag}
            )

        rejected = block_proposal("rejected-tag")
        ck(
            "reject.proposal-newly-pending-block",
            ["pending", "BLOCK", 1],
            [rejected.status.value, rejected.lane.name, rejected.occurrence_count],
        )
        await deliver_block(rejected, block_notifier)
        pre_vocab = (vm_data / "tag_vocabulary.json").read_bytes()
        calls = await harness.callback(reject(rejected), APPROVAL_PEER)
        ck("reject.human-rejected-status", "rejected", queue.get(rejected.proposal_id).status.value)
        ck("reject.vocabulary-untouched", pre_vocab, (vm_data / "tag_vocabulary.json").read_bytes())
        calls = await harness.callback(approve(rejected), APPROVAL_INSIDER)
        ck(
            "reject.later-approve-refused",
            ["rejected", True],
            [
                queue.get(rejected.proposal_id).status.value,
                any("Already resolved" in t for t in tg_texts(calls)),
            ],
        )

        mark = len(session.calls)
        disabled = await deliver_block(
            block_proposal("disabled-tag"), build_block_notifier(Notifier(w.bot, 0))
        )
        ck(
            "delivery.disabled-status-no-request",
            [DeliveryStatus.DISABLED.value, 0],
            [disabled.value, len(session.calls) - mark],
        )
        unavailable = await deliver_block(block_proposal("unavailable-tag"), None)
        ck("delivery.unavailable-status", DeliveryStatus.UNAVAILABLE.value, unavailable.value)
        session.fail_sends = True
        failed_p = block_proposal("failed-tag")
        mark = len(session.calls)
        failed = await deliver_block(failed_p, block_notifier)
        session.fail_sends = False
        ck(
            "delivery.transport-failure-status-attempted-once",
            [DeliveryStatus.FAILED.value, 1],
            [failed.value, len(session.calls) - mark],
        )
        ck(
            "delivery.failed-delivery-leaves-pending",
            "pending",
            queue.get(failed_p.proposal_id).status.value,
        )
        delivery.update(disabled=disabled.value, unavailable=unavailable.value, failed=failed.value)
        skim = queue.propose(
            ProposalKind.TAG_VOCABULARY,
            0.6,
            Impact.MEDIUM,
            "skim",
            {"tag": "skim-lane"},
            lane_override=Lane.SKIM,
        )
        ck(
            "delivery.skim-not-required",
            DeliveryStatus.NOT_REQUIRED.value,
            (await deliver_block(skim, block_notifier)).value,
        )
        ck(
            "delivery.terminal-not-required",
            DeliveryStatus.NOT_REQUIRED.value,
            (await deliver_block(queue.get(p_app.proposal_id), block_notifier)).value,
        )
        for tag in ("disabled-tag", "unavailable-tag", "failed-tag", "skim-lane"):
            queue.reject(
                queue.propose(
                    ProposalKind.TAG_VOCABULARY, 0.1, Impact.HIGH, "x", {"tag": tag}
                ).proposal_id,
                "cleanup",
            )

        # ---- 5. raised real operations: missing note + real filesystem fault ---------
        def tag_task(path: str, tags: list[str]) -> Any:
            return queue.propose(
                ProposalKind.TAG_APPLICATION,
                0.1,
                Impact.HIGH,
                f"tag {path}",
                {"note_path": path, "tags": tags},
            )

        missing = tag_task("restorable.md", ["restored"])
        calls = await harness.callback(approve(missing), APPROVAL_INSIDER)
        got = queue.get(missing.proposal_id)
        probes.append("missing-note-real-applier-raises")
        ck("fault.missing-note-failed-not-applied", "failed", got.status.value)
        ck(
            "fault.missing-note-result-recorded-and-not-success",
            [True, False],
            [bool(got.result), got.status is ProposalStatus.APPLIED],
        )
        ck("fault.missing-note-reported-failed", True, any("Failed" in t for t in tg_texts(calls)))
        calls = await harness.callback(approve(missing), APPROVAL_INSIDER)
        ck(
            "fault.approve-on-failed-never-retries",
            "failed",
            queue.get(missing.proposal_id).status.value,
        )
        (w.vault / "restorable.md").write_text(approval_note("Restorable", body), encoding="utf-8")
        pre = approval_observe(w, vm_data)
        await harness.callback(retry(missing), APPROVAL_OUTSIDER)
        ck(
            "fault.outsider-retry-no-change",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )
        await harness.callback(retry(missing), APPROVAL_INSIDER)
        ck(
            "fault.explicit-retry-after-restore-applied",
            ["applied", ["restored"]],
            [
                queue.get(missing.proposal_id).status.value,
                frontmatter.load(w.vault / "restorable.md")["tags"],
            ],
        )

        locked = w.vault / "locked.md"
        locked.write_text(approval_note("Locked", body, tags="[orig]"), encoding="utf-8")
        before = locked.read_bytes()
        locked.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ck("fault.filesystem-permission-actually-enforced", False, os.access(locked, os.W_OK))
        fault = tag_task("locked.md", ["unlocked"])
        await harness.callback(approve(fault), APPROVAL_INSIDER)
        probes.append("filesystem-permission-denied-real-open")
        got = queue.get(fault.proposal_id)
        ck("fault.filesystem-failed", ["failed", True], [got.status.value, bool(got.result)])
        ck("fault.filesystem-bytes-untouched", before, locked.read_bytes())
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        await harness.callback(retry(fault), APPROVAL_INSIDER)
        ck(
            "fault.filesystem-explicit-retry-applied",
            ["applied", ["orig", "unlocked"]],
            [queue.get(fault.proposal_id).status.value, frontmatter.load(locked)["tags"]],
        )

        # ---- 6. missing operations + informational kinds (bare queue) ----------------
        bare_session = approval_session()
        from aiogram import Bot as TgBot

        bare_bot = TgBot("123456:OFFLINE-ACCEPTANCE", session=bare_session)
        bare = ReviewQueue(w.data / "bare.db", AutonomyThresholds(force_block=True))
        closers.append(bare.close)
        import vaultmind.bot.commands as commands_mod

        bare_handlers = commands_mod.CommandHandlers(
            w.settings, w.store, w.graph, w.parser, w.handlers._ctx.thinking, llm, review_queue=bare
        )
        bare_harness = TelegramHarness(bare_handlers, bare_session, bare_bot)
        mutation_kinds = [
            ProposalKind.TAG_APPLICATION,
            ProposalKind.TAG_VOCABULARY,
            ProposalKind.CONTRADICTION_RESOLUTION,
            ProposalKind.MATURATION_SYNTHESIS,
            ProposalKind.SOURCE_INGESTION,
        ]
        for kind in mutation_kinds:
            p = bare.propose(kind, 0.1, Impact.HIGH, f"bare {kind.value}", {"bare": kind.value})
            calls = await bare_harness.callback(
                callback_data("approve", p.proposal_id), APPROVAL_INSIDER
            )
            got = bare.get(p.proposal_id)
            ck(
                f"missing-operation.{kind.value}-failed",
                True,
                got.status is ProposalStatus.FAILED and bool(got.result),
            )
            ck(
                f"missing-operation.{kind.value}-not-reported-applied",
                True,
                any("Failed" in t for t in tg_texts(calls))
                and not any("Applied" in t for t in tg_texts(calls)),
            )
            await bare_harness.callback(callback_data("retry", p.proposal_id), APPROVAL_INSIDER)
            ck(
                f"missing-operation.{kind.value}-retry-still-failed",
                "failed",
                bare.get(p.proposal_id).status.value,
            )
        for kind in (ProposalKind.CONTRADICTION_ESCALATION, ProposalKind.DUPLICATE_MERGE):
            p = bare.propose(kind, 0.1, Impact.HIGH, f"info {kind.value}", {"info": kind.value})
            calls = await bare_harness.callback(
                callback_data("approve", p.proposal_id), APPROVAL_INSIDER
            )
            ck(
                f"informational.{kind.value}-acknowledged-explicitly",
                [True, "acknowledged"],
                [
                    any("Acknowledged" in t for t in tg_texts(calls)),
                    bare.get(p.proposal_id).status.value,
                ],
            )

        # ---- 7. truthful mixed approve-all -------------------------------------------
        (w.vault / "bulk.md").write_text(approval_note("Bulk", body), encoding="utf-8")
        mixed = {
            "applied": queue.propose(
                ProposalKind.TAG_APPLICATION,
                0.6,
                Impact.MEDIUM,
                "bulk ok",
                {"note_path": "bulk.md", "tags": ["bulk-tag"]},
                lane_override=Lane.SKIM,
            ),
            "raises": queue.propose(
                ProposalKind.TAG_APPLICATION,
                0.6,
                Impact.MEDIUM,
                "bulk gone",
                {"note_path": "gone.md", "tags": ["x"]},
                lane_override=Lane.SKIM,
            ),
            "info": queue.propose(
                ProposalKind.DUPLICATE_MERGE,
                0.8,
                Impact.MEDIUM,
                "bulk dup",
                {"source_path": "a.md", "match_path": "b.md"},
                lane_override=Lane.SKIM,
            ),
            "no-operation": queue.propose(
                ProposalKind.SOURCE_INGESTION,
                0.6,
                Impact.MEDIUM,
                "bulk src",
                {"item_id": "i"},
                lane_override=Lane.SKIM,
            ),
        }

        def reported_counts(calls: list[dict[str, Any]]) -> list[int] | None:
            match = re.search(
                r"(\d+) applied, (\d+) acknowledged.*?(\d+) failed",
                "\n".join(tg_texts(calls)),
            )
            return [int(count) for count in match.groups()] if match else None

        pre = approval_observe(w, vm_data)
        await harness.callback("autonomy_approve_all_skim", APPROVAL_OUTSIDER)
        ck(
            "approve-all.outsider-no-change",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )
        calls = await harness.callback("autonomy_approve_all_skim", APPROVAL_INSIDER)
        ck("approve-all.reported-outcome-counts", [1, 1, 2], reported_counts(calls))
        ck(
            "approve-all.per-proposal-states",
            {
                "applied": "applied",
                "raises": "failed",
                "info": "acknowledged",
                "no-operation": "failed",
            },
            {k: queue.get(v.proposal_id).status.value for k, v in mixed.items()},
        )
        ck("approve-all.real-mutation", ["bulk-tag"], frontmatter.load(w.vault / "bulk.md")["tags"])
        pre = approval_observe(w, vm_data)
        calls = await harness.callback("autonomy_approve_all_skim", APPROVAL_INSIDER)
        ck(
            "approve-all.never-retries-failed",
            [[0, 0, 0], approval_json_digest(pre)],
            [reported_counts(calls), approval_json_digest(approval_observe(w, vm_data))],
        )

        # ---- 8. /edit: owner spoof, auth, confirm/cancel, lineage, index -------------
        edit_body = "quartz gauge calibration drifted past tolerance " * 4
        (w.vault / "editme.md").write_text(approval_note("Edit me", edit_body), encoding="utf-8")
        (w.vault / "deleteme.md").write_text(
            approval_note("Delete me", "obsolete scaffold manifest listing " * 6), encoding="utf-8"
        )
        w.store.index_notes(w.parser.iter_notes(), w.parser)
        llm.reply = "pressure vessel recertification completed on schedule " * 3
        calls = await harness.command(
            "/edit editme.md rewrite about recertification", APPROVAL_PEER
        )
        edit_keyboards = tg_keyboards(calls)
        ck(
            "edit.keyboard-owner-bound",
            [[f"edit_confirm:{APPROVAL_PEER}", f"edit_cancel:{APPROVAL_PEER}"]],
            edit_keyboards,
        )
        ck("edit.pending-owned-by-requester", [APPROVAL_PEER], list(w.handlers._pending_edits))
        edited_before = (w.vault / "editme.md").read_bytes()
        pre = approval_observe(w, vm_data)
        for who, data, label in (
            (APPROVAL_OUTSIDER, f"edit_confirm:{APPROVAL_PEER}", "outsider-confirm"),
            (APPROVAL_OUTSIDER, f"edit_cancel:{APPROVAL_PEER}", "outsider-cancel"),
            (APPROVAL_INSIDER, f"edit_confirm:{APPROVAL_PEER}", "spoof-confirm"),
            (APPROVAL_INSIDER, f"edit_cancel:{APPROVAL_PEER}", "spoof-cancel"),
        ):
            calls = await harness.callback(data, who)
            ck(
                f"edit.{label}-refused",
                True,
                any("Unauthorized" in t or "Not your edit" in t for t in tg_texts(calls)),
            )
            ck(
                f"edit.{label}-no-message-edit",
                [],
                [c for c in calls if c["method"] == "EditMessageText"],
            )
        ck(
            "edit.spoof-and-outsider-leave-pending-and-bytes",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )
        calls = await harness.callback(f"edit_confirm:{APPROVAL_PEER}", APPROVAL_PEER)
        text = (w.vault / "editme.md").read_text(encoding="utf-8")
        ck(
            "edit.owner-confirm-new-body-above-callout-prior-preserved",
            [True, True, True],
            [llm.reply.strip() in text, "[!superseded]" in text, edit_body.strip() in text],
        )
        ck("edit.owner-confirm-consumes-pending", {}, w.handlers._pending_edits)
        ck(
            "edit.lineage-recorded-once",
            ["edited"],
            [e.event for e in w.lineage.get_lineage("editme.md")],
        )
        hits = w.store.search("pressure vessel recertification", n_results=3)
        ck(
            "edit.reindexed-chroma-search",
            "editme.md",
            hits[0]["metadata"]["note_path"] if hits else None,
        )
        pre = approval_observe(w, vm_data)
        await harness.callback(f"edit_confirm:{APPROVAL_PEER}", APPROVAL_PEER)
        ck(
            "edit.replay-confirm-no-write",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )
        await harness.command("/edit editme.md second pass", APPROVAL_PEER)
        pre_cancel = (w.vault / "editme.md").read_bytes()
        await harness.callback(f"edit_cancel:{APPROVAL_PEER}", APPROVAL_PEER)
        ck(
            "edit.owner-cancel-discards-pending-no-write",
            [{}, pre_cancel],
            [w.handlers._pending_edits, (w.vault / "editme.md").read_bytes()],
        )
        ck("edit.baseline-bytes-before-confirm-were-original", True, edited_before != pre_cancel)

        # ---- 9. /delete: auth, cancel, soft-delete, lineage, index, replay -----------
        calls = await harness.command("/delete deleteme.md", APPROVAL_OUTSIDER)
        ck("delete.outsider-command-silent", [], calls)
        calls = await harness.command("/delete deleteme.md", APPROVAL_INSIDER)
        delete_kb = tg_keyboards(calls)
        ck("delete.keyboard", [["delete_confirm:deleteme.md", "delete_cancel"]], delete_kb)
        pre = approval_observe(w, vm_data)
        calls = await harness.callback("delete_confirm:deleteme.md", APPROVAL_OUTSIDER)
        ck(
            "delete.outsider-confirm-refused-no-change",
            [True, approval_json_digest(pre)],
            [
                any("Unauthorized" in t for t in tg_texts(calls)),
                approval_json_digest(approval_observe(w, vm_data)),
            ],
        )
        calls = await harness.callback("delete_cancel", APPROVAL_OUTSIDER)
        ck(
            "delete.outsider-cancel-refused-no-change",
            [True, [], approval_json_digest(pre)],
            [
                any("Unauthorized" in t for t in tg_texts(calls)),
                [c for c in calls if c["method"] == "EditMessageText"],
                approval_json_digest(approval_observe(w, vm_data)),
            ],
        )
        await harness.callback("delete_cancel", APPROVAL_INSIDER)
        ck(
            "delete.cancel-no-change",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )
        original_delete = (w.vault / "deleteme.md").read_text(encoding="utf-8")
        await harness.callback("delete_confirm:deleteme.md", APPROVAL_INSIDER)
        marked = (w.vault / "deleteme.md").read_text(encoding="utf-8")
        ck(
            "delete.soft-delete-callout-and-original-preserved",
            [True, True],
            ["[!superseded]" in marked, original_delete.split("---\n", 2)[2].strip() in marked],
        )
        gone = w.store.search("obsolete scaffold manifest listing", n_results=5)
        ck(
            "delete.removed-from-vector-index",
            False,
            any(h["metadata"]["note_path"] == "deleteme.md" for h in gone),
        )
        ck(
            "delete.lineage-recorded",
            ["deleted"],
            [e.event for e in w.lineage.get_lineage("deleteme.md")],
        )
        pre = approval_observe(w, vm_data)
        await harness.callback("delete_confirm:deleteme.md", APPROVAL_INSIDER)
        ck(
            "delete.replay-no-write",
            approval_json_digest(pre),
            approval_json_digest(approval_observe(w, vm_data)),
        )

        # ---- 10. legacy schema migration, reopen, no automatic retry -----------------
        legacy_commit, legacy_schema = approval_legacy_schema(home)
        legacy_vault_note = "legacy-fail.md"
        statuses = ProposalStatus
        legacy_rows = [
            (
                "legacy-pending",
                ProposalKind.TAG_APPLICATION,
                "BLOCK",
                statuses.PENDING,
                "",
                {"note_path": "legacy-pending.md", "tags": ["lp"]},
            ),
            (
                "legacy-applied",
                ProposalKind.TAG_VOCABULARY,
                "SKIM",
                statuses.APPLIED,
                "'legacy-applied' added to vocabulary",
                {"tag": "legacy-applied"},
            ),
            (
                "legacy-ack",
                ProposalKind.DUPLICATE_MERGE,
                "SKIM",
                statuses.ACKNOWLEDGED,
                "",
                {"source_path": "a.md", "match_path": "b.md"},
            ),
            (
                "legacy-human-reject",
                ProposalKind.TAG_VOCABULARY,
                "BLOCK",
                statuses.REJECTED,
                "not wanted",
                {"tag": "legacy-human-reject"},
            ),
            (
                "legacy-known-failure",
                ProposalKind.TAG_APPLICATION,
                "BLOCK",
                statuses.REJECTED,
                "applier raised — see logs",
                {"note_path": legacy_vault_note, "tags": ["lf"]},
            ),
            (
                "legacy-ambiguous-1",
                ProposalKind.TAG_VOCABULARY,
                "BLOCK",
                statuses.REJECTED,
                "applier raised",
                {"tag": "legacy-ambiguous-1"},
            ),
            (
                "legacy-ambiguous-2",
                ProposalKind.TAG_VOCABULARY,
                "BLOCK",
                statuses.REJECTED,
                "",
                {"tag": "legacy-ambiguous-2"},
            ),
        ]
        columns = (
            "proposal_id,kind,lane,lane_reason,confidence,impact,summary,payload,"
            "status,result,created,last_seen,resolved,occurrence_count"
        )
        base_cols = columns.split(",")
        lid = {row[0]: digest(row[0].encode()) for row in legacy_rows}
        raw_before: dict[str, dict[str, Any]] = {}

        def seed(db_path: Path) -> None:
            import sqlite3

            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as conn:
                conn.executescript(legacy_schema)
                for i, (pid, kind, lane, status, result, payload) in enumerate(legacy_rows):
                    resolved = None if status is statuses.PENDING else f"2026-01-02T00:00:0{i}"
                    conn.execute(
                        f"INSERT INTO review_proposals ({columns}) VALUES ({','.join('?' * 14)})",
                        (
                            lid[pid],
                            kind.value,
                            lane,
                            "legacy fixture",
                            0.5,
                            "medium",
                            f"summary {pid}",
                            json.dumps(payload, sort_keys=True),
                            status.value,
                            result,
                            f"2026-01-01T00:00:0{i}",
                            f"2026-01-01T01:00:0{i}",
                            resolved,
                            i + 1,
                        ),
                    )
                conn.execute("PRAGMA user_version = 0")
                conn.row_factory = sqlite3.Row
                for r in conn.execute("SELECT * FROM review_proposals"):
                    raw_before[r["proposal_id"]] = dict(r)

        old_vocab = (vm_data / "tag_vocabulary.json").read_bytes()
        lw = approval_world(
            home / "legacy",
            force_block=True,
            allowed=[APPROVAL_INSIDER],
            notes={"legacy-pending.md": approval_note("LP", body)},
            llm=llm,
            seed_queue=seed,
            closers=closers,
        )
        import sqlite3

        def dump(db: str) -> dict[str, dict[str, Any]]:
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                return {
                    r["proposal_id"]: dict(r)
                    for r in conn.execute("SELECT * FROM review_proposals")
                }

        db = str(lw.settings.autonomy.db_path)
        rows = dump(db)
        ck("migration.legacy-schema-from-git", True, "legacy_status" not in legacy_schema)
        ck("migration.raw-legacy-rows-captured-before-open", len(legacy_rows), len(raw_before))
        for pid, _kind, _lane, _status, result, _payload in legacy_rows:
            known = pid == "legacy-known-failure"
            expected_row = dict(raw_before[lid[pid]])
            if known:
                expected_row["status"] = "failed"
            ck(
                f"migration.{pid}-all-original-columns-exact",
                expected_row,
                {c: rows[lid[pid]][c] for c in base_cols},
            )
            ck(
                f"migration.{pid}-legacy-evidence",
                ["rejected", result] if known else [None, None],
                [rows[lid[pid]]["legacy_status"], rows[lid[pid]]["legacy_result"]],
            )
        lw.queue.close()
        from vaultmind.config import load_settings

        reopened = app._build_review_queue(load_settings(lw.config))
        closers.append(reopened.close)
        ck(
            "migration.reopen-idempotent",
            approval_json_digest(rows),
            approval_json_digest(dump(db)),
        )
        ck(
            "migration.no-automatic-retry-bytes-and-vocab",
            [old_vocab, False],
            [
                (vm_data / "tag_vocabulary.json").read_bytes(),
                (lw.vault / legacy_vault_note).exists(),
            ],
        )
        ck(
            "migration.pending-stays-pending-and-listed",
            [lid["legacy-pending"]],
            [
                p.proposal_id
                for p in reopened.list_pending()
                if p.kind is ProposalKind.TAG_APPLICATION
            ],
        )
        legacy_handlers = commands_mod.CommandHandlers(
            lw.settings,
            lw.store,
            lw.graph,
            lw.parser,
            lw.handlers._ctx.thinking,
            llm,
            review_queue=reopened,
        )
        legacy_harness = TelegramHarness(legacy_handlers, lw.session, lw.bot)
        for pid in ("legacy-human-reject", "legacy-ambiguous-1", "legacy-ambiguous-2"):
            await legacy_harness.callback(callback_data("retry", lid[pid]), APPROVAL_INSIDER)
            ck(f"migration.{pid}-never-retryable", "rejected", reopened.get(lid[pid]).status.value)
        (lw.vault / legacy_vault_note).write_text(approval_note("LF", body), encoding="utf-8")
        await legacy_harness.callback(
            callback_data("retry", lid["legacy-known-failure"]), APPROVAL_INSIDER
        )
        ck(
            "migration.explicit-retry-of-known-failure-applies",
            ["applied", ["lf"]],
            [
                reopened.get(lid["legacy-known-failure"]).status.value,
                frontmatter.load(lw.vault / legacy_vault_note)["tags"],
            ],
        )
        await legacy_harness.callback(
            callback_data("approve", lid["legacy-pending"]), APPROVAL_INSIDER
        )
        ck(
            "migration.legacy-pending-approves-with-real-applier",
            ["applied", ["lp"]],
            [
                reopened.get(lid["legacy-pending"]).status.value,
                frontmatter.load(lw.vault / "legacy-pending.md")["tags"],
            ],
        )

        # ---- 11. existing producers: duplicate/contradiction/maturation/sources -------
        pw = approval_world(
            home / "producers",
            force_block=True,
            allowed=[APPROVAL_INSIDER],
            notes={
                "dupA.md": approval_note(
                    "Dup A", "vectora " + " ".join(APPROVAL_WORDS), modified='"2026-01-01T00:00:00"'
                )
            },
            llm=llm,
            embedder=ApprovalEmbedder(),
            closers=closers,
        )
        pq, psession, pharness = pw.queue, pw.session, pw.harness
        notifier_cb = build_block_notifier(Notifier(pw.bot, APPROVAL_CHAT))
        pw.store.index_notes([pw.parser.parse_file(pw.vault / "dupA.md")], pw.parser)
        detector = DuplicateDetector(pw.settings.duplicate_detection, pw.store)
        text_b = "vectorb " + " ".join(APPROVAL_WORDS)
        (pw.vault / "dupB.md").write_text(
            approval_note("Dup B", text_b, modified='"2026-01-01T00:00:00"'), encoding="utf-8"
        )
        (pw.vault / "dupC.md").write_text(
            approval_note("Dup C", text_b, modified='"2026-02-01T00:00:00"'), encoding="utf-8"
        )
        note_a, note_b, note_c = (pw.parser.parse_file(pw.vault / f"dup{x}.md") for x in "ABC")
        a_bytes = (pw.vault / "dupA.md").read_bytes()
        merge_hits = [m for m in detector.find_duplicates(note_b) if m.match_path == "dupA.md"]
        detector.clear_results()
        ck(
            "producers.real-detector-sees-exactly-one-merge",
            [["merge"], [0.85]],
            [[m.match_type.value for m in merge_hits], [m.similarity for m in merge_hits]],
        )

        bus = VaultEventBus()
        bus.subscribe(NoteCreatedEvent, app._duplicate_review_subscriber(detector, pq, notifier_cb))  # type: ignore[arg-type]
        mark = len(psession.calls)
        await bus.publish(NoteCreatedEvent(path=note_b.path, note=note_b))
        dups = [p for p in pq.list_pending() if p.kind is ProposalKind.DUPLICATE_MERGE]
        ck(
            "producer.duplicate-subscriber-event-path-newly-pending-forced-block",
            [1, "BLOCK"],
            [len(dups), dups[0].lane.name if dups else None],
        )
        ck(
            "producer.duplicate-one-generic-keyboard",
            [
                [
                    callback_data("approve", dups[0].proposal_id),
                    callback_data("reject", dups[0].proposal_id),
                ]
            ],
            tg_keyboards(psession.calls[mark:]),
        )
        await bus.publish(NoteCreatedEvent(path=note_b.path, note=note_b))
        ck(
            "producer.duplicate-dedup-occurrence-and-silence",
            [2, 1],
            [
                pq.get(dups[0].proposal_id).occurrence_count,
                len(tg_keyboards(psession.calls[mark:])),
            ],
        )
        await pharness.callback(callback_data("approve", dups[0].proposal_id), APPROVAL_INSIDER)
        ck(
            "producer.duplicate-informational-acknowledged-no-mutation",
            ["acknowledged", a_bytes],
            [pq.get(dups[0].proposal_id).status.value, (pw.vault / "dupA.md").read_bytes()],
        )
        mark = len(psession.calls)
        await bus.publish(NoteCreatedEvent(path=note_b.path, note=note_b))
        ck(
            "producer.duplicate-terminal-replay-no-resend",
            0,
            len(tg_keyboards(psession.calls[mark:])),
        )

        gaps = GapStore(pw.data / "gaps.db")
        closers.append(gaps._conn.close)
        contradiction = ContradictionDetector(
            pw.settings.contradiction,
            detector,
            llm,
            "scripted-model",
            pw.vault,
            pw.parser,
            gap_store=gaps,
            review_queue=pq,
            on_block=notifier_cb,
        )
        llm.reply = json.dumps(
            {"materially_conflicts": True, "reasoning": "scripted: valve count disagrees"}
        )
        mark = len(psession.calls)
        await contradiction.on_note_changed(NoteCreatedEvent(path=note_b.path, note=note_b))
        esc = [p for p in pq.list_pending() if p.kind is ProposalKind.CONTRADICTION_ESCALATION]
        ck(
            "producer.contradiction-escalation-newly-pending-block",
            [1, "BLOCK", "pending"],
            [len(esc), esc[0].lane.name if esc else None, esc[0].status.value if esc else None],
        )
        kbs = tg_keyboards(psession.calls[mark:])
        ck(
            "producer.contradiction-escalation-one-generic-keyboard",
            [
                [
                    callback_data("approve", esc[0].proposal_id),
                    callback_data("reject", esc[0].proposal_id),
                ]
            ],
            kbs,
        )
        await contradiction.on_note_changed(NoteCreatedEvent(path=note_b.path, note=note_b))
        ck(
            "producer.contradiction-dedup-no-resend",
            [2, 1],
            [pq.get(esc[0].proposal_id).occurrence_count, len(tg_keyboards(psession.calls[mark:]))],
        )
        await pharness.callback(callback_data("approve", esc[0].proposal_id), APPROVAL_INSIDER)
        ck(
            "producer.contradiction-escalation-ack-explicit",
            ["acknowledged", a_bytes],
            [pq.get(esc[0].proposal_id).status.value, (pw.vault / "dupA.md").read_bytes()],
        )
        mark = len(psession.calls)
        await contradiction.on_note_changed(NoteCreatedEvent(path=note_b.path, note=note_b))
        ck(
            "producer.contradiction-terminal-replay-no-resend",
            [0, "acknowledged"],
            [len(tg_keyboards(psession.calls[mark:])), pq.get(esc[0].proposal_id).status.value],
        )

        match_c = next(m for m in detector.find_duplicates(note_c) if m.match_path == "dupA.md")
        outcome = resolve_conflict(note_c, note_a)
        mark = len(psession.calls)
        await contradiction._apply_resolution(
            note_c, match_c, outcome
        )  # private seam: auto_resolve stays false
        res = [p for p in pq.list_pending() if p.kind is ProposalKind.CONTRADICTION_RESOLUTION]
        ck(
            "producer.contradiction-resolution-newly-pending-block-unapplied",
            [1, "BLOCK", False],
            [
                len(res),
                res[0].lane.name if res else None,
                bool(frontmatter.load(pw.vault / "dupA.md").get("contradicted_by")),
            ],
        )
        ck(
            "producer.contradiction-resolution-one-keyboard",
            1,
            len(tg_keyboards(psession.calls[mark:])),
        )
        await contradiction._apply_resolution(note_c, match_c, outcome)
        ck(
            "producer.contradiction-resolution-dedup-silent",
            [2, 1],
            [pq.get(res[0].proposal_id).occurrence_count, len(tg_keyboards(psession.calls[mark:]))],
        )
        await pharness.callback(callback_data("approve", res[0].proposal_id), APPROVAL_INSIDER)
        marked_a = frontmatter.load(pw.vault / "dupA.md")
        ck(
            "producer.contradiction-resolution-approved-real-marking",
            ["applied", ["dupC.md"], True],
            [
                pq.get(res[0].proposal_id).status.value,
                marked_a.get("contradicted_by"),
                " ".join(APPROVAL_WORDS) in marked_a.content,
            ],
        )

        mat_text = "mycelium network nutrient exchange between forest trees " * 3
        for i in range(3):
            (pw.vault / f"mat{i}.md").write_text(
                approval_note(f"Mat {i}", mat_text + f"variant{i}", type="fleeting"),
                encoding="utf-8",
            )
        pw.store.index_notes(pw.parser.iter_notes(), pw.parser)
        files_before_job = sorted(str(p) for p in pw.vault.rglob("*.md"))

        def pipeline_for(world: SimpleNamespace) -> Any:
            return MaturationPipeline(
                world.settings.maturation,
                world.store._collection,
                world.graph,
                llm,
                world.vault,
                state_path=world.data / "maturation_state.json",
            )

        pipeline = pipeline_for(pw)
        clusters = pipeline.discover()
        ck("producer.maturation-real-cluster-discovered", True, len(clusters) >= 1)
        job = app._maturation_review_job(pipeline, pq, notifier_cb)
        mark = len(psession.calls)
        await job()
        mats = [p for p in pq.list_pending() if p.kind is ProposalKind.MATURATION_SYNTHESIS]
        ck(
            "producer.maturation-forced-block-pending-one-keyboard-each",
            [len(clusters), ["BLOCK"] * len(clusters), len(clusters)],
            [len(mats), [p.lane.name for p in mats], len(tg_keyboards(psession.calls[mark:]))],
        )
        ck(
            "producer.maturation-no-synthesis-before-approval",
            files_before_job,
            sorted(str(p) for p in pw.vault.rglob("*.md")),
        )
        await job()
        ck(
            "producer.maturation-dedup-silent",
            len(clusters),
            len(tg_keyboards(psession.calls[mark:])),
        )
        fingerprint = mats[0].payload["fingerprint"]
        pipeline.dismiss(fingerprint)
        calls = await pharness.callback(
            callback_data("approve", mats[0].proposal_id), APPROVAL_INSIDER
        )
        got = pq.get(mats[0].proposal_id)
        probes.append("maturation-stale-cluster-real-applier")
        ck(
            "producer.maturation-stale-cluster-failed-honestly",
            [True, True],
            [got.status is ProposalStatus.FAILED, bool(got.result)],
        )
        ck(
            "producer.maturation-stale-not-reported-applied",
            False,
            any("Applied" in t for t in tg_texts(calls)),
        )
        dw = approval_world(
            home / "default-maturation",
            force_block=False,
            allowed=[APPROVAL_INSIDER],
            notes={
                f"mat{i}.md": approval_note(f"Mat {i}", mat_text + f"variant{i}", type="fleeting")
                for i in range(3)
            },
            llm=llm,
            closers=closers,
        )
        dw.store.index_notes(dw.parser.iter_notes(), dw.parser)
        dpipe = pipeline_for(dw)
        djob = app._maturation_review_job(
            dpipe, dw.queue, build_block_notifier(Notifier(dw.bot, APPROVAL_CHAT))
        )
        await djob()
        dmats = dw.queue.list_pending()
        ck(
            "producer.maturation-default-stays-skim-no-keyboard",
            [True, {"SKIM"}, 0],
            [bool(dmats), {p.lane.name for p in dmats}, len(tg_keyboards(dw.session.calls))],
        )
        llm.reply = "malformed scripted synthesis: not a valid permanent note"
        pre_files = sorted(str(p) for p in dw.vault.rglob("*.md"))
        await dw.harness.callback(callback_data("approve", dmats[0].proposal_id), APPROVAL_INSIDER)
        got = dw.queue.get(dmats[0].proposal_id)
        probes.append("maturation-malformed-synthesis-real-pipeline")
        ck(
            "producer.maturation-malformed-synthesis-failed-not-applied",
            [True, False],
            [got.status is ProposalStatus.FAILED, "applied" in got.status.value],
        )
        ck(
            "producer.maturation-malformed-synthesis-wrote-no-note",
            pre_files,
            sorted(str(p) for p in dw.vault.rglob("*.md")),
        )

        feed = pw.data / "feed.xml"
        feed.write_text(
            '<?xml version="1.0"?><rss version="2.0"><channel><title>Fixture</title>'
            "<item><guid>fixture-1</guid><title>Fixture item</title><link>https://example.invalid/1</link>"
            "<description>Local fixture feed body for acceptance.</description>"
            "<pubDate>Mon, 05 Jan 2026 10:00:00 +0000</pubDate></item></channel></rss>",
            encoding="utf-8",
        )
        source_feed_sha256 = digest(feed.read_bytes())
        instance = SourceInstance(name="fixture-feed", kind="rss", target=str(feed), enabled=True)
        pq.register_applier(
            ProposalKind.SOURCE_INGESTION,
            make_applier(vault_root=pw.vault, llm_client=llm, model="scripted-model"),
        )
        source_store = SourceStore(pw.data / "sources.db")
        closers.append(source_store.close)
        mark = len(psession.calls)
        run = await run_connector_once(
            instance,
            source_store=source_store,
            review_queue=pq,
            parser=pw.parser,
            store=pw.store,
            vault_root=pw.vault,
            event_bus=VaultEventBus(),
            on_block=notifier_cb,
        )
        srcs = [p for p in pq.list_pending() if p.kind is ProposalKind.SOURCE_INGESTION]
        ck("producer.source-run-real-rss-local-file-fetched", 1, run.items_fetched)
        ck(
            "producer.source-newly-pending-block-one-keyboard",
            [1, "BLOCK", 1],
            [
                len(srcs),
                srcs[0].lane.name if srcs else None,
                len(tg_keyboards(psession.calls[mark:])),
            ],
        )
        ck(
            "producer.source-no-note-written-before-approval",
            False,
            (pw.vault / "00-inbox" / "sources").exists(),
        )
        second_store = SourceStore(pw.data / "sources-second.db")
        closers.append(second_store.close)
        await run_connector_once(
            instance,
            source_store=second_store,
            review_queue=pq,
            parser=pw.parser,
            store=pw.store,
            vault_root=pw.vault,
            event_bus=VaultEventBus(),
            on_block=notifier_cb,
        )
        ck(
            "producer.source-dedup-silent",
            [2, 1, False],
            [
                pq.get(srcs[0].proposal_id).occurrence_count,
                len(tg_keyboards(psession.calls[mark:])),
                (pw.vault / "00-inbox" / "sources").exists(),
            ],
        )
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(old_level)
        for close in reversed(closers):
            close()

    ck(
        "probes.all-fault-probes-executed",
        [
            "filesystem-permission-denied-real-open",
            "maturation-malformed-synthesis-real-pipeline",
            "maturation-stale-cluster-real-applier",
            "missing-note-real-applier-raises",
        ],
        sorted(probes),
    )
    worlds = (w, lw, pw, dw)
    initial_worlds = {str(world.root.relative_to(home)): world.input_identity for world in worlds}
    corpus = {
        "initial_worlds": initial_worlds,
        "seed_vocabulary_sha256": seed_vocabulary_sha256,
        "source_feed_sha256": source_feed_sha256,
        "source_instance": asdict(instance),
        "legacy_rows_sha256": approval_json_digest(raw_before),
        "scripted_llm_calls": llm.calls,
    }
    final_files = {
        str(path.relative_to(home)): digest(path.read_bytes())
        for world in worlds
        for path in sorted(world.vault.rglob("*.md"))
    }
    receipt = {
        "import_origin": os.path.realpath(vaultmind.__file__),
        "interpreter": {"executable": sys.executable, "version": sys.version},
        "git": approval_git_binding(home),
        "cli": {
            "argv": ["--config", "<w>/config.toml", "auto-tag", "--apply"],
            "exit_code": cli.exit_code,
            "raw_output": cli.output,
        },
        "raw_log": log_records,
        "configuration_files_sha256": {
            name: identity["files_sha256"]["config.toml"]
            for name, identity in initial_worlds.items()
        },
        "config_sha256": approval_json_digest(
            {name: identity["effective_config_sha256"] for name, identity in initial_worlds.items()}
        ),
        "corpus_inputs": corpus,
        "corpus_sha256": approval_json_digest(corpus),
        "final_corpus_files_sha256": final_files,
        "legacy_schema": {
            "commit": legacy_commit,
            "sha256": digest(legacy_schema.encode()),
            "label_to_proposal_id": lid,
        },
        "duplicate_fixture_vectors": {
            "cosine": APPROVAL_MERGE_COSINE,
            "vectors_sha256": approval_json_digest(
                {token: ApprovalEmbedder().embed_query(token) for token in ("vectora", "vectorb")}
            ),
            "boundary": "scripted provider-boundary embedding vectors, not model quality",
        },
        "delivery_statuses": delivery,
        "fault_probes_executed": sorted(probes),
        "scripted_llm_calls": len(llm.calls),
        "source_run": {
            "items_fetched": run.items_fetched,
            "items_ingested": run.items_ingested,
            "error": run.error,
        },
        "failed_checks": failures,
        "fixture_limits": (
            "Fixture-only offline proof. Source write/index/finalization (M15) is NOT exercised or "
            "claimed: SOURCE_INGESTION is only proven pending/unapplied with one keyboard. "
            "Contradiction resolution is driven via ContradictionDetector._apply_resolution "
            "because [contradiction].auto_resolve stays false. No native model, provider, "
            "embedding-quality or real Telegram qualification."
        ),
    }
    if failures:
        checks_fail = AssertionError(
            f"approval acceptance failed {len(failures)} check(s): {failures}"
        )
        checks_fail.receipt = receipt  # type: ignore[attr-defined]
        raise checks_fail
    return receipt


async def config_mcp_policy_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """M21: real Click/SDK policy calls; terminal transport and embeddings are substituted."""
    from importlib import import_module
    from unittest.mock import patch

    from click.testing import CliRunner
    from mcp.types import CallToolRequest, CallToolRequestParams
    from pydantic import ValidationError

    import vaultmind.indexer as indexer
    from vaultmind.config import load_settings

    cli_module = import_module("vaultmind.cli")
    check = make_checker(checks)
    root = home / "config-mcp-case"
    selected = root / "selected-vault"
    sentinel = root / "sentinel-vault"
    data = root / "selected-data"
    forbidden = root / "forbidden-data"
    for vault, marker in ((selected, "SELECTED-POLICY"), (sentinel, "SENTINEL-POLICY")):
        for folder in ("allowed", "dotenv-only", "toml-only"):
            target = vault / folder / "policy.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"---\ntitle: Policy\n---\n\n{marker} {folder}\n", encoding="utf-8")

    def snapshot(vault: Path) -> dict[str, Any]:
        return {
            str(path.relative_to(vault)): (
                path.is_dir(),
                path.stat().st_mtime_ns,
                None if path.is_dir() else digest(path.read_bytes()),
            )
            for path in [vault, *sorted(vault.rglob("*"))]
        }

    before = snapshot(sentinel)
    config = root / "policy.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(sentinel))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        f"[chroma]\npersist_dir = {json.dumps(str(forbidden / 'chroma'))}\n"
        f"[graph]\npersist_path = {json.dumps(str(forbidden / 'graph.json'))}\n"
        "[embedding]\ncache_enabled = false\n"
        "[mcp.profiles.config-policy]\n"
        'description = "source-precedence policy fixture"\n'
        'allowed_tools = ["vault_read"]\nfolder_scope = ["toml-only"]\n'
        "write_enabled = true\nmax_note_size_bytes = 4096\n",
        encoding="utf-8",
    )
    dotenv = home / ".env"
    dotenv.write_text(
        "VAULTMIND_MCP__PROFILES__CONFIG-POLICY__WRITE_ENABLED=false\n"
        'VAULTMIND_MCP__PROFILES__CONFIG-POLICY__ALLOWED_TOOLS=["vault_read","vault_stats"]\n'
        'VAULTMIND_MCP__PROFILES__CONFIG-POLICY__FOLDER_SCOPE=["dotenv-only"]\n'
        "VAULTMIND_MCP__PROFILES__CONFIG-POLICY__MAX_NOTE_SIZE_BYTES=1024\n",
        encoding="utf-8",
    )
    prefix = "VAULTMIND_MCP__PROFILES__CONFIG-POLICY__"
    overrides = {
        "VAULTMIND_VAULT__PATH": str(selected),
        "VAULTMIND_CHROMA__PERSIST_DIR": str(data / "chroma"),
        "VAULTMIND_GRAPH__PERSIST_PATH": str(data / "graph.json"),
        "VAULTMIND_OPENAI_API_KEY": "acceptance-only-not-a-provider-key",
        prefix + "WRITE_ENABLED": "false",
        prefix + "ALLOWED_TOOLS": json.dumps(["vault_read", "vault_write", "vault_search"]),
        prefix + "FOLDER_SCOPE": '["allowed"]',
        prefix + "MAX_NOTE_SIZE_BYTES": "256",
    }
    requests: list[tuple[str, dict[str, Any]]] = []
    responses: list[dict[str, Any]] = []
    starts = 0

    async def terminal_transport(server: Any, _stdio_server: Any) -> None:
        """Terminal stdio/run seam only; exercise the actual SDK-registered handler."""
        nonlocal starts
        starts += 1
        handler = server.request_handlers[CallToolRequest]
        for tool, arguments in requests:
            request = CallToolRequest(
                method="tools/call", params=CallToolRequestParams(name=tool, arguments=arguments)
            )
            result = (await handler(request)).root
            responses.append(json.loads(result.content[0].text))

    async def serve(calls: list[tuple[str, dict[str, Any]]]) -> Any:
        requests[:] = calls
        responses.clear()
        # CliRunner invokes the real command in a thread because mcp-serve owns asyncio.run.
        # No settings, profile, enforcer, store, persistence, or SDK handler is substituted.
        result = await asyncio.to_thread(
            CliRunner().invoke,
            cli_module.cli,
            ["--config", str(config), "mcp-serve", "--profile", "config-policy"],
        )
        return result

    written = selected / "allowed" / "written.md"
    content = "---\ntitle: Policy write\n---\n\nharbor policy persisted\n"
    receipts: dict[str, Any] = {}
    try:
        with (
            patch.dict(os.environ, overrides),
            patch.object(indexer, "Embedder", side_effect=lambda *a, **k: DeterministicEmbedder()),
            patch.object(cli_module, "_run_mcp", terminal_transport),
        ):
            result = await serve(
                [
                    ("vault_read", {"path": "allowed/policy.md"}),
                    ("vault_write", {"path": "allowed/written.md", "content": content}),
                    ("vault_read", {"path": "dotenv-only/policy.md"}),
                    ("vault_read", {"path": "toml-only/policy.md"}),
                    ("vault_stats", {}),
                ]
            )
            check("config.mcp.false.cli-exit", 0, result.exit_code)
            check("config.mcp.false.transport-starts", 1, starts)
            check("config.mcp.false.response-count", 5, len(responses))
            selected_read, denied_write, dotenv_read, toml_read, denied_tool = responses
            check(
                "config.mcp.selected-vault-read",
                "SELECTED-POLICY allowed",
                selected_read.get("content", "").split("\n\n")[-1].strip(),
            )
            check(
                "config.mcp.false.write-denied",
                True,
                "Write operations disabled" in denied_write.get("error", ""),
            )
            check("config.mcp.false.no-persisted-write", False, written.exists())
            for label, payload in (("dotenv", dotenv_read), ("toml", toml_read)):
                check(
                    f"config.mcp.env-folder-denies-{label}",
                    True,
                    bool(payload.get("error")) and "content" not in payload,
                )
            check("config.mcp.env-tool-denies-dotenv", True, bool(denied_tool.get("error")))
            receipts["false"] = list(responses)
            del os.environ[prefix + "WRITE_ENABLED"]
            result = await serve(
                [("vault_write", {"path": "allowed/written.md", "content": content})]
            )
            check("config.mcp.dotenv-false.cli-exit", 0, result.exit_code)
            check("config.mcp.dotenv-false.transport-starts", 2, starts)
            check("config.mcp.dotenv-false.response-count", 1, len(responses))
            check(
                "config.mcp.dotenv-false.over-toml-true",
                True,
                "Write operations disabled" in responses[0].get("error", ""),
            )
            check("config.mcp.dotenv-false.no-persisted-write", False, written.exists())
            receipts["dotenv_false"] = list(responses)

            # Explicit env true overrides dotenv false; replay the exact denied arguments.
            os.environ[prefix + "WRITE_ENABLED"] = "true"
            result = await serve(
                [
                    ("vault_write", {"path": "allowed/written.md", "content": content}),
                    ("vault_read", {"path": "allowed/written.md"}),
                    ("vault_search", {"query": "harbor policy persisted", "n_results": 1}),
                    ("vault_write", {"path": "allowed/oversize.md", "content": "x" * 257}),
                    ("vault_write", {"path": "dotenv-only/written.md", "content": content}),
                ]
            )
            check("config.mcp.true.cli-exit", 0, result.exit_code)
            check("config.mcp.true.transport-starts", 3, starts)
            check("config.mcp.true.response-count", 5, len(responses))
            allowed_write, reread, search, oversized, out_of_scope = responses
            check(
                "config.mcp.true.write-success",
                {"status": "ok", "path": "allowed/written.md"},
                allowed_write,
            )
            check(
                "config.mcp.true.persisted-content",
                content,
                written.read_text(encoding="utf-8") if written.exists() else None,
            )
            check("config.mcp.true.sdk-reread", content, reread.get("content"))
            indexed = search.get("results", [])
            check("config.mcp.true.indexed-count", 1, len(indexed))
            check(
                "config.mcp.true.indexed-paths",
                ["allowed/written.md"],
                [hit["metadata"]["note_path"] for hit in indexed],
            )
            check(
                "config.mcp.true.indexed-content",
                True,
                bool(indexed) and "harbor policy persisted" in indexed[0]["content"],
            )
            check(
                "config.mcp.env-size-denied",
                True,
                "exceeds limit 256" in oversized.get("error", ""),
            )
            check(
                "config.mcp.env-size-not-persisted",
                False,
                (selected / "allowed" / "oversize.md").exists(),
            )
            check("config.mcp.true.scope-denied", True, bool(out_of_scope.get("error")))
            check(
                "config.mcp.true.scope-not-persisted",
                False,
                (selected / "dotenv-only" / "written.md").exists(),
            )
            check("config.mcp.selected-chroma-created", True, (data / "chroma").is_dir())
            receipts["true"] = list(responses)
            stable = snapshot(selected)
            for label, leaf, value in (
                ("bool", "WRITE_ENABLED", "not-a-boolean"),
                ("tool-list", "ALLOWED_TOOLS", '{"not":"a-list"}'),
                ("tool-leaf", "ALLOWED_TOOLS", '["vault_read",9]'),
                ("folder-list", "FOLDER_SCOPE", '{"not":"a-list"}'),
                ("folder-leaf", "FOLDER_SCOPE", '["allowed",9]'),
                ("negative-size", "MAX_NOTE_SIZE_BYTES", "-1"),
                ("invalid-size", "MAX_NOTE_SIZE_BYTES", "not-a-size"),
            ):
                with patch.dict(os.environ, {prefix + leaf: value}):
                    invalid = False
                    try:
                        load_settings(config)
                    except ValidationError:
                        invalid = True
                    check(f"config.mcp.invalid-{label}.load-rejected", True, invalid)
                    result = await serve([])
                    check(f"config.mcp.invalid-{label}.cli-failed", True, result.exit_code != 0)
                    check(
                        f"config.mcp.invalid-{label}.cli-validation",
                        True,
                        isinstance(result.exception, ValidationError),
                    )
                    check(f"config.mcp.invalid-{label}.before-server-start", 3, starts)
                    check(f"config.mcp.invalid-{label}.vault-unchanged", stable, snapshot(selected))
    finally:
        check("config.mcp.sentinel-unchanged", before, snapshot(sentinel))
        check("config.mcp.forbidden-data-absent", False, forbidden.exists())
        dotenv.unlink(missing_ok=True)
    return {
        "boundary": (
            "real Click mcp-serve assembly and SDK CallToolRequest handler; terminal stdio/run "
            "boundary replaced; deterministic embedding provider boundary only"
        ),
        "policy_responses": receipts,
    }


async def config_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """M21: source precedence reaches real CLI indexing and retrieval computation."""
    from unittest.mock import patch

    from click.testing import CliRunner
    from pydantic import ValidationError

    import vaultmind.indexer as indexer
    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.bot.handlers.recall import handle_recall
    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.cli import cli
    from vaultmind.config import Settings, load_settings
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

    check = make_checker(checks)
    root = home / "config-case"
    sentinel = root / "sentinel-vault"
    selected = root / "selected-vault"
    forbidden = root / "forbidden-data"
    data = root / "selected-data"
    sentinel.mkdir(parents=True)
    check("config.isolation.home", str(home), os.environ.get("HOME"))
    check("config.isolation.cwd", True, home.resolve() == Path.cwd().resolve())
    selected.mkdir()
    data.mkdir()
    (sentinel / "private.md").write_text("Sentinel must never be indexed.\n", encoding="utf-8")
    for name in ("alpha", "beta", "gamma"):
        (selected / f"{name}.md").write_text(
            f"---\ntitle: {name}\n---\n\nharbor lantern {name} inspections\n",
            encoding="utf-8",
        )

    def snapshot() -> dict[str, Any]:
        return {
            str(path.relative_to(sentinel)): (
                path.is_dir(),
                path.stat().st_mtime_ns,
                None if path.is_dir() else digest(path.read_bytes()),
            )
            for path in [sentinel, *sorted(sentinel.rglob("*"))]
        }

    before = snapshot()
    # Explicit old lists fail actionably; defaults are not a silent repair.
    legacy_config = root / "legacy-missing-exclusions.toml"
    legacy_bytes = (
        f"[vault]\npath = {json.dumps(str(sentinel))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "_wiki"]\n'
    ).encode()
    legacy_config.write_bytes(legacy_bytes)
    for args in (["ask", "private"], ["search-sources", "private"]):
        result = CliRunner().invoke(cli, ["--config", str(legacy_config), *args])
        check("config.old-exclusions." + args[0] + ".fails", True, result.exit_code != 0)
        check(
            "config.old-exclusions." + args[0] + ".actionable",
            True,
            "excluded_folders" in result.output
            and "wiki" in result.output
            and "_wiki" in result.output,
        )
        check(
            "config.old-exclusions." + args[0] + ".not-rewritten",
            True,
            legacy_bytes == legacy_config.read_bytes(),
        )
    check("config.old-exclusions.no-workspace", False, (sentinel / "_wiki").exists())
    check("config.old-exclusions.no-index", False, forbidden.exists())
    config = root / "explicit.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(sentinel))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        f"[chroma]\npersist_dir = {json.dumps(str(forbidden / 'chroma'))}\n"
        f"[graph]\npersist_path = {json.dumps(str(forbidden / 'graph.json'))}\n"
        "[search]\nhybrid_enabled = true\nmax_results = 3\npage_size = 7\n"
        f"bm25_db_path = {json.dumps(str(forbidden / 'bm25.db'))}\n"
        "cache_enabled = false\n[embedding]\ncache_enabled = false\n",
        encoding="utf-8",
    )
    dotenv = home / ".env"
    dotenv.write_text("VAULTMIND_SEARCH__MAX_RESULTS=2\n", encoding="utf-8")
    fixture_secret = "acceptance-only-not-a-provider-key"
    overrides = {
        "VAULTMIND_VAULT__PATH": str(selected),
        "VAULTMIND_CHROMA__PERSIST_DIR": str(data / "chroma"),
        "VAULTMIND_GRAPH__PERSIST_PATH": str(data / "graph.json"),
        "VAULTMIND_SEARCH__BM25_DB_PATH": str(data / "bm25.db"),
        "VAULTMIND_SEARCH__MAX_RESULTS": "1",
        "VAULTMIND_OPENAI_API_KEY": fixture_secret,
    }
    captured: list[str] = []
    try:
        with patch.dict(os.environ, overrides):
            settings = load_settings(config)
            check("config.env.vault", True, settings.vault.path == selected.resolve())
            check("config.env.chroma", True, settings.chroma.persist_dir == data / "chroma")
            check("config.env.graph", True, settings.graph.persist_path == data / "graph.json")
            check("config.env.bm25", True, Path(settings.search.bm25_db_path) == data / "bm25.db")
            check("config.toml.unset-leaf-preserved", 7, settings.search.page_size)
            check("config.env.over-dotenv", 1, settings.search.max_results)
            deliberate = Settings.from_toml(config, search={"max_results": 4})
            check("config.constructor.over-env", 4, deliberate.search.max_results)
            check("config.constructor.nested-merge", 7, deliberate.search.page_size)

            # This is the only substituted production boundary. CLI settings loading,
            # parsing, chunking, Chroma persistence and BM25 indexing remain real.
            with patch.object(
                indexer, "Embedder", side_effect=lambda *a, **k: DeterministicEmbedder()
            ):
                result = CliRunner().invoke(cli, ["--config", str(config), "index"])
            captured.append(result.output)
            check("config.cli.index-exit", 0, result.exit_code)
            check("config.cli.chroma-created", True, (data / "chroma").is_dir())
            check("config.cli.bm25-created", True, (data / "bm25.db").is_file())
            bm25 = BM25Index(data / "bm25.db")
            try:
                store = VaultStore(
                    settings.chroma, cast("Embedder", DeterministicEmbedder()), bm25=bm25
                )

                async def retrieve(current: Any) -> list[dict[str, Any]]:
                    llm = ScriptedLLM()
                    context = HandlerContext(
                        current,
                        store,
                        KnowledgeGraph(current.graph),
                        VaultParser(current.vault),
                        ThinkingPartner(current, llm_factory=lambda: llm),
                        llm,
                        current.vault.path,
                    )
                    message = FakeMessage()
                    sessions: dict[int, Any] = {}
                    await handle_recall(
                        context, cast("TelegramMessage", message), "harbor lantern", sessions
                    )
                    captured.extend(message.answers)
                    return sessions[42].results

                hits = await retrieve(settings)
                check("config.consumer.env-result-count", 1, len(hits))
                check(
                    "config.consumer.selected-note",
                    True,
                    all(
                        hit["metadata"]["note_path"] in {"alpha.md", "beta.md", "gamma.md"}
                        for hit in hits
                    ),
                )
                check(
                    "config.consumer.bm25-indexed",
                    ["alpha.md", "beta.md", "gamma.md"],
                    sorted(hit["note_path"] for hit in bm25.search("harbor", 3)),
                )
                del os.environ["VAULTMIND_SEARCH__MAX_RESULTS"]
                dotenv_settings = load_settings(config)
                check("config.dotenv.over-toml", 2, dotenv_settings.search.max_results)
                check(
                    "config.consumer.dotenv-result-count", 2, len(await retrieve(dotenv_settings))
                )
                dotenv.unlink()
                toml_settings = load_settings(config)
                check("config.toml.absent-env", 3, toml_settings.search.max_results)
                toml_hits = await retrieve(toml_settings)
                check("config.consumer.toml-result-count", 3, len(toml_hits))
                check(
                    "config.consumer.toml-note-paths",
                    ["alpha.md", "beta.md", "gamma.md"],
                    sorted(hit["metadata"]["note_path"] for hit in toml_hits),
                )
                # Index with hybrid enabled above so the independent BM25 proof
                # remains real; disable it only for this loaded consumer run.
                with patch.dict(os.environ, {"VAULTMIND_SEARCH__HYBRID_ENABLED": "false"}):
                    semantic_settings = load_settings(config)
                    check(
                        "config.env.semantic-only", False, semantic_settings.search.hybrid_enabled
                    )
                    semantic_hits = await retrieve(semantic_settings)
                check(
                    "config.consumer.semantic-note-paths",
                    ["alpha.md", "beta.md", "gamma.md"],
                    sorted(hit["metadata"]["note_path"] for hit in semantic_hits),
                )
                check("config.consumer.chroma-document-count", 3, store._collection.count())
                check(
                    "config.consumer.semantic-indexed-content",
                    ["alpha.md", "beta.md", "gamma.md"],
                    sorted(
                        hit["metadata"]["note_path"]
                        for hit in semantic_hits
                        if (
                            f"harbor lantern {Path(hit['metadata']['note_path']).stem} inspections"
                            in hit["content"]
                        )
                    ),
                )
                check(
                    "config.consumer.semantic-vector-evidence",
                    3,
                    sum(
                        isinstance(hit.get("distance"), (int, float))
                        and not isinstance(hit["distance"], bool)
                        and math.isfinite(hit["distance"])
                        and 0 <= hit["distance"] < 1
                        and bool(hit.get("chunk_id"))
                        for hit in semantic_hits
                    ),
                )
            finally:
                bm25.close()

            graph = KnowledgeGraph(settings.graph)
            graph.add_entity("Harbor", "project", source_note="alpha.md")
            graph.save()
            check("config.consumer.graph-created", True, (data / "graph.json").is_file())
            reloaded = KnowledgeGraph(settings.graph)
            entity = reloaded.get_entity("Harbor")
            check(
                "config.consumer.graph-reload",
                ["alpha.md"],
                entity["source_notes"] if entity is not None else None,
            )
            os.environ["VAULTMIND_SEARCH__MAX_RESULTS"] = "not-an-integer"
            invalid = False
            try:
                load_settings(config)
            except ValidationError:
                invalid = True
            check("config.invalid-env-rejected", True, invalid)
            # Invalid input must fail at the real command's loading boundary too.
            result = CliRunner().invoke(cli, ["--config", str(config), "index"])
            captured.append(result.output)
            check("config.cli.invalid-env-rejected", True, result.exit_code != 0)
            check(
                "config.cli.invalid-env-validation",
                True,
                isinstance(result.exception, ValidationError),
            )
            check(
                "config.secret-free-output",
                False,
                fixture_secret in json.dumps(checks) + "".join(captured),
            )
            del os.environ["VAULTMIND_SEARCH__MAX_RESULTS"]
            os.environ["VAULTMIND_VAULT__PATH"] = str(root / "missing-vault")
            invalid_vault = False
            try:
                load_settings(config)
            except ValidationError:
                invalid_vault = True
            check("config.invalid-vault-rejected", True, invalid_vault)
            for key in overrides:
                os.environ.pop(key, None)
            fallback = load_settings(config)
            check("config.toml.vault-preserved", True, fallback.vault.path == sentinel.resolve())
            check(
                "config.toml.chroma-preserved",
                True,
                fallback.chroma.persist_dir == forbidden / "chroma",
            )
            check(
                "config.toml.graph-preserved",
                True,
                fallback.graph.persist_path == forbidden / "graph.json",
            )
            check(
                "config.toml.bm25-preserved",
                True,
                Path(fallback.search.bm25_db_path) == forbidden / "bm25.db",
            )
    finally:
        check("config.isolation.sentinel-unchanged", True, snapshot() == before)
        check("config.isolation.forbidden-data-absent", False, forbidden.exists())
        if dotenv.exists():
            dotenv.unlink()
    mcp_policy = await config_mcp_policy_case(home, checks)
    return {
        "fixture": "explicit temporary TOML, cwd dotenv, HOME, sentinel and selected vault/data",
        "consumer": (
            "real Click index; semantic-only recall over persisted Chroma; BM25 contents; "
            "graph save/reload; real Click mcp-serve SDK policy and persistence"
        ),
        "mcp_policy": mcp_policy,
    }


async def wiki_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise the public manual wiki commands with synthetic state and a scripted provider."""
    import difflib
    import sqlite3
    import stat
    from unittest.mock import patch

    from click.testing import CliRunner

    import vaultmind.cli as app
    from vaultmind.config import VAULTMIND_HOME
    from vaultmind.llm.client import LLMResponse
    from vaultmind.wiki.workspace import WikiWorkspace, proposal_digest

    home = home.resolve()
    check = make_checker(checks)
    check("wiki.isolation.home", str(home), str(Path(os.environ["HOME"]).resolve()))
    check("wiki.isolation.cwd", str(home), str(Path.cwd().resolve()))
    check(
        "wiki.isolation.application-home",
        str(home / ".vaultmind"),
        str(VAULTMIND_HOME.resolve()),
    )
    root = home / "wiki-case"
    vault = root / "vault"
    vault.mkdir(parents=True)
    forbidden = root / "forbidden-indexes"
    config = root / "config.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = [".obsidian", ".git", ".trash", "wiki", "_wiki"]\n'
        '[llm]\nprovider = "ollama"\nthinking_model = "fixture-wiki-model"\nmax_tokens = 1777\n'
        f"[chroma]\npersist_dir = {json.dumps(str(forbidden / 'chroma'))}\n"
        f"[search]\nbm25_db_path = {json.dumps(str(forbidden / 'bm25.db'))}\n"
        f"[graph]\npersist_path = {json.dumps(str(forbidden / 'graph.json'))}\n",
        encoding="utf-8",
    )
    config_before = config.read_bytes()
    source_a = root / "source-a.md"
    source_b = root / "source-b.md"
    source_a.write_bytes(b"# Initial cache policy\r\n\r\nThe project note recommends 30 days.\r\n")
    source_b.write_bytes(
        b"# Operations policy\n\nThe operations note recommends 7 days; scope is unresolved.\n"
    )
    initial_body = (
        "# Report cache policy\n\n## Current understanding\n\n"
        "The project note recommends 30 days. [[_wiki/sources/source-a]]\n\n"
        "[Original source](../_wiki/sources/source-a.md#initial-cache-policy)\n"
    )
    conflict_body = (
        "# Report cache policy\n\n## Current understanding\n\n"
        "Retention duration is unresolved across the two sources.\n\n"
        "## Conflicting information\n\n"
        "- The project note recommends 30 days. [[_wiki/sources/source-a]]\n"
        "- The operations note recommends 7 days, with unresolved scope. "
        "[[_wiki/sources/source-b]]\n"
    )
    page = vault / "wiki/report-cache.md"
    control = vault / "_wiki"
    runner = CliRunner()
    model_calls: list[dict[str, Any]] = []
    factory_calls: list[tuple[str, str, int]] = []
    responses: list[str | Exception] = []

    class WikiLLM:
        provider_name = "scripted-wiki-acceptance"

        def complete(
            self,
            messages: list[Message],
            model: str,
            max_tokens: int = 4096,
            system: str | None = None,
        ) -> LLMResponse:
            model_calls.append(
                {
                    "messages": [m.content for m in messages],
                    "system": system,
                    "model": model,
                    "max_tokens": max_tokens,
                }
            )
            if not responses:
                raise AssertionError("Unexpected additional wiki completion")
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return LLMResponse(text=response, model=model, usage={})

        def complete_multimodal(
            self,
            messages: list[MultimodalMessage],
            model: str,
            max_tokens: int = 4096,
            system: str | None = None,
        ) -> LLMResponse:
            raise AssertionError("Wiki must not request a multimodal completion")

    def factory(settings: Any) -> WikiLLM:
        factory_calls.append(
            (settings.llm.provider, settings.llm.thinking_model, settings.llm.max_tokens)
        )
        return WikiLLM()

    def command(*args: str, ok: bool = True) -> Any:
        result = runner.invoke(app.cli, ["--config", str(config), "wiki", *args])
        check("wiki.cli." + ".".join(args[:1]) + ".exit", ok, result.exit_code == 0)
        if not ok:
            check("wiki.cli.error-is-explicit", True, bool(result.output.strip()))
        return result

    def queued(proposal_id: str) -> dict[str, Any]:
        with sqlite3.connect(control / "review_queue.db") as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM review_proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is None:
            raise AssertionError("CLI printed a proposal that is absent from its queue")
        return {**dict(row), "payload": json.loads(row["payload"])}

    def propose(body: str, *, generated: bool = False, source: str = "source-a") -> str:
        args = ["propose", "--page", "wiki/report-cache.md", "--source", source]
        if generated:
            responses.append(body)
        else:
            draft = root / "draft.md"
            draft.write_text(body, encoding="utf-8")
            args += ["--draft-file", str(draft)]
        result = command(*args)
        match = re.search(r"^Proposal: ([0-9a-f]{64})$", result.output, re.M)
        if match is None:
            raise AssertionError("CLI did not return an unambiguous proposal identity")
        return match.group(1)

    def reviewed(proposal_id: str) -> tuple[dict[str, Any], str]:
        row = queued(proposal_id)
        payload = row["payload"]
        review = command("review", proposal_id).output
        expected_digest = proposal_digest(payload)
        check("wiki.review.digest", True, f"SHA256: {expected_digest}" in review)
        before = payload["before"]
        after = payload["after"]
        exact_diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="a/wiki/report-cache.md",
                tofile="b/wiki/report-cache.md",
            )
        )
        check("wiki.review.exact-diff", True, exact_diff in review)
        for source in payload["sources"]:
            check("wiki.review.source-id", True, source["id"] in review)
            check("wiki.review.full-source-hash", True, source["sha256"] in review)
        check("wiki.review.block", "BLOCK", row["lane"])
        check("wiki.review.pending", "pending", row["status"])
        return payload, expected_digest

    def history() -> dict[str, str]:
        return {
            str(path.relative_to(control)): digest(path.read_bytes())
            for path in (control / "history").rglob("*.md")
        }

    with patch.object(app, "_create_llm_client", side_effect=factory):
        added = command("add", str(source_a), "--id", "source-a", "--uri", "urn:example:project")
        check("wiki.add.full-hash-printed", True, digest(source_a.read_bytes()) in added.output)
        check(
            "wiki.add.original-bytes",
            True,
            source_a.read_bytes() == (control / "sources/source-a.md").read_bytes(),
        )
        first_id = propose(initial_body, generated=True)
        first, first_digest = reviewed(first_id)
        check("wiki.create.not-applied-before-review", False, page.exists())
        check(
            "wiki.create.exact-original-source-context",
            True,
            source_a.read_bytes().decode() in "\n".join(model_calls[0]["messages"]),
        )
        command("apply", first_id, "--expect", first_digest)
        first_bytes = first["after"].encode()
        check("wiki.create.accepted-exact-candidate", True, page.read_bytes() == first_bytes)
        check("wiki.create.queue-real-applied", "applied", queued(first_id)["status"])
        check(
            "wiki.create.source-still-original",
            True,
            source_a.read_bytes() == (control / "sources/source-a.md").read_bytes(),
        )
        markdown_target = page.parent / "../_wiki/sources/source-a.md"
        check(
            "wiki.markdown.source-resolves-from-actual-page",
            str((control / "sources/source-a.md").resolve()),
            str(markdown_target.resolve()),
        )
        check(
            "wiki.markdown.resolved-original-bytes",
            True,
            markdown_target.read_bytes() == source_a.read_bytes(),
        )
        fifo = root / "nonregular-source.md"
        os.mkfifo(fifo)
        queue_before_fifo = (control / "review_queue.db").read_bytes()
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import json, socket, sys\n"
                "attempts = []\n"
                "def guard(event, args):\n"
                "    if event in {'socket.getaddrinfo', 'socket.gethostbyname', "
                "'socket.gethostbyaddr'} or (event in {'socket.connect', 'socket.sendto'} "
                "and args[0].family != socket.AF_UNIX):\n"
                "        attempts.append(event)\n"
                "        raise RuntimeError('Network forbidden in wiki FIFO acceptance')\n"
                "sys.addaudithook(guard)\n"
                "from click.testing import CliRunner\n"
                "from vaultmind.cli import cli\n"
                "result = CliRunner().invoke(cli, sys.argv[1:])\n"
                "print(json.dumps({'exit_code': result.exit_code, 'output': result.output, "
                "'network_attempts': attempts}))\n",
                "--config",
                str(config),
                "wiki",
                "add",
                str(fifo),
                "--id",
                "nonregular",
            ],
            cwd=home,
            env={
                "HOME": str(home),
                "PATH": os.environ["PATH"],
                "ANONYMIZED_TELEMETRY": "False",
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        check("wiki.nonregular.subprocess-completes", 0, child.returncode)
        refusal = json.loads(child.stdout)
        check("wiki.nonregular.child-network-attempts", [], refusal["network_attempts"])
        check("wiki.nonregular.explicit-refusal", True, bool(refusal["output"].strip()))
        check("wiki.nonregular.unsuccessful", True, refusal["exit_code"] != 0)
        check("wiki.nonregular.still-fifo", True, stat.S_ISFIFO(fifo.lstat().st_mode))
        check("wiki.nonregular.current-unchanged", True, page.read_bytes() == first_bytes)
        check(
            "wiki.nonregular.queue-unchanged",
            True,
            (control / "review_queue.db").read_bytes() == queue_before_fifo,
        )
        check(
            "wiki.nonregular.original-source-unchanged",
            True,
            (control / "sources/source-a.md").read_bytes() == source_a.read_bytes(),
        )
        check(
            "wiki.nonregular.no-source-record", False, (control / "sources/nonregular.md").exists()
        )

        command("add", str(source_b), "--id", "source-b", "--uri", "urn:example:operations")
        update_id = propose(conflict_body, generated=True, source="source-b")
        update, update_digest = reviewed(update_id)
        check("wiki.update.before-exact-current", True, update["before"].encode() == first_bytes)
        check(
            "wiki.update.retained-and-selected-provenance",
            ["source-a", "source-b"],
            sorted(s["id"] for s in update["sources"]),
        )
        context = "\n".join(model_calls[1]["messages"])
        for name, data in (
            ("retained-source", source_a.read_bytes()),
            ("selected-source", source_b.read_bytes()),
            ("current-page", first_bytes),
        ):
            check("wiki.update.generator-has-complete-" + name, True, data.decode() in context)
        check("wiki.update.pending-leaves-current-exact", True, page.read_bytes() == first_bytes)
        command("apply", update_id, "--expect", update_digest)
        accepted = update["after"].encode()
        check("wiki.update.accepted-exact-reviewed-candidate", True, page.read_bytes() == accepted)
        check("wiki.update.conflict-body-preserved", True, conflict_body.strip() in update["after"])
        check(
            "wiki.update.only-one-current-page",
            ["report-cache.md"],
            sorted(p.name for p in page.parent.glob("*.md")),
        )
        check(
            "wiki.update.complete-prior-page-preserved",
            True,
            (control / "history/report-cache" / f"{update_id}.md").read_bytes() == first_bytes,
        )
        for identifier, original in (("source-a", source_a), ("source-b", source_b)):
            check(
                "wiki.update.original-" + identifier,
                True,
                original.read_bytes() == (control / "sources" / f"{identifier}.md").read_bytes(),
            )

        original_history = history()
        metadata = (control / "sources/source-a.json").read_bytes()
        command("add", str(source_a), "--id", "source-a")
        command("apply", update_id, "--expect", update_digest)
        check("wiki.noop.current-unchanged", True, page.read_bytes() == accepted)
        check("wiki.noop.history-unchanged", original_history, history())
        check(
            "wiki.noop.source-metadata-unchanged",
            True,
            metadata == (control / "sources/source-a.json").read_bytes(),
        )
        command("apply", update_id, "--expect", "0" * 64, ok=False)
        check("wiki.wrong-digest.current-unchanged", True, page.read_bytes() == accepted)

        # Every negative uses a different payload so no failed proposal is silently retried.
        stale_id = propose(
            conflict_body + "\nReviewer note: stale-base check.\n", source="source-b"
        )
        _, stale_digest = reviewed(stale_id)
        edited = accepted + b"\nManual edit that must survive stale rejection.\n"
        page.write_bytes(edited)
        command("apply", stale_id, "--expect", stale_digest, ok=False)
        check("wiki.stale.edit-preserved", True, page.read_bytes() == edited)
        page.write_bytes(accepted)
        command("apply", stale_id, "--expect", stale_digest, ok=False)
        check("wiki.stale.failed-not-retried", True, page.read_bytes() == accepted)

        for target in ("source-bytes", "source-metadata"):
            proposal_id = propose(
                conflict_body + f"\nReviewer note: {target}.\n", source="source-b"
            )
            _, expected = reviewed(proposal_id)
            path = control / (
                "sources/source-b.md" if target == "source-bytes" else "sources/source-b.json"
            )
            original = path.read_bytes()
            if target == "source-bytes":
                path.write_bytes(original + b"\nchanged after review\n")
            else:
                changed = json.loads(original)
                changed["uri"] = "urn:example:tampered"
                path.write_text(json.dumps(changed), encoding="utf-8")
            command("apply", proposal_id, "--expect", expected, ok=False)
            check("wiki.tamper." + target + ".unchanged-page", True, page.read_bytes() == accepted)
            path.write_bytes(original)

        tampered_id = propose(
            conflict_body + "\nReviewer note: payload identity.\n", source="source-b"
        )
        payload, _ = reviewed(tampered_id)
        payload["after"] += "\nUnreviewed persisted payload replacement.\n"
        payload["after_sha256"] = digest(payload["after"].encode())
        with sqlite3.connect(control / "review_queue.db") as db:
            db.execute(
                "UPDATE review_proposals SET payload = ? WHERE proposal_id = ?",
                (json.dumps(payload), tampered_id),
            )
        command("apply", tampered_id, "--expect", proposal_digest(payload), ok=False)
        check(
            "wiki.tamper.recomputed-digest-does-not-reidentify-proposal",
            True,
            page.read_bytes() == accepted,
        )

        failure_secret = "PRIVATE_PROVIDER_EXCEPTION_BODY"
        responses.append(RuntimeError(failure_secret))
        failed = command(
            "propose", "--page", "wiki/report-cache.md", "--source", "source-b", ok=False
        )
        check("wiki.provider.error-body-not-disclosed", False, failure_secret in failed.output)
        check("wiki.provider.no-fallback-page", True, page.read_bytes() == accepted)
        check("wiki.provider.single-call-no-retry", 3, len(model_calls))
        check(
            "wiki.factory.configured-values",
            [("ollama", "fixture-wiki-model", 1777)] * 3,
            factory_calls,
        )

        invalid = root / "invalid.md"
        for target in ("../outside.md", "wiki/nested/page.md", "_wiki/history/old.md"):
            invalid.write_text(initial_body, encoding="utf-8")
            command(
                "propose",
                "--page",
                target,
                "--source",
                "source-a",
                "--draft-file",
                str(invalid),
                ok=False,
            )
        for link in (
            "[[_wiki/sources/missing]]",
            "[[../outside]]",
            "[[_wiki/history/report-cache/old]]",
            "[bad](../../outside.md)",
        ):
            invalid.write_text(conflict_body + "\n" + link + "\n", encoding="utf-8")
            command(
                "propose",
                "--page",
                "wiki/report-cache.md",
                "--source",
                "source-b",
                "--draft-file",
                str(invalid),
                ok=False,
            )
        command("add", str(source_a), "--id", "../escape", ok=False)
        check(
            "wiki.invalid-references-and-paths.current-unchanged",
            True,
            page.read_bytes() == accepted,
        )

        # Retention refusal must not persist the marker/body in source, queue or logs.
        rejected_body = "PRIVATE_DO_NOT_RETAIN_CONTENT"
        retained_files = {
            str(p.relative_to(vault)): p.read_bytes() for p in vault.rglob("*") if p.is_file()
        }
        for marker in (
            "do_not_retain: true",
            "retention: do-not-retain",
            'do_not_retain: "true"',
            "retention: [do-not-retain]",
        ):
            invalid.write_text(f"---\n{marker}\n---\n{rejected_body}", encoding="utf-8")
            refused = command("add", str(invalid), "--id", "do-not-retain", ok=False)
            check(
                "wiki.retention.no-source",
                False,
                (control / "sources/do-not-retain.md").exists(),
            )
            check("wiki.retention.no-body-output", False, rejected_body in refused.output)
        check(
            "wiki.retention.no-body-persisted",
            False,
            any(rejected_body.encode() in p.read_bytes() for p in vault.rglob("*") if p.is_file()),
        )
        check(
            "wiki.retention.accepted-state-unchanged",
            True,
            all((vault / name).read_bytes() == data for name, data in retained_files.items()),
        )

        # A symlink swap after review must fail before touching its external target.
        symlink_id = propose(
            conflict_body + "\nReviewer note: output symlink.\n", source="source-b"
        )
        _, symlink_digest = reviewed(symlink_id)
        outside = root / "outside.md"
        outside.write_bytes(b"external sentinel\n")
        page.unlink()
        page.symlink_to(outside)
        command("apply", symlink_id, "--expect", symlink_digest, ok=False)
        check(
            "wiki.symlink.external-target-unchanged",
            True,
            outside.read_bytes() == b"external sentinel\n",
        )
        page.unlink()
        page.write_bytes(accepted)

        # Simulate process loss after the real replacement, before the completion receipt.
        interrupted_id = propose(
            conflict_body + "\nReviewer note: interrupted operation.\n", source="source-b"
        )
        interrupted, interrupted_digest = reviewed(interrupted_id)
        replace = WikiWorkspace._replace_page

        def interrupted_replace(self: Any, target: Path, content: bytes) -> None:
            replace(self, target, content)
            raise RuntimeError("synthetic interruption after replacement")

        with patch.object(WikiWorkspace, "_replace_page", interrupted_replace):
            command("apply", interrupted_id, "--expect", interrupted_digest, ok=False)
        actual_after_interruption = page.read_bytes()
        check(
            "wiki.interruption.replacement-really-happened",
            True,
            actual_after_interruption == interrupted["after"].encode(),
        )
        state = json.loads((control / "operations" / f"{interrupted_id}.json").read_text())
        check("wiki.interruption.prepared-not-completed", "prepared", state["state"])
        check(
            "wiki.interruption.queue-not-applied",
            False,
            queued(interrupted_id)["status"] == "applied",
        )
        reopened_review = command("review", interrupted_id).output.lower()
        check("wiki.interruption.reopen-reports-incomplete", True, "incomplete" in reopened_review)
        command("apply", interrupted_id, "--expect", interrupted_digest, ok=False)
        check(
            "wiki.interruption.no-blind-replay",
            True,
            page.read_bytes() == actual_after_interruption,
        )
        check(
            "wiki.interruption.original-prior-page-preserved",
            True,
            (control / "history/report-cache" / f"{interrupted_id}.md").read_bytes() == accepted,
        )

    from vaultmind.config import load_settings
    from vaultmind.knowledge.intake import SourceIntake
    from vaultmind.knowledge.sources import search_sources
    from vaultmind.wiki.workspace import WikiError

    settings = load_settings(config)
    capture_bytes = b"# Intake evidence\r\n\r\norioncapture exact submitted text\r\n"
    capture_args = {
        "original_name": "submitted.txt",
        "origin": "urn:acceptance:submitted",
        "adapter": "acceptance",
        "event_id": "event-1",
        "revision": "revision-1",
    }
    effects: list[str] = []
    with SourceIntake(settings) as intake:
        captured = intake.preserve_text(capture_bytes, **capture_args)
        record = intake.read(captured.source_id)
        check("wiki.capture.exact-original", True, capture_bytes == record.content)
        check(
            "wiki.capture.no-implicit-index-or-approval",
            [True, False, False],
            [captured.preserved, captured.indexed, captured.approved],
        )
        check("wiki.capture.actual-origin", capture_args["origin"], record.provenance["origin"])
        check("wiki.capture.original-retention", True, record.provenance["original_retained"])
        projection = intake.project_text(captured, settings.vault.inbox_folder)
        check("wiki.capture.projection-succeeded", "succeeded", projection.status)
        check(
            "wiki.capture.projection-exact-text",
            True,
            capture_bytes == (vault / projection.value).read_bytes(),
        )

        def fail_index() -> None:
            effects.append("index")
            raise RuntimeError("PRIVATE-DOWNSTREAM-FAILURE-BODY")

        failed = captured.run_stage("index", fail_index)
        check("wiki.capture.partial-index-failed", "failed", failed.status)
        check(
            "wiki.capture.partial-still-preserved",
            True,
            capture_bytes == intake.read(captured.source_id).content,
        )
        check(
            "wiki.capture.partial-error-redacted",
            False,
            "PRIVATE-DOWNSTREAM-FAILURE-BODY" in json.dumps(failed.as_dict()),
        )
        replay = intake.preserve_text(capture_bytes, **capture_args)
        replay_index = replay.run_stage("index", fail_index)
        check(
            "wiki.capture.same-event-replay",
            [captured.source_id, "unchanged", "failed", ["index"]],
            [replay.source_id, replay.status, replay_index.status, effects],
        )
        changed = intake.preserve_text(capture_bytes + b"new bytes\n", **capture_args)
        distinct = intake.preserve_text(
            capture_bytes,
            **{**capture_args, "event_id": "event-2"},
        )
        check(
            "wiki.capture.changed-and-distinct-events-immutable",
            3,
            len({captured.source_id, changed.source_id, distinct.source_id}),
        )
        parent = intake.preserve_attachment(
            b"\x89PNG\r\nsynthetic-original",
            original_name="original.png",
            origin="urn:acceptance:photo",
            adapter="acceptance",
            media_type="image/png",
        )
        child = intake.preserve_text(
            "orioncapture derived description",
            original_name="description.md",
            origin="urn:acceptance:vision",
            adapter="acceptance",
            role="description",
            attachments=(parent.source_id,),
        )
        check(
            "wiki.capture.derivative-attachment-closure",
            [parent.source_id],
            [dependency.source_id for dependency in intake.read(child.source_id).dependencies],
        )
        with SourceIntake(
            settings,
            allows_path=lambda path: not path.startswith(f"_wiki/sources/{parent.source_id}"),
        ) as restricted_intake:
            try:
                restricted_intake.read(child.source_id)
            except WikiError:
                closure_denied = True
            else:
                closure_denied = False
        check("wiki.capture.unauthorized-parent-closure-denied", True, closure_denied)
        source_path = vault / parent.path
        source_path.write_bytes(b"tampered attachment")
        try:
            intake.read(child.source_id)
        except WikiError:
            tampered_refused = True
        else:
            tampered_refused = False
        check("wiki.capture.tampered-parent-refused", True, tampered_refused)
        before_retention = {
            path.name: digest(path.read_bytes())
            for path in (control / "sources").iterdir()
            if path.is_file()
        }
        try:
            intake.preserve_text(
                "---\ndo_not_retain: true\n---\nPRIVATE-INTAKE-RETENTION-BODY",
                original_name="private.md",
                origin="urn:acceptance:private",
                adapter="acceptance",
            )
        except WikiError:
            retention_refused = True
        else:
            retention_refused = False
        check("wiki.capture.retention-refused", True, retention_refused)
        check(
            "wiki.capture.retention-no-persistence",
            before_retention,
            {
                path.name: digest(path.read_bytes())
                for path in (control / "sources").iterdir()
                if path.is_file()
            },
        )
    current_note = vault / "ordinary.md"
    current_note.write_text("orioncapture current ordinary body\n", encoding="utf-8")
    old_index = BoundaryStore(
        [
            {
                "chunk_id": "ordinary.md::0",
                "content": "STALE-INDEX-BODY",
                "metadata": {"note_path": "ordinary.md"},
                "distance": 0.0,
            }
        ]
    )
    hits = search_sources(settings, "orioncapture", store=old_index, n_results=100)
    check(
        "wiki.source-search.current-and-preserved",
        ["ordinary", "preserved"],
        sorted({hit["source_kind"] for hit in hits}),
    )
    check(
        "wiki.source-search.no-control-history-wiki-or-stale-excerpts",
        True,
        all(
            hit["approved"] is False
            and hit["status"] == "source-only"
            and "STALE-INDEX-BODY" not in hit["content"]
            and not hit["path"].startswith(("wiki/", "_wiki/history/", "_wiki/operations/"))
            and hit["distance"] is None
            for hit in hits
        ),
    )
    check(
        "wiki.source-search.tampered-closure-not-discoverable",
        False,
        any(hit.get("source_id") == child.source_id for hit in hits),
    )
    ordinary_hit = next(hit for hit in hits if hit["path"] == "ordinary.md")
    check(
        "wiki.source-search.current-digest",
        digest(current_note.read_bytes()),
        ordinary_hit["current_digest"],
    )
    cli_search = runner.invoke(
        app.cli, ["--config", str(config), "search-sources", "orioncapture", "--limit", "100"]
    )
    check("wiki.source-cli.exit", 0, cli_search.exit_code)
    check(
        "wiki.source-cli.shared-current-policy",
        search_sources(settings, "orioncapture", n_results=100),
        json.loads(cli_search.output)["results"],
    )
    capture_draft = root / "capture-draft.md"
    capture_draft.write_text(
        "# Captured evidence\n\n## Current understanding\n\n"
        f"Submitted orioncapture evidence. [[_wiki/sources/{captured.source_id}]]\n",
        encoding="utf-8",
    )
    pending_capture = command(
        "propose",
        "--page",
        "wiki/captured.md",
        "--source",
        captured.source_id,
        "--draft-file",
        str(capture_draft),
    )
    check(
        "wiki.capture.explicit-source-curation-input", True, "Proposal:" in pending_capture.output
    )
    check("wiki.capture.proposal-is-not-approval", False, (vault / "wiki/captured.md").exists())
    # These real forbidden namespace bodies match the query; absence must come
    # from policy, not an accidentally irrelevant fixture.
    check(
        "wiki.source-search.current-wiki-relevance-control", True, b"Retention" in page.read_bytes()
    )
    namespace_hits = search_sources(settings, "Retention", n_results=100)
    check(
        "wiki.source-search.relevant-wiki-and-history-still-isolated",
        True,
        all(
            not hit["path"].startswith(
                ("wiki/", "_wiki/history/", "_wiki/pending/", "_wiki/operations/")
            )
            for hit in namespace_hits
        ),
    )
    pending_hits = search_sources(settings, "orioncapture", n_results=100)
    check(
        "wiki.source-search.pending-proposal-not-a-source",
        True,
        all(
            not hit["path"].startswith(("wiki/", "_wiki/pending/", "_wiki/operations/"))
            for hit in pending_hits
        ),
    )

    check("wiki.no-index-side-effects", False, forbidden.exists())
    check("wiki.config-not-edited", True, config.read_bytes() == config_before)
    return {
        "fixture": (
            "synthetic CRLF/local UTF8 sources, explicit excluded wiki/_wiki, private HOME/cwd"
        ),
        "consumer": (
            "actual Click wiki commands -> configured client factory "
            "-> real generator/queue/applier"
        ),
        "model_calls": len(model_calls),
        "limits": (
            "No live provider, embedding, private vault, editor, watcher "
            "or installed-wheel qualification."
        ),
    }


async def wiki_query_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Observe public read-only consumers against real accepted synthetic pages."""
    from unittest.mock import patch

    from click.testing import CliRunner

    import vaultmind.cli as app
    from vaultmind.llm.client import LLMResponse

    check = make_checker(checks)
    vault = home / "vault"
    vault.mkdir()
    config = home / "query.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = ["wiki", "_wiki"]\n'
        '[llm]\nprovider = "ollama"\nthinking_model = "query-offline"\nmax_tokens = 1733\n',
        encoding="utf-8",
    )
    source = home / "source.md"
    source.write_text("# Policy\n\n19 days versus 7 days; unresolved.\n", encoding="utf-8")
    page = "wiki/policy.md"
    late = "Late verified retention is 19 days."
    draft = home / "draft.md"
    calls: list[dict[str, Any]] = []
    response: dict[str, Any] = {}

    class QueryLLM:
        provider_name = "scripted-offline-query"

        def complete(
            self,
            messages: list[Message],
            model: str,
            max_tokens: int = 4096,
            system: str | None = None,
        ) -> LLMResponse:
            calls.append(
                {
                    "prompt": json.loads(messages[0].content),
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                }
            )
            return LLMResponse(text=json.dumps(response), model=model, usage={})

        def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
            raise AssertionError("Unexpected multimodal completion")

    def accept(body: str, target: str = page, source_id: str = "policy") -> None:
        draft.write_text(body, encoding="utf-8")
        proposed = command(
            "propose", "--page", target, "--source", source_id, "--draft-file", str(draft)
        )
        identity = re.search(r"^Proposal: ([0-9a-f]{64})$", proposed.output, re.M)
        if identity is None:
            raise AssertionError("CLI did not return an unambiguous proposal identity")
        proposal_id = identity.group(1)
        reviewed = command("review", proposal_id)
        digest = re.search(r"^SHA256: ([0-9a-f]{64})$", reviewed.output, re.M)
        if digest is None:
            raise AssertionError("CLI did not return an unambiguous reviewed digest")
        command("apply", proposal_id, "--expect", digest.group(1))

    def files() -> dict[str, bytes]:
        return {
            str(path.relative_to(vault)): path.read_bytes()
            for path in vault.rglob("*")
            if path.is_file()
        }

    runner = CliRunner()

    def command(*args: str, ok: bool = True) -> Any:
        result = runner.invoke(app.cli, ["--config", str(config), "wiki", *args])
        check("wiki-query.cli." + args[0] + ".exit", ok, result.exit_code == 0)
        if not ok:
            check("wiki-query.explicit-error", True, bool(result.output.strip()))
        return result

    def query(*, evidence: bool = False, ok: bool = True) -> Any:
        return command(
            "query",
            "What is the retention?",
            "--page",
            page,
            *(["--evidence-only"] if evidence else []),
            ok=ok,
        )

    def ask(
        question: str = "What is the retention?", *, evidence: bool = False, ok: bool = True
    ) -> Any:
        result = runner.invoke(
            app.cli,
            ["--config", str(config), "ask", question, *(["--evidence-only"] if evidence else [])],
        )
        check("wiki-query.cli.ask.exit", ok, result.exit_code == 0)
        if not ok:
            check("wiki-query.ask.explicit-error", True, bool(result.output.strip()))
        return result

    with patch.object(app, "_create_llm_client", side_effect=AssertionError("Unexpected model")):
        empty_before = files()
        empty = json.loads(ask().output)
        check("wiki-query.empty.status", "unsupported", empty["status"])
        check("wiki-query.empty.reason", "no_eligible_pages", empty["reason"])
        check("wiki-query.empty.readonly", True, empty_before == files())

    command("add", str(source), "--id", "policy", "--uri", "urn:offline:policy")
    accept(
        "# Policy\n\n## Current understanding\n\n"
        + "Background. " * 60
        + late
        + " [[_wiki/sources/policy]]\n"
    )
    gardening_source = home / "gardening.md"
    gardening_source.write_text("# Gardening\n\nWater seedlings weekly.\n", encoding="utf-8")
    command("add", str(gardening_source), "--id", "gardening", "--uri", "urn:offline:garden")
    gardening_page = "wiki/gardening.md"
    accept(
        "# Gardening\n\n## Current understanding\n\n"
        "Water seedlings weekly. [[_wiki/sources/gardening]]\n",
        gardening_page,
        "gardening",
    )
    before = files()
    with patch.object(app, "_create_llm_client", side_effect=AssertionError("Unexpected model")):
        listed = json.loads(command("list").output)
        check(
            "wiki-query.list.current-only",
            [gardening_page, page],
            [row["path"] for row in listed],
        )
        check(
            "wiki-query.lint.valid",
            ["valid", "valid"],
            [row["status"] for row in json.loads(command("lint").output)],
        )
        evidence = json.loads(query(evidence=True).output)
        automatic = json.loads(ask(evidence=True).output)
        for field in ("evidence", "citations", "selected_pages", "sources", "unresolved_conflicts"):
            check("wiki-query.automatic-explicit." + field, evidence[field], automatic[field])
        check("wiki-query.automatic.selected", [page], automatic["selection"]["selected_pages"])
        check("wiki-query.automatic.mode", "automatic", automatic["selection"]["mode"])
        no_match = json.loads(ask("xylophone zeppelin").output)
        check("wiki-query.no-match.status", "unsupported", no_match["status"])
        check("wiki-query.no-match.reason", "no_lexical_support", no_match["reason"])
        ask("q" * 4097, ok=False)
        invalid = vault / gardening_page
        accepted_gardening = invalid.read_bytes()
        invalid.write_bytes(accepted_gardening + b"\nUnreviewed unrelated edit.\n")
        try:
            ask(evidence=True, ok=False)
        finally:
            invalid.write_bytes(accepted_gardening)
    late_row = next(row for row in evidence["evidence"] if late in row["text"])
    check("wiki-query.full-late-evidence", True, late_row["text"].index(late) > 500)
    check("wiki-query.source-provenance", "policy", late_row["sources"][0]["id"])
    citation = late_row["id"]
    with patch.object(app, "_create_llm_client", return_value=QueryLLM()):
        for status in ("supported", "unsupported"):
            response = {
                "status": status,
                "answer": "19 days." if status == "supported" else "Unknown.",
                "citations": [citation] if status == "supported" else [],
            }
            result = json.loads(query().output)
            check("wiki-query.generated." + status, status, result["status"])
            automatic_result = json.loads(ask().output)
            check(
                "wiki-query.automatic-generated." + status,
                result["status"],
                automatic_result["status"],
            )
            check(
                "wiki-query.generated-citation-parity",
                result["citations"],
                automatic_result["citations"],
            )
        check("wiki-query.configured-model", "query-offline", calls[-1]["model"])
        check("wiki-query.configured-tokens", 1733, calls[-1]["max_tokens"])
        check(
            "wiki-query.prompt-full-evidence",
            True,
            any(late in row["text"] for row in calls[-1]["prompt"]["evidence"]),
        )
        response = {"status": "supported", "answer": "Invented.", "citations": ["invented"]}
        query(ok=False)
    check("wiki-query.readonly", True, before == files())
    preserved = vault / "_wiki/sources/policy.md"
    original = preserved.read_bytes()
    preserved.unlink()
    command("lint", ok=False)
    query(evidence=True, ok=False)
    preserved.write_bytes(original)
    check("wiki-query.restored", "evidence", json.loads(query(evidence=True).output)["status"])
    accept(
        "# Policy\n\n## Current understanding\n\nRetention remains unresolved.\n\n"
        "## Conflicting information\n\nRetention is 19 versus 7 days. "
        "[[_wiki/sources/policy]]\n"
    )
    conflict = json.loads(query(evidence=True).output)
    check("wiki-query.conflict-page", [page], conflict["unresolved_conflicts"])
    check(
        "wiki-query.lint.unresolved",
        ["valid", "unresolved"],
        [row["status"] for row in json.loads(command("lint").output)],
    )
    ids = [row["id"] for row in conflict["evidence"] if "## Conflicting information" in row["text"]]
    conflict_before = files()
    with patch.object(app, "_create_llm_client", return_value=QueryLLM()):
        response = {"status": "conflicting", "answer": "19 and 7 days conflict.", "citations": ids}
        check("wiki-query.conflict-attribution", ids, json.loads(query().output)["citations"])
        check("wiki-query.automatic-conflict", ids, json.loads(ask().output)["citations"])
        response = {"status": "supported", "answer": "19 days wins.", "citations": ids}
        query(ok=False)
        ask(ok=False)
    check("wiki-query.conflict-readonly", True, conflict_before == files())
    for index in range(8):
        accept(
            f"# Retention alternative {index}\n\n## Current understanding\n\n"
            "Retention remains unresolved. [[_wiki/sources/policy]]\n",
            f"wiki/retention-{index}.md",
        )
    bounded_before = files()
    with patch.object(app, "_create_llm_client", side_effect=AssertionError("Unexpected model")):
        bounded = json.loads(ask(evidence=True).output)
    check("wiki-query.bounded.count", 8, len(bounded["selection"]["selected_pages"]))
    check("wiki-query.bounded.disclosed", True, bounded["selection"]["limited"])
    check("wiki-query.lexical-limit.disclosed", True, bool(bounded["selection"]["limitation"]))
    check("wiki-query.bounded-readonly", True, bounded_before == files())
    return {
        "consumer": "real Click add/propose/review/apply/list/ask/query/lint, configured client",
        "model_calls": len(calls),
        "observed_statuses": ["evidence", "supported", "unsupported", "conflicting"],
        "synthetic_limits": (
            "Deterministic lexical discovery only; scripted generator, not model quality. "
            "No live provider, embeddings, Telegram, MCP cutover or private-vault evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=["retrieval", "mcp", "approval", "config", "wiki", "wiki-query"],
    )
    parser.add_argument("--offline", required=True, action="store_true")
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    receipt_path = args.receipt_out.resolve() if args.receipt_out else None
    receipt: dict[str, Any] = {
        "case": args.case,
        "command": sys.argv,
        "python": sys.version,
        "checks": [],
        "transport": "in-process real Chroma/BM25/handlers",
        "boundary_substitutions": [
            "deterministic token embeddings",
            "scripted llm.client",
            "fake Telegram delivery",
            "scripted exact-floor raw hits",
            "direct BM25 fixture insert: lexical::0 (not parser-indexed or vector-indexed)",
        ],
        "limitations": "No hosted embedding quality, model quality or Telegram delivery evidence.",
    }
    if args.case == "config":
        receipt.update(
            transport=(
                "in-process real Click index, semantic recall, graph persistence and Click "
                "mcp-serve assembly with real SDK registered call handlers"
            ),
            boundary_substitutions=[
                "deterministic token embeddings at indexer.Embedder and query embedder",
                "recording FakeMessage instead of Telegram delivery",
                "unused ScriptedLLM dependency of ThinkingPartner; no LLM invocation",
                "MCP terminal stdio/run boundary only; real settings, policy and persistence",
            ],
            limitations=(
                "Temporary fixture only; no hosted embedding quality, native model, Telegram, "
                "or MCP transport qualification. Graph persistence is exercised directly, "
                "not entity extraction (which requires a model)."
            ),
        )
    if args.case == "mcp":
        receipt.update(
            transport=(
                "subprocess: sibling `vaultmind` console script via runpy in isolated Python "
                "-> installed MCP SDK stdio server; client is the installed SDK ClientSession "
                "over stdio_client"
            ),
            boundary_substitutions=[
                "temporary HOME/vault/Chroma/graph/SQLite/audit paths and explicit fixture TOML",
                "empty OpenAI API key for Embedder construction; no hosted provider use",
                "root and derived fixtures: deterministic two-dimensional vectors at "
                "Embedder._embed_texts_uncached; real persisted Chroma query/filter/ranking",
                "read-only Python profiling of real dispatch/producers and in-memory buffers",
                "launcher-side network guard (audit hook) with a blocked-lookup self-test",
            ],
            limitations=(
                "No hosted embedding, model, watcher or Telegram evidence; synthetic "
                "root-note, traversal, adversarial derived graph and SQLite fixtures, not "
                "a private vault. Optional producers use real CLI-configured backing stores; "
                "restricted denial controls are compared with substantive full-profile output."
            ),
            network_instrumentation={
                "parent": "sys.addaudithook: socket.getaddrinfo, non-AF_UNIX socket.connect",
                "child": "python -I launcher audit hook: getaddrinfo/gethostbyname/"
                "gethostbyaddr, non-AF_UNIX connect/sendto; expected zero non-probe attempts",
            },
        )
    if args.case == "approval":
        receipt.update(
            transport=(
                "in-process: real click CLI (auto-tag), real ReviewQueue/appliers, real "
                "CommandHandlers behind the router registered by bot.telegram.register_handlers; "
                "real aiogram Update/Dispatcher/Notifier over a recording fake BaseSession"
            ),
            boundary_substitutions=[
                "scripted llm.client at cli._create_llm_client and handler/detector/pipeline "
                "clients (provider boundary only)",
                "deterministic token embeddings (no hosted embedding)",
                "recording aiogram BaseSession: no Telegram API call is made",
                "local RSS file as the connector fetch target (no network)",
                "temporary HOME/VAULTMIND_HOME and explicit fixture TOML per world",
            ],
            limitations="Fixture-only offline proof; not native model/provider/Telegram "
            "qualification; source finalization (M15) not claimed.",
        )
    if args.case == "wiki":
        receipt.update(
            transport="in-process real Click wiki add/propose/review/apply and persisted queue",
            boundary_substitutions=[
                "scripted LLM at configured cli._create_llm_client only",
                "synthetic sources, isolated HOME/cwd and explicit fixture TOML",
                "interruption injected at page replacement after real replacement",
            ],
            limitations=(
                "Manual lifecycle engineering only; no live provider, embeddings, private vault, "
                "editor, watcher, installed artifact or OS-wide concurrency proof."
            ),
        )
    attempts: list[str] = []
    if args.case == "wiki-query":
        receipt.update(
            transport=(
                "in-process real Click wiki list/query/lint and approved current-page validation"
            ),
            boundary_substitutions=[
                "scripted LLM at configured cli._create_llm_client only",
                "synthetic accepted pages/sources, isolated HOME/cwd and explicit TOML",
            ],
            limitations=(
                "No live provider/model quality, embeddings, private vault "
                "or installed artifact proof."
            ),
        )

    def forbid_network(event: str, arguments: tuple[Any, ...]) -> None:
        if event == "socket.getaddrinfo" or (
            event == "socket.connect" and arguments[0].family != socket.AF_UNIX
        ):
            attempts.append(event)
            raise RuntimeError("Network forbidden in offline acceptance")

    sys.addaudithook(forbid_network)
    old_cwd = Path.cwd()
    try:
        prefix = {
            "retrieval": "vm-m11-retrieval-",
            "mcp": "vm-m12-mcp-",
            "approval": "vm-m13-approval-",
            "config": "vm-m21-config-",
            "wiki": "vm-m31-wiki-",
            "wiki-query": "vm-m32-wiki-query-",
        }[args.case]
        with (
            tempfile.TemporaryDirectory(prefix=prefix) as directory,
            offline_environment_scope(offline_environment(Path(directory))),
        ):
            home = Path(directory)
            os.chdir(home)
            receipt["artifact"] = artifact_identity(home)
            if not receipt["artifact"]["runtime_matches_reviewed"]:
                raise ValueError("Imported VaultMind code does not match the reviewed source")
            if args.case == "retrieval":
                case = retrieval(home, receipt["checks"])
            elif args.case == "approval":
                case = approval_case(home, receipt["checks"])
            elif args.case == "config":
                case = config_case(home, receipt["checks"])
            elif args.case == "wiki":
                case = wiki_case(home, receipt["checks"])
            elif args.case == "wiki-query":
                case = wiki_query_case(home, receipt["checks"])
            else:
                case = mcp_case(home, receipt["checks"])
            receipt.update(asyncio.run(case))
            if attempts:
                raise RuntimeError(f"Unexpected network attempts: {attempts}")
        receipt["result"] = "pass"
    except Exception as exc:
        receipt.update(result="fail", failure=f"{type(exc).__name__}: {exc}")
        receipt.update(getattr(exc, "receipt", {}))
        raise
    finally:
        os.chdir(old_cwd)
        receipt["network_attempts"] = attempts
        text = json.dumps(receipt, indent=2)
        if receipt_path:
            receipt_path.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
