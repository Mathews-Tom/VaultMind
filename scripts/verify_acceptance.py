#!/usr/bin/env python3
"""Offline acceptance: M11 retrieval, M12 MCP stdio, M13 approval, M14 lineage; labeled seams."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
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


def import_binding() -> dict[str, Any]:
    """Observe where `vaultmind` is imported from and whether it equals the tracked source."""
    import importlib.util

    spec = importlib.util.find_spec("vaultmind")
    if spec is None or spec.origin is None:
        raise RuntimeError("vaultmind is not importable")
    package = Path(spec.origin).resolve().parent
    source = (ROOT / "src" / "vaultmind").resolve()
    mismatched = sorted(
        str(path.relative_to(source))
        for path in source.rglob("*.py")
        if not (package / path.relative_to(source)).is_file()
        or digest((package / path.relative_to(source)).read_bytes()) != digest(path.read_bytes())
    )
    extra = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if not (source / path.relative_to(package)).is_file()
    )
    return {
        "origin": str(package / "__init__.py"),
        "editable_source": package == source,
        "mismatched_files": mismatched,
        "extra_installed_files": extra,
        "identical_to_tracked_source": not mismatched and not extra,
    }


def artifact_identity() -> dict[str, Any]:
    files = sorted((ROOT / "src" / "vaultmind").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).splitlines()
    return {
        "plan_id": "VM-REMEDIATION-2026-10-11-R1",
        "revision": "R1-A+R1-B",
        "head": head,
        "tree": tree,
        "import_binding": import_binding(),
        "working_tree_dirty": bool(changes),
        "working_tree_changes": changes,
        "code_sha256": digest(json.dumps(hashes, sort_keys=True).encode()),
        "file_sha256": hashes,
        "dependency_sha256": {
            name: digest((ROOT / name).read_bytes()) for name in ("pyproject.toml", "uv.lock")
        },
        "contract_sha256": {
            name: digest((ROOT / ".docs" / name).read_bytes())
            for name in ("DEVELOPMENT_PLAN.md", "EXECUTION_PROMPTS.md")
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
        f'[chroma]\npersist_dir = "{root / "data/chroma"}"\n'
        f'[graph]\npersist_path = "{root / "data/graph.json"}"\n'
        f"[search]\nhybrid_enabled = {str(hybrid).lower()}\nmax_results = 10\npage_size = 10\n"
        f'bm25_db_path = "{root / "data/bm25.db"}"\ncache_enabled = false\n'
        f"[bench]\nk = 10\nscore_floor = {FLOOR}\n"
        f'trend_path = "{root / "data/trend.jsonl"}"\n'
        f'[gaps]\nenabled = true\ndb_path = "{root / "data/gaps.db"}"\n'
        "[embedding]\ncache_enabled = false\n[llm]\ngraph_context_enabled = false\n",
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
    from vaultmind.memory.gaps import GapStore
    from vaultmind.vault.parser import VaultParser

    def check(name: str, expected: Any, observed: Any) -> None:
        checks.append({"id": name, "expected": expected, "actual": observed})
        if expected != observed:
            raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")

    check(
        "installed-code-origin",
        True,
        import_binding()["identical_to_tracked_source"],
    )
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
    notes = parser.iter_notes(strict=True)
    store.index_notes(notes, parser)
    exact = parser.chunk_note(parser.parse_file(settings.vault.path / ALPHA))[0].content
    bm25.upsert("lexical::0", "lexical.md", "Lexical", "kestrelquartz field memo")
    corpus_hashes["lexical"] = digest(b"lexical::0 lexical.md Lexical kestrelquartz field memo")
    corpus_hashes["exact-query"] = digest(exact.encode())
    gaps: list[Any] = []

    def context(cfg: Any, target: Any) -> Any:
        gap = GapStore(Path(cfg.gaps.db_path), stale_after_days=cfg.gaps.stale_after_days)
        gaps.append(gap)
        llm = ScriptedLLM()
        thinking = ThinkingPartner(
            cfg.llm, cfg.telegram, llm_client=llm, gap_store=gap, score_floor=cfg.bench.score_floor
        )
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
            ctx.store,
            k=ctx.settings.search.max_results,
            hybrid_enabled=ctx.settings.search.hybrid_enabled,
            score_floor=ctx.settings.bench.score_floor,
            ranking_config=ctx.settings.ranking,
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
                ctx.store,
                query,
                ctx.settings.search.max_results,
                ctx.settings.search.hybrid_enabled,
                ctx.settings.ranking,
            ),
        )
        if expected == "unknown":
            check(label + ".unknown-rendered", True, "relevance unknown" in message.answers[-1])
        return hits

    try:
        for hybrid in (False, True):
            mode = "hybrid" if hybrid else "vector"
            cfg = fixture_config(home, mode, hybrid)
            ctx = context(cfg, store)
            hits = await observe(mode + ".self-match", ctx, exact, "strong")
            alpha = next(h for h in hits if h["metadata"]["note_path"] == ALPHA)
            check(mode + ".raw-self-distance", True, abs(alpha["distance"]) < 0.01)
            raw = run_query(store, exact, 10, hybrid, RankingConfig(enabled=False))
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
            await observe(mode + ".lexical", ctx, "kestrelquartz", "unknown" if hybrid else "weak")
            empty_cfg = fixture_config(home, mode + "-empty", hybrid)
            empty_bm25 = BM25Index(Path(empty_cfg.search.bm25_db_path))
            try:
                empty = VaultStore(empty_cfg.chroma, embedder, bm25=empty_bm25)
                await observe(mode + ".empty", context(empty_cfg, empty), "unanswered", "weak")
            finally:
                empty_bm25.close()

        cfg = fixture_config(home, "boundary", True)
        for name, distances, expected in BOUNDARIES + [("missing", [None], "unknown")]:
            hits = [
                {
                    "chunk_id": f"{i}::0",
                    "content": "boundary input",
                    "metadata": {"note_path": f"{i}.md"},
                    "rrf_score": 0.02,
                    **({"distance": distance} if name != "missing" else {}),
                }
                for i, distance in enumerate(distances)
            ]
            ctx = context(cfg, BoundaryStore(hits))
            await observe("boundary." + name, ctx, name, expected)
            before = {g.gap_id for g in ctx.gap_store.list_open()}
            await ctx.thinking.think(42, "thinking-" + name, ctx.store, ctx.graph)
            new = [g for g in ctx.gap_store.list_open() if g.gap_id not in before]
            check(
                "thinking." + name,
                ["unanswered_question"] if expected == "weak" else [],
                [str(g.kind) for g in new],
            )
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
# an empty OpenAI constructor key (never used: metadata tools make no embedding call) and a
# launcher-side network guard. No fake transport, no source injection, no PYTHONPATH.
# ---------------------------------------------------------------------------

MCP_SCENARIOS = (
    "planner-scoped",
    "planner-configured-override",
    "researcher-default-control",
    "full-control",
)
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
import json
import os
import socket
import sys
import threading

state = sys.argv[1]
cli_args = sys.argv[2:]
log = open(os.path.join(state, "launcher.jsonl"), "a", buffering=1, encoding="utf-8")


def emit(**fields):
    log.write(json.dumps(fields, sort_keys=True) + "\n")
    log.flush()


probing = [False]
HOST_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
PEER_EVENTS = {"socket.connect", "socket.sendto"}


def guard(event, args):
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

prefixes = ("VAULTMIND_", "PYTHONPATH", "OPENAI", "ANTHROPIC", "GEMINI", "VOYAGE", "TELEGRAM")
emit(
    kind="environment",
    home=os.environ.get("HOME"),
    forbidden=sorted(k for k in os.environ if k.upper().startswith(prefixes) or "API_KEY" in k),
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
cli = entry.load()
import vaultmind

dist = importlib.metadata.distribution("vaultmind")
emit(
    kind="entrypoint",
    name=entry.name,
    group=entry.group,
    value=entry.value,
    cli_type=type(cli).__module__ + "." + type(cli).__name__,
    dist_version=dist.version,
    direct_url=dist.read_text("direct_url.json"),
    module_file=os.path.realpath(vaultmind.__file__),
    executable=sys.executable,
    python=sys.version.split()[0],
    sys_path=sys.path,
)

code = 0
try:
    sys.argv = ["vaultmind", *cli_args]
    cli()
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
        f"[vault]\npath = {json.dumps(str(root / 'vault'))}\n",
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
    launcher: str = MCP_LAUNCHER,
) -> dict[str, Any]:
    """Spawn the registered CLI and drive initialize/list_tools/call_tool over SDK stdio."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client import stdio as sdk_stdio

    state = root / "launcher-state"
    state.mkdir()
    # COLUMNS keeps Rich from wrapping a caller argument across log lines, so the per-call
    # scan can mask exactly that call's own strings without any substring exemption.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-c", launcher, str(state), *cli_args],
        env={"HOME": str(root), "ANONYMIZED_TELEMETRY": "False", "COLUMNS": "2000"},
        cwd=str(root),
    )
    frames_in: list[dict[str, Any]] = []
    frames_out: list[dict[str, Any]] = []
    errors: list[str] = []
    results: list[Any] = []
    stderr_chunks: list[bytes] = []
    stderr_path = state / "stderr.txt"

    def stderr_since(offset: int) -> tuple[bytes, int]:
        data = stderr_path.read_bytes()
        return data[offset:], len(data)

    with stderr_path.open("w", encoding="utf-8") as errlog:
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
                    for tool, arguments in calls:
                        results.append(await session.call_tool(tool, arguments))
                        chunk, offset = stderr_since(offset)
                        stderr_chunks.append(chunk)
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
    }


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
        stderr_contains = [
            "Profile: planner (Read/write access for project planning)",
            "Folder scope: 02-projects, 00-inbox",
            "DEBUG",
        ]
        stderr_absent = ["defaulting to 'researcher'"]
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
        stderr_contains = [
            "Profile: planner (Fixture override)",
            "Folder scope: 00-inbox",
        ]
        stderr_absent = ["Read/write access for project planning"]
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
        if full:
            stderr_contains = [
                "Profile: full (Unrestricted access)",
                "Folder scope: *",
            ]
            stderr_absent = ["defaulting to 'researcher'"]
        else:
            stderr_contains = [
                "Warning: No --profile specified; defaulting to 'researcher' (read-only)",
                "Profile: researcher (Read-only access for research and Q&A tasks)",
                "Folder scope: *",
            ]
            stderr_absent = []
    return {
        "name": name,
        "profile": profile,
        "scoped": scoped,
        "override": override,
        "cli_args": cli_args,
        "calls": calls,
        "forbidden": forbidden,
        "stderr_contains": stderr_contains,
        "stderr_absent": stderr_absent,
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
    if tool == "follow_links":
        for key in ("out_links", "backlinks"):
            if isinstance(out.get(key), list):
                out[key] = sorted(
                    out[key], key=lambda link: (str(link.get("path")), str(link.get("title")))
                )
    return out


async def run_mcp_scenario(name: str, root: Path, check: Any) -> dict[str, Any]:
    """One actual subprocess lifecycle with per-call, per-frame and per-origin assertions."""
    from mcp.types import JSONRPCMessage

    spec = mcp_scenario(name, root)
    fixture = mcp_build_fixture(root, spec["scoped"], consumer=True)
    config = mcp_write_config(root, spec["override"])
    calls = spec["calls"]
    try:
        run = await mcp_stdio_session(
            root, spec["cli_args"], [(c["tool"], c["arguments"]) for c in calls]
        )
    finally:
        mcp_restore_permissions(fixture)
    state: Path = run["state"]
    events = [
        json.loads(line)
        for line in (state / "launcher.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    raw = (state / "stdout.raw").read_bytes().decode("utf-8")
    stderr = mcp_trim((state / "stderr.txt").read_text(encoding="utf-8"))
    stderr_startup = mcp_trim(run["stderr_startup"])
    stderr_chunks = [mcp_trim(chunk) for chunk in run["stderr_chunks"]]
    prefix = f"{name}."

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
            check(label + ".payload", mcp_normalize(tool, expected_call["expected"]), actual)
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
                "kind": kind,
                "actual": actual,
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
        expected_reason = (
            observed_calls[position]["actual"].get("error", "")
            if expected_result in ("DENIED", "ERROR")
            else ""
        )
        expected_rows.append((spec["profile"], c["tool"], expected_result, expected_reason))
    check(
        prefix + "audit.rows",
        expected_rows,
        [(row["profile"], row["tool"], row["result"], row.get("reason", "")) for row in audit],
    )
    # Independent per-row audit oracle: each row is bound to ITS call's arguments (never the
    # union of other calls') and scanned for unknown physical targets and path residue.
    for position, (c, row) in enumerate(zip(calls, audit, strict=True)):
        arguments = c["arguments"]
        label = f"{prefix}{position:02d}.{c['tool']}"
        reason = row.get("reason", "")
        check(
            label + ".audit.params-are-this-call",
            {key: str(value)[:200] for key, value in arguments.items()},
            row["params"],
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
    check(
        prefix + "stderr.diagnostics",
        [True] * len(spec["stderr_contains"]),
        [needle in stderr for needle in spec["stderr_contains"]],
    )
    check(
        prefix + "stderr.absent",
        [],
        [needle for needle in spec["stderr_absent"] if needle in stderr],
    )
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
        ["vaultmind", "console_scripts", "vaultmind.cli:cli", "click.core.Group"],
        [entry["name"], entry["group"], entry["value"], entry["cli_type"]],
    )
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
        "stdout_lines": lines,
        "stderr": stderr,
        "audit": audit,
        "calls": observed_calls,
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
    from vaultmind.config import VaultConfig
    from vaultmind.mcp.auth import ProfileEnforcer, ProfileError
    from vaultmind.mcp.profiles import load_profile
    from vaultmind.mcp.server import _dispatch_tool
    from vaultmind.vault.parser import VaultParser

    vault = mcp_build_fixture(root, scoped=False)["vault"]
    parser = VaultParser(VaultConfig(path=vault))
    hub = "02-projects/hub.md"
    secret = "03-private/secret.md"

    def dispatch(tool: str, args: dict[str, Any], **context: Any) -> Any:
        try:
            return _dispatch_tool(tool, args, vault, object(), object(), parser, **context)
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


async def mcp_factory_none_handlers(root: Path, check: Any) -> dict[str, Any]:
    """Real ``create_mcp_server(enforcer=None)`` SDK request handler: named tools fail closed.

    Each request goes through the SDK-registered ``CallToolRequest`` handler (the same
    handler the stdio server runs). The identical arguments are replayed against a server
    built with a real full-profile enforcer, so a failure to start/validate cannot read
    as a green denial.
    """
    from mcp.types import CallToolRequest, CallToolRequestParams

    from vaultmind.config import VaultConfig
    from vaultmind.mcp.auth import ProfileEnforcer
    from vaultmind.mcp.profiles import load_profile
    from vaultmind.mcp.server import create_mcp_server
    from vaultmind.vault.parser import VaultParser

    vault = mcp_build_fixture(root, scoped=False)["vault"]
    parser = VaultParser(VaultConfig(path=vault))
    none_server = create_mcp_server(vault, object(), object(), parser, enforcer=None)
    full_server = create_mcp_server(
        vault, object(), object(), parser, enforcer=ProfileEnforcer(load_profile("full"), vault)
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
    _unused, unrelated_payload, _text = await invoke(
        none_server, "vault_list", {"folder": "00-inbox"}
    )
    check(
        "factory.unrelated-tool-without-enforcer-unchanged",
        {"notes": ["00-inbox/inbox-note.md"], "count": 1},
        unrelated_payload,
    )
    return receipt


async def mcp_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    home = home.resolve()
    check = make_checker(checks)
    scenarios: dict[str, Any] = {}
    for name in MCP_SCENARIOS:
        scenarios[name] = await run_mcp_scenario(name, home / name, check)
    mcp_direct_dispatcher(home / "direct-dispatcher", check)
    factory = await mcp_factory_none_handlers(home / "factory-none", check)
    origins = {s["import_origin"] for s in scenarios.values()}
    check("origin.consistent-across-launches", 1, len(origins))
    return {
        "mcp_sdk": importlib.metadata.version("mcp"),
        "import_origin": origins.pop(),
        "factory_none_handlers": factory,
        "scenarios": scenarios,
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

    async def text(self, text: str, actor: int) -> list[dict[str, Any]]:
        return await self._feed(
            {
                "message": {
                    "message_id": self.n + 10,
                    "date": 1700000000,
                    "chat": {"id": actor, "type": "private"},
                    "from": {"id": actor, "is_bot": False, "first_name": "t"},
                    "text": text,
                }
            }
        )

    async def photo(self, actor: int, size: int = 1000) -> list[dict[str, Any]]:
        return await self._feed(
            {
                "message": {
                    "message_id": self.n + 10,
                    "date": 1700000000,
                    "chat": {"id": actor, "type": "private"},
                    "from": {"id": actor, "is_bot": False, "first_name": "t"},
                    "photo": [
                        {
                            "file_id": "F1",
                            "file_unique_id": "U1",
                            "width": 10,
                            "height": 10,
                            "file_size": size,
                        }
                    ],
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
        f'[autonomy]\nforce_block = {str(force_block).lower()}\ndb_path = "{data / "review.db"}"\n'
        f"[telegram]\nallowed_user_ids = [{', '.join(map(str, allowed))}]\n"
        f'[chroma]\npersist_dir = "{data / "chroma"}"\n'
        f'[graph]\npersist_path = "{data / "graph.json"}"\n'
        f'[search]\nbm25_db_path = "{data / "fts.db"}"\n'
        "[embedding]\ncache_enabled = false\n[llm]\ngraph_context_enabled = false\n",
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
    thinking = ThinkingPartner(settings.llm, settings.telegram, llm)
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


def approval_git_binding() -> dict[str, Any]:
    def git(*argv: str) -> str | None:
        done = subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    refs: dict[str, Any] = {}
    for ref in ("fix/vm-m13-review-appliers", "origin/main", "main"):
        commit = git("rev-parse", "--verify", "--quiet", ref)
        if commit:
            refs[ref] = {"commit": commit, "merge_base_with_head": git("merge-base", "HEAD", ref)}
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "parents": (git("rev-list", "--parents", "-n1", "HEAD") or "").split()[1:],
        "tree": git("rev-parse", "HEAD^{tree}"),
        "reference_commits": refs,
    }


def approval_legacy_schema() -> tuple[str, str]:
    """The newest committed review_proposals schema that predates the legacy_* columns."""
    path = "src/vaultmind/services/review_queue.py"
    log = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--", path], text=True
    ).split()
    for commit in log:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"], text=True
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
        w.store.index_notes(w.parser.iter_notes(strict=True), w.parser)
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
        legacy_commit, legacy_schema = approval_legacy_schema()
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
        pw.store.index_notes(pw.parser.iter_notes(strict=True), pw.parser)
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
        dw.store.index_notes(dw.parser.iter_notes(strict=True), dw.parser)
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
        "git": approval_git_binding(),
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


LINEAGE_ACTOR = 42
LINEAGE_BASE_COMMIT = "7ef84c83b1379bf3bdb825af8c1e80032302c3af"
LINEAGE_OFFLINE_KEY = "OFFLINE-SENTINEL-NOT-CREDENTIAL"
LINEAGE_FAILURE = re.compile(r"fail|error|not indexed|\u26a0", re.IGNORECASE)
LINEAGE_SUCCESS = re.compile(
    r"Edit applied|Superseded \(not deleted\)|captured|Indexed \d+ chunks|Ingested|Bookmarked"
    r"|Created qa-artifact",
    re.IGNORECASE,
)
LINEAGE_SURFACES = (
    "vector_rows",
    "keyword_rows",
    "vector_search",
    "hybrid_search",
    "keyword_search",
    "keyword_matched_ids",
)
LINEAGE_ABSENT: dict[str, list[str]] = {surface: [] for surface in LINEAGE_SURFACES}
LINEAGE_FOUND = dict.fromkeys(LINEAGE_SURFACES, True)


class LineageEmbedder(DeterministicEmbedder):
    """Synthetic embedder with a switchable outage; counts batches reaching the boundary."""

    def __init__(self) -> None:
        self.outage = False
        self.batches = 0
        self.failures = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches += 1
        if self.outage:
            self.failures += 1
            raise RuntimeError("synthetic embedding outage")
        return super().embed_texts(texts)


class LineageKeywordFault:
    """Instance-level fault at the keyword-index write boundary over the real SQLite index."""

    def __init__(self, bm25: Any) -> None:
        self.armed: set[str] = set()
        self.raised: list[str] = []
        self.methods = sorted(
            name
            for name, _ in inspect.getmembers(type(bm25), inspect.isfunction)
            if not name.startswith("_") and name not in ("search", "close")
        )
        for name in self.methods:
            setattr(bm25, name, self._wrap(name, getattr(bm25, name)))

    def _wrap(self, name: str, real: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            if "*" in self.armed or name in self.armed:
                self.raised.append(name)
                raise sqlite3.OperationalError(f"synthetic keyword-index outage in {name}")
            return real(*args, **kwargs)

        return call

    def arm(self, *names: str) -> None:
        self.armed = set(names) or {"*"}

    def disarm(self) -> None:
        self.armed = set()


def lineage_note(title: str, body: str, **meta: str) -> str:
    return approval_note(title, body, **meta)


def lineage_legacy_writer() -> Any:
    """The pre-M14 marker writer executed from the base commit (generated legacy bytes)."""
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{LINEAGE_BASE_COMMIT}:src/vaultmind/contradiction/marking.py",
        ],
        text=True,
    )
    namespace: dict[str, Any] = {"__name__": "lineage_legacy_marking"}
    exec(compile(source, "lineage_legacy_marking.py", "exec"), namespace)
    return namespace["write_superseded_block"]


def lineage_legacy_delete(path: Path, moment: Any) -> None:
    """Exactly what base bot/handlers/delete.py wrote on confirmation."""
    lineage_legacy_writer()(
        path,
        callout_type="superseded",
        title="Deleted via bot",
        rationale="Requested via Telegram `/delete` \u2014 content preserved below.",
        frontmatter_key="superseded_at",
        frontmatter_value="bot:delete",
        dated=True,
        timestamp=moment,
    )


def lineage_legacy_edit(path: Path, new_body: str, instruction: str, moment: Any) -> None:
    """Exactly what base bot/handlers/edit.py wrote on confirmation."""
    lineage_legacy_writer()(
        path,
        callout_type="superseded",
        title="Edited via bot",
        rationale=f"Instruction: {instruction}",
        frontmatter_key="superseded_at",
        frontmatter_value=moment.isoformat(),
        dated=True,
        timestamp=moment,
        new_content=new_body,
    )


def lineage_replies(calls: list[dict[str, Any]]) -> list[str]:
    methods = ("SendMessage", "EditMessageText", "AnswerCallbackQuery")
    return [c["data"].get("text", "") for c in calls if c["method"] in methods]


def lineage_confirm_data(calls: list[dict[str, Any]], prefix: str) -> str:
    found = [data for kb in tg_keyboards(calls) for data in kb if data.startswith(prefix)]
    if len(found) != 1:
        raise RuntimeError(f"expected one {prefix} button, got {found}: {lineage_replies(calls)}")
    return found[0]


async def lineage_delete(world: SimpleNamespace, rel: str) -> list[str]:
    """Real /delete command then the confirmation callback through the aiogram dispatcher."""
    prompt = await world.harness.command(f"/delete {rel}", LINEAGE_ACTOR)
    try:
        confirm = lineage_confirm_data(prompt, "delete_confirm:")
    except RuntimeError:
        return lineage_replies(prompt)
    return lineage_replies(await world.harness.callback(confirm, LINEAGE_ACTOR))


async def lineage_edit(
    world: SimpleNamespace, rel: str, body: str, instruction: str
) -> tuple[list[str], list[str]]:
    """Real /edit (generating via scripted LLM) then confirmation; returns (prompt, confirm)."""
    world.llm.reply = body
    prompt = await world.harness.command(f"/edit {rel} {instruction}", LINEAGE_ACTOR)
    texts = lineage_replies(prompt)
    try:
        confirm = lineage_confirm_data(prompt, "edit_confirm:")
    except RuntimeError:
        return texts, []
    return texts, lineage_replies(await world.harness.callback(confirm, LINEAGE_ACTOR))


def lineage_config_text(
    vault: Path,
    data: Path,
    stability: bool,
    ttl: int,
    extra: str,
    *,
    embedding_cache: bool = False,
) -> str:
    """Explicit absolute-path TOML shared by the bot, CLI and MCP assemblies of every world."""
    return (
        f'openai_api_key = "{LINEAGE_OFFLINE_KEY}"\n'
        f'anthropic_api_key = "{LINEAGE_OFFLINE_KEY}"\n'
        f'[vault]\npath = "{vault}"\n'
        f'[chroma]\npersist_dir = "{data / "chroma"}"\ncollection_name = "lineage_chunks"\n'
        f'[graph]\npersist_path = "{data / "graph.json"}"\n'
        f'[search]\nhybrid_enabled = true\nbm25_db_path = "{data / "bm25.db"}"\n'
        "cache_enabled = false\n"
        f"[watch]\ndebounce_ms = 60\nhash_stability_check = {str(stability).lower()}\n"
        "reextract_graph = false\n"
        f"[telegram]\nallowed_user_ids = [{LINEAGE_ACTOR}]\nthinking_session_ttl = {ttl}\n"
        f"[embedding]\ncache_enabled = {str(embedding_cache).lower()}\n"
        "[llm]\ngraph_context_enabled = false\n" + extra
    )


def lineage_world(
    home: Path,
    name: str,
    notes: dict[str, str],
    llm: ApprovalLLM,
    closers: list[Any],
    *,
    stability: bool = True,
    ttl: int = 3600,
    extra_toml: str = "",
    cli_seed: bool = False,
    embedding_cache: bool = False,
) -> SimpleNamespace:
    """Explicit temporary TOML + real parser/Chroma/SQLite BM25/handlers over isolated files."""
    from aiogram import Bot

    from vaultmind.bot.commands import CommandHandlers
    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import load_settings
    from vaultmind.graph.evolution import LineageStore
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

    root = (home / name).resolve()
    vault, data = root / "vault", root / "data"
    vault.mkdir(parents=True)
    data.mkdir()
    for rel, text in notes.items():
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text, encoding="utf-8")
    config = root / "config.toml"
    config.write_text(
        lineage_config_text(
            vault, data, stability, ttl, extra_toml, embedding_cache=embedding_cache
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    embedder = LineageEmbedder()
    indexes_before = {
        "chroma": settings.chroma.persist_dir.exists(),
        "keyword": (data / "bm25.db").exists(),
    }
    seed_result = None
    if cli_seed:
        # The installed command creates the indexes before any verifier handle opens them.
        seed_result = lineage_cli_run(SimpleNamespace(config=config, embedder=embedder), ["index"])
        if seed_result.exit_code:
            raise RuntimeError(f"Command-built seed failed: {seed_result.output}")
    bm25 = BM25Index(data / "bm25.db")
    fault = LineageKeywordFault(bm25)
    store = VaultStore(settings.chroma, cast("Embedder", embedder), bm25=bm25)
    parser = VaultParser(settings.vault)
    graph = KnowledgeGraph(settings.graph)
    lineage = LineageStore(data / "lineage.json")
    session = approval_session()
    bot = Bot("123456:OFFLINE-ACCEPTANCE", session=session)
    thinking = ThinkingPartner(
        settings.llm,
        settings.telegram,
        llm,
        vault_root=settings.vault.path,
        parser=parser,
        distill_config=settings.distill,
        score_floor=settings.bench.score_floor,
    )
    handlers = CommandHandlers(settings, store, graph, parser, thinking, llm, lineage_store=lineage)
    closers.append(bm25.close)
    return SimpleNamespace(
        name=name,
        root=root,
        vault=vault,
        data=data,
        config=config,
        settings=settings,
        indexes_before=indexes_before,
        cli_seed_result=seed_result,
        embedder=embedder,
        bm25=bm25,
        fault=fault,
        store=store,
        parser=parser,
        graph=graph,
        lineage=lineage,
        llm=llm,
        session=session,
        bot=bot,
        handlers=handlers,
        harness=TelegramHarness(handlers, session, bot),
        thinking=thinking,
        input_identity={
            "files_sha256": {
                str(path.relative_to(root)): digest(path.read_bytes())
                for path in [config, *sorted(vault.rglob("*.md"))]
            },
            "effective_config_sha256": approval_json_digest(settings.model_dump(mode="json")),
        },
    )


def lineage_chroma_client(world: SimpleNamespace) -> Any:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=str(world.settings.chroma.persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def lineage_vector_rows(world: SimpleNamespace) -> dict[str, str]:
    collection = lineage_chroma_client(world).get_collection(world.settings.chroma.collection_name)
    got = collection.get(include=["documents"])
    return dict(zip(got["ids"], got["documents"], strict=True))


def lineage_keyword_rows(world: SimpleNamespace) -> dict[str, str]:
    con = sqlite3.connect(f"file:{world.data / 'bm25.db'}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): str(row[1])
            for row in con.execute("SELECT chunk_id, content FROM fts_chunks")
        }
    finally:
        con.close()


def lineage_state(world: SimpleNamespace) -> dict[str, dict[str, str]]:
    return {"vector": lineage_vector_rows(world), "keyword": lineage_keyword_rows(world)}


def lineage_keyword_note_rows(world: SimpleNamespace, rel: str) -> dict[str, str]:
    """Keyword-index rows of one note, read-only and parameterized."""
    con = sqlite3.connect(f"file:{world.data / 'bm25.db'}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): str(row[1])
            for row in con.execute(
                "SELECT chunk_id, content FROM fts_chunks WHERE note_path = ?", (rel,)
            )
        }
    finally:
        con.close()


def lineage_keyword_matched(world: SimpleNamespace, token: str) -> list[str]:
    """Chunk ids the FTS5 MATCH operator itself selects for `token` (read-only, parameterized)."""
    con = sqlite3.connect(f"file:{world.data / 'bm25.db'}?mode=ro", uri=True)
    try:
        return sorted(
            str(row[0])
            for row in con.execute(
                "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH ?", (f'"{token}"',)
            )
        )
    finally:
        con.close()


def lineage_exposure(world: SimpleNamespace, token: str) -> dict[str, list[str]]:
    """Every retrieval surface (stored rows, vector, hybrid, keyword) that carries `token`."""
    state = lineage_state(world)
    vector = world.store.search(token, n_results=50)
    hybrid = world.store.hybrid_search(token, n_results=50)
    keyword = world.bm25.search(token, n_results=50)
    return {
        "vector_rows": sorted(k for k, v in state["vector"].items() if token in v),
        "keyword_rows": sorted(k for k, v in state["keyword"].items() if token in v),
        "vector_search": sorted(h["chunk_id"] for h in vector if token in h["content"]),
        "hybrid_search": sorted(h["chunk_id"] for h in hybrid if token in h["content"]),
        "keyword_search": sorted(str(h["chunk_id"]) for h in keyword),
        "keyword_matched_ids": lineage_keyword_matched(world, token),
    }


def lineage_found(world: SimpleNamespace, token: str) -> dict[str, bool]:
    return {k: bool(v) for k, v in lineage_exposure(world, token).items()}


def lineage_ids(world: SimpleNamespace, rel: str) -> dict[str, list[str]]:
    state = lineage_state(world)
    return {
        kind: sorted(k for k in rows if k.startswith(f"{rel}::")) for kind, rows in state.items()
    }


def lineage_expected_ids(world: SimpleNamespace, rel: str) -> dict[str, list[str]]:
    """Chunk ids of the shared current-content projection (the parser's own chunking)."""
    note = world.parser.parse_file(world.vault / rel)
    ids = sorted(c.chunk_id for c in world.parser.chunk_note(note))
    return {"vector": ids, "keyword": ids}


def lineage_index_full(world: SimpleNamespace) -> int:
    return int(world.store.index_notes(world.parser.iter_notes(strict=True), world.parser))


def lineage_seed_stale(world: SimpleNamespace, rel: str, texts: list[str]) -> list[str]:
    """Rows a pre-upgrade index held for `rel`, written through independent handles."""
    from vaultmind.vault.models import NoteChunk

    chunks = [
        NoteChunk(note_path=rel, note_title="Stale", chunk_idx=i, content=text)
        for i, text in enumerate(texts)
    ]
    collection = lineage_chroma_client(world).get_collection(world.settings.chroma.collection_name)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=DeterministicEmbedder().embed_texts(texts),  # type: ignore[arg-type]
        metadatas=[c.to_chroma_metadata() for c in chunks],
    )
    con = sqlite3.connect(str(world.data / "bm25.db"))
    try:
        with con:
            for chunk in chunks:
                con.execute("DELETE FROM fts_chunks WHERE chunk_id = ?", (chunk.chunk_id,))
                con.execute(
                    "INSERT INTO fts_chunks(chunk_id, note_path, note_title, content)"
                    " VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, rel, "Stale", chunk.content),
                )
    finally:
        con.close()
    return [c.chunk_id for c in chunks]


def lineage_entry_point() -> Any:
    """The registered `vaultmind` console script, resolved from installed distribution metadata."""
    matches = [
        ep
        for ep in importlib.metadata.entry_points(group="console_scripts")
        if ep.name == "vaultmind"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one registered vaultmind console script: {matches}")
    return matches[0]


def lineage_entry_identity() -> dict[str, Any]:
    entry = lineage_entry_point()
    dist = importlib.metadata.distribution("vaultmind")
    return {
        "name": entry.name,
        "group": entry.group,
        "value": entry.value,
        "dist_version": dist.version,
        "direct_url": dist.read_text("direct_url.json"),
    }


LINEAGE_CLI_RESOURCES = ("bm25", "cache", "gap_store")


def lineage_resource_state(resource: Any) -> str:
    """'open' or 'closed' from the resource's own SQLite connection (anything else is surfaced)."""
    connection = getattr(resource, "_con", None)
    if connection is None:
        connection = resource._conn
    try:
        connection.execute("SELECT 1").fetchall()
    except sqlite3.ProgrammingError as exc:
        return "closed" if "closed database" in str(exc) else f"unreadable: {exc}"
    return "open"


class LineageResourceObserver:
    """Passive observation of the configured resources one installed command creates.

    A read-only profile hook reads the command's own frame locals; it replaces no event loop,
    callback, resource or close. It STRONGLY RETAINS each configured resource (keyword index,
    embedding cache, gap ledger) so it can neither be collected nor mistaken for closed, and reads
    that resource's own SQLite connection state when the command's frame returns (``final``) and,
    for a command that runs its own event loop, when that loop starts (``live``). A resource the
    command never created is never reported, so an unobserved resource cannot pass as closed.
    """

    def __init__(self, command: str, module: str = "vaultmind.cli") -> None:
        self.command = command
        self.module = module
        self.retained: dict[str, Any] = {}
        self.live: dict[str, str] = {}
        self.final: dict[str, str] = {}
        self.previous = sys.getprofile()

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if self.previous is not None:
            self.previous(frame, event, arg)
        if event not in ("call", "return") or frame.f_globals.get("__name__") != self.module:
            return
        name = frame.f_code.co_name
        if event == "return" and name == self.command:
            self._take(frame, self.final)
        elif event == "call" and name == "_run_watch":
            caller = frame.f_back
            while caller is not None and not (
                caller.f_code.co_name == self.command
                and caller.f_globals.get("__name__") == self.module
            ):
                caller = caller.f_back
            if caller is not None:
                self._take(caller, self.live)

    def _take(self, frame: Any, into: dict[str, str]) -> None:
        for key, value in dict(frame.f_locals).items():
            if key in LINEAGE_CLI_RESOURCES and value is not None:
                self.retained[key] = value
                into[key] = lineage_resource_state(value)


def lineage_observer_selftest(scratch: Path) -> dict[str, dict[str, str]]:
    """Run the observer over real resources that are left open, closed, or never created.

    The probe functions are ordinary functions observed exactly like a CLI command; their real
    SQLite resources are leaked, closed, or absent, so a passive observer that could not tell a
    leak from a close (or reported an unobserved resource as closed) would fail here.
    """
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.embedding_cache import EmbeddingCache
    from vaultmind.memory.gaps import GapStore

    scratch.mkdir(parents=True)

    def leak() -> None:
        bm25 = BM25Index(scratch / "leak-bm25.db")
        cache = EmbeddingCache(scratch / "leak-cache.db")
        gap_store = GapStore(scratch / "leak-gaps.db")
        # Real use of each resource, then returned without closing: a leak.
        _ = bm25.count
        cache.stats()
        gap_store.list_open()

    def release() -> None:
        bm25 = BM25Index(scratch / "release-bm25.db")
        cache = EmbeddingCache(scratch / "release-cache.db")
        gap_store = GapStore(scratch / "release-gaps.db")
        bm25.close()
        cache.close()
        gap_store.close()

    def absent() -> None:
        return None

    outcome: dict[str, dict[str, str]] = {}
    for probe in (leak, release, absent):
        observer = LineageResourceObserver(probe.__name__, module=__name__)
        sys.setprofile(observer)
        try:
            probe()
        finally:
            sys.setprofile(observer.previous)
        outcome[probe.__name__] = observer.final
        for resource in observer.retained.values():
            resource.close()
    return outcome


def lineage_cli_run(world: SimpleNamespace, args: list[str], embedder: Any = None) -> Any:
    """Run the registered `vaultmind --config <explicit TOML> <args>` Click group (blocking).

    ``args[0]`` is the command name. Only the external Embedder factory is substituted. The
    command itself creates the configured Chroma store, SQLite keyword index, embedding cache and
    gap ledger (nothing here builds, patches or injects BM25, VaultStore, the watcher or the
    handler, and no ``close`` is replaced). ``resources`` is the passive
    `LineageResourceObserver` account of what the command created and its connection state.
    """
    from unittest.mock import patch

    from click.testing import CliRunner

    import vaultmind.indexer as indexer

    use = embedder or world.embedder
    cli = lineage_entry_point().load()
    observer = LineageResourceObserver(args[0])
    sys.setprofile(observer)
    try:
        with patch.object(indexer, "Embedder", lambda *args, **kwargs: use):
            result = CliRunner().invoke(cli, ["--config", str(world.config), *args])
    finally:
        sys.setprofile(observer.previous)
    return SimpleNamespace(
        exit_code=result.exit_code,
        output=result.output,
        exception=result.exception,
        resources=observer,
    )


async def lineage_cli_index(world: SimpleNamespace, embedder: LineageEmbedder | None = None) -> Any:
    """Installed `vaultmind index`; only the embedding provider boundary is replaced."""
    return await asyncio.to_thread(lineage_cli_run, world, ["index"], embedder)


@contextlib.contextmanager
def lineage_readonly_keyword(world: SimpleNamespace) -> Any:
    """Real keyword-index write fault for a command that opens its own handle: read-only file."""
    db = world.data / "bm25.db"
    mode = db.stat().st_mode & 0o777
    db.chmod(0o444)
    try:
        yield
    finally:
        db.chmod(mode)


async def lineage_until(predicate: Any, timeout: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return bool(predicate())


class LineageDirectWatch:
    """Verifier-assembled watcher rig (direct VaultWatcher + IncrementalWatchHandler).

    NOT the shipped `vaultmind watch` assembly: this rig builds the handler/watcher by hand
    over the verifier-wired world store (which already carries its own BM25 index) so that
    failure injection, event publication and the tracked hash are observable. It proves the
    handler's own error semantics only; the standalone command is exercised separately by
    `LineageCliWatch`.
    """

    def __init__(self, world: SimpleNamespace) -> None:
        from vaultmind.vault.events import (
            NoteCreatedEvent,
            NoteDeletedEvent,
            NoteModifiedEvent,
            VaultEventBus,
        )
        from vaultmind.vault.watch_handler import IncrementalWatchHandler
        from vaultmind.vault.watcher import VaultWatcher

        self.world = world
        self.events: list[tuple[str, str, int | None]] = []
        bus = VaultEventBus()

        async def record(event: Any) -> None:
            try:
                rel = str(event.path.relative_to(world.vault))
            except ValueError:
                rel = str(event.path)
            self.events.append((type(event).__name__, rel, getattr(event, "chunks_indexed", None)))

        for kind in (NoteCreatedEvent, NoteModifiedEvent, NoteDeletedEvent):
            bus.subscribe(kind, record)
        self.handler = IncrementalWatchHandler(
            config=world.settings.watch,
            parser=world.parser,
            store=world.store,
            event_bus=bus,
        )
        self.watcher = VaultWatcher(
            config=world.settings.vault, on_change=self.handler.handle_change
        )
        self.watcher.start()

    def of(self, rel: str, *kinds: str) -> list[tuple[str, str, int | None]]:
        return [e for e in self.events if e[1] == rel and (not kinds or e[0] in kinds)]

    def hash_of(self, rel: str) -> str | None:
        return self.handler.indexed_hashes.get(self.world.vault / rel)

    async def wait_events(self, rel: str, count: int, *kinds: str) -> bool:
        return await lineage_until(lambda: len(self.of(rel, *kinds)) >= count)

    async def wait_fault(self, count: int) -> bool:
        return await lineage_until(lambda: len(self.world.fault.raised) >= count)

    async def settle(self) -> None:
        await asyncio.sleep(0.6)

    def stop(self) -> None:
        self.watcher.stop()


LINEAGE_CANARY = "cli-watch-canary.md"
LT_CANARY = "ketchcanaryclaim"


class LineageCliWatch:
    """The installed `vaultmind watch` command running against a world's explicit TOML.

    The command assembles its own Chroma store, SQLite keyword index, parser, event bus,
    IncrementalWatchHandler and VaultWatcher; only the external Embedder factory is
    substituted. It runs in a worker thread so its own `asyncio.run` owns the loop that
    watchdog events are marshalled onto. Readiness and every outcome are observed through
    the configured indexes (read-only SQLite and Chroma queries), never through a hook in
    the command. Stopping cancels the command's main task, which is what SIGINT does to
    `asyncio.run`; the command's own `finally` then stops the watcher and closes its indexes.
    """

    def __init__(self, world: SimpleNamespace) -> None:
        self.world = world
        self.loop: asyncio.AbstractEventLoop | None = None
        self.main_task: asyncio.Task[Any] | None = None
        self.future: asyncio.Future[Any] | None = None
        self.canary = world.vault / LINEAGE_CANARY
        self.revision = 0

    def _loop(self) -> asyncio.AbstractEventLoop | None:
        return self.loop if self.loop is not None and self.loop.is_running() else None

    def _run(self) -> Any:
        # Read-only profiling observes the product loop/task; no event-loop or callback replacement.
        def observe(frame: Any, event: str, arg: Any) -> None:
            if (
                event == "call"
                and frame.f_code.co_name == "_run_watch"
                and frame.f_globals.get("__name__") == "vaultmind.cli"
            ):
                self.loop = asyncio.get_running_loop()
                self.main_task = asyncio.current_task()

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            return lineage_cli_run(self.world, ["watch"])
        finally:
            sys.setprofile(previous)

    async def start(self) -> bool:
        self.future = asyncio.ensure_future(asyncio.to_thread(self._run))
        future = self.future
        return await lineage_until(lambda: self._loop() is not None or future.done(), 30.0) and (
            self._loop() is not None
        )

    async def beat(self, timeout: float = 3.0) -> bool:
        """Rewrite the canary note and wait until the command's index shows that revision."""
        self.revision += 1
        marker = f"{LT_CANARY} revision {self.revision}"
        self.canary.write_text(lineage_note("Watch Canary", f"{marker}."), encoding="utf-8")
        return await lineage_until(
            lambda: any(
                marker in text
                for text in lineage_keyword_note_rows(self.world, LINEAGE_CANARY).values()
            ),
            timeout,
        )

    async def ready(self) -> bool:
        """Observer-live barrier: a real filesystem event must reach the configured indexes."""
        for _ in range(8):
            if await self.beat():
                return True
        return False

    async def stop(self) -> SimpleNamespace:
        """Bounded graceful cancellation of the command; returns its outcome."""
        assert self.future is not None
        loop = self._loop()
        if loop is not None:

            def cancel() -> None:
                if self.main_task is not None:
                    self.main_task.cancel()

            loop.call_soon_threadsafe(cancel)
        done, _ = await asyncio.wait({self.future}, timeout=20.0)
        result = self.future.result() if done else None
        return SimpleNamespace(stopped=bool(done), result=result)


def lineage_launcher() -> str:
    """MCP_LAUNCHER (network guard, installed entry point, env audit) + the Embedder factory."""
    marker = "code = 0\ntry:\n"
    if MCP_LAUNCHER.count(marker) != 1:
        raise RuntimeError("launcher template changed: embedder substitution point not found")
    substitution = (
        "import hashlib\nimport re\n\n"
        + inspect.getsource(DeterministicEmbedder)
        + "\n\nimport vaultmind.indexer as _indexer\n\n"
        "_indexer.Embedder = lambda *args, **kwargs: DeterministicEmbedder()\n"
        'emit(kind="embedder-substitution", factory="DeterministicEmbedder")\n\n'
    )
    return MCP_LAUNCHER.replace(marker, substitution + marker)


async def lineage_launch(root: Path, state_name: str, cli_args: list[str]) -> SimpleNamespace:
    """Run one registered `vaultmind` command in an isolated child (non-MCP commands)."""
    state = root / state_name
    state.mkdir()
    done = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-I", "-c", lineage_launcher(), str(state), *cli_args],
        env={"HOME": str(root), "ANONYMIZED_TELEMETRY": "False", "COLUMNS": "2000"},
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    return SimpleNamespace(
        code=done.returncode, stdout=done.stdout, stderr=done.stderr, state=state
    )


def lineage_launcher_checks(ck: Any, prefix: str, root: Path, state: Path) -> dict[str, Any]:
    """Identity, isolation and network-guard checks over one child's launcher log."""
    import vaultmind

    events = [
        json.loads(line)
        for line in (state / "launcher.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(event["kind"], []).append(event)
    entry = by_kind["entrypoint"][0]
    ck(
        prefix + "origin.registered-entry-point",
        [["vaultmind.cli:cli"], "vaultmind", "console_scripts", "click.core.Group"],
        [
            [e["matches"] for e in by_kind["entrypoint-lookup"]][0],
            entry["name"],
            entry["group"],
            entry["cli_type"],
        ],
    )
    ck(
        prefix + "origin.same-interpreter-and-package-as-verifier",
        [sys.executable, os.path.realpath(vaultmind.__file__)],
        [entry["executable"], entry["module_file"]],
    )
    environment = by_kind["environment"][0]
    ck(
        prefix + "environment.home-and-no-inherited-credentials",
        [str(root), []],
        [environment["home"], environment["forbidden"]],
    )
    ck(
        prefix + "network.guard-blocked-self-test-and-no-outbound-attempt",
        {"probe": ["blocked"], "outbound": []},
        {
            "probe": [e["result"] for e in by_kind["probe"]],
            "outbound": [e for e in by_kind.get("network", []) if not e["probe"]],
        },
    )
    ck(
        prefix + "lifecycle.clean-exit",
        [[0, True]],
        [[e["code"], e["tee_joined"]] for e in by_kind.get("exit", [])],
    )
    return {
        "entry_point": {
            key: entry[key]
            for key in ("name", "group", "value", "dist_version", "direct_url", "module_file")
        },
        "executable": entry["executable"],
        "python": entry["python"],
        "boundary_metadata_not_evidence": {
            "note": (
                "Launcher-owned label of the one substituted external boundary (the Embedder "
                "factory); recorded by the launcher itself, so it is metadata and not an "
                "acceptance predicate. State evidence is the index/Chroma/SQLite comparisons."
            ),
            "embedder_substitution": [
                e["factory"] for e in by_kind.get("embedder-substitution", [])
            ],
        },
        "launcher_events": events,
    }


def lineage_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


LT_DELETE = "quaydeletedclaim"
LT_KEEP = "quaykeptclaim"
LT_ORIG = "tidaloriginalclaim"
LT_V1 = "tidalfirstclaim"
LT_V2 = "tidalsecondclaim"
LT_AUTHOR = "tidalauthoredquote"
LT_CLOSE = "tidalclosingclaim"


def lineage_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_sha256": digest(value)}
    if isinstance(value, dict):
        return {str(key): lineage_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [lineage_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [lineage_json(item) for item in sorted(value, key=repr)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def lineage_edges(world: SimpleNamespace, rel: str) -> list[str]:
    """Lineage as persisted on disk (fresh store), not the in-memory handler object."""
    from vaultmind.graph.evolution import LineageStore

    return [e.event for e in LineageStore(world.data / "lineage.json").get_lineage(rel)]


def lineage_vault_digests(world: SimpleNamespace) -> dict[str, str]:
    return {
        str(path.relative_to(world.vault)): digest(path.read_bytes())
        for path in sorted(world.vault.rglob("*.md"))
    }


def lineage_failure_in(replies: list[str]) -> bool:
    return any(LINEAGE_FAILURE.search(text) for text in replies)


def lineage_success_in(replies: list[str]) -> bool:
    return any(LINEAGE_SUCCESS.search(text) for text in replies)


async def lineage_delete_scenario(env: SimpleNamespace) -> None:
    """Acceptance 1: bot delete, replay, then CLI/API full reindex and watch/incremental."""
    ck = env.ck
    body = f"Dock inspection record: {LT_DELETE} stays on file."
    notes = {
        "del.md": lineage_note("Quay Ledger", body),
        "keep.md": lineage_note("Quay Keeper", f"Control record: {LT_KEEP}."),
    }
    w = lineage_world(env.home, "delete", notes, env.llm, env.closers)
    env.worlds.append(w)
    path = w.vault / "del.md"

    count = lineage_index_full(w)
    ck("delete.baseline-full-index-returns-int", ["int", True], [type(count).__name__, count >= 2])
    ck("delete.baseline-retired-text-is-retrievable", LINEAGE_FOUND, lineage_found(w, LT_DELETE))
    ck("delete.baseline-control-is-retrievable", LINEAGE_FOUND, lineage_found(w, LT_KEEP))

    replies = await lineage_delete(w, "del.md")
    marked = path.read_bytes()
    note = w.parser.parse_file(path)
    ck("delete.confirm-reported-no-failure", False, lineage_failure_in(replies))
    ck("delete.original-body-preserved-on-disk", True, body in marked.decode())
    ck("delete.raw-note-content-keeps-original", True, body in note.content)
    ck("delete.current-projection-empty", "", note.current_content().strip())
    ck(
        "delete.retired-text-absent-from-every-surface",
        LINEAGE_ABSENT,
        lineage_exposure(w, LT_DELETE),
    )
    ck(
        "delete.zero-chunk-note-has-no-rows",
        {"vector": [], "keyword": []},
        lineage_ids(w, "del.md"),
    )
    ck("delete.control-still-retrievable", LINEAGE_FOUND, lineage_found(w, LT_KEEP))
    ck("delete.exactly-one-lineage-edge", ["deleted"], lineage_edges(w, "del.md"))

    replay = await lineage_delete(w, "del.md")
    ck("delete.replay-leaves-file-bytes-unchanged", digest(marked), digest(path.read_bytes()))
    ck("delete.replay-keeps-one-lineage-edge", ["deleted"], lineage_edges(w, "del.md"))
    ck("delete.replay-reported-no-failure", False, lineage_failure_in(replay))
    ck("delete.replay-retired-text-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_DELETE))

    cli = await lineage_cli_index(w)
    ck("delete.cli-full-reindex-exit-code", 0, cli.exit_code)
    ck(
        "delete.cli-full-reindex-reports-counts",
        True,
        bool(re.search(r"Indexed \d+ chunks from \d+ notes", cli.output)),
    )
    ck("delete.cli-full-retired-text-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_DELETE))
    ck("delete.cli-full-control-retrievable", LINEAGE_FOUND, lineage_found(w, LT_KEEP))
    ck("delete.cli-full-file-bytes-unchanged", digest(marked), digest(path.read_bytes()))

    count = lineage_index_full(w)
    expected = len(lineage_expected_ids(w, "keep.md")["vector"])
    ck(
        "delete.api-full-count-is-active-chunks-only",
        [expected, "int"],
        [count, type(count).__name__],
    )
    ck("delete.api-full-retired-text-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_DELETE))

    stale = [f"note: Quay Ledger (type: fleeting)\n\n{body}"]
    lineage_seed_stale(w, "del.md", stale)
    ck("delete.seeded-stale-rows-are-visible", LINEAGE_FOUND, lineage_found(w, LT_DELETE))
    incremental = w.store.index_single_note(w.parser.parse_file(path), w.parser)
    ck("delete.incremental-index-returns-zero", 0, incremental)
    ck(
        "delete.incremental-retires-vector-and-keyword",
        LINEAGE_ABSENT,
        lineage_exposure(w, LT_DELETE),
    )

    lineage_seed_stale(w, "del.md", stale)
    rig = LineageDirectWatch(w)
    try:
        await asyncio.sleep(1.0)
        path.write_bytes(marked)
        seen = await rig.wait_events("del.md", 1, "NoteCreatedEvent", "NoteModifiedEvent")
        ck("delete.watch-event-published", True, seen)
        ck(
            "delete.watch-zero-chunk-event",
            [0],
            [e[2] for e in rig.of("del.md", "NoteCreatedEvent", "NoteModifiedEvent")][:1],
        )
        ck(
            "delete.watch-retires-vector-and-keyword",
            LINEAGE_ABSENT,
            lineage_exposure(w, LT_DELETE),
        )
        ck("delete.watch-hash-tracks-file", lineage_hash(path), rig.hash_of("del.md"))
    finally:
        rig.stop()

    recall_gone = lineage_replies(await w.harness.command(f"/recall {LT_DELETE}", LINEAGE_ACTOR))
    joined = "\n".join(recall_gone)
    search_session = w.handlers._search_sessions.get(LINEAGE_ACTOR)
    hits = search_session.results if search_session is not None else []
    ck(
        "delete.bot-recall-omits-retired-note",
        [[], False],
        [
            [
                hit["metadata"]["note_path"]
                for hit in hits
                if hit["metadata"]["note_path"] == "del.md" or LT_DELETE in hit["content"]
            ],
            "Quay Ledger" in joined,
        ],
    )
    ck("delete.final-file-bytes-unchanged", digest(marked), digest(path.read_bytes()))
    ck("delete.final-lineage-single-edge", ["deleted"], lineage_edges(w, "del.md"))


async def lineage_edit_scenario(env: SimpleNamespace) -> None:
    """Acceptance 2: two real edit flows keep history, only the latest claim is retrievable."""
    ck = env.ck
    original = f"Initial statement {LT_ORIG} about tide tables."
    first = f"Revised statement {LT_V1} about tide tables."
    second = (
        f"Final statement {LT_V2} about tide tables.\n\n"
        "> [!superseded] Authored quotation \u2014 2020-01-01\n"
        f"> Quoted outside text {LT_AUTHOR}.\n\n"
        f"Closing remark {LT_CLOSE}."
    )
    notes = {
        "edit.md": lineage_note("Tide Tables", original),
        "other.md": lineage_note("Other Notes", f"Control record: {LT_KEEP}."),
    }
    w = lineage_world(env.home, "edits", notes, env.llm, env.closers)
    env.worlds.append(w)
    path = w.vault / "edit.md"
    lineage_index_full(w)
    ck("edit.baseline-original-retrievable", LINEAGE_FOUND, lineage_found(w, LT_ORIG))
    calls_before = len(env.llm.calls)

    prompt, reply = await lineage_edit(w, "edit.md", first, "revise the statement")
    ck("edit.first-reported-no-failure", False, lineage_failure_in(prompt + reply))
    ck("edit.first-original-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_ORIG))
    ck("edit.first-latest-retrievable", LINEAGE_FOUND, lineage_found(w, LT_V1))
    ck(
        "edit.first-index-ids-match-projection",
        lineage_expected_ids(w, "edit.md"),
        lineage_ids(w, "edit.md"),
    )

    prompt, reply = await lineage_edit(w, "edit.md", second, "rewrite with a quotation")
    after_second = path.read_bytes()
    ck("edit.second-reported-no-failure", False, lineage_failure_in(prompt + reply))
    ck("edit.second-original-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_ORIG))
    ck("edit.second-first-revision-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_V1))
    ck("edit.second-latest-retrievable", LINEAGE_FOUND, lineage_found(w, LT_V2))
    ck("edit.authored-lookalike-in-current-text-stays", LINEAGE_FOUND, lineage_found(w, LT_AUTHOR))
    ck("edit.text-below-authored-lookalike-stays", LINEAGE_FOUND, lineage_found(w, LT_CLOSE))
    ck(
        "edit.second-index-ids-match-projection",
        lineage_expected_ids(w, "edit.md"),
        lineage_ids(w, "edit.md"),
    )
    ck("edit.scripted-llm-called-per-edit", 2, len(env.llm.calls) - calls_before)
    ck("edit.lineage-has-two-edited-edges", ["edited", "edited"], lineage_edges(w, "edit.md"))

    def history_kept() -> dict[str, list[bool]]:
        text = path.read_text(encoding="utf-8")
        content = w.parser.parse_file(path).content
        wanted = (original, first, second)
        return {
            "file_text": [part in text for part in wanted],
            "note_content": [part in content for part in wanted],
        }

    ck(
        "edit.raw-file-keeps-original-v1-v2",
        {"file_text": [True] * 3, "note_content": [True] * 3},
        history_kept(),
    )
    read = "\n".join(lineage_replies(await w.harness.command("/read edit.md", LINEAGE_ACTOR)))
    ck(
        "edit.bot-read-shows-complete-history",
        [True, True, True],
        [token in read for token in (LT_ORIG, LT_V1, LT_V2)],
    )

    stale = [
        f"note: Tide Tables (type: fleeting)\n\n{original}",
        f"note: Tide Tables (type: fleeting)\n\n{first}",
        f"note: Tide Tables (type: fleeting)\n\nlonger stale tail {LT_ORIG} {LT_V1}",
    ]

    def current_ok(surface: str) -> None:
        ck(f"edit.{surface}-original-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_ORIG))
        ck(f"edit.{surface}-first-revision-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_V1))
        ck(f"edit.{surface}-latest-retrievable", LINEAGE_FOUND, lineage_found(w, LT_V2))
        ck(
            f"edit.{surface}-ids-exactly-projection",
            lineage_expected_ids(w, "edit.md"),
            lineage_ids(w, "edit.md"),
        )

    lineage_seed_stale(w, "edit.md", stale)
    ck("edit.seeded-stale-visible-before-incremental", LINEAGE_FOUND, lineage_found(w, LT_ORIG))
    n = w.store.index_single_note(w.parser.parse_file(path), w.parser)
    ck(
        "edit.incremental-returns-projection-chunk-count",
        len(lineage_expected_ids(w, "edit.md")["vector"]),
        n,
    )
    current_ok("incremental")

    lineage_seed_stale(w, "edit.md", stale)
    n = lineage_index_full(w)
    ck("edit.api-full-returns-active-chunk-count", True, isinstance(n, int) and n >= 2)
    current_ok("full-api")

    lineage_seed_stale(w, "edit.md", stale)
    cli = await lineage_cli_index(w)
    ck("edit.cli-full-exit-code", 0, cli.exit_code)
    current_ok("full-cli")

    lineage_seed_stale(w, "edit.md", stale)
    rig = LineageDirectWatch(w)
    try:
        await asyncio.sleep(1.0)
        path.write_bytes(after_second)
        ck(
            "edit.watch-event-published",
            True,
            await rig.wait_events("edit.md", 1, "NoteCreatedEvent", "NoteModifiedEvent"),
        )
        current_ok("watch")
        ck("edit.watch-hash-tracks-file", lineage_hash(path), rig.hash_of("edit.md"))
    finally:
        rig.stop()

    ck(
        "edit.indexing-never-rewrote-canonical-file",
        digest(after_second),
        digest(path.read_bytes()),
    )
    ck(
        "edit.history-still-complete-after-all-surfaces",
        {"file_text": [True] * 3, "note_content": [True] * 3},
        history_kept(),
    )
    ck("edit.lineage-unchanged-by-indexing", ["edited", "edited"], lineage_edges(w, "edit.md"))


LT_GONE = "wharfgoneclaim"
LT_STAY = "wharfstayclaim"
LT_SHOAL = ("shoalpartone", "shoalparttwo", "shoalpartthree")
LT_REEF = ("reefpartone", "reefparttwo", "reefpartthree")
LT_FOLD = "buoyoldclaim"
LT_FNEW = "buoynewclaim"
LT_FEDIT = "buoyeditedclaim"
LT_GOLD = "buoygoldclaim"
LT_GNEW = "buoygnewclaim"
LT_GLAST = "buoyglastclaim"
LT_HGONE = "buoyhgoneclaim"
LT_NEWNOTE = "buoyfreshnoteclaim"
LT_KILL = "buoykillclaim"


def lineage_parts(tokens: tuple[str, ...]) -> str:
    return "\n\n".join(f"## Part {i}\n\n{token} detail line." for i, token in enumerate(tokens, 1))


def lineage_try(call: Any) -> tuple[Exception | None, Any]:
    try:
        return None, call()
    except Exception as exc:
        return exc, None


def lineage_failure_shape(exc: Exception | None) -> dict[str, Any]:
    """External shape of an index failure: explicit type, operation, stage, applied stages."""
    if exc is None:
        return {"type": None}
    return {
        "type": type(exc).__name__,
        "operation": getattr(exc, "operation", None),
        "stage": str(getattr(exc, "stage", None)),
        "completed": [str(stage) for stage in getattr(exc, "completed_stages", ())],
        "cause": type(exc.__cause__).__name__ if exc.__cause__ else None,
    }


def lineage_siblings(world: SimpleNamespace) -> dict[str, Any]:
    """State the indexes must never reconcile: lineage edges and non-configured indexes."""
    from vaultmind.graph.evolution import LineageStore

    other = lineage_chroma_client(world).get_collection("lineage_other_derived")
    con = sqlite3.connect(f"file:{world.data / 'other_bm25.db'}?mode=ro", uri=True)
    try:
        other_keyword = sorted(str(r[0]) for r in con.execute("SELECT chunk_id FROM fts_chunks"))
    finally:
        con.close()
    store = LineageStore(world.data / "lineage.json")
    return {
        "lineage": [(e.edge_id, e.event) for e in store.get_lineage("stay.md")],
        "other_collection": sorted(other.get()["ids"]),
        "other_keyword": other_keyword,
    }


async def lineage_reconcile_scenario(env: SimpleNamespace) -> None:
    """Acceptance 3: file removal, empty snapshot (configured indexes only), shorter revision."""
    from vaultmind.indexer.bm25 import BM25Index

    ck = env.ck
    notes = {
        "gone.md": lineage_note("Wharf Gone", f"Retired wharf memo {LT_GONE}."),
        "stay.md": lineage_note("Wharf Stay", f"Standing wharf memo {LT_STAY}."),
        "long.md": lineage_note("Shoal Log", lineage_parts(LT_SHOAL)),
        "long2.md": lineage_note("Reef Log", lineage_parts(LT_REEF)),
    }
    w = lineage_world(env.home, "reconcile", notes, env.llm, env.closers)
    env.worlds.append(w)
    other_text = "other collection evidence"
    other = lineage_chroma_client(w).get_or_create_collection("lineage_other_derived")
    other.upsert(
        ids=["other::0"],
        documents=[other_text],
        embeddings=[DeterministicEmbedder().embed_query(other_text)],  # type: ignore[arg-type]
    )
    other_keyword = BM25Index(w.data / "other_bm25.db")
    env.closers.append(other_keyword.close)
    other_keyword.upsert("other::0", "other.md", "Other", "other keyword evidence")
    w.lineage.record("stay.md", "edited", "synthetic prior history fixture")
    siblings = lineage_siblings(w)
    ck(
        "reconcile.sibling-fixture-present",
        [1, ["other::0"], ["other::0"]],
        [
            len(siblings["lineage"]),
            siblings["other_collection"],
            siblings["other_keyword"],
        ],
    )

    cli = await lineage_cli_index(w)
    ck("reconcile.baseline-cli-full-exit-code", 0, cli.exit_code)
    ck(
        "reconcile.baseline-gone-indexed",
        lineage_expected_ids(w, "gone.md"),
        lineage_ids(w, "gone.md"),
    )
    shoal_before = lineage_ids(w, "long.md")
    ck(
        "reconcile.baseline-multi-chunk-note",
        [lineage_expected_ids(w, "long.md"), True],
        [
            shoal_before,
            len(shoal_before["vector"]) >= 3,
        ],
    )

    (w.vault / "gone.md").unlink()
    cli = await lineage_cli_index(w)
    ck("reconcile.removal-cli-full-exit-code", 0, cli.exit_code)
    ck("reconcile.removed-file-rows-gone", {"vector": [], "keyword": []}, lineage_ids(w, "gone.md"))
    ck("reconcile.removed-file-text-unretrievable", LINEAGE_ABSENT, lineage_exposure(w, LT_GONE))
    ck("reconcile.surviving-note-retrievable", LINEAGE_FOUND, lineage_found(w, LT_STAY))
    ck("reconcile.removal-leaves-siblings-intact", siblings, lineage_siblings(w))

    (w.vault / "long.md").write_text(
        lineage_note("Shoal Log", lineage_parts(LT_SHOAL[:1])), encoding="utf-8"
    )
    count = lineage_index_full(w)
    ck("reconcile.shorter-full-returns-int", "int", type(count).__name__)
    ck(
        "reconcile.shorter-full-ids-exactly-projection",
        lineage_expected_ids(w, "long.md"),
        lineage_ids(w, "long.md"),
    )
    ck("reconcile.shorter-full-keeps-one-chunk", 1, len(lineage_ids(w, "long.md")["vector"]))
    for stale in LT_SHOAL[1:]:
        ck(f"reconcile.shorter-full-{stale}-gone", LINEAGE_ABSENT, lineage_exposure(w, stale))
    ck("reconcile.shorter-full-current-part-found", LINEAGE_FOUND, lineage_found(w, LT_SHOAL[0]))
    ck(
        "reconcile.shorter-full-other-note-ids",
        lineage_expected_ids(w, "stay.md"),
        lineage_ids(w, "stay.md"),
    )

    reef = w.vault / "long2.md"
    reef.write_text(lineage_note("Reef Log", lineage_parts(LT_REEF[:1])), encoding="utf-8")
    count = w.store.index_single_note(w.parser.parse_file(reef), w.parser)
    ck("reconcile.shorter-incremental-returns-chunk-count", 1, count)
    ck(
        "reconcile.shorter-incremental-ids-exactly-projection",
        lineage_expected_ids(w, "long2.md"),
        lineage_ids(w, "long2.md"),
    )
    for stale in LT_REEF[1:]:
        ck(
            f"reconcile.shorter-incremental-{stale}-gone",
            LINEAGE_ABSENT,
            lineage_exposure(w, stale),
        )
    ck(
        "reconcile.incremental-leaves-other-note-rows",
        lineage_expected_ids(w, "long.md"),
        lineage_ids(w, "long.md"),
    )

    files_before = lineage_vault_digests(w)
    empty = w.store.index_notes([], w.parser)
    ck("reconcile.empty-snapshot-returns-zero-int", [0, "int"], [empty, type(empty).__name__])
    ck(
        "reconcile.empty-snapshot-clears-configured-indexes",
        {"vector": {}, "keyword": {}},
        lineage_state(w),
    )
    ck(
        "reconcile.empty-snapshot-searches-return-nothing",
        [[], []],
        [w.store.search(LT_STAY, n_results=5), w.store.hybrid_search(LT_STAY, n_results=5)],
    )
    ck("reconcile.empty-snapshot-never-touches-vault-files", files_before, lineage_vault_digests(w))
    ck("reconcile.empty-snapshot-leaves-lineage-and-other-indexes", siblings, lineage_siblings(w))

    cli = await lineage_cli_index(w)
    ck("reconcile.rebuild-from-files-exit-code", 0, cli.exit_code)
    ck("reconcile.rebuild-restores-derived-rows", LINEAGE_FOUND, lineage_found(w, LT_STAY))

    for md in sorted(w.vault.glob("*.md")):
        md.unlink()
    cli = await lineage_cli_index(w)
    ck("reconcile.all-files-removed-cli-exit-code", 0, cli.exit_code)
    ck(
        "reconcile.all-files-removed-cli-reports-zero",
        True,
        "Indexed 0 chunks from 0 notes" in re.sub(r"\s+", " ", cli.output),
    )
    ck(
        "reconcile.all-files-removed-derived-indexes-empty",
        {"vector": {}, "keyword": {}},
        lineage_state(w),
    )
    ck("reconcile.all-files-removed-leaves-siblings", siblings, lineage_siblings(w))


async def lineage_failure_scenario(env: SimpleNamespace) -> None:
    """Acceptance 4+5: embedder failure keeps last-good evidence; cross-index failure explicit."""
    from vaultmind.indexer.store import IndexStage

    ck = env.ck
    notes = {
        "f.md": lineage_note("Fault Subject", f"Baseline buoy memo {LT_FOLD}."),
        "g.md": lineage_note("Gauge Subject", f"Gauge baseline {LT_GOLD}."),
        "h.md": lineage_note("Harbor Subject", f"Soon removed memo {LT_HGONE}."),
    }
    w = lineage_world(env.home, "faults", notes, env.llm, env.closers)
    env.worlds.append(w)
    lineage_index_full(w)
    good = lineage_state(w)
    ck(
        "fault.baseline-all-notes-indexed",
        True,
        all(lineage_ids(w, rel)["vector"] for rel in ("f.md", "g.md", "h.md")),
    )

    f_path = w.vault / "f.md"
    f_path.write_text(
        lineage_note("Fault Subject", f"Replacement buoy memo {LT_FNEW}."), encoding="utf-8"
    )
    f_bytes = f_path.read_bytes()
    w.embedder.outage = True
    batches = w.embedder.batches
    exc, _ = lineage_try(lambda: w.store.index_single_note(w.parser.parse_file(f_path), w.parser))
    shape = lineage_failure_shape(exc)
    ck(
        "fault.embedder-incremental-explicit-failure",
        ["IndexUpdateError", "index_single_note", IndexStage.EMBED.value, [], "RuntimeError"],
        [
            shape["type"],
            shape.get("operation"),
            shape.get("stage"),
            shape.get("completed"),
            shape.get("cause"),
        ],
    )
    ck("fault.embedder-incremental-hit-boundary", True, w.embedder.batches > batches)
    ck("fault.embedder-incremental-vector-keyword-unchanged", good, lineage_state(w))
    ck("fault.embedder-incremental-last-good-retrievable", LINEAGE_FOUND, lineage_found(w, LT_FOLD))
    ck(
        "fault.embedder-incremental-new-claim-not-indexed",
        LINEAGE_ABSENT,
        lineage_exposure(w, LT_FNEW),
    )
    ck("fault.embedder-incremental-canonical-file-kept", f_bytes, f_path.read_bytes())

    (w.vault / "h.md").unlink()
    (w.vault / "fresh.md").write_text(
        lineage_note("Fresh Subject", f"Fresh note memo {LT_NEWNOTE}."), encoding="utf-8"
    )
    exc, _ = lineage_try(lambda: lineage_index_full(w))
    shape = lineage_failure_shape(exc)
    ck(
        "fault.embedder-full-explicit-failure",
        ["IndexUpdateError", "index_notes", IndexStage.EMBED.value, []],
        [shape["type"], shape.get("operation"), shape.get("stage"), shape.get("completed")],
    )
    ck("fault.embedder-full-no-destructive-write-before-failure", good, lineage_state(w))
    ck(
        "fault.embedder-full-removed-file-rows-still-last-good",
        LINEAGE_FOUND,
        lineage_found(w, LT_HGONE),
    )
    ck(
        "fault.embedder-full-hybrid-evidence-intact",
        True,
        bool(w.store.hybrid_search(LT_FOLD, n_results=5)),
    )

    cli = await lineage_cli_index(w)
    ck("fault.embedder-full-cli-exits-nonzero", True, cli.exit_code != 0)
    ck("fault.embedder-full-cli-prints-no-success", False, bool(LINEAGE_SUCCESS.search(cli.output)))
    ck("fault.embedder-full-cli-state-unchanged", good, lineage_state(w))

    prompt, reply = await lineage_edit(w, "f.md", f"Edited memo {LT_FEDIT}.", "tweak the memo")
    f_text = f_path.read_text(encoding="utf-8")
    ck(
        "fault.edit-handler-reports-failure",
        [True, False],
        [lineage_failure_in(reply), lineage_success_in(reply)],
    )
    ck(
        "fault.edit-handler-file-keeps-new-and-prior-text",
        [True, True],
        [LT_FEDIT in f_text, LT_FNEW in f_text],
    )
    ck("fault.edit-handler-lineage-edge-kept", ["edited"], lineage_edges(w, "f.md"))
    ck("fault.edit-handler-index-unchanged", good, lineage_state(w))

    w.embedder.outage = False
    count = lineage_index_full(w)
    ck("fault.recovery-full-returns-int", "int", type(count).__name__)
    ck(
        "fault.recovery-edited-claim-only",
        [LINEAGE_FOUND, LINEAGE_ABSENT, LINEAGE_ABSENT],
        [
            lineage_found(w, LT_FEDIT),
            lineage_exposure(w, LT_FOLD),
            lineage_exposure(w, LT_FNEW),
        ],
    )
    ck(
        "fault.recovery-removed-file-gone-fresh-file-present",
        [LINEAGE_ABSENT, LINEAGE_FOUND],
        [
            lineage_exposure(w, LT_HGONE),
            lineage_found(w, LT_NEWNOTE),
        ],
    )

    g_path = w.vault / "g.md"
    g_path.write_text(lineage_note("Gauge Subject", f"Gauge revision {LT_GNEW}."), encoding="utf-8")
    g_bytes = g_path.read_bytes()
    w.fault.arm("replace_note")
    exc, _ = lineage_try(lambda: w.store.index_single_note(w.parser.parse_file(g_path), w.parser))
    shape = lineage_failure_shape(exc)
    ck(
        "fault.cross-index-incremental-explicit-failure",
        [
            "IndexUpdateError",
            "index_single_note",
            IndexStage.KEYWORD_REPLACE.value,
            "OperationalError",
        ],
        [shape["type"], shape.get("operation"), shape.get("stage"), shape.get("cause")],
    )
    ck(
        "fault.cross-index-incremental-reports-vector-stage-applied",
        True,
        IndexStage.VECTOR_UPSERT.value in shape.get("completed", []),
    )
    ck("fault.cross-index-incremental-fault-fired-once", ["replace_note"], w.fault.raised[:1])
    ck("fault.cross-index-incremental-canonical-file-kept", g_bytes, g_path.read_bytes())
    ck("fault.cross-index-incremental-no-lineage-invented", [], lineage_edges(w, "g.md"))
    w.fault.disarm()
    count = w.store.index_single_note(w.parser.parse_file(g_path), w.parser)
    ck(
        "fault.cross-index-retry-returns-chunk-count",
        len(lineage_expected_ids(w, "g.md")["vector"]),
        count,
    )
    ck(
        "fault.cross-index-retry-converges-both-indexes",
        [LINEAGE_FOUND, LINEAGE_ABSENT],
        [
            lineage_found(w, LT_GNEW),
            lineage_exposure(w, LT_GOLD),
        ],
    )

    g_path.write_text(lineage_note("Gauge Subject", f"Gauge final {LT_GLAST}."), encoding="utf-8")
    g_bytes = g_path.read_bytes()
    w.fault.arm("replace_all")
    exc, _ = lineage_try(lambda: lineage_index_full(w))
    shape = lineage_failure_shape(exc)
    ck(
        "fault.cross-index-full-explicit-failure",
        ["IndexUpdateError", "index_notes", IndexStage.KEYWORD_REPLACE.value],
        [shape["type"], shape.get("operation"), shape.get("stage")],
    )
    with lineage_readonly_keyword(w):
        cli = await lineage_cli_index(w)
    ck("fault.cross-index-full-cli-exits-nonzero", True, cli.exit_code != 0)
    ck(
        "fault.cross-index-full-cli-prints-no-success",
        False,
        bool(LINEAGE_SUCCESS.search(cli.output)),
    )
    ck("fault.cross-index-full-canonical-file-kept", g_bytes, g_path.read_bytes())
    w.fault.disarm()
    cli = await lineage_cli_index(w)
    ck("fault.cross-index-full-cli-retry-exit-code", 0, cli.exit_code)
    ck(
        "fault.cross-index-full-retry-converges",
        [LINEAGE_FOUND, LINEAGE_ABSENT],
        [
            lineage_found(w, LT_GLAST),
            lineage_exposure(w, LT_GNEW),
        ],
    )

    kill = w.vault / "kill.md"
    kill_body = f"Doomed buoy memo {LT_KILL}."
    kill.write_text(lineage_note("Doomed Subject", kill_body), encoding="utf-8")
    lineage_index_full(w)
    w.fault.arm("delete_note")
    replies = await lineage_delete(w, "kill.md")
    marked = kill.read_bytes()
    ck(
        "fault.delete-handler-reports-failure",
        [True, False],
        [lineage_failure_in(replies), lineage_success_in(replies)],
    )
    ck(
        "fault.delete-handler-keeps-original-and-marks-file",
        [True, ""],
        [
            kill_body in marked.decode(),
            w.parser.parse_file(kill).current_content().strip(),
        ],
    )
    ck("fault.delete-handler-lineage-edge-kept", ["deleted"], lineage_edges(w, "kill.md"))
    ck("fault.delete-handler-keyword-fault-fired", True, "delete_note" in w.fault.raised)
    w.fault.disarm()
    replay = await lineage_delete(w, "kill.md")
    ck("fault.delete-replay-retries-retirement-without-failure", False, lineage_failure_in(replay))
    ck("fault.delete-replay-retired-text-absent", LINEAGE_ABSENT, lineage_exposure(w, LT_KILL))
    ck(
        "fault.delete-replay-keeps-file-and-single-edge",
        [marked, ["deleted"]],
        [
            kill.read_bytes(),
            lineage_edges(w, "kill.md"),
        ],
    )


async def lineage_watch_scenario(env: SimpleNamespace, stability: bool) -> None:
    """Acceptance 5: real filesystem events; failures keep hash/events truthful and retry."""
    from vaultmind.vault.events import NoteDeletedEvent

    ck = env.ck
    mode = "stable" if stability else "direct"
    label = f"watch-{mode}"
    changed = ("NoteCreatedEvent", "NoteModifiedEvent")
    first = f"Watched baseline {LT_FOLD}."
    second = f"Watched revision {LT_FNEW}."
    third = f"Watched final {LT_GLAST}."
    w = lineage_world(
        env.home,
        label,
        {"w.md": lineage_note("Watch Subject", first)},
        env.llm,
        env.closers,
        stability=stability,
    )
    env.worlds.append(w)
    path = w.vault / "w.md"
    rig = LineageDirectWatch(w)
    try:
        await asyncio.sleep(1.0)
        baseline = path.read_bytes()
        path.write_bytes(baseline)
        ck(f"{label}.first-event-indexes-note", True, await rig.wait_events("w.md", 1, *changed))
        ck(f"{label}.first-event-hash-recorded", lineage_hash(path), rig.hash_of("w.md"))
        ck(f"{label}.first-event-baseline-retrievable", LINEAGE_FOUND, lineage_found(w, LT_FOLD))
        good_hash, good_state = rig.hash_of("w.md"), lineage_state(w)
        published = len(rig.events)

        w.fault.arm("replace_note")
        path.write_text(lineage_note("Watch Subject", second), encoding="utf-8")
        failed_bytes = path.read_bytes()
        ck(f"{label}.cross-index-failure-attempted", True, await rig.wait_fault(1))
        await rig.settle()
        ck(f"{label}.failure-publishes-no-event", published, len(rig.events))
        ck(f"{label}.failure-keeps-last-good-hash", good_hash, rig.hash_of("w.md"))
        ck(
            f"{label}.failure-keeps-canonical-file",
            True,
            second in path.read_text(encoding="utf-8"),
        )
        w.fault.disarm()

        path.write_bytes(failed_bytes)
        ck(f"{label}.same-content-event-retries", True, await rig.wait_events("w.md", 2, *changed))
        retried = rig.of("w.md", *changed)[-1]
        ck(
            f"{label}.retry-event-reports-chunks",
            len(lineage_expected_ids(w, "w.md")["vector"]),
            retried[2],
        )
        ck(f"{label}.retry-advances-hash-after-success", lineage_hash(path), rig.hash_of("w.md"))
        ck(
            f"{label}.retry-converges-both-indexes",
            [LINEAGE_FOUND, LINEAGE_ABSENT],
            [
                lineage_found(w, LT_FNEW),
                lineage_exposure(w, LT_FOLD),
            ],
        )
        good_state, published = lineage_state(w), len(rig.events)
        batches = w.embedder.batches
        path.write_bytes(failed_bytes)
        await rig.settle()
        ck(
            f"{label}.identical-successful-content-not-reindexed",
            [published, batches],
            [len(rig.events), w.embedder.batches],
        )

        if not stability:
            w.fault.arm("replace_note")
            faults = len(w.fault.raised)
            path.write_text(lineage_note("Watch Subject", third), encoding="utf-8")
            ck(f"{label}.revert-case-failure-attempted", True, await rig.wait_fault(faults + 1))
            await rig.settle()
            w.fault.disarm()
            path.write_bytes(failed_bytes)
            ck(
                f"{label}.bytes-reverting-to-last-good-hash-still-retry",
                True,
                await rig.wait_events("w.md", 3, *changed),
            )
            ck(
                f"{label}.revert-retry-reconciles-partial-vector-write",
                [LINEAGE_ABSENT, LINEAGE_FOUND, lineage_hash(path)],
                [
                    lineage_exposure(w, LT_GLAST),
                    lineage_found(w, LT_FNEW),
                    rig.hash_of("w.md"),
                ],
            )
        if stability:
            good_hash = rig.hash_of("w.md")
            w.embedder.outage = True
            failures = w.embedder.failures
            path.write_text(lineage_note("Watch Subject", third), encoding="utf-8")
            third_bytes = path.read_bytes()
            await lineage_until(lambda: w.embedder.failures > failures)
            await rig.settle()
            ck(f"{label}.embedder-outage-publishes-no-event", published, len(rig.events))
            ck(f"{label}.embedder-outage-keeps-last-good-hash", good_hash, rig.hash_of("w.md"))
            ck(f"{label}.embedder-outage-keeps-last-good-rows", good_state, lineage_state(w))
            w.embedder.outage = False
            path.write_bytes(third_bytes)
            ck(
                f"{label}.embedder-recovery-event-retries",
                True,
                await rig.wait_events("w.md", 3, *changed),
            )
            ck(
                f"{label}.embedder-recovery-converges",
                [LINEAGE_FOUND, LINEAGE_ABSENT],
                [
                    lineage_found(w, LT_GLAST),
                    lineage_exposure(w, LT_FNEW),
                ],
            )

        path.unlink()
        ck(
            f"{label}.filesystem-removal-event",
            True,
            await lineage_until(lambda: any(e[0] == "NoteDeletedEvent" for e in rig.of("w.md"))),
        )
        ck(f"{label}.removal-clears-rows", {"vector": [], "keyword": []}, lineage_ids(w, "w.md"))
        ck(f"{label}.removal-drops-hash", None, rig.hash_of("w.md"))

        path.write_text(lineage_note("Watch Subject", first), encoding="utf-8")
        await rig.wait_events("w.md", 4, *changed)
        w.fault.disarm()
        w.fault.arm("delete_note")
        deleted_before = len(rig.of("w.md", NoteDeletedEvent.__name__))
        path.unlink()
        ck(f"{label}.removal-failure-attempted", True, await rig.wait_fault(1))
        await rig.settle()
        ck(
            f"{label}.removal-failure-publishes-no-event",
            deleted_before,
            len(rig.of("w.md", NoteDeletedEvent.__name__)),
        )
        w.fault.disarm()
        lineage_index_full(w)
        ck(
            f"{label}.full-reindex-reconciles-failed-removal",
            {"vector": [], "keyword": []},
            lineage_ids(w, "w.md"),
        )
    finally:
        rig.stop()


LT_AUTH_A = "dredgeopeningauth"
LT_AUTH_B = "dredgeclosingauth"
LT_ASIDE = "dredgeauthoraside"
LT_QUOTE = "dredgequotedtext"
LT_QUOTE_AFTER = "dredgeafterquote"
LT_UNPAIRED = "dredgeunpairedmark"
LT_LEG_DEL = "dredgelegacydel"
LT_LEG_O = "dredgelegacyorig"
LT_LEG_1 = "dredgelegacyone"
LT_LEG_2 = "dredgelegacytwo"
LT_MIX = "dredgelegacymix"
LT_AMB_DEL = "dredgeambiguousdel"
LT_AMB_V1 = "dredgeambiguousvone"
LT_AMB_NEW = "dredgeambiguousnew"
LT_AMB_X1 = "dredgeambiguousxone"


async def lineage_grammar_scenario(env: SimpleNamespace) -> None:
    """Acceptance 6: authored lookalikes stay evidence, exact legacy works, ambiguity fails."""
    from datetime import UTC, datetime

    import frontmatter

    from vaultmind.vault.projection import AmbiguousHistoryError

    ck = env.ck
    callout = "> [!superseded] Replaced by newer analysis \u2014 2024-05-01"
    bot_delete_quote = (
        "> [!superseded] Deleted via bot \u2014 2026-10-01\n"
        "> Requested via Telegram `/delete` \u2014 content preserved below."
    )
    notes = {
        "authored.md": lineage_note(
            "Authored Dredge Log",
            f"Opening evidence {LT_AUTH_A}.\n\n{callout}\n> Author aside kept {LT_ASIDE}.\n\n"
            f"Closing evidence {LT_AUTH_B}.",
        ),
        "quote.md": lineage_note(
            "Quoted Dredge Log",
            f"Quote of bot output {LT_QUOTE}:\n\n{bot_delete_quote}\n\n"
            "> [!superseded] Edited via bot \u2014 2026-10-01\n"
            "> Instruction: quoted instruction\n\n"
            f"Still current text {LT_QUOTE_AFTER}.",
        ),
        "unpaired.md": lineage_note(
            "Unpaired Dredge Log",
            "<!-- vaultmind:supersede id=0123456789abcdef0123456789abcdef -->\n"
            f"Unpaired marker text {LT_UNPAIRED}.",
        ),
    }
    w = lineage_world(env.home, "grammar", notes, env.llm, env.closers)
    env.worlds.append(w)
    moment_1 = datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=UTC)
    moment_2 = datetime(2026, 10, 2, 12, 0, 0, 654321, tzinfo=UTC)
    legacy_del = w.vault / "legacy-del.md"
    legacy_del.write_text(
        lineage_note("Legacy Gone", f"Legacy retired memo {LT_LEG_DEL}."), encoding="utf-8"
    )
    lineage_legacy_delete(legacy_del, moment_1)
    legacy_edit = w.vault / "legacy-edit.md"
    legacy_edit.write_text(
        lineage_note("Legacy Edited", f"Legacy original {LT_LEG_O}."), encoding="utf-8"
    )
    lineage_legacy_edit(
        legacy_edit, f"Legacy first revision {LT_LEG_1}.", "first rewrite", moment_1
    )
    lineage_legacy_edit(
        legacy_edit, f"Legacy second revision {LT_LEG_2}.", "second rewrite", moment_2
    )
    legacy_text = legacy_del.read_text(encoding="utf-8")
    ck(
        "grammar.legacy-fixtures-written-by-base-commit-writer",
        [True, True, 2],
        [
            "bot:delete" in legacy_text and "Deleted via bot" in legacy_text,
            "Edited via bot" in legacy_edit.read_text(encoding="utf-8"),
            legacy_edit.read_text(encoding="utf-8").count("[!superseded] Edited via bot"),
        ],
    )

    count = lineage_index_full(w)
    ck("grammar.full-index-returns-int", "int", type(count).__name__)
    authored = (LT_AUTH_A, LT_ASIDE, LT_AUTH_B, LT_QUOTE, LT_QUOTE_AFTER, LT_UNPAIRED)
    ck(
        "grammar.authored-lookalikes-remain-evidence",
        dict.fromkeys(authored, LINEAGE_FOUND),
        {token: lineage_found(w, token) for token in authored},
    )
    ck(
        "grammar.authored-current-projection-equals-whole-body",
        [True, True, True],
        [
            w.parser.parse_file(w.vault / rel).current_content().strip()
            == w.parser.parse_file(w.vault / rel).content.strip()
            for rel in ("authored.md", "quote.md", "unpaired.md")
        ],
    )

    del_note = w.parser.parse_file(legacy_del)
    ck("grammar.legacy-delete-retired-everywhere", LINEAGE_ABSENT, lineage_exposure(w, LT_LEG_DEL))
    ck(
        "grammar.legacy-delete-projection-empty-raw-intact",
        ["", True],
        [del_note.current_content().strip(), LT_LEG_DEL in del_note.content],
    )
    edit_note = w.parser.parse_file(legacy_edit)
    current = edit_note.current_content()
    ck(
        "grammar.legacy-edit-projection-is-newest-only",
        [True, False, False, True, True, True],
        [
            LT_LEG_2 in current,
            LT_LEG_O in current,
            LT_LEG_1 in current,
            LT_LEG_O in edit_note.content,
            LT_LEG_1 in edit_note.content,
            LT_LEG_2 in edit_note.content,
        ],
    )
    ck("grammar.legacy-edit-newest-retrievable", LINEAGE_FOUND, lineage_found(w, LT_LEG_2))
    ck(
        "grammar.legacy-edit-older-text-retired",
        [LINEAGE_ABSENT, LINEAGE_ABSENT],
        [lineage_exposure(w, LT_LEG_O), lineage_exposure(w, LT_LEG_1)],
    )
    ck(
        "grammar.legacy-edit-ids-match-projection",
        lineage_expected_ids(w, "legacy-edit.md"),
        lineage_ids(w, "legacy-edit.md"),
    )
    cli = await lineage_cli_index(w)
    ck("grammar.legacy-cli-full-exit-code", 0, cli.exit_code)
    ck(
        "grammar.legacy-cli-full-same-evidence",
        [LINEAGE_ABSENT, LINEAGE_FOUND],
        [lineage_exposure(w, LT_LEG_1), lineage_found(w, LT_LEG_2)],
    )

    legacy_bytes = legacy_del.read_bytes()
    calls_before = len(env.llm.calls)
    prompt, reply = await lineage_edit(w, "legacy-del.md", f"Revived {LT_MIX}.", "try to revive")
    ck(
        "grammar.edit-of-bot-deleted-note-refused-before-llm",
        [[], True, 0, legacy_bytes, []],
        [
            reply,
            bool(re.search(r"not applied|fail|error", " ".join(prompt), re.IGNORECASE)),
            len(env.llm.calls) - calls_before,
            legacy_del.read_bytes(),
            lineage_edges(w, "legacy-del.md"),
        ],
    )

    prompt, reply = await lineage_edit(
        w, "legacy-edit.md", f"Mixed history revision {LT_MIX}.", "continue the history"
    )
    mixed = legacy_edit.read_text(encoding="utf-8")
    ck("grammar.mixed-edit-reported-no-failure", False, lineage_failure_in(prompt + reply))
    ck(
        "grammar.mixed-edit-newest-new-boundary-wins",
        [LINEAGE_FOUND, LINEAGE_ABSENT, LINEAGE_ABSENT, LINEAGE_ABSENT],
        [
            lineage_found(w, LT_MIX),
            lineage_exposure(w, LT_LEG_2),
            lineage_exposure(w, LT_LEG_1),
            lineage_exposure(w, LT_LEG_O),
        ],
    )
    ck(
        "grammar.mixed-edit-keeps-whole-history-on-disk",
        [True] * 4,
        [token in mixed for token in (LT_LEG_O, LT_LEG_1, LT_LEG_2, LT_MIX)],
    )
    ck("grammar.mixed-edit-one-lineage-edge", ["edited"], lineage_edges(w, "legacy-edit.md"))

    good = lineage_state(w)
    ambiguous = {
        "amb-del.md": LT_AMB_DEL,
        "amb-edit.md": LT_AMB_V1,
        "amb-new.md": LT_AMB_NEW,
        "amb-extra.md": LT_AMB_X1,
    }
    (w.vault / "amb-del.md").write_text(
        lineage_note(
            "Ambiguous Delete",
            f"Ambiguous retired memo {LT_AMB_DEL}.",
            superseded_at="[bot:delete]",
        ),
        encoding="utf-8",
    )
    amb_edit = w.vault / "amb-edit.md"
    amb_edit.write_text(lineage_note("Ambiguous Edit", "Ambiguous original."), encoding="utf-8")
    lineage_legacy_edit(amb_edit, f"Ambiguous current {LT_AMB_V1}.", "amb rewrite", moment_1)
    post = frontmatter.load(str(amb_edit))
    post.content = "\n".join(
        line
        for line in post.content.splitlines()
        if not line.startswith(("> [!superseded]", "> Instruction:"))
    )
    amb_edit.write_text(frontmatter.dumps(post), encoding="utf-8")
    stamp = "2026-10-02T12:00:00.123456+00:00"
    (w.vault / "amb-new.md").write_text(
        lineage_note(
            "Ambiguous Boundary",
            f"Malformed boundary memo {LT_AMB_NEW}.",
            superseded_at=f"['{stamp}']",
            supersede_boundaries=f"['nothex={stamp}']",
        ),
        encoding="utf-8",
    )
    amb_extra = w.vault / "amb-extra.md"
    amb_extra.write_text(lineage_note("Ambiguous Extra", "Extra original."), encoding="utf-8")
    lineage_legacy_edit(amb_extra, f"Extra current {LT_AMB_X1}.", "extra rewrite", moment_1)
    post = frontmatter.load(str(amb_extra))
    post.content = (
        "> [!superseded] Edited via bot \u2014 2026-10-03\n> Instruction: quoted\n\n" + post.content
    )
    amb_extra.write_text(frontmatter.dumps(post), encoding="utf-8")
    raw = {rel: (w.vault / rel).read_bytes() for rel in ambiguous}

    for rel, token in ambiguous.items():
        label = rel.removesuffix(".md")
        note = w.parser.parse_file(w.vault / rel)
        ck(f"grammar.{label}-raw-read-intact", True, token in note.content)
        exc, _ = lineage_try(note.current_content)
        ck(
            f"grammar.{label}-projection-reports-ambiguity",
            [True, True],
            [isinstance(exc, AmbiguousHistoryError), isinstance(exc, ValueError)],
        )
        exc, _ = lineage_try(lambda n=note: w.store.index_single_note(n, w.parser))
        ck(f"grammar.{label}-incremental-index-fails-explicitly", True, isinstance(exc, ValueError))
        ck(
            f"grammar.{label}-no-whole-text-fallback",
            [LINEAGE_ABSENT, {"vector": [], "keyword": []}],
            [lineage_exposure(w, token), lineage_ids(w, rel)],
        )
    ck("grammar.ambiguous-incremental-leaves-index-unchanged", good, lineage_state(w))

    exc, _ = lineage_try(lambda: lineage_index_full(w))
    ck("grammar.ambiguous-full-index-fails-explicitly", True, isinstance(exc, ValueError))
    ck("grammar.ambiguous-full-index-leaves-index-unchanged", good, lineage_state(w))
    cli = await lineage_cli_index(w)
    ck(
        "grammar.ambiguous-cli-fails-without-success",
        [True, False],
        [cli.exit_code != 0, bool(LINEAGE_SUCCESS.search(cli.output))],
    )
    ck("grammar.ambiguous-cli-leaves-index-unchanged", good, lineage_state(w))

    rig = LineageDirectWatch(w)
    try:
        await asyncio.sleep(1.0)
        for rel in ambiguous:
            (w.vault / rel).write_bytes(raw[rel])
        await asyncio.sleep(2.0)
        ck(
            "grammar.ambiguous-watch-publishes-nothing-and-tracks-no-hash",
            [[], [None] * len(ambiguous)],
            [
                [e for e in rig.events if e[1] in ambiguous],
                [rig.hash_of(rel) for rel in ambiguous],
            ],
        )
    finally:
        rig.stop()
    ck("grammar.ambiguous-watch-leaves-index-unchanged", good, lineage_state(w))

    calls_before = len(env.llm.calls)
    prompt, reply = await lineage_edit(w, "amb-edit.md", "unused", "edit ambiguous history")
    ck(
        "grammar.edit-of-ambiguous-note-refused-before-llm",
        [[], True, 0],
        [
            reply,
            bool(re.search(r"not applied|fail|error", " ".join(prompt), re.IGNORECASE)),
            len(env.llm.calls) - calls_before,
        ],
    )
    replies = await lineage_delete(w, "amb-del.md")
    ck(
        "grammar.delete-of-ambiguous-note-fails-explicitly",
        [True, False, [], []],
        [
            lineage_failure_in(replies),
            lineage_success_in(replies),
            lineage_edges(w, "amb-del.md"),
            lineage_edges(w, "amb-edit.md"),
        ],
    )
    ck(
        "grammar.ambiguous-files-byte-identical-throughout",
        raw,
        {rel: (w.vault / rel).read_bytes() for rel in ambiguous},
    )

    for rel in ambiguous:
        (w.vault / rel).unlink()
    count = lineage_index_full(w)
    ck(
        "grammar.removing-ambiguous-notes-unblocks-full-index",
        ["int", LINEAGE_FOUND],
        [type(count).__name__, lineage_found(w, LT_MIX)],
    )


class LineageLLM(ApprovalLLM):
    """Scripted llm.client boundary that also records the multimodal (vision) call."""

    provider_name = "scripted-lineage-acceptance"

    def complete_multimodal(
        self,
        messages: list[MultimodalMessage],
        model: str,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        from vaultmind.llm.client import LLMResponse

        self.calls.append(
            {"multimodal": True, "model": model, "response_sha256": digest(self.reply.encode())}
        )
        return LLMResponse(text=self.reply, model=model, usage={})


def lineage_photo_session(session: Any) -> None:
    """Telegram file boundary only: GetFile metadata and the downloaded bytes are synthetic."""
    from aiogram.types import File

    real = session.make_request

    async def make_request(bot: Any, method: Any, timeout: int | None = None) -> Any:
        if type(method).__name__ == "GetFile":
            data = method.model_dump(exclude_none=True)
            session.calls.append({"method": "GetFile", "data": data})
            return File(file_id=data["file_id"], file_unique_id="U1", file_path="photos/a.jpg")
        return await real(bot, method, timeout)

    async def stream_content(*args: Any, **kwargs: Any) -> Any:
        yield b"\xff\xd8\xff\xe0OFFLINE-SYNTHETIC-JPEG"

    session.make_request = make_request
    session.stream_content = stream_content


class LineagePage:
    """urllib response stand-in for the article fetch network boundary."""

    def __init__(self, title: str) -> None:
        self.title = title

    def read(self) -> bytes:
        page = f"<html><title>{self.title}</title><body>Lineage article text.</body></html>"
        return page.encode()


async def lineage_consumer_scenario(env: SimpleNamespace) -> None:
    """Every changed index_single_note consumer reports failure, keeps its saved file."""
    import logging
    from unittest.mock import patch

    from youtube_transcript_api import YouTubeTranscriptApi

    from vaultmind.mcp.server import _dispatch_tool
    from vaultmind.research.pipeline import (
        ResearchConfig,
        ResearchIndexError,
        run_research_pipeline,
    )
    from vaultmind.research.searcher import SearchResult

    ck = env.ck
    cause = "synthetic keyword-index outage"
    notes = {"keep.md": lineage_note("Consumer Keeper", f"Control record {LT_KEEP}.")}
    w = lineage_world(env.home, "consumers", notes, env.llm, env.closers)
    env.worlds.append(w)
    lineage_photo_session(w.session)
    lineage_index_full(w)
    seen: dict[tuple[str, str], tuple[list[str], list[Path]]] = {}

    def md_files() -> set[Path]:
        return set(w.vault.rglob("*.md"))

    async def probe(label: str, drive: Any) -> None:
        for mode in ("control", "fault"):
            await asyncio.sleep(1.1)
            before = md_files()
            if mode == "fault":
                w.fault.arm("*")
            try:
                replies = await drive(mode)
            finally:
                w.fault.disarm()
            created = sorted(md_files() - before)
            rels = [str(path.relative_to(w.vault)) for path in created]
            seen[(label, mode)] = (replies, created)
            joined = "\n".join(replies)
            if mode == "control":
                ck(
                    f"consumer.{label}-control-saves-indexes-and-reports-success",
                    [False, True, 1, [True]],
                    [
                        lineage_failure_in(replies),
                        lineage_success_in(replies),
                        len(created),
                        [bool(lineage_ids(w, rel)["keyword"]) for rel in rels],
                    ],
                )
            else:
                ck(
                    f"consumer.{label}-fault-reports-explicit-failure-not-success",
                    [True, False, True],
                    [lineage_failure_in(replies), lineage_success_in(replies), cause in joined],
                )
                ck(
                    f"consumer.{label}-fault-keeps-one-saved-file-and-names-it",
                    [1, True],
                    [len(created), all(rel in joined for rel in rels)],
                )

    async def capture_text(mode: str) -> list[str]:
        return lineage_replies(await w.harness.text(f"note: lineage {mode} capture", LINEAGE_ACTOR))

    async def capture_url(mode: str) -> list[str]:
        page = LineagePage(f"Lineage {mode} Article")
        with patch("urllib.request.urlopen", lambda req, timeout=15: page):
            text = f"note: read https://example.test/lineage-{mode}"
            return lineage_replies(await w.harness.text(text, LINEAGE_ACTOR))

    async def capture_photo(mode: str) -> list[str]:
        w.llm.reply = f"Photo description lineage {mode} claim"
        return lineage_replies(await w.harness.photo(LINEAGE_ACTOR))

    await probe("text-capture", capture_text)
    await probe("url-capture", capture_url)
    await probe("photo-capture", capture_photo)
    for mode in ("control", "fault"):
        replies, created = seen[("url-capture", mode)]
        texts = [path.read_text(encoding="utf-8") for path in created]
        ck(
            f"consumer.url-capture-{mode}-no-plain-text-fallback-duplicate",
            [False, True],
            [
                any("Capturing as plain text" in reply for reply in replies),
                all("source: telegram" not in text for text in texts),
            ],
        )

    for mode in ("control", "fault"):
        if mode == "fault":
            w.fault.arm("*")
        results = {
            "vault_write": _dispatch_tool(
                "vault_write",
                {
                    "path": f"00-inbox/mcp-write-{mode}.md",
                    "content": lineage_note(f"MCP Write {mode}", f"MCP written {mode} body."),
                },
                w.vault,
                w.store,
                w.graph,
                w.parser,
                enforcer=None,
            ),
            "capture_note": _dispatch_tool(
                "capture_note",
                {"content": f"MCP captured {mode} body", "title": f"MCP Cap {mode}"},
                w.vault,
                w.store,
                w.graph,
                w.parser,
                enforcer=None,
            ),
        }
        w.fault.disarm()
        for tool, result in results.items():
            saved = (w.vault / result["path"]).exists() if "path" in result else False
            if mode == "control":
                ck(
                    f"consumer.mcp-{tool}-control-ok",
                    [True, False, True],
                    ["status" in result and result["status"] == "ok", "error" in result, saved],
                )
            else:
                ck(
                    f"consumer.mcp-{tool}-fault-returns-error-with-saved-file",
                    [True, True, False, True, True],
                    [
                        "error" in result,
                        result.get("saved"),
                        "status" in result,
                        cause in str(result.get("error")),
                        saved,
                    ],
                )

    async def fake_search(query: str, max_results: int = 5) -> list[Any]:
        return [
            SearchResult(
                video_id="lineagevid",
                title=f"Lineage Video {query.split()[-1]}",
                url="https://youtu.be/lineagevid",
                channel="c",
                duration_seconds=1,
                view_count=1,
                description="d",
            )
        ]

    for mode in ("control", "fault"):
        w.llm.reply = "Scripted research analysis lineageresearchclaim."
        if mode == "fault":
            w.fault.arm("*")
        raised: Exception | None = None
        result = None
        with (
            patch("vaultmind.research.pipeline.search_youtube", fake_search),
            patch.object(
                YouTubeTranscriptApi,
                "get_transcript",
                staticmethod(
                    lambda video_id, languages=None, mode=mode: [{"text": f"transcript {mode}"}]
                ),
            ),
        ):
            try:
                result = await run_research_pipeline(
                    f"lineage {mode}",
                    w.vault,
                    w.llm,
                    w.store,
                    w.parser,
                    config=ResearchConfig(max_results=1),
                )
            except Exception as exc:
                raised = exc
        w.fault.disarm()
        written = sorted((w.vault / "research").rglob(f"*{mode}*.md"))
        index_error = raised.__cause__ if isinstance(raised, ResearchIndexError) else raised
        outcome = type(index_error).__name__ if index_error else "completed"
        if mode == "control":
            summary_rel = str(result.summary_path.relative_to(w.vault)) if result else ""
            ck(
                "consumer.research-control-completes-and-indexes-summary",
                [None, True],
                [raised and repr(raised), bool(lineage_ids(w, summary_rel)["keyword"])],
            )
        else:
            ck(
                "consumer.research-fault-propagates-index-error-never-completes",
                ["IndexUpdateError", None, True],
                [outcome, result, bool(written)],
            )
            saved_paths = sorted((w.vault / "research" / f"lineage-{mode}").rglob("*.md"))
            ck(
                "consumer.research-fault-reports-all-saved-and-unattempted-paths",
                [
                    True,
                    [str(path) for path in saved_paths],
                    [],
                    [str(path) for path in saved_paths if path.name == "summary.md"],
                ],
                [
                    isinstance(raised, ResearchIndexError),
                    sorted(str(path) for path in getattr(raised, "saved_paths", ())),
                    [str(path) for path in getattr(raised, "indexed_paths", ())],
                    sorted(str(path) for path in getattr(raised, "unattempted_paths", ())),
                ],
            )

    for mode in ("control", "fault"):
        with lineage_readonly_keyword(w) if mode == "fault" else contextlib.nullcontext():
            cli = await lineage_cli_index(w)
        if mode == "control":
            ck("consumer.cli-index-control-exit-code", 0, cli.exit_code)
        else:
            ck(
                "consumer.cli-index-fault-nonzero-without-success",
                [True, False, True],
                [
                    cli.exit_code != 0,
                    bool(LINEAGE_SUCCESS.search(cli.output)),
                    "readonly database" in cli.output + repr(cli.exception),
                ],
            )

    tw = lineage_world(
        env.home,
        "consumers-think",
        {"keep.md": lineage_note("Think Keeper", f"Control record {LT_KEEP}.")},
        env.llm,
        env.closers,
        extra_toml="[distill]\nenabled = true\nmin_turns = 1\n",
    )
    env.worlds.append(tw)
    distill_json = (
        '{"question":"Lineage QMODE?","summary":"S lineage","resolution":"R lineage",'
        '"systems":[],"participants":[]}'
    )

    async def thinking_probe(label: str, command: str, reply: str) -> None:
        for mode in ("control", "fault"):
            await asyncio.sleep(1.1)
            before = set(tw.vault.rglob("*.md"))
            tw.llm.reply = reply.replace("MODE", mode)
            await tw.harness.command(f"/think explore: lineage {label} {mode} topic", LINEAGE_ACTOR)
            if mode == "fault":
                tw.fault.arm("*")
            try:
                replies = lineage_replies(
                    await tw.harness.command(command.replace("MODE", mode), LINEAGE_ACTOR)
                )
            finally:
                tw.fault.disarm()
            created = sorted(set(tw.vault.rglob("*.md")) - before)
            rels = [str(path.relative_to(tw.vault)) for path in created]
            joined = "\n".join(replies)
            if mode == "control":
                ck(
                    f"consumer.{label}-control-saves-indexes-and-reports-success",
                    [False, True, 1, [True]],
                    [
                        lineage_failure_in(replies),
                        lineage_success_in(replies),
                        len(created),
                        [bool(lineage_ids(tw, rel)["keyword"]) for rel in rels],
                    ],
                )
            else:
                ck(
                    f"consumer.{label}-fault-reports-explicit-failure-not-success",
                    [True, False, True],
                    [lineage_failure_in(replies), lineage_success_in(replies), cause in joined],
                )
                ck(
                    f"consumer.{label}-fault-keeps-one-saved-file-and-names-it",
                    [1, True],
                    [len(created), all(rel in joined for rel in rels)],
                )

    await thinking_probe("bookmark", "/bookmark lineage MODE bookmark", "Think answer.")
    await thinking_probe("distill", "/distill", distill_json)

    dw = lineage_world(
        env.home,
        "consumers-deferred",
        {"keep.md": lineage_note("Deferred Keeper", f"Control record {LT_KEEP}.")},
        env.llm,
        env.closers,
        ttl=1,
        extra_toml="[distill]\nenabled = true\nmin_turns = 1\n",
    )
    env.worlds.append(dw)
    records: list[logging.LogRecord] = []

    class Records(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    thinking_log = logging.getLogger("vaultmind.bot.thinking")
    handler = Records(level=logging.ERROR)
    thinking_log.addHandler(handler)
    try:
        for mode in ("control", "fault"):
            records.clear()
            before = set(dw.vault.rglob("*.md"))
            dw.llm.reply = distill_json.replace("MODE", f"deferred {mode}")
            user = 7 if mode == "control" else 8
            await dw.thinking.think(user, f"first {mode} topic", dw.store, dw.graph)
            await asyncio.sleep(1.5)
            if mode == "fault":
                dw.fault.arm("*")
            try:
                await dw.thinking.think(user, f"second {mode} topic", dw.store, dw.graph)
                others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                await asyncio.wait_for(asyncio.gather(*others, return_exceptions=True), 60)
            finally:
                dw.fault.disarm()
            created = sorted(set(dw.vault.rglob("*.md")) - before)
            rels = [str(path.relative_to(dw.vault)) for path in created]
            logged = [r for r in records if r.exc_info and cause in str(r.exc_info[1])]
            if mode == "control":
                ck(
                    "consumer.deferred-distill-control-indexes-artifact-without-error-log",
                    [1, [True], []],
                    [
                        len(created),
                        [bool(lineage_ids(dw, rel)["keyword"]) for rel in rels],
                        [r.getMessage() for r in records],
                    ],
                )
            else:
                ck(
                    "consumer.deferred-distill-fault-keeps-artifact-and-logs-explicit-failure",
                    [1, True],
                    [len(created), len(logged) == 1 and rels[0] in logged[0].getMessage()],
                )
            dw.thinking.clear_session(user)
    finally:
        thinking_log.removeHandler(handler)


CW_OLD = "ketchoriginalauth"
CW_NEW = "ketchrevisedauth"
CW_LDEL = "ketchlegacydel"
CW_LO = "ketchlegacyorig"
CW_LN = "ketchlegacynew"
CW_GONE = "ketchremovedfile"
CW_BDEL = "ketchbotdeleted"
CW_FRESH = "ketchfreshfile"


async def lineage_cli_watch_scenario(env: SimpleNamespace) -> None:
    """Acceptance 6: installed `index` then `watch` commands reconcile BOTH configured indexes.

    The commands build their own Chroma store and SQLite keyword index from explicit absolute
    TOML; only the Embedder factory is substituted. Every retirement below is performed by the
    command's own watcher/handler/store on real filesystem events, observed read-only.
    """
    from datetime import UTC, datetime

    ck = env.ck
    moment = datetime(2026, 10, 12, 12, 0, 0, 111111, tzinfo=UTC)
    notes = {
        "auth.md": lineage_note("Authored Subject", f"Authored baseline {CW_OLD}."),
        "ldel.md": lineage_note("Legacy Delete", f"Legacy delete subject {CW_LDEL}."),
        "ledit.md": lineage_note("Legacy Edit", f"Legacy edit original {CW_LO}."),
        "gone.md": lineage_note("Removed Subject", f"Soon removed {CW_GONE}."),
        "bdel.md": lineage_note("Bot Delete", f"Bot delete subject {CW_BDEL}."),
        "keep.md": lineage_note("Watch Keeper", f"Control record {LT_KEEP}."),
    }
    w = lineage_world(
        env.home,
        "cli-watch",
        notes,
        env.llm,
        env.closers,
        extra_toml="[duplicate_detection]\nenabled = false\n",
        cli_seed=True,
    )
    env.worlds.append(w)
    ck(
        "cli-watch.configured-index-files-absent-before-command-seed",
        {"chroma": False, "keyword": False},
        w.indexes_before,
    )
    ck(
        "cli-watch.resource-observer-discriminates-leaked-closed-and-absent",
        {
            "leak": dict.fromkeys(LINEAGE_CLI_RESOURCES, "open"),
            "release": dict.fromkeys(LINEAGE_CLI_RESOURCES, "closed"),
            "absent": {},
        },
        lineage_observer_selftest(env.home / "observer-selftest"),
    )
    cli = w.cli_seed_result
    ck("cli-watch.baseline-index-exit-code", 0, cli.exit_code)
    ck(
        "cli-watch.baseline-index-closes-its-keyword-connection",
        {"bm25": "closed"},
        cli.resources.final,
    )
    ck(
        "cli-watch.baseline-every-note-indexed-by-the-command",
        {rel: lineage_expected_ids(w, rel) for rel in notes},
        {rel: lineage_ids(w, rel) for rel in notes},
    )
    claims = (CW_OLD, CW_LDEL, CW_LO, CW_GONE, CW_BDEL, LT_KEEP)
    ck(
        "cli-watch.baseline-claims-retrievable-on-every-surface",
        [LINEAGE_FOUND] * len(claims),
        [lineage_found(w, claim) for claim in claims],
    )

    batches = w.embedder.batches
    rig = LineageCliWatch(w)
    outcome = SimpleNamespace(stopped=False, result=None)
    try:
        if not await rig.start():
            raise RuntimeError("installed watch command never started its event loop")
        ck("cli-watch.observer-live-via-real-filesystem-event", True, await rig.ready())
        ck("cli-watch.command-embeds-through-external-factory", True, w.embedder.batches > batches)

        async def settles(*, gone: tuple[str, ...], present: tuple[str, ...]) -> bool:
            return await lineage_until(
                lambda: (
                    all(lineage_exposure(w, t) == LINEAGE_ABSENT for t in gone)
                    and all(lineage_found(w, t) == LINEAGE_FOUND for t in present)
                )
            )

        paths = {rel: w.vault / rel for rel in notes}
        paths["auth.md"].write_text(
            lineage_note("Authored Subject", f"Authored revision {CW_NEW}."), encoding="utf-8"
        )
        settled = await settles(gone=(CW_OLD,), present=(CW_NEW,))
        ck(
            "cli-watch.authored-edit-replaces-both-indexes",
            [True, LINEAGE_ABSENT, LINEAGE_FOUND, lineage_expected_ids(w, "auth.md")],
            [
                settled,
                lineage_exposure(w, CW_OLD),
                lineage_found(w, CW_NEW),
                lineage_ids(w, "auth.md"),
            ],
        )

        lineage_legacy_delete(paths["ldel.md"], moment)
        settled = await settles(gone=(CW_LDEL,), present=(LT_KEEP,))
        ck(
            "cli-watch.legacy-bot-delete-retired-from-vector-keyword-hybrid",
            [True, LINEAGE_ABSENT, {"vector": [], "keyword": []}, True],
            [
                settled,
                lineage_exposure(w, CW_LDEL),
                lineage_ids(w, "ldel.md"),
                CW_LDEL in paths["ldel.md"].read_text(encoding="utf-8"),
            ],
        )

        lineage_legacy_edit(paths["ledit.md"], f"Legacy edit current {CW_LN}.", "rewrite", moment)
        settled = await settles(gone=(CW_LO,), present=(CW_LN,))
        ck(
            "cli-watch.legacy-bot-edit-indexes-only-current-text",
            [True, LINEAGE_ABSENT, LINEAGE_FOUND, lineage_expected_ids(w, "ledit.md"), True],
            [
                settled,
                lineage_exposure(w, CW_LO),
                lineage_found(w, CW_LN),
                lineage_ids(w, "ledit.md"),
                CW_LO in paths["ledit.md"].read_text(encoding="utf-8"),
            ],
        )

        paths["gone.md"].unlink()
        settled = await settles(gone=(CW_GONE,), present=())
        ck(
            "cli-watch.removed-file-retired-from-vector-keyword-hybrid",
            [True, LINEAGE_ABSENT, {"vector": [], "keyword": []}],
            [settled, lineage_exposure(w, CW_GONE), lineage_ids(w, "gone.md")],
        )

        (w.vault / "fresh.md").write_text(
            lineage_note("Fresh Subject", f"Fresh file {CW_FRESH}."), encoding="utf-8"
        )
        settled = await settles(gone=(), present=(CW_FRESH,))
        ck(
            "cli-watch.new-file-indexed-into-both-indexes",
            [True, lineage_expected_ids(w, "fresh.md")],
            [settled, lineage_ids(w, "fresh.md")],
        )

        replies = await lineage_delete(w, "bdel.md")
        marked = paths["bdel.md"].read_bytes()
        beat = await rig.beat()
        ck(
            "cli-watch.bot-delete-stays-retired-while-watching",
            [False, True, LINEAGE_ABSENT, {"vector": [], "keyword": []}, ["deleted"], True],
            [
                lineage_failure_in(replies),
                beat,
                lineage_exposure(w, CW_BDEL),
                lineage_ids(w, "bdel.md"),
                lineage_edges(w, "bdel.md"),
                CW_BDEL.encode() in marked,
            ],
        )
        ck(
            "cli-watch.bot-delete-leaves-file-bytes-unchanged-by-the-watcher",
            digest(marked),
            digest(paths["bdel.md"].read_bytes()),
        )

        rig.canary.unlink()
        retired = await lineage_until(
            lambda: lineage_ids(w, LINEAGE_CANARY) == {"vector": [], "keyword": []}
        )
        ck("cli-watch.deleted-event-retires-canary-rows", True, retired)
        ck(
            "cli-watch.control-note-untouched",
            [lineage_expected_ids(w, "keep.md"), LINEAGE_FOUND],
            [lineage_ids(w, "keep.md"), lineage_found(w, LT_KEEP)],
        )
    finally:
        outcome = await rig.stop()

    result = outcome.result or SimpleNamespace(
        exit_code=None,
        output="",
        exception=None,
        resources=SimpleNamespace(live={}, final={}),
    )
    ck("cli-watch.graceful-cancellation-within-bound", True, outcome.stopped)
    ck(
        "cli-watch.command-exits-zero-without-exception",
        [0, None],
        [result.exit_code, repr(result.exception) if result.exception else None],
    )
    ck("cli-watch.command-reported-watching", True, "Watching" in result.output)
    ck(
        "cli-watch.command-closes-its-keyword-connection",
        {"live": {"bm25": "open"}, "final": {"bm25": "closed"}},
        {"live": result.resources.live, "final": result.resources.final},
    )
    retired_rels = ("ldel.md", "gone.md", "bdel.md", LINEAGE_CANARY)
    ck(
        "cli-watch.final-ids-equal-current-projection",
        {
            "auth.md": lineage_expected_ids(w, "auth.md"),
            "ledit.md": lineage_expected_ids(w, "ledit.md"),
            "fresh.md": lineage_expected_ids(w, "fresh.md"),
            "keep.md": lineage_expected_ids(w, "keep.md"),
            "retired": {"vector": [], "keyword": []},
        },
        {
            "auth.md": lineage_ids(w, "auth.md"),
            "ledit.md": lineage_ids(w, "ledit.md"),
            "fresh.md": lineage_ids(w, "fresh.md"),
            "keep.md": lineage_ids(w, "keep.md"),
            "retired": {
                kind: [k for rel in retired_rels for k in lineage_ids(w, rel)[kind]]
                for kind in ("vector", "keyword")
            },
        },
    )
    env.owning["cli_watch"] = {
        "entry_point": lineage_entry_identity(),
        "config_sha256": digest(w.config.read_bytes()),
        "canary_revisions": rig.revision,
        "command_output": result.output,
    }


MS_OLD = "harborovwoldclaim"
MS_NEW = "harborovwnewclaim"
MS_RET = "harborretireclaim"
MS_CAP = "harborcaptureclaim"


async def lineage_mcp_serve_scenario(env: SimpleNamespace) -> None:
    """Acceptance 7: registered MCP mutation consumers over the real `mcp-serve` assembly.

    Installed `index` seeds the configured vector and keyword indexes in a child; the installed
    `mcp-serve` then runs in an isolated child behind the SDK stdio client with only the
    Embedder factory substituted, and `vault_write` (overwrite and retire) and `capture_note`
    mutate through the assembly's own store. Old keyword rows must not survive.
    """
    from datetime import UTC, datetime

    from vaultmind.config import load_settings
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.store import VaultStore
    from vaultmind.vault.parser import VaultParser

    ck = env.ck
    root = (env.home / "mcp-serve").resolve()
    vault, data, scratch = root / "vault", root / "data", root / "scratch"
    for directory in (vault, data, scratch):
        directory.mkdir(parents=True)
    notes = {
        "ow.md": lineage_note("Serve Overwrite", f"Overwrite baseline {MS_OLD}."),
        "ret.md": lineage_note("Serve Retire", f"Retire baseline {MS_RET}."),
        "keep.md": lineage_note("Serve Keeper", f"Control record {LT_KEEP}."),
    }
    for rel, text in notes.items():
        (vault / rel).write_text(text, encoding="utf-8")
    config = root / "config.toml"
    config.write_text(lineage_config_text(vault, data, True, 3600, ""), encoding="utf-8")
    settings = load_settings(config)
    # An already-generated legacy bot deletion of ret.md, produced by the base-commit writer.
    moment = datetime(2026, 10, 12, 12, 0, 0, 222222, tzinfo=UTC)
    (scratch / "ret.md").write_text(notes["ret.md"], encoding="utf-8")
    lineage_legacy_delete(scratch / "ret.md", moment)
    retired_text = (scratch / "ret.md").read_text(encoding="utf-8")
    keyword_db_before = (data / "bm25.db").exists()

    seed = await lineage_launch(root, "launcher-state-index", ["--config", str(config), "index"])
    ck("mcp-serve.seed-index-exit-code", 0, seed.code)
    seed_identity = lineage_launcher_checks(ck, "mcp-serve.seed.", root, seed.state)
    indexed = re.search(r"Indexed (\d+) chunks from (\d+) notes", re.sub(r"\s+", " ", seed.stdout))
    view = SimpleNamespace(data=data, vault=vault)
    baseline = {rel: lineage_keyword_note_rows(view, rel) for rel in notes}
    ck(
        "mcp-serve.seed-keyword-index-created-and-filled-by-installed-command",
        [False, 3, True, True, True],
        [
            keyword_db_before,
            indexed and int(indexed.group(2)),
            all(baseline.values()),
            any(MS_OLD in text for text in baseline["ow.md"].values()),
            any(MS_RET in text for text in baseline["ret.md"].values()),
        ],
    )

    calls = [
        (
            "vault_write",
            {
                "path": "ow.md",
                "content": lineage_note("Serve Overwrite", f"Overwrite replacement {MS_NEW}."),
            },
        ),
        ("vault_write", {"path": "ret.md", "content": retired_text}),
        ("capture_note", {"content": f"MCP captured claim {MS_CAP}.", "title": "Serve Capture"}),
        ("vault_search", {"query": MS_OLD, "n_results": 20}),
        ("vault_search", {"query": MS_RET, "n_results": 20}),
        ("vault_search", {"query": LT_KEEP, "n_results": 20}),
    ]
    run = await mcp_stdio_session(
        root,
        ["--config", str(config), "mcp-serve", "--profile", "full"],
        calls,
        launcher=lineage_launcher(),
    )
    session_identity = lineage_launcher_checks(ck, "mcp-serve.session.", root, run["state"])
    payloads = [json.loads(result.content[0].text) for result in run["results"]]
    stderr = (run["state"] / "stderr.txt").read_text(encoding="utf-8")
    ck(
        "mcp-serve.session-protocol-clean",
        [[], [False] * len(calls), False],
        [run["errors"], [bool(r.isError) for r in run["results"]], "Traceback" in stderr],
    )
    ck(
        "mcp-serve.mutation-tools-report-saved-and-indexed",
        [
            {"status": "ok", "path": "ow.md"},
            {"status": "ok", "path": "ret.md"},
            ["ok", "Serve Capture"],
        ],
        [payloads[0], payloads[1], [payloads[2].get("status"), payloads[2].get("title")]],
    )

    bm25 = BM25Index(data / "bm25.db")
    env.closers.append(bm25.close)
    parser = VaultParser(settings.vault)
    store = VaultStore(settings.chroma, cast("Embedder", DeterministicEmbedder()), bm25=bm25)
    after = SimpleNamespace(
        settings=settings, data=data, vault=vault, store=store, bm25=bm25, parser=parser
    )
    captured = payloads[2]["path"]
    ck(
        "mcp-serve.overwrite-replaces-old-vector-keyword-and-hybrid-evidence",
        [LINEAGE_ABSENT, LINEAGE_FOUND, lineage_expected_ids(after, "ow.md")],
        [
            lineage_exposure(after, MS_OLD),
            lineage_found(after, MS_NEW),
            lineage_ids(after, "ow.md"),
        ],
    )
    ck(
        "mcp-serve.retire-removes-vector-keyword-and-hybrid-evidence-keeps-file",
        [LINEAGE_ABSENT, {"vector": [], "keyword": []}, retired_text, ""],
        [
            lineage_exposure(after, MS_RET),
            lineage_ids(after, "ret.md"),
            (vault / "ret.md").read_text(encoding="utf-8"),
            parser.parse_file(vault / "ret.md").current_content().strip(),
        ],
    )
    ck(
        "mcp-serve.capture-indexed-and-control-untouched",
        [LINEAGE_FOUND, lineage_expected_ids(after, captured), LINEAGE_FOUND],
        [
            lineage_found(after, MS_CAP),
            lineage_ids(after, captured),
            lineage_found(after, LT_KEEP),
        ],
    )
    searches = [payload["results"] for payload in payloads[3:]]
    ck(
        "mcp-serve.vault-search-never-returns-retired-or-overwritten-text",
        [False, False, True],
        [
            any(MS_OLD in hit["content"] for hit in searches[0]),
            any(MS_RET in hit["content"] for hit in searches[1]),
            any(LT_KEEP in hit["content"] for hit in searches[2]),
        ],
    )
    env.owning["mcp_serve"] = {
        "config_sha256": digest(config.read_bytes()),
        "initial_files_sha256": {rel: digest(text.encode()) for rel, text in notes.items()},
        "final_files_sha256": lineage_vault_digests(after),
        "seed": seed_identity,
        "session": session_identity,
        "tool_results": payloads,
        "seed_output": seed.stdout,
    }


SX_LOG_1 = "merelogpartone"
SX_LOG_2 = "merelogparttwo"
SX_LOG_3 = "merelogpartthree"
SX_LOG_FIXED = "merelogrepaired"
SX_GOOD_OLD = "meregoodbaseline"
SX_GOOD_NEW = "meregoodrevision"
SX_MALFORMED = b"---\ntitle: [unclosed\n---\n\nmalformed body text.\n"


def lineage_note_vector_state(world: SimpleNamespace, rel: str) -> list[list[Any]]:
    """Raw Chroma rows of one note: id, document, metadata and embedding (independent handle)."""
    collection = lineage_chroma_client(world).get_collection(world.settings.chroma.collection_name)
    got = collection.get(where={"note_path": rel}, include=["documents", "metadatas", "embeddings"])
    embeddings = [[float(value) for value in row] for row in got["embeddings"]]
    return sorted(
        [chunk_id, document, dict(metadata), embedding]
        for chunk_id, document, metadata, embedding in zip(
            got["ids"], got["documents"], got["metadatas"], embeddings, strict=True
        )
    )


def lineage_note_keyword_state(world: SimpleNamespace, rel: str) -> list[list[str]]:
    """Raw keyword-index rows of one note (read-only, parameterized)."""
    con = sqlite3.connect(f"file:{world.data / 'bm25.db'}?mode=ro", uri=True)
    try:
        return [
            [str(value) for value in row]
            for row in con.execute(
                "SELECT chunk_id, note_path, note_title, content FROM fts_chunks"
                " WHERE note_path = ? ORDER BY chunk_id",
                (rel,),
            )
        ]
    finally:
        con.close()


def lineage_error_chain(result: Any, kind: type[BaseException]) -> Any:
    """The typed error behind a command's Click exit, found through its cause/context chain."""
    error = result.exception
    while error is not None and not isinstance(error, kind):
        error = error.__cause__ or error.__context__
    return error


async def lineage_strict_cli_scenario(env: SimpleNamespace) -> None:
    """Strict full-index CLI parse failure keeps last-good rows and bytes; lenient CLI works."""
    from vaultmind.vault.parser import VaultParseError

    ck = env.ck
    log_tokens = (SX_LOG_1, SX_LOG_2, SX_LOG_3)
    notes = {
        "keep.md": lineage_note("Strict Keeper", f"Control record {LT_KEEP}."),
        "good.md": lineage_note("Strict Good", f"Good baseline {SX_GOOD_OLD}."),
        "log.md": lineage_note("Strict Log", lineage_parts(log_tokens)),
    }
    w = lineage_world(env.home, "strict-cli", notes, env.llm, env.closers, cli_seed=True)
    env.worlds.append(w)
    seed = w.cli_seed_result
    ck(
        "strict-cli.seed-by-installed-index-exits-zero-and-closes-keyword-connection",
        [0, {"bm25": "closed"}],
        [seed.exit_code, seed.resources.final],
    )
    tokens = (LT_KEEP, SX_GOOD_OLD, *log_tokens)
    ck(
        "strict-cli.seed-every-note-indexed-and-retrievable-on-every-surface",
        [{rel: lineage_expected_ids(w, rel) for rel in notes}, [LINEAGE_FOUND] * len(tokens)],
        [{rel: lineage_ids(w, rel) for rel in notes}, [lineage_found(w, t) for t in tokens]],
    )
    exposure_before = {t: lineage_exposure(w, t) for t in (*tokens, SX_GOOD_NEW)}
    rows_before = {
        rel: [lineage_note_vector_state(w, rel), lineage_note_keyword_state(w, rel)]
        for rel in notes
    }

    revised = lineage_note("Strict Good", f"Good revision {SX_GOOD_NEW}.").encode()
    (w.vault / "log.md").write_bytes(SX_MALFORMED)
    (w.vault / "good.md").write_bytes(revised)
    batches = w.embedder.batches
    cli = await lineage_cli_index(w)
    typed = lineage_error_chain(cli, VaultParseError)
    ck(
        "strict-cli.parse-failure-exits-nonzero-typed-without-traceback-or-success",
        [True, True, ["log.md"], True, False, False],
        [
            cli.exit_code != 0,
            "Indexing failed" in cli.output,
            [str(path) for path, _ in typed.failures] if typed else None,
            "log.md" in cli.output,
            "Traceback" in cli.output,
            bool(LINEAGE_SUCCESS.search(cli.output)),
        ],
    )
    ck(
        "strict-cli.parse-failure-reaches-no-embedding-and-leaves-raw-rows-unchanged",
        [0, rows_before, exposure_before],
        [
            w.embedder.batches - batches,
            {
                rel: [lineage_note_vector_state(w, rel), lineage_note_keyword_state(w, rel)]
                for rel in notes
            },
            {t: lineage_exposure(w, t) for t in (*tokens, SX_GOOD_NEW)},
        ],
    )
    ck(
        "strict-cli.parse-failure-keeps-malformed-and-edited-source-bytes",
        [SX_MALFORMED, revised],
        [(w.vault / "log.md").read_bytes(), (w.vault / "good.md").read_bytes()],
    )
    ck(
        "strict-cli.parse-failure-closes-its-keyword-connection",
        {"bm25": "closed"},
        cli.resources.final,
    )

    lenient = await asyncio.to_thread(lineage_cli_run, w, ["stats"])
    ck(
        "strict-cli.unrelated-lenient-cli-stays-usable-over-the-readable-notes",
        [0, True, False],
        [lenient.exit_code, "Total notes: 2" in lenient.output, "Traceback" in lenient.output],
    )
    ck(
        "strict-cli.lenient-cli-leaves-rows-and-malformed-bytes-unchanged",
        [rows_before, SX_MALFORMED],
        [
            {
                rel: [lineage_note_vector_state(w, rel), lineage_note_keyword_state(w, rel)]
                for rel in notes
            },
            (w.vault / "log.md").read_bytes(),
        ],
    )

    (w.vault / "log.md").write_text(
        lineage_note("Strict Log", lineage_parts((SX_LOG_1, SX_LOG_2, SX_LOG_FIXED))),
        encoding="utf-8",
    )
    repaired = await lineage_cli_index(w)
    ck(
        "strict-cli.repaired-vault-full-index-reconciles-both-indexes",
        [0, {"bm25": "closed"}, LINEAGE_ABSENT, LINEAGE_FOUND, LINEAGE_ABSENT, LINEAGE_FOUND]
        + [{rel: lineage_expected_ids(w, rel) for rel in notes}],
        [
            repaired.exit_code,
            repaired.resources.final,
            lineage_exposure(w, SX_GOOD_OLD),
            lineage_found(w, SX_GOOD_NEW),
            lineage_exposure(w, SX_LOG_3),
            lineage_found(w, SX_LOG_FIXED),
            {rel: lineage_ids(w, rel) for rel in notes},
        ],
    )
    env.owning["strict_index_cli"] = {
        "entry_point": lineage_entry_identity(),
        "config_sha256": digest(w.config.read_bytes()),
        "malformed_sha256": digest(SX_MALFORMED),
        "failure_output": cli.output,
        "lenient_output": lenient.output,
    }


TH_V0 = "rulesvzeroclaim"
TH_V0_RULE = "rulesolderrule"
TH_V1 = "rulesvoneclaim"
TH_V1_RULE = "rulesmiddlerule"
TH_V2 = "rulesvtwoclaim"
TH_V2_RULE = "rulesfinalrule"


async def lineage_thematic_edit_scenario(env: SimpleNamespace) -> None:
    """Two real bot edits of a note led by Markdown thematic rules index only the latest text."""
    ck = env.ck
    original = f"---\n\nv0 claim {TH_V0}.\n\n---\n\nOlder rule section {TH_V0_RULE}."
    first = f"---\n\nv1 claim {TH_V1}.\n\n---\n\nMiddle rule section {TH_V1_RULE}."
    second = f"---\n\nv2 claim {TH_V2}.\n\n---\n\nFinal rule section {TH_V2_RULE}."
    notes = {
        "rules.md": lineage_note("Thematic Rules", "\n" + original),
        "other.md": lineage_note("Other Notes", f"Control record: {LT_KEEP}."),
    }
    w = lineage_world(env.home, "thematic-edit", notes, env.llm, env.closers)
    env.worlds.append(w)
    path = w.vault / "rules.md"
    lineage_index_full(w)
    ck(
        "thematic-edit.baseline-leading-rules-are-current-text-and-indexed",
        [original, LINEAGE_FOUND, LINEAGE_FOUND, lineage_expected_ids(w, "rules.md")],
        [
            w.parser.parse_file(path).current_content(),
            lineage_found(w, TH_V0),
            lineage_found(w, TH_V0_RULE),
            lineage_ids(w, "rules.md"),
        ],
    )
    calls_before = len(env.llm.calls)

    prompt, reply = await lineage_edit(w, "rules.md", first, "rewrite the rule sections")
    ck(
        "thematic-edit.first-applied-without-failure",
        [False, True],
        [lineage_failure_in(prompt + reply), lineage_success_in(reply)],
    )
    ck(
        "thematic-edit.first-only-v1-indexed",
        [[LINEAGE_ABSENT] * 2, [LINEAGE_FOUND] * 2, lineage_expected_ids(w, "rules.md")],
        [
            [lineage_exposure(w, TH_V0), lineage_exposure(w, TH_V0_RULE)],
            [lineage_found(w, TH_V1), lineage_found(w, TH_V1_RULE)],
            lineage_ids(w, "rules.md"),
        ],
    )

    prompt, reply = await lineage_edit(w, "rules.md", second, "rewrite the rule sections again")
    ck(
        "thematic-edit.second-applied-without-failure",
        [False, True],
        [lineage_failure_in(prompt + reply), lineage_success_in(reply)],
    )
    older = (TH_V0, TH_V0_RULE, TH_V1, TH_V1_RULE)
    ck(
        "thematic-edit.second-only-v2-indexed-on-every-surface",
        [[LINEAGE_ABSENT] * 4, [LINEAGE_FOUND] * 2, lineage_expected_ids(w, "rules.md")],
        [
            [lineage_exposure(w, token) for token in older],
            [lineage_found(w, TH_V2), lineage_found(w, TH_V2_RULE)],
            lineage_ids(w, "rules.md"),
        ],
    )
    ck("thematic-edit.scripted-llm-called-per-edit", 2, len(env.llm.calls) - calls_before)
    ck(
        "thematic-edit.lineage-has-two-edited-edges",
        ["edited", "edited"],
        lineage_edges(w, "rules.md"),
    )
    text = path.read_text(encoding="utf-8")
    note = w.parser.parse_file(path)
    ck(
        "thematic-edit.raw-file-keeps-original-v1-v2-and-title",
        [[True] * 3, [True] * 3, second, "Thematic Rules"],
        [
            [part in text for part in (original, first, second)],
            [part in note.content for part in (original, first, second)],
            note.current_content(),
            note.title,
        ],
    )
    cli = await lineage_cli_index(w)
    ck(
        "thematic-edit.full-index-cli-keeps-only-v2",
        [0, {"bm25": "closed"}, [LINEAGE_ABSENT] * 4, [LINEAGE_FOUND] * 2]
        + [lineage_expected_ids(w, "rules.md")],
        [
            cli.exit_code,
            cli.resources.final,
            [lineage_exposure(w, token) for token in older],
            [lineage_found(w, TH_V2), lineage_found(w, TH_V2_RULE)],
            lineage_ids(w, "rules.md"),
        ],
    )
    env.owning["thematic_edit"] = {
        "files_sha256": {
            "original": digest(original.encode()),
            "v1": digest(first.encode()),
            "v2": digest(second.encode()),
            "final_file": digest(path.read_bytes()),
        },
        "config_sha256": digest(w.config.read_bytes()),
    }


RS_PRIOR = "harborpriorgood"
RS_NEW = "harborreplacement"
RS_OTHER = "harborothersource"
RS_SUMMARY = "harborsummaryclaim"
RS_TITLES = ("Alpha", "Beta")
RS_VIDEOS = ("a", "b")


class LineageResearchEmbedder(LineageEmbedder):
    """External embedding boundary with a content-selective synthetic outage."""

    def __init__(self, fail_on: str | None = None) -> None:
        super().__init__()
        self.fail_on = fail_on

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on is not None and any(self.fail_on in text for text in texts):
            self.batches += 1
            self.failures += 1
            raise RuntimeError("synthetic research embedding outage")
        return super().embed_texts(texts)


@contextlib.contextmanager
def lineage_research_boundaries(llm: Any, transcripts: dict[str, str]) -> Any:
    """External seams of the research command ONLY: yt-dlp search, transcript API, LLM client.

    Parser, projection, store, keyword index, CLI assembly, pipeline, gap ledger and resource
    handling all stay real; the command builds its own indexes, cache and gap store.
    """
    from unittest.mock import patch

    from youtube_transcript_api import YouTubeTranscriptApi

    from vaultmind.research.searcher import SearchResult

    def search(query: str, max_results: int) -> list[Any]:
        return [
            SearchResult(
                video_id=video,
                title=title,
                url=f"https://example.invalid/{video}",
                channel="fixture",
                duration_seconds=1,
                view_count=1,
                description="fixture",
            )
            for video, title in zip(RS_VIDEOS, RS_TITLES, strict=True)
        ]

    def transcript(video_id: str, languages: list[str] | None = None) -> list[dict[str, str]]:
        return [{"text": transcripts[video_id]}]

    with (
        patch("vaultmind.cli._create_llm_client", lambda settings: llm),
        patch("vaultmind.research.searcher._run_yt_search", search),
        patch.object(YouTubeTranscriptApi, "get_transcript", staticmethod(transcript)),
    ):
        yield


def lineage_gap_rows(world: SimpleNamespace, gap_id: str) -> list[list[Any]]:
    """The gap ledger row, read-only from the configured SQLite file."""
    con = sqlite3.connect(f"file:{world.data / 'gaps.db'}?mode=ro", uri=True)
    try:
        return [
            list(row)
            for row in con.execute(
                "SELECT gap_id, kind, question, status, evidence_ref, resolution_ref, created,"
                " last_seen, resolved, occurrence_count FROM gaps WHERE gap_id = ?",
                (gap_id,),
            )
        ]
    finally:
        con.close()


def lineage_rows_outside(world: SimpleNamespace, prefix: str) -> dict[str, dict[str, str]]:
    """Raw stored vector and keyword rows of every note outside the `prefix` folder."""
    return {
        kind: {key: text for key, text in rows.items() if not key.startswith(prefix)}
        for kind, rows in lineage_state(world).items()
    }


async def lineage_research_cli(
    env: SimpleNamespace,
    w: SimpleNamespace,
    label: str,
    failing: int | None,
) -> SimpleNamespace:
    """One installed `vaultmind research <query>` over the configured world; returns evidence.

    `failing` is the position of the source whose replacement text the external embedding
    boundary rejects (None: no outage). A prior-good row for that source is indexed first, and
    a gap for the query is minted first, both through the real stores.
    """
    from vaultmind.memory.gaps import GapKind, GapStore

    query = f"harbor{label}"
    folder = w.vault / "research" / query
    sources = [folder / f"{title}.md" for title in RS_TITLES]
    summary = folder / "summary.md"
    tokens = SimpleNamespace(
        prior=RS_PRIOR + label,
        new=RS_NEW + label,
        other=RS_OTHER + label,
        summary=RS_SUMMARY + label,
    )
    if failing is not None:
        folder.mkdir(parents=True)
        sources[failing].write_text(
            lineage_note(RS_TITLES[failing], f"Prior good source {tokens.prior}."),
            encoding="utf-8",
        )
        w.store.index_single_note(w.parser.parse_file(sources[failing]), w.parser)
    gaps = GapStore(w.data / "gaps.db")
    gap = gaps.mint(query, GapKind.UNANSWERED_QUESTION)
    gaps.close()
    prefix = f"research/{query}/"
    failed_rel = str(sources[failing].relative_to(w.vault)) if failing is not None else ""
    before = SimpleNamespace(
        outside=lineage_rows_outside(w, prefix),
        gap=lineage_gap_rows(w, gap.gap_id),
        failed_vector=lineage_note_vector_state(w, failed_rel),
        failed_keyword=lineage_note_keyword_state(w, failed_rel),
    )
    transcripts = {
        video: (
            f"{tokens.new} replacement transcript"
            if failing is not None and index == failing
            else f"{tokens.other} other source transcript"
        )
        for index, video in enumerate(RS_VIDEOS)
    }
    env.llm.reply = json.dumps(
        {
            "summary": f"{tokens.summary} fixture analysis",
            "key_themes": ["fixture"],
            "comparative_insights": "fixture",
            "gaps": "fixture",
            "recommendations": "fixture",
        }
    )
    embedder = LineageResearchEmbedder(tokens.new if failing is not None else None)
    with lineage_research_boundaries(env.llm, transcripts):
        cli = await asyncio.to_thread(lineage_cli_run, w, ["research", query], embedder)
    return SimpleNamespace(
        query=query,
        cli=cli,
        embedder=embedder,
        saved=[*sources, summary],
        tokens=tokens,
        gap_id=gap.gap_id,
        prefix=prefix,
        before=before,
        failing=failing,
    )


async def lineage_research_cli_scenario(env: SimpleNamespace) -> None:
    """Installed `research` CLI: control, then FIRST and SECOND source index failures."""
    from vaultmind.indexer.store import IndexUpdateError
    from vaultmind.research.pipeline import ResearchIndexError

    ck = env.ck
    data = (env.home / "research-cli").resolve() / "data"
    w = lineage_world(
        env.home,
        "research-cli",
        {"keep.md": lineage_note("Research Keeper", f"Control record {LT_KEEP}.")},
        env.llm,
        env.closers,
        extra_toml=f'[gaps]\nenabled = true\ndb_path = "{data / "gaps.db"}"\n',
        embedding_cache=True,
    )
    env.worlds.append(w)
    lineage_index_full(w)
    configured = {"bm25": "closed", "cache": "closed", "gap_store": "closed"}

    def rel(path: Path) -> str:
        return str(path.relative_to(w.vault))

    control = await lineage_research_cli(env, w, "control", None)
    saved = control.saved
    gap = lineage_gap_rows(w, control.gap_id)
    ck(
        "research-cli.control-completes-closes-gap-and-releases-every-resource",
        [0, True, True, ["answered", rel(saved[2])], configured],
        [
            control.cli.exit_code,
            "Research complete" in control.cli.output,
            "Closed gap" in control.cli.output,
            [gap[0][3], gap[0][5]] if gap else None,
            control.cli.resources.final,
        ],
    )
    ck(
        "research-cli.control-indexes-every-saved-note-searchable",
        [{rel(path): lineage_expected_ids(w, rel(path)) for path in saved}]
        + [LINEAGE_FOUND, LINEAGE_FOUND],
        [
            {rel(path): lineage_ids(w, rel(path)) for path in saved},
            lineage_found(w, control.tokens.other),
            lineage_found(w, control.tokens.summary),
        ],
    )

    outcomes: dict[str, Any] = {}
    for label, position in (("first", 0), ("second", 1)):
        run = await lineage_research_cli(env, w, label, position)
        cli, tokens = run.cli, run.tokens
        saved = run.saved
        failed = saved[position]
        typed = lineage_error_chain(cli, ResearchIndexError)
        update = typed.__cause__ if typed is not None else None
        ck(
            f"research-cli.{label}-failure-exits-nonzero-typed-without-traceback-or-success",
            [True, True, "embed", [], 1, True, True, False, False],
            [
                cli.exit_code != 0,
                isinstance(typed, ResearchIndexError),
                update.stage.value if isinstance(update, IndexUpdateError) else None,
                list(update.completed_stages) if isinstance(update, IndexUpdateError) else None,
                run.embedder.failures,
                "Error:" in cli.output,
                "no index changes were applied" in cli.output,
                "Traceback" in cli.output,
                bool(re.search(r"Research complete|Closed gap", cli.output)),
            ],
        )
        ck(
            f"research-cli.{label}-failure-reports-exact-saved-indexed-failed-unattempted-paths",
            [
                [str(path) for path in saved],
                [str(path) for path in saved[:position]],
                str(failed),
                [str(path) for path in saved[position + 1 :]],
                True,
            ],
            [
                [str(path) for path in getattr(typed, "saved_paths", ())],
                [str(path) for path in getattr(typed, "indexed_paths", ())],
                str(getattr(typed, "failed_path", "")),
                [str(path) for path in getattr(typed, "unattempted_paths", ())],
                all(str(path) in cli.output for path in saved),
            ],
        )
        ck(
            f"research-cli.{label}-failure-keeps-every-saved-file-with-its-new-bytes",
            [True, True, True],
            [
                all(path.is_file() for path in saved),
                tokens.new in failed.read_text(encoding="utf-8"),
                tokens.summary in saved[2].read_text(encoding="utf-8"),
            ],
        )
        last_good_ids = sorted(row[0] for row in run.before.failed_keyword)
        ck(
            f"research-cli.{label}-failed-source-last-good-rows-unchanged-and-searchable",
            [
                run.before.failed_vector,
                run.before.failed_keyword,
                LINEAGE_FOUND,
                LINEAGE_ABSENT,
                last_good_ids,
                [],
            ],
            [
                lineage_note_vector_state(w, rel(failed)),
                lineage_note_keyword_state(w, rel(failed)),
                lineage_found(w, tokens.prior),
                lineage_exposure(w, tokens.new),
                lineage_keyword_matched(w, tokens.prior),
                lineage_keyword_matched(w, tokens.new),
            ],
        )
        indexed = saved[:position]
        ck(
            f"research-cli.{label}-indexed-prefix-searchable-and-unattempted-absent",
            [
                {
                    rel(path): lineage_expected_ids(w, rel(path))
                    if path in indexed
                    else {"vector": [], "keyword": []}
                    for path in saved
                    if path != failed
                },
                LINEAGE_FOUND if indexed else LINEAGE_ABSENT,
                LINEAGE_ABSENT,
            ],
            [
                {rel(path): lineage_ids(w, rel(path)) for path in saved if path != failed},
                lineage_found(w, tokens.other) if indexed else lineage_exposure(w, tokens.other),
                lineage_exposure(w, tokens.summary),
            ],
        )
        ck(
            f"research-cli.{label}-failure-leaves-prior-good-notes-and-gap-unchanged-open",
            [run.before.outside, run.before.gap, "open", ""],
            [
                lineage_rows_outside(w, run.prefix),
                lineage_gap_rows(w, run.gap_id),
                lineage_gap_rows(w, run.gap_id)[0][3],
                lineage_gap_rows(w, run.gap_id)[0][5],
            ],
        )
        ck(
            f"research-cli.{label}-failure-closes-every-configured-resource",
            configured,
            cli.resources.final,
        )
        outcomes[label] = {
            "position": position,
            "exit_code": cli.exit_code,
            "output": cli.output,
            "saved": [rel(path) for path in saved],
            "stage": update.stage.value if isinstance(update, IndexUpdateError) else None,
        }
    env.owning["research_cli"] = {
        "entry_point": lineage_entry_identity(),
        "config_sha256": digest(w.config.read_bytes()),
        "control_output": control.cli.output,
        "failures": outcomes,
    }


LINEAGE_SCENARIOS = (
    ("delete", lineage_delete_scenario),
    ("edit", lineage_edit_scenario),
    ("reconcile", lineage_reconcile_scenario),
    ("failure", lineage_failure_scenario),
    ("grammar", lineage_grammar_scenario),
    ("consumers", lineage_consumer_scenario),
    ("cli-watch", lineage_cli_watch_scenario),
    ("mcp-serve", lineage_mcp_serve_scenario),
    ("strict-cli", lineage_strict_cli_scenario),
    ("thematic-edit", lineage_thematic_edit_scenario),
    ("research-cli", lineage_research_cli_scenario),
)


async def lineage_case(home: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    import logging
    import traceback

    home = home.resolve()
    failures: list[str] = []

    def ck(name: str, expected: Any, actual: Any) -> None:
        checks.append(
            {"id": name, "expected": lineage_json(expected), "actual": lineage_json(actual)}
        )
        if expected != actual:
            failures.append(name)

    config_preloaded = "vaultmind.config" in sys.modules
    ck("sanitization.config-not-imported-before-temp-home", False, config_preloaded)
    if config_preloaded:
        raise RuntimeError("Refusing lineage acceptance: configuration predates temporary HOME")
    import vaultmind
    from vaultmind.config import VAULTMIND_HOME

    ck(
        "sanitization.vaultmind-home-is-temp",
        str(home / ".vaultmind"),
        str(VAULTMIND_HOME.resolve()),
    )
    ck(
        "sanitization.code-imported-from-worktree",
        True,
        import_binding()["identical_to_tracked_source"],
    )
    if VAULTMIND_HOME.resolve() != home / ".vaultmind":
        raise RuntimeError("Refusing lineage acceptance: application state is outside fixture HOME")
    log_records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(f"{record.levelname} {record.name}: {record.getMessage()}")

    capture = Capture()
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.addHandler(capture)
    root_logger.setLevel(logging.INFO)
    env = SimpleNamespace(ck=ck, home=home, llm=LineageLLM(), closers=[], worlds=[], owning={})
    completed: list[str] = []
    try:
        for name, scenario in LINEAGE_SCENARIOS:
            try:
                await scenario(env)
                completed.append(name)
            except Exception:
                ck(f"{name}.scenario-ran-to-completion", None, traceback.format_exc())
        for stability in (True, False):
            name = f"watch-{'stable' if stability else 'direct'}"
            try:
                await lineage_watch_scenario(env, stability)
                completed.append(name)
            except Exception:
                ck(f"{name}.scenario-ran-to-completion", None, traceback.format_exc())
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(old_level)
        for close in reversed(env.closers):
            close()

    expected_scenarios = [name for name, _ in LINEAGE_SCENARIOS] + ["watch-stable", "watch-direct"]
    ck("scenarios.all-completed", expected_scenarios, completed)
    worlds = env.worlds
    initial = {world.name: world.input_identity for world in worlds}
    legacy_source = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{LINEAGE_BASE_COMMIT}:src/vaultmind/contradiction/marking.py",
        ],
        text=True,
    )
    corpus = {
        "initial_worlds": initial,
        "legacy_writer": {
            "commit": LINEAGE_BASE_COMMIT,
            "path": "src/vaultmind/contradiction/marking.py",
            "sha256": digest(legacy_source.encode()),
        },
        "scripted_llm_calls": env.llm.calls,
    }
    receipt = {
        "import_origin": os.path.realpath(vaultmind.__file__),
        "interpreter": {"executable": sys.executable, "version": sys.version},
        "git": approval_git_binding(),
        "fixture_roots": {world.name: str(world.root.relative_to(home)) for world in worlds},
        "configuration_files_sha256": {
            name: identity["files_sha256"]["config.toml"] for name, identity in initial.items()
        },
        "config_sha256": approval_json_digest(
            {name: identity["effective_config_sha256"] for name, identity in initial.items()}
        ),
        "corpus_inputs": corpus,
        "corpus_sha256": approval_json_digest(corpus),
        "final_corpus_files_sha256": {
            f"{world.name}/{rel}": value
            for world in worlds
            for rel, value in lineage_vault_digests(world).items()
        },
        "raw_log": log_records,
        "scenarios_completed": completed,
        "owning_surfaces": env.owning,
        "scripted_llm_calls": len(env.llm.calls),
        "failed_checks": failures,
        "fixture_limits": (
            "Fixture-only offline proof over synthetic temporary vaults and explicit TOML. Real "
            "parser, Chroma, SQLite BM25, hybrid search, Telegram handlers over the aiogram "
            "dispatcher and lineage store run. The installed `vaultmind` console script runs "
            "`index`, `stats`, `research` and `watch` in-process and `index`/`mcp-serve` in "
            "isolated children (MCP over the SDK stdio client); each command builds its own "
            "configured Chroma "
            "store and SQLite keyword index. The direct-watcher scenarios use a verifier-"
            "assembled VaultWatcher/IncrementalWatchHandler over a verifier-wired store and "
            "prove only the handler's own error semantics. Only the LLM, embedding provider, "
            "Telegram transport, article fetch, YouTube search/transcript and Telegram file "
            "boundaries are replaced; faults are injected at the SQLite keyword-index (direct "
            "store: instance wrapper; installed commands: read-only index file) and embedder "
            "boundaries. Legacy bytes come from executing the base commit's marker writer. No "
            "M18 campaign, no native model, provider, embedding-quality or Telegram "
            "qualification."
        ),
    }
    if failures:
        checks_fail = AssertionError(
            f"lineage acceptance failed {len(failures)} check(s): {failures}"
        )
        checks_fail.receipt = receipt  # type: ignore[attr-defined]
        raise checks_fail
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, choices=["retrieval", "mcp", "approval", "lineage"]
    )
    parser.add_argument("--offline", required=True, action="store_true")
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    receipt_path = args.receipt_out.resolve() if args.receipt_out else None
    receipt: dict[str, Any] = {
        "case": args.case,
        "artifact": artifact_identity(),
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
    if args.case == "mcp":
        receipt.update(
            transport=(
                "subprocess: registered `vaultmind` console_scripts entry point (Click group) "
                "-> installed MCP SDK stdio server; client is the installed SDK ClientSession "
                "over stdio_client"
            ),
            boundary_substitutions=[
                "temporary HOME/vault/Chroma/graph/audit paths and explicit fixture TOML",
                "empty OpenAI API key for Embedder construction only; metadata tools make no "
                "embedding request",
                "launcher-side network guard (audit hook) with a blocked-lookup self-test",
            ],
            limitations=(
                "No hosted embedding, model or Telegram evidence; read-only traversal tools "
                "only; fixture vault, not a private vault."
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
    if args.case == "lineage":
        receipt.update(
            transport=(
                "in-process: real VaultParser/Chroma/SQLite BM25/hybrid VaultStore, real "
                "CommandHandlers behind bot.telegram.register_handlers over aiogram "
                "Update/Dispatcher, real LineageStore; installed `vaultmind` console script "
                "(Click group from installed metadata) running `index`, `stats`, `research` and "
                "`watch` in-process (watch on its own asyncio.run loop in a worker thread, real "
                "watchdog events, cancelled gracefully; resources observed passively by a "
                "read-only profile hook that retains them) and `index`/`mcp-serve` in isolated "
                "children (MCP over "
                "the SDK stdio client); verifier-assembled direct VaultWatcher + "
                "IncrementalWatchHandler scenarios are labelled direct-watcher and prove only "
                "handler error semantics"
            ),
            boundary_substitutions=[
                "synthetic token embeddings at the external embedding Embedder factory "
                "(vaultmind.indexer.Embedder replaced only for installed commands; switchable "
                "outage in-process)",
                "scripted llm.client (generating and confirming bot edits, distill, vision)",
                "recording aiogram BaseSession plus synthetic GetFile/stream bytes (no Telegram)",
                "urllib.request.urlopen, yt-dlp search (`_run_yt_search`), transcript API and "
                "`vaultmind.cli._create_llm_client` replaced for consumers and the research "
                "command",
                "content-selective synthetic embedding outage at the Embedder boundary for the "
                "research command",
                "fault injection at SQLite keyword-index write methods (direct stores) or a "
                "read-only index file (installed commands), and the embedder only",
                "launcher-recorded Embedder-substitution identity is metadata, not evidence",
                "temporary HOME/VAULTMIND_HOME and explicit absolute fixture TOML per world",
                "isolated children run behind a launcher-side network audit hook with a "
                "blocked-lookup self-test",
            ],
            limitations="Fixture-only offline proof; not native model/provider/Telegram "
            "qualification; no M18 campaign; sources finalization (M15) not claimed.",
        )
    attempts: list[str] = []

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
            "lineage": "vm-m14-lineage-",
        }[args.case]
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            home = Path(directory)
            for key in list(os.environ):
                if key.startswith("VAULTMIND_"):
                    del os.environ[key]
            if args.case in ("approval", "lineage"):
                secret = ("OPENAI", "ANTHROPIC", "GEMINI", "VOYAGE", "TELEGRAM")
                for key in list(os.environ):
                    if key.upper().startswith(secret) or "API_KEY" in key:
                        del os.environ[key]
            os.environ["HOME"] = str(home)
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.chdir(home)
            if args.case == "retrieval":
                case = retrieval(home, receipt["checks"])
            elif args.case == "approval":
                case = approval_case(home, receipt["checks"])
            elif args.case == "lineage":
                case = lineage_case(home, receipt["checks"])
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
