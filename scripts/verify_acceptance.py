#!/usr/bin/env python3
"""Offline acceptance: M11 retrieval and M12 MCP stdio, real consumers, labeled seams."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
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


def artifact_identity() -> dict[str, Any]:
    files = sorted((ROOT / "src" / "vaultmind").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).splitlines()
    return {
        "plan_id": "VM-REMEDIATION-2026-10-11-R1",
        "revision": "R1-A+R1-B",
        "head": head,
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
        Path(vaultmind.__file__).resolve().is_relative_to(ROOT / "src"),
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
    notes = parser.iter_notes()
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
        args=["-I", "-c", MCP_LAUNCHER, str(state), *cli_args],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["retrieval", "mcp"])
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
        prefix = "vm-m11-retrieval-" if args.case == "retrieval" else "vm-m12-mcp-"
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            home = Path(directory)
            for key in list(os.environ):
                if key.startswith("VAULTMIND_"):
                    del os.environ[key]
            os.environ["HOME"] = str(home)
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.chdir(home)
            if args.case == "retrieval":
                case = retrieval(home, receipt["checks"])
            else:
                case = mcp_case(home, receipt["checks"])
            receipt.update(asyncio.run(case))
            if attempts:
                raise RuntimeError(f"Unexpected network attempts: {attempts}")
        receipt["result"] = "pass"
    except Exception as exc:
        receipt.update(result="fail", failure=f"{type(exc).__name__}: {exc}")
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
