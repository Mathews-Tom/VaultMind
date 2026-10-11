#!/usr/bin/env python3
"""M11 offline retrieval acceptance: real storage/consumers, labeled provider seams."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["retrieval"])
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
        with tempfile.TemporaryDirectory(prefix="vm-m11-retrieval-") as directory:
            home = Path(directory)
            for key in list(os.environ):
                if key.startswith("VAULTMIND_"):
                    del os.environ[key]
            os.environ["HOME"] = str(home)
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.chdir(home)
            receipt.update(asyncio.run(retrieval(home, receipt["checks"])))
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
