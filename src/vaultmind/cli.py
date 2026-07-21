"""CLI entry point for VaultMind.

Commands:
    vaultmind index       — Full index of the vault
    vaultmind bot         — Start the Telegram bot
    vaultmind mcp-serve   — Start the MCP server
    vaultmind graph-build — Build/rebuild the knowledge graph
    vaultmind graph-report — Generate graph analytics report
    vaultmind stats       — Show vault statistics
    vaultmind bench        — Score /recall against a golden question set
    vaultmind init        — Initialize vault structure
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.logging import RichHandler

from vaultmind import __version__

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.embedding_cache import EmbeddingCache
    from vaultmind.llm.client import LLMClient
    from vaultmind.services.review_queue import Applier, ProposalKind, ReviewQueue

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _require_llm_key(settings: object) -> None:
    """Validate that the configured LLM provider has an API key set."""
    from vaultmind.config import Settings

    assert isinstance(settings, Settings)
    provider = settings.llm.provider
    if provider == "ollama":
        return
    key = settings.llm_api_key
    if not key:
        env_var = f"VAULTMIND_{provider.upper()}_API_KEY"
        console.print(f"[red]✗[/red] {provider.capitalize()} API key not set. Set {env_var}.")
        sys.exit(1)


def _create_bm25_index(settings: object) -> BM25Index | None:
    """Create a BM25Index if hybrid search is enabled, otherwise return None."""
    from vaultmind.config import VAULTMIND_HOME, Settings
    from vaultmind.indexer.bm25 import BM25Index

    assert isinstance(settings, Settings)
    if not settings.search.hybrid_enabled:
        return None

    bm25_path = Path(settings.search.bm25_db_path or str(VAULTMIND_HOME / "data" / "bm25.db"))
    return BM25Index(bm25_path)


def _create_embedding_cache(settings: object) -> EmbeddingCache | None:
    """Create an EmbeddingCache if caching is enabled, otherwise return None."""
    from vaultmind.config import VAULTMIND_HOME, Settings
    from vaultmind.indexer.embedding_cache import EmbeddingCache

    assert isinstance(settings, Settings)
    if not settings.embedding.cache_enabled:
        return None

    cache_path = VAULTMIND_HOME / "data" / "embedding_cache.db"
    return EmbeddingCache(cache_path)


def _create_llm_client(settings: object) -> LLMClient:
    """Create an LLM client from settings."""
    from vaultmind.config import Settings
    from vaultmind.llm import create_llm_client

    assert isinstance(settings, Settings)
    base_url = None
    if settings.llm.provider == "ollama":
        base_url = settings.llm.ollama_base_url
    return create_llm_client(
        provider=settings.llm.provider,
        api_key=settings.llm_api_key,
        base_url=base_url,
    )


def _build_review_queue(
    settings: object,
    appliers: Mapping[ProposalKind, Applier] | None = None,
) -> ReviewQueue:
    """Construct the M7 `ReviewQueue` from `[autonomy]` settings.

    Shared by every call site that routes automated mutation proposals
    through the queue (`watch`, `bot`, `auto-tag`).
    """
    from vaultmind.config import VAULTMIND_HOME, Settings
    from vaultmind.services.review_queue import AutonomyThresholds, ReviewQueue

    assert isinstance(settings, Settings)
    queue_db = (
        Path(settings.autonomy.db_path)
        if settings.autonomy.db_path
        else VAULTMIND_HOME / "data" / "review_queue.db"
    )
    return ReviewQueue(
        queue_db,
        AutonomyThresholds(
            block_below=settings.autonomy.block_below,
            skim_below=settings.autonomy.skim_below,
            force_block=settings.autonomy.force_block,
        ),
        appliers=appliers,
    )


def _duplicate_review_subscriber(
    detector: object,
    queue: ReviewQueue,
) -> Callable[[object], Awaitable[None]]:
    """Wrap a `DuplicateDetector` event-bus callback so merge-band matches
    are also routed into the review queue as `DUPLICATE_MERGE` proposals.

    Replaces subscribing `detector.on_note_changed` directly — this wrapper
    calls it first (unchanged detection/caching behavior), then mints queue
    proposals from the freshly cached results.
    """
    from vaultmind.indexer.duplicate_detector import DuplicateDetector
    from vaultmind.services.review_queue import ReviewQueue, mint_duplicate_proposals

    assert isinstance(detector, DuplicateDetector)
    assert isinstance(queue, ReviewQueue)

    async def _handle(event: object) -> None:
        await detector.on_note_changed(event)  # type: ignore[arg-type]
        note = getattr(event, "note", None)
        if note is not None:
            matches = detector.get_results(str(note.path))
            mint_duplicate_proposals(queue, note, matches)

    return _handle


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option("-c", "--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str | None) -> None:
    """VaultMind — AI-powered personal knowledge management."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize ~/.vaultmind directory structure."""
    from vaultmind.config import VAULTMIND_HOME, load_settings

    # Create base structure before loading settings (vault path must exist)
    vault_dir = VAULTMIND_HOME / "vault"
    data_dir = VAULTMIND_HOME / "data"

    vault_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(ctx.obj.get("config_path"))
    vault = settings.vault.path

    vault_folders = [
        "00-inbox",
        "01-daily",
        "02-projects",
        "03-areas",
        "04-resources",
        "05-archive",
        "06-templates",
        "_meta/graph-exports",
        "_meta/indexes",
    ]

    for folder in vault_folders:
        (vault / folder).mkdir(parents=True, exist_ok=True)

    console.print(f"[green]✓[/green] VaultMind initialized at {VAULTMIND_HOME}")
    console.print(f"  vault:  {vault} ({len(vault_folders)} folders)")
    console.print(f"  data:   {data_dir}")

    # Create ChromaDB and graph parent dirs
    settings.chroma.persist_dir.mkdir(parents=True, exist_ok=True)
    settings.graph.persist_path.parent.mkdir(parents=True, exist_ok=True)


@cli.command("suggest-links")
@click.pass_context
def suggest_links(ctx: click.Context) -> None:
    """Scan vault for link suggestions between notes."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.indexer.note_suggester import NoteSuggester
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not api_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)

    graph: KnowledgeGraph | None = None
    if settings.graph.persist_path.exists():
        graph = KnowledgeGraph(settings.graph)

    suggester = NoteSuggester(settings.note_suggestions, store, graph=graph)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status(f"Finding link suggestions for {len(notes)} notes..."):
        results = suggester.scan_vault(notes)

    if not results:
        console.print("[green]✓[/green] No link suggestions found.")
    else:
        total = sum(len(v) for v in results.values())
        for note_path, suggestions in sorted(results.items()):
            console.print(f"\n[bold]{note_path}[/bold]")
            for s in suggestions:
                extras = []
                if s.shared_entities:
                    extras.append(f"shared: {', '.join(s.shared_entities[:3])}")
                if s.graph_distance is not None:
                    extras.append(f"graph: {s.graph_distance}")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                console.print(
                    f"  [cyan]link[/cyan] (score {s.composite_score:.2f},"
                    f" sim {s.similarity:.0%}): {s.target_path}{extra_str}"
                )

        console.print(
            f"\n[bold]Summary:[/bold] {len(results)} notes with suggestions ({total} total links)"
        )

    if cache is not None:
        cache.close()


@cli.command("digest")
@click.option("--days", default=None, type=int, help="Override period_days from config")
@click.option(
    "--save/--no-save",
    default=None,
    help="Override save_to_vault from config",
)
@click.pass_context
def digest(ctx: click.Context, days: int | None, save: bool | None) -> None:
    """Generate a vault digest report."""
    from rich.table import Table

    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.indexer.digest import DigestGenerator
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not api_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    digest_config = settings.digest
    if days is not None:
        from vaultmind.config import DigestConfig

        digest_config = DigestConfig(
            enabled=digest_config.enabled,
            period_days=days,
            schedule_hour=digest_config.schedule_hour,
            timezone=digest_config.timezone,
            save_to_vault=digest_config.save_to_vault,
            send_telegram=digest_config.send_telegram,
            max_trending=digest_config.max_trending,
            max_suggestions=digest_config.max_suggestions,
            connection_threshold_low=digest_config.connection_threshold_low,
            connection_threshold_high=digest_config.connection_threshold_high,
        )

    should_save = digest_config.save_to_vault if save is None else save

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)
    graph = KnowledgeGraph(settings.graph)

    from vaultmind.memory.gaps import GapStore

    gap_store: GapStore | None = None
    if settings.gaps.enabled:
        gap_db = (
            Path(settings.gaps.db_path)
            if settings.gaps.db_path
            else VAULTMIND_HOME / "data" / "gaps.db"
        )
        gap_store = GapStore(gap_db, stale_after_days=settings.gaps.stale_after_days)

    review_queue = _build_review_queue(settings) if settings.autonomy.enabled else None

    generator = DigestGenerator(
        store=store,
        graph=graph,
        parser=parser,
        config=digest_config,
        gap_store=gap_store,
        review_queue=review_queue,
    )

    with console.status("Generating digest..."):
        report = generator.generate()

    # Rich console output
    date_str = report.generated_at.strftime("%Y-%m-%d")
    console.print(f"\n[bold]Daily Digest -- {date_str}[/bold]")
    console.print(
        f"  Period: last {report.period_days} days"
        f" | {report.total_notes} notes"
        f" | {report.total_entities} entities\n"
    )

    if report.new_notes or report.modified_notes:
        console.print("[bold]Activity[/bold]")
        for title in report.new_notes:
            console.print(f"  [green]+[/green] {title}")
        for title in report.modified_notes:
            console.print(f"  [cyan]~[/cyan] {title}")
        console.print()

    if report.trending_entities:
        console.print("[bold]Trending Topics[/bold]")
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Entity")
        table.add_column("Current", justify="right")
        table.add_column("Previous", justify="right")
        table.add_column("Delta", justify="right")
        for entity in report.trending_entities:
            table.add_row(
                entity.name,
                str(entity.current_count),
                str(entity.previous_count),
                f"[green]+{entity.delta}[/green]",
            )
        console.print(table)
        console.print()

    if report.suggested_connections:
        console.print("[bold]Suggested Connections[/bold]")
        for conn in report.suggested_connections:
            console.print(
                f"  [cyan]{conn.note_a}[/cyan] <-> [cyan]{conn.note_b}[/cyan]"
                f" ({conn.similarity:.0%})"
            )
        console.print()

    if report.orphan_notes:
        console.print("[bold]Orphan Notes[/bold]")
        for title in report.orphan_notes:
            console.print(f"  [yellow]{title}[/yellow]")
        console.print()

    if report.skim_pending_count:
        console.print(f"[bold]Pending Review[/bold] ({report.skim_pending_count} SKIM item(s))")
        for summary in report.skim_pending:
            console.print(f"  [magenta]\u2022[/magenta] {summary}")
        console.print()

    if not any(
        [
            report.new_notes,
            report.modified_notes,
            report.trending_entities,
            report.suggested_connections,
            report.orphan_notes,
            report.skim_pending_count,
        ]
    ):
        console.print(f"[dim]No activity in the last {report.period_days} days.[/dim]")

    if should_save:
        with console.status("Saving digest to vault..."):
            dest = generator.save_to_vault(report, settings.vault.path)
        console.print(f"[green]✓[/green] Digest saved to {dest}")

    if cache is not None:
        cache.close()


@cli.command("scan-duplicates")
@click.pass_context
def scan_duplicates(ctx: click.Context) -> None:
    """Scan vault for semantic duplicates and merge candidates."""
    from vaultmind.config import load_settings
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.indexer.duplicate_detector import DuplicateDetector, MatchType
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not api_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)
    detector = DuplicateDetector(settings.duplicate_detection, store)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status(f"Scanning {len(notes)} notes for duplicates..."):
        results = detector.scan_vault(notes)

    if not results:
        console.print("[green]✓[/green] No duplicates or merge candidates found.")
    else:
        total_dup = 0
        total_merge = 0
        for note_path, matches in sorted(results.items()):
            dups = [m for m in matches if m.match_type == MatchType.DUPLICATE]
            merges = [m for m in matches if m.match_type == MatchType.MERGE]
            total_dup += len(dups)
            total_merge += len(merges)

            console.print(f"\n[bold]{note_path}[/bold]")
            for m in dups:
                console.print(f"  [red]duplicate[/red] ({m.similarity:.0%}): {m.match_path}")
            for m in merges:
                console.print(f"  [yellow]merge[/yellow] ({m.similarity:.0%}): {m.match_path}")

        console.print(
            f"\n[bold]Summary:[/bold] {len(results)} notes with matches"
            f" ({total_dup} duplicates, {total_merge} merge candidates)"
        )

    if cache is not None:
        cache.close()


@cli.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    """Full index of the vault into ChromaDB."""
    from vaultmind.config import load_settings
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not api_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    bm25 = _create_bm25_index(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key, cache=cache)
    store = VaultStore(settings.chroma, embedder, bm25=bm25)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status(f"Indexing {len(notes)} notes..."):
        count = store.index_notes(notes, parser)

    console.print(f"[green]✓[/green] Indexed {count} chunks from {len(notes)} notes")

    if cache is not None:
        from vaultmind.indexer.embedding_cache import EmbeddingCache

        assert isinstance(cache, EmbeddingCache)
        cs = cache.stats()
        console.print(
            f"  Embedding cache: {cs['total_entries']} entries, "
            f"{cs['total_size_bytes'] / 1024:.1f} KB"
        )
        cache.close()


@cli.command()
@click.pass_context
def bot(ctx: click.Context) -> None:
    """Start the Telegram bot."""
    from vaultmind.bot.commands import CommandHandlers
    from vaultmind.bot.telegram import create_bot, register_handlers
    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    if not settings.telegram.bot_token:
        console.print("[red]✗[/red] Telegram bot token not set. Set VAULTMIND_TELEGRAM__BOT_TOKEN.")
        sys.exit(1)

    _require_llm_key(settings)

    # Initialize components
    is_openai = settings.embedding.provider == "openai"
    embed_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    cache = _create_embedding_cache(settings)
    bm25 = _create_bm25_index(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, embed_key, cache=cache)
    store = VaultStore(settings.chroma, embedder, bm25=bm25)
    graph = KnowledgeGraph(settings.graph)

    from vaultmind.bot.session_store import SessionStore
    from vaultmind.config import VAULTMIND_HOME

    llm_client = _create_llm_client(settings)
    session_store = SessionStore(VAULTMIND_HOME / "data" / "sessions.db")

    # Knowledge gap ledger (constructed early — wired into both ThinkingPartner and CommandHandlers)
    from vaultmind.memory.gaps import GapStore

    gap_store: GapStore | None = None
    if settings.gaps.enabled:
        gap_db = (
            Path(settings.gaps.db_path)
            if settings.gaps.db_path
            else VAULTMIND_HOME / "data" / "gaps.db"
        )
        gap_store = GapStore(gap_db, stale_after_days=settings.gaps.stale_after_days)

    # Graph context builder for thinking partner
    from vaultmind.graph.context import GraphContextBuilder

    graph_ctx: GraphContextBuilder | None = None
    if settings.llm.graph_context_enabled:
        graph_ctx = GraphContextBuilder(
            knowledge_graph=graph,
            llm_client=llm_client,
            fast_model=settings.llm.fast_model,
        )

    thinking = ThinkingPartner(
        settings.llm,
        settings.telegram,
        llm_client,
        session_store=session_store,
        graph_context_builder=graph_ctx,
        vault_root=settings.vault.path,
        parser=parser,
        distill_config=settings.distill,
        gap_store=gap_store,
        score_floor=settings.bench.score_floor,
    )

    # Duplicate detection
    from vaultmind.indexer.duplicate_detector import DuplicateDetector

    detector: DuplicateDetector | None = None
    if settings.duplicate_detection.enabled:
        detector = DuplicateDetector(settings.duplicate_detection, store)

    # Review queue (M7) — unifies tag/duplicate/contradiction/maturation
    # mutation proposals behind one AUTO/SKIM/BLOCK lane model.
    review_queue: ReviewQueue | None = None
    if settings.autonomy.enabled:
        review_queue = _build_review_queue(settings)

    # Note suggestions
    from vaultmind.indexer.note_suggester import NoteSuggester

    suggester: NoteSuggester | None = None
    if settings.note_suggestions.enabled:
        suggester = NoteSuggester(settings.note_suggestions, store, graph=graph)

    # Voice transcription
    from vaultmind.bot.transcribe import Transcriber

    transcriber: Transcriber | None = None
    if settings.voice.enabled and settings.openai_api_key:
        transcriber = Transcriber(settings.voice, settings.openai_api_key)

    # Evolution detector
    from vaultmind.graph.evolution import EvolutionDetector

    evolution_detector: EvolutionDetector | None = None
    if settings.evolution.enabled:
        evolution_detector = EvolutionDetector(
            knowledge_graph=graph,
            confidence_drift_threshold=settings.evolution.confidence_drift_threshold,
            stale_days=settings.evolution.stale_days,
            min_confidence_for_stale=settings.evolution.min_confidence_for_stale,
        )

    # Maturation pipeline
    from vaultmind.pipeline.maturation import MaturationPipeline

    maturation_pipeline: MaturationPipeline | None = None
    if settings.maturation.enabled:
        synthesis_model = settings.maturation.synthesis_model or settings.llm.thinking_model
        mat_config = settings.maturation
        # Override synthesis_model with resolved value
        mat_config_dict = mat_config.model_dump()
        mat_config_dict["synthesis_model"] = synthesis_model
        from vaultmind.config import MaturationConfig

        resolved_mat_config = MaturationConfig(**mat_config_dict)
        maturation_pipeline = MaturationPipeline(
            config=resolved_mat_config,
            collection=store._collection,
            knowledge_graph=graph,
            llm=llm_client,
            vault_root=settings.vault.path,
        )

    # Episodic memory
    from vaultmind.memory.store import EpisodeStore

    episode_store: EpisodeStore | None = None
    if settings.episodic.enabled:
        episode_db = (
            Path(settings.episodic.db_path)
            if settings.episodic.db_path
            else VAULTMIND_HOME / "data" / "episodes.db"
        )
        episode_store = EpisodeStore(episode_db)

    # Procedural memory
    from vaultmind.memory.procedural import ProceduralMemory

    procedural_memory: ProceduralMemory | None = None
    if settings.procedural.enabled:
        procedural_db = (
            Path(settings.procedural.db_path)
            if settings.procedural.db_path
            else VAULTMIND_HOME / "data" / "procedural.db"
        )
        procedural_memory = ProceduralMemory(procedural_db)
        console.print("[green]✓[/green] Procedural memory enabled")

    # Preference store (for insight loop + bot handlers)
    from vaultmind.tracking.preferences import PreferenceStore

    pref_db = (
        Path(settings.tracking.db_path)
        if settings.tracking.db_path
        else VAULTMIND_HOME / "data" / "preferences.db"
    )
    preference_store: PreferenceStore | None = (
        PreferenceStore(pref_db) if settings.tracking.enabled else None
    )

    handlers = CommandHandlers(
        settings=settings,
        store=store,
        graph=graph,
        parser=parser,
        thinking=thinking,
        llm_client=llm_client,
        duplicate_detector=detector,
        note_suggester=suggester,
        transcriber=transcriber,
        evolution_detector=evolution_detector,
        maturation_pipeline=maturation_pipeline,
        episode_store=episode_store,
        procedural_memory=procedural_memory,
        gap_store=gap_store,
        review_queue=review_queue,
    )

    tg_bot, dp = create_bot(settings.telegram.bot_token)
    register_handlers(handlers)

    # Proactive notifier — hoisted here (not gated on maturation) so
    # contradiction escalation can notify regardless of maturation config.
    from vaultmind.bot.notifier import Notifier

    notifier: Notifier | None = None
    if settings.telegram.notification_chat_id:
        notifier = Notifier(bot=tg_bot, chat_id=settings.telegram.notification_chat_id)

    # Contradiction detection
    from vaultmind.contradiction.detector import ContradictionDetector

    contradiction_detector: ContradictionDetector | None = None
    if settings.contradiction.enabled and detector is not None:
        on_escalate = None
        if notifier is not None:
            from vaultmind.bot.handlers.autonomy import build_block_notifier

            on_escalate = build_block_notifier(notifier)
        contradiction_model = settings.contradiction.detection_model or settings.llm.fast_model
        contradiction_detector = ContradictionDetector(
            settings.contradiction,
            detector,
            llm_client,
            contradiction_model,
            settings.vault.path,
            parser,
            ranking_config=settings.ranking,
            gap_store=gap_store,
            on_escalate=on_escalate,
            review_queue=review_queue,
        )

    # Wire up incremental watch mode
    from vaultmind.graph.maintenance import GraphMaintainer
    from vaultmind.vault.events import (
        AnyVaultEvent,
        NoteCreatedEvent,
        NoteDeletedEvent,
        NoteModifiedEvent,
        VaultEventBus,
    )
    from vaultmind.vault.watch_handler import IncrementalWatchHandler
    from vaultmind.vault.watcher import VaultWatcher

    event_bus = VaultEventBus()

    # Subscribe duplicate detection and note suggestions to watch events
    if detector is not None:
        if review_queue is not None:
            dup_subscriber = _duplicate_review_subscriber(detector, review_queue)
            event_bus.subscribe(NoteCreatedEvent, dup_subscriber)  # type: ignore[arg-type]
            event_bus.subscribe(NoteModifiedEvent, dup_subscriber)  # type: ignore[arg-type]
        else:
            event_bus.subscribe(NoteCreatedEvent, detector.on_note_changed)  # type: ignore[arg-type]
            event_bus.subscribe(NoteModifiedEvent, detector.on_note_changed)  # type: ignore[arg-type]
    if suggester is not None:
        event_bus.subscribe(NoteCreatedEvent, suggester.on_note_changed)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, suggester.on_note_changed)  # type: ignore[arg-type]
    if contradiction_detector is not None:
        event_bus.subscribe(
            NoteCreatedEvent,
            contradiction_detector.on_note_changed,  # type: ignore[arg-type]
        )
        event_bus.subscribe(
            NoteModifiedEvent,
            contradiction_detector.on_note_changed,  # type: ignore[arg-type]
        )

    # Subscribe graph maintenance to note deletion events
    maintainer = GraphMaintainer(graph)
    event_bus.subscribe(NoteDeletedEvent, maintainer.on_note_deleted)  # type: ignore[arg-type]

    watch_handler = IncrementalWatchHandler(
        config=settings.watch,
        parser=parser,
        store=store,
        event_bus=event_bus,
    )
    watcher = VaultWatcher(config=settings.vault, on_change=watch_handler.handle_change)

    provider = settings.llm.provider
    model = settings.llm.thinking_model
    console.print(f"[green]✓[/green] Starting Telegram bot (LLM: {provider}/{model})...")
    console.print(f"  Watch mode: debounce={settings.watch.debounce_ms}ms")

    # Maturation scheduler
    from datetime import timedelta

    from vaultmind.services.scheduler import ScheduledJob, SchedulerService

    scheduler: SchedulerService | None = None
    if maturation_pipeline is not None:
        if review_queue is not None:
            from vaultmind.services.review_queue import ProposalKind

            def _maturation_applier(payload: dict[str, Any]) -> str:
                assert maturation_pipeline is not None
                fingerprint = str(payload["fingerprint"])
                match = next(
                    (c for c in maturation_pipeline.discover() if c.fingerprint == fingerprint),
                    None,
                )
                if match is None:
                    return "cluster no longer available (dismissed or already synthesized)"
                return maturation_pipeline.synthesize(match)

            review_queue.register_applier(ProposalKind.MATURATION_SYNTHESIS, _maturation_applier)

        async def _maturation_digest() -> None:
            assert maturation_pipeline is not None
            clusters = maturation_pipeline.discover()
            if clusters:
                logging.getLogger(__name__).info(
                    "Maturation digest: %d clusters ready", len(clusters)
                )
                if review_queue is not None:
                    from vaultmind.services.review_queue import Impact, Lane, ProposalKind

                    for cluster in clusters:
                        review_queue.propose(
                            ProposalKind.MATURATION_SYNTHESIS,
                            cluster.score,
                            Impact.MEDIUM,
                            f"Synthesize permanent note from {len(cluster.note_paths)} "
                            f"notes ({cluster.top_entity})",
                            {"fingerprint": cluster.fingerprint},
                            lane_override=Lane.SKIM,
                        )
            maturation_pipeline.mark_run()

        maturation_job = ScheduledJob.legacy(
            name="maturation_digest",
            interval=timedelta(days=7),
            execute=_maturation_digest,
        )
        sched_state = (
            Path(settings.scheduler.state_path)
            if settings.scheduler.state_path
            else Path.home() / ".vaultmind" / "data" / "scheduler_state.json"
        )

        # Compound loop jobs (notifier used below was constructed above,
        # hoisted out of this block so contradiction escalation can use it too)
        jobs: list[ScheduledJob] = [maturation_job]

        if settings.loops.insight_enabled and preference_store is not None:
            from vaultmind.services.loops.insight_loop import create_insight_executor
            from vaultmind.services.scheduler import resolve_cron_expr

            insight_exec = create_insight_executor(preference_store)
            insight_cron = resolve_cron_expr(
                settings.loops.insight_schedule,
                settings.loops.insight_interval_days,
            )
            jobs.append(
                ScheduledJob(
                    name="insight_loop",
                    interval=timedelta(days=settings.loops.insight_interval_days),
                    cron_expr=insight_cron,
                    execute=insight_exec,
                )
            )

        if settings.loops.evolution_enabled and evolution_detector is not None:
            from vaultmind.services.loops.evolution_loop import create_evolution_executor
            from vaultmind.services.scheduler import resolve_cron_expr

            evolution_exec = create_evolution_executor(evolution_detector)
            evolution_cron = resolve_cron_expr(
                settings.loops.evolution_schedule,
                settings.loops.evolution_interval_days,
            )
            jobs.append(
                ScheduledJob(
                    name="evolution_loop",
                    interval=timedelta(days=settings.loops.evolution_interval_days),
                    cron_expr=evolution_cron,
                    execute=evolution_exec,
                )
            )

        if (
            settings.loops.procedural_enabled
            and settings.procedural.enabled
            and procedural_memory is not None
            and episode_store is not None
        ):
            from vaultmind.services.loops.procedural_loop import create_procedural_executor
            from vaultmind.services.scheduler import resolve_cron_expr

            proc_model = settings.procedural.synthesis_model or settings.llm.fast_model
            procedural_exec = create_procedural_executor(
                procedural_memory=procedural_memory,
                episode_store=episode_store,
                llm_client=llm_client,
                model=proc_model,
            )
            procedural_cron = resolve_cron_expr(
                settings.loops.procedural_schedule,
                settings.loops.procedural_interval_days,
            )
            jobs.append(
                ScheduledJob(
                    name="procedural_loop",
                    interval=timedelta(days=settings.loops.procedural_interval_days),
                    cron_expr=procedural_cron,
                    execute=procedural_exec,
                )
            )
        scheduler = SchedulerService(
            jobs=jobs,
            state_path=sched_state,
            notifier=notifier,
        )

    # Bridge vault events to scheduler for event-triggered loops
    if scheduler is not None:

        async def _on_note_event_for_scheduler(event: AnyVaultEvent) -> None:
            scheduler.record_event(type(event).__name__)

        event_bus.subscribe(NoteCreatedEvent, _on_note_event_for_scheduler)
        event_bus.subscribe(NoteModifiedEvent, _on_note_event_for_scheduler)

    async def _run_bot_with_watcher() -> None:
        watcher.start()
        scheduler_task = None
        if scheduler is not None:
            scheduler_task = asyncio.create_task(scheduler.run())
        try:
            await dp.start_polling(tg_bot)
        finally:
            if scheduler is not None:
                scheduler.stop()
            if scheduler_task is not None:
                scheduler_task.cancel()
            watcher.stop()

    asyncio.run(_run_bot_with_watcher())


@cli.command()
@click.option("--graph", "with_graph", is_flag=True, help="Enable graph re-extraction")
@click.pass_context
def watch(ctx: click.Context, with_graph: bool) -> None:
    """Watch vault for changes and incrementally re-index."""
    from vaultmind.config import load_settings
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.vault import VaultParser
    from vaultmind.vault.events import VaultEventBus
    from vaultmind.vault.watch_handler import IncrementalWatchHandler
    from vaultmind.vault.watcher import VaultWatcher

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    embed_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not embed_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, embed_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)

    event_bus = VaultEventBus()

    # Duplicate detection and note suggestions
    from vaultmind.indexer.duplicate_detector import DuplicateDetector
    from vaultmind.indexer.note_suggester import NoteSuggester
    from vaultmind.vault.events import NoteCreatedEvent, NoteModifiedEvent

    if settings.duplicate_detection.enabled:
        detector = DuplicateDetector(settings.duplicate_detection, store)
        if settings.autonomy.enabled:
            dup_subscriber = _duplicate_review_subscriber(detector, _build_review_queue(settings))
            event_bus.subscribe(NoteCreatedEvent, dup_subscriber)  # type: ignore[arg-type]
            event_bus.subscribe(NoteModifiedEvent, dup_subscriber)  # type: ignore[arg-type]
        else:
            event_bus.subscribe(NoteCreatedEvent, detector.on_note_changed)  # type: ignore[arg-type]
            event_bus.subscribe(NoteModifiedEvent, detector.on_note_changed)  # type: ignore[arg-type]

    # Optional graph re-extraction
    graph = None
    extractor = None
    watch_config = settings.watch
    if with_graph:
        from vaultmind.config import WatchConfig
        from vaultmind.graph import EntityExtractor, KnowledgeGraph

        _require_llm_key(settings)
        graph = KnowledgeGraph(settings.graph)
        extraction_model = settings.graph.extraction_model or settings.llm.thinking_model
        llm_client = _create_llm_client(settings)
        extractor = EntityExtractor(settings.graph, llm_client, model=extraction_model)
        # Force reextract_graph on since user explicitly opted in via --graph
        watch_config = WatchConfig(
            debounce_ms=settings.watch.debounce_ms,
            hash_stability_check=settings.watch.hash_stability_check,
            reextract_graph=True,
            batch_graph_interval_seconds=settings.watch.batch_graph_interval_seconds,
        )

    # Note suggestions (graph available only if --graph was used)
    if settings.note_suggestions.enabled:
        watch_suggester = NoteSuggester(settings.note_suggestions, store, graph=graph)
        event_bus.subscribe(NoteCreatedEvent, watch_suggester.on_note_changed)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, watch_suggester.on_note_changed)  # type: ignore[arg-type]

    handler = IncrementalWatchHandler(
        config=watch_config,
        parser=parser,
        store=store,
        event_bus=event_bus,
        graph=graph,
        extractor=extractor,
    )
    watcher = VaultWatcher(config=settings.vault, on_change=handler.handle_change)

    console.print(f"[green]✓[/green] Watching {settings.vault.path}")
    console.print(f"  Debounce: {watch_config.debounce_ms}ms")
    console.print(f"  Hash stability: {watch_config.hash_stability_check}")
    console.print(f"  Graph re-extraction: {watch_config.reextract_graph}")

    async def _run_watch() -> None:
        watcher.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            watcher.stop()
            if cache is not None:
                cache.close()

    try:
        asyncio.run(_run_watch())
    except KeyboardInterrupt:
        watcher.stop()
        if cache is not None:
            cache.close()
        console.print("\n[yellow]Watcher stopped.[/yellow]")


def _load_tag_vocabulary(path: Path) -> set[str]:
    """Load the durable approved-tag vocabulary (M7 review-queue applier state)."""
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("approved", []))


def _save_tag_vocabulary(path: Path, tags: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"approved": sorted(tags)}, indent=2))


@cli.command("auto-tag")
@click.option("--apply", "do_apply", is_flag=True, help="Route suggestions through the queue")
@click.option("--approve-skim", is_flag=True, help="Approve all pending SKIM-lane tag proposals")
@click.option("--show-pending", is_flag=True, help="Show pending tag proposals and exit")
@click.pass_context
def auto_tag(
    ctx: click.Context,
    do_apply: bool,
    approve_skim: bool,
    show_pending: bool,
) -> None:
    """Auto-tag vault notes using LLM classification, via the review queue.

    By default runs in dry-run mode — shows suggestions without proposing
    them. Use --apply to route suggestions through the review queue:
    already-known-vocabulary tags apply immediately (AUTO lane); novel tags
    are queued for review (SKIM lane) — see --show-pending/--approve-skim,
    the bot's /review command, or the next `vaultmind digest`.
    """
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.indexer.auto_tagger import AutoTagger
    from vaultmind.services.review_queue import (
        Impact,
        Lane,
        ProposalKind,
        migrate_quarantine,
    )
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))
    _require_llm_key(settings)

    model = settings.auto_tag.tagging_model or settings.llm.fast_model
    llm_client = _create_llm_client(settings)
    tagger = AutoTagger(settings.auto_tag, llm_client, model)

    vocab_path = VAULTMIND_HOME / "data" / "tag_vocabulary.json"

    def _apply_tag_application(payload: dict[str, Any]) -> str:
        note_path = settings.vault.path / str(payload["note_path"])
        tags = list(payload["tags"])
        if not note_path.exists():
            raise FileNotFoundError(note_path)
        tagger.apply_tags(note_path, tags)
        return f"applied {len(tags)} tag(s) to {payload['note_path']}"

    def _apply_tag_vocabulary(payload: dict[str, Any]) -> str:
        tag = str(payload["tag"])
        approved = _load_tag_vocabulary(vocab_path)
        approved.add(tag)
        _save_tag_vocabulary(vocab_path, approved)
        return f"'{tag}' added to vocabulary"

    queue = _build_review_queue(
        settings,
        appliers={
            ProposalKind.TAG_APPLICATION: _apply_tag_application,
            ProposalKind.TAG_VOCABULARY: _apply_tag_vocabulary,
        },
    )

    quarantine_path = VAULTMIND_HOME / "data" / "tag_quarantine.json"
    migrated = migrate_quarantine(queue, quarantine_path)
    if migrated:
        console.print(
            f"[cyan]Migrated {migrated} pending tag(s) from quarantine to the review queue.[/cyan]"
        )

    if approve_skim:
        approved = queue.approve_all(lane=Lane.SKIM)
        console.print(f"[green]✓[/green] Approved {len(approved)} SKIM-lane tag proposal(s).")
        return

    if show_pending:
        pending = queue.list_pending()
        if not pending:
            console.print("[green]No pending tag proposals.[/green]")
            return
        console.print("[bold]Pending tag proposals:[/bold]")
        for p in pending:
            console.print(f"  [{p.lane.label}] {p.summary}")
        return

    parser = VaultParser(settings.vault)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    vault_tags = tagger.collect_vault_tags(notes) | _load_tag_vocabulary(vocab_path)
    console.print(f"  Vault tag vocabulary: {len(vault_tags)} unique tags")

    # Filter to notes without many tags (likely need tagging)
    candidates = [n for n in notes if len(n.tags) <= 1]
    console.print(f"  Candidates (≤1 tag): {len(candidates)} of {len(notes)} notes")

    if not candidates:
        console.print("[green]✓[/green] All notes are well-tagged.")
        return

    with console.status(f"Classifying {len(candidates)} notes..."):
        suggestions = tagger.suggest_batch(candidates, vault_tags)

    if not suggestions:
        console.print("[green]✓[/green] No tag suggestions generated.")
        return

    for s in suggestions:
        tag_str = ", ".join(s.suggested_tags) if s.suggested_tags else "(none from vocab)"
        new_str = f" + new: {', '.join(s.new_tags)}" if s.new_tags else ""
        console.print(f"  [bold]{s.note_title}[/bold]: [cyan]{tag_str}[/cyan]{new_str}")

    console.print(f"\n[bold]Summary:[/bold] {len(suggestions)} notes with suggestions")

    if not do_apply:
        console.print("  Run with --apply to route these through the review queue.")
        return

    auto_count = 0
    skim_count = 0
    for s in suggestions:
        if s.suggested_tags:
            proposal = queue.propose(
                ProposalKind.TAG_APPLICATION,
                confidence=1.0,
                impact=Impact.LOW,
                summary=f"Apply {len(s.suggested_tags)} known tag(s) to '{s.note_title}'",
                payload={"note_path": s.note_path, "tags": s.suggested_tags},
            )
            if proposal.lane is Lane.AUTO:
                auto_count += 1
            else:
                skim_count += 1
        for tag in s.new_tags:
            vocab_proposal = queue.propose(
                ProposalKind.TAG_VOCABULARY,
                confidence=0.6,
                impact=Impact.LOW,
                summary=f"New tag vocabulary: '{tag}'",
                payload={"tag": tag},
            )
            if vocab_proposal.lane is not Lane.AUTO:
                skim_count += 1

    console.print(
        f"[green]✓[/green] {auto_count} tag-application(s) applied immediately (AUTO); "
        f"{skim_count} queued for review (SKIM) — see `auto-tag --show-pending`, "
        "the bot's /review command, or the next `vaultmind digest`."
    )


@cli.command("tag-synonyms")
@click.option(
    "--min-similarity",
    default=0.75,
    show_default=True,
    help="Minimum string similarity threshold (0.0–1.0)",
)
@click.option(
    "--min-co-occurrence",
    default=0.5,
    show_default=True,
    help="Minimum co-occurrence ratio threshold (0.0–1.0)",
)
@click.pass_context
def tag_synonyms(ctx: click.Context, min_similarity: float, min_co_occurrence: float) -> None:
    """Detect likely tag synonyms and suggest merges (zero LLM cost)."""
    from rich.table import Table

    from vaultmind.config import load_settings
    from vaultmind.indexer.tag_analyzer import compute_tag_stats, find_synonyms
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))
    parser = VaultParser(settings.vault)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status("Analysing tags..."):
        tag_counts, co_occurrences = compute_tag_stats(notes)
        synonyms = find_synonyms(
            tag_counts,
            co_occurrences,
            min_similarity=min_similarity,
            min_co_occurrence=min_co_occurrence,
        )

    console.print(
        f"\n[bold]Tag vocabulary:[/bold] {len(tag_counts)} unique tags across {len(notes)} notes"
    )

    if not synonyms:
        console.print("[green]✓[/green] No synonym candidates found.")
        return

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Tag A")
    table.add_column("Tag B")
    table.add_column("Similarity", justify="right")
    table.add_column("Co-occur ratio", justify="right")
    table.add_column("Suggested canonical")

    for s in synonyms:
        table.add_row(
            s.tag_a,
            s.tag_b,
            f"{s.similarity:.0%}",
            f"{s.co_occurrence_ratio:.0%}",
            f"[green]{s.suggested_canonical}[/green]",
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {len(synonyms)} synonym candidate(s) found.")


@cli.command("graph-build")
@click.option("--full", is_flag=True, help="Rebuild entire graph from scratch")
@click.pass_context
def graph_build(ctx: click.Context, full: bool) -> None:
    """Build or rebuild the knowledge graph from vault notes."""
    from vaultmind.config import load_settings
    from vaultmind.graph import EntityExtractor, KnowledgeGraph
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    _require_llm_key(settings)

    parser = VaultParser(settings.vault)
    graph = KnowledgeGraph(settings.graph)

    # Resolve extraction model: graph config override or LLM thinking model
    extraction_model = settings.graph.extraction_model or settings.llm.thinking_model

    llm_client = _create_llm_client(settings)
    extractor = EntityExtractor(
        settings.graph,
        llm_client,
        model=extraction_model,
    )

    if full:
        console.print("[yellow]Rebuilding graph from scratch...[/yellow]")
        graph._graph.clear()

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status(f"Extracting entities from {len(notes)} notes..."):
        extraction_stats = extractor.extract_and_update_graph(notes, graph)

    console.print(
        f"[green]✓[/green] Graph updated: "
        f"+{extraction_stats['entities_added']} entities, "
        f"+{extraction_stats['relationships_added']} relationships"
    )
    gs = graph.stats
    console.print(f"  Total: {gs['nodes']} nodes, {gs['edges']} edges")


@cli.command("graph-maintain")
@click.pass_context
def graph_maintain(ctx: click.Context) -> None:
    """Run graph maintenance: prune stale references and remove orphans."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.graph.maintenance import GraphMaintainer
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    if not settings.graph.persist_path.exists():
        console.print("[yellow]No graph found. Run `graph-build` first.[/yellow]")
        return

    parser = VaultParser(settings.vault)
    graph = KnowledgeGraph(settings.graph)
    maintainer = GraphMaintainer(graph)

    before = graph.stats

    with console.status("Parsing vault for existing note paths..."):
        notes = parser.iter_notes()
        existing_paths = {str(n.path) for n in notes}

    with console.status("Running graph maintenance..."):
        stats = maintainer.full_maintenance(existing_paths)

    after = graph.stats

    console.print("[green]✓[/green] Graph maintenance complete")
    console.print(f"  Stale node refs pruned: {stats['nodes_pruned']}")
    console.print(f"  Stale edge refs pruned: {stats['edges_pruned']}")
    console.print(f"  Orphan entities removed: {stats['orphans_removed']}")
    console.print(
        f"  Graph: {before['nodes']} → {after['nodes']} nodes,"
        f" {before['edges']} → {after['edges']} edges"
    )


@cli.command("graph-report")
@click.pass_context
def graph_report(ctx: click.Context) -> None:
    """Generate and save a knowledge graph analytics report."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph

    settings = load_settings(ctx.obj.get("config_path"))
    graph = KnowledgeGraph(settings.graph)

    report = graph.to_markdown_summary()
    report_path = settings.vault.path / settings.vault.meta_folder / "graph-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    console.print(f"[green]✓[/green] Report written to {report_path}")
    console.print(report)


@cli.command()
@click.argument("query")
@click.option("--max-results", default=None, type=int, help="Override max results from config")
@click.pass_context
def research(ctx: click.Context, query: str, max_results: int | None) -> None:
    """Run a research pipeline: search YouTube, fetch transcripts, analyze, create vault notes."""
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.research.pipeline import ResearchConfig as PipelineConfig
    from vaultmind.research.pipeline import run_research_pipeline
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))
    _require_llm_key(settings)

    is_openai = settings.embedding.provider == "openai"
    embed_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not embed_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, embed_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)
    llm_client = _create_llm_client(settings)

    pipeline_config = PipelineConfig(
        max_results=max_results or settings.research.max_results,
        output_folder=settings.research.output_folder,
        youtube_language=settings.research.youtube_language,
    )

    model = settings.llm.thinking_model

    from vaultmind.memory.gaps import GapStore

    gap_store: GapStore | None = None
    if settings.gaps.enabled:
        gap_db = (
            Path(settings.gaps.db_path)
            if settings.gaps.db_path
            else VAULTMIND_HOME / "data" / "gaps.db"
        )
        gap_store = GapStore(gap_db, stale_after_days=settings.gaps.stale_after_days)

    async def _run() -> None:
        with console.status(f"Researching: {query}..."):
            result = await run_research_pipeline(
                query=query,
                vault_root=settings.vault.path,
                llm_client=llm_client,
                store=store,
                parser=parser,
                config=pipeline_config,
                model=model,
            )

        console.print(f"\n[green]✓[/green] Research complete: {query}")
        console.print(f"  Sources created: {len(result.sources_created)}")
        for p in result.sources_created:
            console.print(f"    {p.relative_to(settings.vault.path)}")
        if result.summary_path.exists():
            console.print(f"  Summary: {result.summary_path.relative_to(settings.vault.path)}")
        console.print(f"\n[bold]Summary:[/bold] {result.analysis_summary}")

        if gap_store is not None and result.summary_path.exists():
            resolution_ref = str(result.summary_path.relative_to(settings.vault.path))
            closed = gap_store.close_from_research(query, resolution_ref)
            if closed is not None:
                console.print(
                    f"[green]✓[/green] Closed gap {closed.gap_id[:8]} -> {resolution_ref}"
                )

    asyncio.run(_run())

    if cache is not None:
        cache.close()
    if gap_store is not None:
        gap_store.close()


@cli.command()
@click.option("--days", default=30, type=int, help="Analysis period in days")
@click.option("--save", "save_report", is_flag=True, help="Save report to vault")
@click.pass_context
def learn(ctx: click.Context, days: int, save_report: bool) -> None:
    """Analyze usage patterns and generate preference insights."""
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.tracking import (
        PreferenceStore,
        analyze_preferences,
        generate_preference_report,
    )

    settings = load_settings(ctx.obj.get("config_path"))

    db_path = (
        Path(settings.tracking.db_path)
        if settings.tracking.db_path
        else VAULTMIND_HOME / "data" / "preferences.db"
    )

    if not db_path.exists():
        console.print(
            "[yellow]No preference data yet.[/yellow] Use the bot to generate interactions."
        )
        return

    review_queue = _build_review_queue(settings) if settings.autonomy.enabled else None
    store = PreferenceStore(db_path)
    insights = analyze_preferences(store, days=days, review_queue=review_queue)
    report = generate_preference_report(insights)

    console.print(report)

    if save_report:
        report_path = settings.vault.path / settings.vault.meta_folder / "usage-insights.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        console.print(f"\n[green]✓[/green] Report saved to {report_path}")

    if insights.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in insights.recommendations:
            console.print(f"  {rec}")

    store.close()


@cli.command("synthesize-workflows")
@click.option(
    "--min-episodes",
    default=3,
    show_default=True,
    type=int,
    help="Minimum episodes per pattern",
)
@click.pass_context
def synthesize_workflows(ctx: click.Context, min_episodes: int) -> None:
    """Mine episodic memory for workflow patterns."""
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.memory.procedural import ProceduralMemory
    from vaultmind.memory.store import EpisodeStore

    settings = load_settings(ctx.obj.get("config_path"))

    if not settings.procedural.enabled:
        console.print(
            "[yellow]Procedural memory is disabled.[/yellow] "
            "Set [procedural] enabled = true in config to use this feature."
        )
        return

    _require_llm_key(settings)

    episode_db = (
        Path(settings.episodic.db_path)
        if settings.episodic.db_path
        else VAULTMIND_HOME / "data" / "episodes.db"
    )
    if not episode_db.exists():
        console.print(
            "[yellow]No episodic memory found.[/yellow] "
            "Use /decide and /outcome in the bot to create episodes first."
        )
        return

    procedural_db = (
        Path(settings.procedural.db_path)
        if settings.procedural.db_path
        else VAULTMIND_HOME / "data" / "procedural.db"
    )

    synthesis_model = settings.procedural.synthesis_model or settings.llm.fast_model
    llm_client = _create_llm_client(settings)
    episode_store = EpisodeStore(episode_db)
    procedural = ProceduralMemory(procedural_db)

    resolved_count = len(episode_store.query_resolved())
    console.print(f"  Resolved episodes available: {resolved_count}")

    with console.status("Synthesizing workflows..."):
        workflows = procedural.synthesize_workflows(
            episode_store=episode_store,
            llm_client=llm_client,
            model=synthesis_model,
            min_episodes=min_episodes,
        )

    if not workflows:
        console.print(
            "[yellow]No workflow patterns found.[/yellow] "
            f"Need >= {min_episodes} resolved episodes with shared entities."
        )
    else:
        console.print(f"[green]✓[/green] Synthesized {len(workflows)} workflow(s):")
        for wf in workflows:
            console.print(f"\n  [bold]{wf.name}[/bold] ({wf.workflow_id})")
            console.print(f"  {wf.description}")
            console.print(f"  Trigger: {wf.trigger_pattern}")
            console.print(f"  Steps: {len(wf.steps)}")
            for i, step in enumerate(wf.steps, 1):
                console.print(f"    {i}. {step}")
            console.print(f"  Source episodes: {len(wf.source_episodes)}")

    episode_store.close()
    procedural.close()


@cli.command("mcp-serve")
@click.option(
    "--profile",
    default=None,
    type=str,
    help="MCP profile name (e.g., researcher, planner, full). Default: researcher (read-only)",
)
@click.pass_context
def mcp_serve(ctx: click.Context, profile: str | None) -> None:
    """Start the MCP server for agent integration."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.mcp.server import create_mcp_server
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    from vaultmind.config import VAULTMIND_HOME
    from vaultmind.mcp.auth import AuditLogger, ProfileEnforcer
    from vaultmind.mcp.profiles import load_profile

    profile_name = profile or "researcher"
    if profile is None:
        import sys as _sys

        logging.getLogger(__name__).warning(
            "No --profile specified; defaulting to 'researcher' (read-only)"
        )
        print(
            "Warning: No --profile specified; defaulting to 'researcher' (read-only)",
            file=_sys.stderr,
        )

    config_profiles = settings.mcp.profiles if settings.mcp.profiles else None
    policy = load_profile(profile_name, config_profiles=config_profiles)
    enforcer = ProfileEnforcer(policy, settings.vault.path)
    audit_log = AuditLogger(VAULTMIND_HOME / "data" / "mcp_audit.jsonl")

    console.print(f"  Profile: {profile_name} ({policy.description})")
    console.print(f"  Write: {'enabled' if policy.write_enabled else 'disabled'}")
    console.print(f"  Folder scope: {', '.join(policy.folder_scope)}")

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)
    graph = KnowledgeGraph(settings.graph)

    server = create_mcp_server(
        settings.vault.path,
        store,
        graph,
        parser,
        enforcer=enforcer,
        audit_logger=audit_log,
    )

    console.print("[green]✓[/green] MCP server starting...")

    from mcp.server.stdio import stdio_server

    asyncio.run(_run_mcp(server, stdio_server))


async def _run_mcp(server: object, stdio_server: object) -> None:
    """Run MCP server with stdio transport."""
    async with stdio_server() as (read, write):  # type: ignore[operator]
        init_opts = server.create_initialization_options()  # type: ignore[attr-defined]
        await server.run(read, write, init_opts)  # type: ignore[attr-defined]


@cli.command()
@click.option(
    "--metadata-audit", is_flag=True, help="Audit metadata coverage across indexed chunks"
)
@click.pass_context
def stats(ctx: click.Context, metadata_audit: bool) -> None:
    """Show vault and graph statistics."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    parser = VaultParser(settings.vault)
    notes = parser.iter_notes()

    # Count by type
    type_counts: dict[str, int] = {}
    for note in notes:
        t = note.note_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    console.print("\n[bold]VaultMind Statistics[/bold]\n")
    console.print(f"[bold]Vault:[/bold] {settings.vault.path}")
    console.print(f"  Total notes: {len(notes)}")
    for ntype, count in sorted(type_counts.items()):
        console.print(f"  {ntype}: {count}")

    if settings.graph.persist_path.exists():
        graph = KnowledgeGraph(settings.graph)
        gs = graph.stats
        console.print("\n[bold]Knowledge Graph:[/bold]")
        console.print(f"  Nodes: {gs['nodes']}")
        console.print(f"  Edges: {gs['edges']}")
        console.print(f"  Density: {gs['density']:.3f}")
        console.print(f"  Components: {gs['components']}")
        console.print(f"  Orphans: {gs['orphans']}")

    from vaultmind.config import VAULTMIND_HOME

    cache_db = VAULTMIND_HOME / "data" / "embedding_cache.db"
    if cache_db.exists():
        from vaultmind.indexer.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_db)
        cs = cache.stats()
        console.print("\n[bold]Embedding Cache:[/bold]")
        console.print(f"  Entries: {cs['total_entries']}")
        console.print(f"  Size: {cs['total_size_bytes'] / 1024:.1f} KB")
        cache.close()

    if metadata_audit:
        from vaultmind.indexer import Embedder, VaultStore

        is_openai = settings.embedding.provider == "openai"
        api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
        audit_cache = _create_embedding_cache(settings)
        embedder = Embedder(settings.embedding, api_key, cache=audit_cache)
        store = VaultStore(settings.chroma, embedder)

        all_chunks = store._collection.get(include=["metadatas"])  # type: ignore[list-item]
        total = len(all_chunks["ids"])
        metadatas = all_chunks["metadatas"] or []

        if total == 0:
            console.print(
                "\n[yellow]No indexed chunks found. Run `vaultmind index` first.[/yellow]"
            )
        else:
            fields = ["note_type", "created", "status"]
            console.print(f"\n[bold]Metadata Audit[/bold] ({total} chunks)")
            for field in fields:
                count = sum(1 for m in metadatas if m.get(field) and m[field] != "")
                pct = count / total * 100
                color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
                console.print(
                    f"  {field:12s}: [{color}]{pct:5.1f}%[/{color}] ({count}/{total} chunks)"
                )

            console.print()
            if any(
                sum(1 for m in metadatas if m.get(f) and m[f] != "") / total < 0.8 for f in fields
            ):
                console.print(
                    "[yellow]Warning: metadata coverage below 80% for some fields."
                    " Features depending on this metadata may degrade.[/yellow]"
                )

        if audit_cache is not None:
            audit_cache.close()


@cli.command()
@click.option(
    "--golden",
    "golden_path",
    type=click.Path(),
    default=None,
    help="Path to the golden question set (default: [bench].golden_path)",
)
@click.option(
    "--k",
    "k_override",
    type=int,
    default=None,
    help="Override top-k for retrieval (default: [bench].k)",
)
@click.option(
    "--bundle",
    "bundle_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=(
        "Run against a deterministic fixture bundle directory "
        "(golden.yaml + retrieval.yaml) instead of the live vault"
    ),
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="Additionally score cite-or-decline correctness via one LLM call per question",
)
@click.option(
    "--trend-path",
    "trend_path_override",
    type=click.Path(),
    default=None,
    help="Override the JSONL trend record path (default: [bench].trend_path)",
)
@click.pass_context
def bench(
    ctx: click.Context,
    golden_path: str | None,
    k_override: int | None,
    bundle_dir: str | None,
    use_llm: bool,
    trend_path_override: str | None,
) -> None:
    """Score the live /recall retrieval path against a golden question set.

    Exits non-zero if any configured threshold is not met. By default
    computes recall@k/MRR with zero LLM calls; pass --llm to additionally
    score cite-or-decline correctness. Appends one JSONL trend record per run.
    """
    from vaultmind.bench.fixture_store import BundleError, load_bundle
    from vaultmind.bench.golden import GoldenSetError, load_golden_set
    from vaultmind.bench.runner import RetrievalStore, passes_thresholds, run_bench
    from vaultmind.bench.trend import append_trend_record, build_trend_record, default_trend_path
    from vaultmind.config import load_settings

    settings = load_settings(ctx.obj.get("config_path"))
    bench_cfg = settings.bench
    hybrid_enabled = settings.search.hybrid_enabled

    llm_model = ""
    decline_scorer = None
    if use_llm:
        _require_llm_key(settings)
        llm_client = _create_llm_client(settings)
        llm_model = bench_cfg.llm_model or settings.llm.fast_model
        from vaultmind.bench.llm_score import make_decline_scorer

        decline_scorer = make_decline_scorer(llm_client, llm_model)

    cache = None
    store: RetrievalStore
    try:
        if bundle_dir is not None:
            try:
                loaded_bundle = load_bundle(Path(bundle_dir))
            except BundleError as exc:
                console.print(f"[red]\u2717[/red] {exc}")
                sys.exit(1)
            store = loaded_bundle.store
            golden_file = Path(golden_path) if golden_path else loaded_bundle.golden_path
        else:
            from vaultmind.indexer import Embedder, VaultStore

            is_openai = settings.embedding.provider == "openai"
            api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
            if not api_key:
                console.print(
                    "[red]\u2717[/red] Embedding API key not set."
                    " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
                )
                sys.exit(1)
            cache = _create_embedding_cache(settings)
            bm25 = _create_bm25_index(settings)
            embedder = Embedder(settings.embedding, api_key, cache=cache)
            store = VaultStore(settings.chroma, embedder, bm25=bm25)
            golden_file = Path(golden_path) if golden_path else Path(bench_cfg.golden_path)

        try:
            golden = load_golden_set(golden_file)
        except GoldenSetError as exc:
            console.print(f"[red]\u2717[/red] {exc}")
            sys.exit(1)

        k = k_override if k_override is not None else bench_cfg.k

        with console.status("Running bench..."):
            report = run_bench(
                golden,
                store,
                k=k,
                hybrid_enabled=hybrid_enabled,
                score_floor=bench_cfg.score_floor,
                decline_scorer=decline_scorer,
                ranking_config=settings.ranking,
            )

        passed = passes_thresholds(
            report,
            recall_at_k_threshold=bench_cfg.recall_at_k_threshold,
            mrr_threshold=bench_cfg.mrr_threshold,
            retrieval_decline_threshold=bench_cfg.retrieval_decline_threshold,
            llm_decline_threshold=bench_cfg.llm_decline_threshold if use_llm else None,
        )

        console.print(f"\n[bold]Bench results[/bold] (k={report.k})")
        console.print(
            f"  Questions: {report.n_answerable} answerable, {report.n_unanswerable} unanswerable"
        )
        console.print(f"  recall@{report.k}: {report.recall_at_k:.2f}  MRR: {report.mrr:.2f}")
        if report.retrieval_decline_accuracy is not None:
            console.print(f"  Retrieval decline accuracy: {report.retrieval_decline_accuracy:.2f}")
        if report.llm_cite_or_decline_accuracy is not None:
            console.print(
                f"  LLM cite-or-decline accuracy: {report.llm_cite_or_decline_accuracy:.2f}"
            )

        trend_path = (
            Path(trend_path_override)
            if trend_path_override
            else (Path(bench_cfg.trend_path) if bench_cfg.trend_path else default_trend_path())
        )
        append_trend_record(build_trend_record(report, passed), trend_path)

        if not passed:
            console.print("[red]\u2717[/red] Bench thresholds not met.")
            sys.exit(1)
        console.print("[green]\u2713[/green] Bench thresholds met.")
    finally:
        if cache is not None:
            cache.close()


@cli.group("eval")
def eval_group() -> None:
    """Evaluation commands for LLM-gated detection surfaces."""


@eval_group.command("contradict")
@click.option(
    "--eval-path",
    "eval_path_override",
    type=click.Path(),
    default=None,
    help="Path to the labeled contradiction eval set (default: [contradiction].eval_path)",
)
@click.pass_context
def eval_contradict(ctx: click.Context, eval_path_override: str | None) -> None:
    """Score the contradiction conflict-detector against a labeled eval set.

    Reports precision/recall/F1 for the LLM detector alongside a trivial
    always-escalate baseline. The detector must beat the baseline before
    `[contradiction].auto_resolve` is safe to enable (Kosha's Gate-0 lesson:
    ship escalate-only until a real eval proves the detector out).
    """
    from vaultmind.config import load_settings
    from vaultmind.contradiction.eval import ContradictEvalError, load_eval_set, run_eval

    settings = load_settings(ctx.obj.get("config_path"))
    _require_llm_key(settings)
    llm_client = _create_llm_client(settings)
    model = settings.contradiction.detection_model or settings.llm.fast_model

    eval_path = (
        Path(eval_path_override) if eval_path_override else Path(settings.contradiction.eval_path)
    )
    try:
        pairs = load_eval_set(eval_path)
    except ContradictEvalError as exc:
        console.print(f"[red]\u2717[/red] {exc}")
        sys.exit(1)

    with console.status(f"Evaluating {len(pairs)} pairs..."):
        report = run_eval(pairs, llm_client, model, max_tokens=settings.contradiction.max_tokens)

    console.print(f"\n[bold]Contradiction eval[/bold] ({report.n_pairs} pairs)")
    console.print(
        f"  Detector — precision: {report.detector.precision:.2f}"
        f"  recall: {report.detector.recall:.2f}  F1: {report.detector.f1:.2f}"
    )
    console.print(
        f"  Baseline (always-escalate) — precision: {report.baseline.precision:.2f}"
        f"  recall: {report.baseline.recall:.2f}  F1: {report.baseline.f1:.2f}"
    )
    if report.beats_baseline:
        console.print("[green]\u2713[/green] Detector beats the trivial always-escalate baseline.")
    else:
        console.print(
            "[red]\u2717[/red] Detector does NOT beat baseline"
            " — auto-resolution should stay disabled."
        )
        sys.exit(1)


@cli.group("source")
def source_group() -> None:
    """Source connector commands (rss/youtube-channel/github-activity, M8)."""


def _sources_config_path(settings: object) -> Path:
    from vaultmind.config import Settings

    assert isinstance(settings, Settings)
    return (
        Path(settings.sources.config_path)
        if settings.sources.config_path
        else Path("config/sources.toml")
    )


def _sources_db_path(settings: object) -> Path:
    from vaultmind.config import VAULTMIND_HOME, Settings

    assert isinstance(settings, Settings)
    return (
        Path(settings.sources.db_path)
        if settings.sources.db_path
        else VAULTMIND_HOME / "data" / "sources.db"
    )


@source_group.command("list")
@click.pass_context
def source_list(ctx: click.Context) -> None:
    """List every configured connector instance and its cursor state."""
    from vaultmind.config import load_settings
    from vaultmind.sources.registry import load_source_instances
    from vaultmind.sources.store import SourceStore

    settings = load_settings(ctx.obj.get("config_path"))
    sources_config = _sources_config_path(settings)
    instances = load_source_instances(sources_config)
    if not instances:
        console.print(f"[yellow]No connector instances configured in {sources_config}[/yellow]")
        return

    source_store = SourceStore(_sources_db_path(settings))
    console.print(f"[bold]Configured connector instances[/bold] ({sources_config})")
    for inst in instances:
        state = source_store.get_state(inst.name)
        status = "[green]enabled[/green]" if inst.enabled else "[dim]disabled[/dim]"
        last_run = state.last_run.isoformat() if state.last_run else "never"
        console.print(
            f"  [bold]{inst.name}[/bold] ({inst.kind}) — {status}"
            f" | target: {inst.target} | runs: {state.run_count} | last run: {last_run}"
        )
    source_store.close()


@source_group.command("status")
@click.argument("name", required=False)
@click.pass_context
def source_status(ctx: click.Context, name: str | None) -> None:
    """Show durable cursor state and recent run history for one or every instance."""
    from vaultmind.config import load_settings
    from vaultmind.sources.registry import load_source_instances
    from vaultmind.sources.store import SourceStore

    settings = load_settings(ctx.obj.get("config_path"))
    sources_config = _sources_config_path(settings)
    instances = {inst.name: inst for inst in load_source_instances(sources_config)}
    if name is not None and name not in instances:
        console.print(f"[red]✗[/red] Unknown connector instance {name!r} in {sources_config}")
        sys.exit(1)

    names = [name] if name is not None else list(instances)
    if not names:
        console.print(f"[yellow]No connector instances configured in {sources_config}[/yellow]")
        return

    source_store = SourceStore(_sources_db_path(settings))
    for inst_name in names:
        state = source_store.get_state(inst_name)
        console.print(f"\n[bold]{inst_name}[/bold]")
        console.print(
            f"  cursor: last_seen_id={state.last_seen_id!r} last_seen_at={state.last_seen_at}"
        )
        console.print(f"  runs: {state.run_count}  last_run: {state.last_run}")
        for run in source_store.list_runs(inst_name, limit=5):
            run_status = f"[red]error: {run.error}[/red]" if run.error else "[green]ok[/green]"
            console.print(
                f"    {run.finished.isoformat()} — fetched {run.items_fetched},"
                f" ingested {run.items_ingested} {run_status}"
            )
    source_store.close()


@source_group.command("run")
@click.argument("name")
@click.pass_context
def source_run(ctx: click.Context, name: str) -> None:
    """Run one connector instance once, outside the scheduler.

    Fetches new items since the stored cursor, routes each through the
    review queue (M7) and distillation (M4-style) exactly as the scheduler
    would, then advances the durable cursor and records a run summary —
    the same `run_connector_once` orchestration `cli.py::bot`'s per-instance
    scheduled jobs use.
    """
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.indexer.duplicate_detector import DuplicateDetector
    from vaultmind.memory.gaps import GapStore
    from vaultmind.services.review_queue import ProposalKind
    from vaultmind.sources.pipeline import make_applier, run_connector_once
    from vaultmind.sources.registry import load_source_instances
    from vaultmind.sources.store import SourceStore
    from vaultmind.vault import VaultParser
    from vaultmind.vault.events import NoteCreatedEvent, NoteModifiedEvent, VaultEventBus

    settings = load_settings(ctx.obj.get("config_path"))
    if not settings.sources.enabled:
        console.print("[red]✗[/red] Source connectors are disabled ([sources].enabled = false).")
        sys.exit(1)

    sources_config = _sources_config_path(settings)
    instances = {inst.name: inst for inst in load_source_instances(sources_config)}
    if name not in instances:
        console.print(f"[red]✗[/red] Unknown connector instance {name!r} in {sources_config}")
        sys.exit(1)
    instance = instances[name]

    _require_llm_key(settings)
    is_openai = settings.embedding.provider == "openai"
    embed_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    if not embed_key:
        console.print(
            "[red]✗[/red] Embedding API key not set."
            " Set VAULTMIND_OPENAI_API_KEY or VAULTMIND_VOYAGE_API_KEY."
        )
        sys.exit(1)

    cache = _create_embedding_cache(settings)
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, embed_key, cache=cache)
    store = VaultStore(settings.chroma, embedder)
    llm_client = _create_llm_client(settings)

    event_bus = VaultEventBus()
    detector: DuplicateDetector | None = None
    if settings.duplicate_detection.enabled:
        detector = DuplicateDetector(settings.duplicate_detection, store)

    review_queue = _build_review_queue(settings)
    distill_model = settings.distill.model or settings.llm.thinking_model

    gap_store: GapStore | None = None
    if settings.gaps.enabled:
        gap_db = (
            Path(settings.gaps.db_path)
            if settings.gaps.db_path
            else VAULTMIND_HOME / "data" / "gaps.db"
        )
        gap_store = GapStore(gap_db, stale_after_days=settings.gaps.stale_after_days)

    review_queue.register_applier(
        ProposalKind.SOURCE_INGESTION,
        make_applier(
            vault_root=settings.vault.path,
            llm_client=llm_client,
            model=distill_model,
            gap_store=gap_store,
        ),
    )

    if detector is not None:
        dup_subscriber = _duplicate_review_subscriber(detector, review_queue)
        event_bus.subscribe(NoteCreatedEvent, dup_subscriber)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, dup_subscriber)  # type: ignore[arg-type]

    if settings.contradiction.enabled and detector is not None:
        from vaultmind.contradiction.detector import ContradictionDetector

        contradiction_model = settings.contradiction.detection_model or settings.llm.fast_model
        contradiction_detector = ContradictionDetector(
            settings.contradiction,
            detector,
            llm_client,
            contradiction_model,
            settings.vault.path,
            parser,
            ranking_config=settings.ranking,
            gap_store=gap_store,
            review_queue=review_queue,
        )
        on_changed = contradiction_detector.on_note_changed
        event_bus.subscribe(NoteCreatedEvent, on_changed)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, on_changed)  # type: ignore[arg-type]

    source_store = SourceStore(_sources_db_path(settings))

    with console.status(f"Running connector '{name}'..."):
        result = asyncio.run(
            run_connector_once(
                instance,
                source_store=source_store,
                review_queue=review_queue,
                parser=parser,
                store=store,
                vault_root=settings.vault.path,
                event_bus=event_bus,
            )
        )

    if cache is not None:
        cache.close()
    if gap_store is not None:
        gap_store.close()
    source_store.close()
    review_queue.close()

    if result.error:
        console.print(f"[red]✗[/red] Connector '{name}' failed: {result.error}")
        sys.exit(1)

    console.print(
        f"[green]✓[/green] Connector '{name}' run complete:"
        f" {result.items_fetched} fetched, {result.items_ingested} ingested"
    )


if __name__ == "__main__":
    cli()
