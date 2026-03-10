"""CLI entry point for VaultMind.

Commands:
    vaultmind index       — Full index of the vault
    vaultmind bot         — Start the Telegram bot
    vaultmind mcp-serve   — Start the MCP server
    vaultmind graph-build — Build/rebuild the knowledge graph
    vaultmind graph-report — Generate graph analytics report
    vaultmind stats       — Show vault statistics
    vaultmind init        — Initialize vault structure
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.logging import RichHandler

from vaultmind import __version__

if TYPE_CHECKING:
    from vaultmind.indexer.bm25 import BM25Index
    from vaultmind.indexer.embedding_cache import EmbeddingCache
    from vaultmind.llm.client import LLMClient

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

    from vaultmind.config import load_settings
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

    generator = DigestGenerator(store=store, graph=graph, parser=parser, config=digest_config)

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

    if not any(
        [
            report.new_notes,
            report.modified_notes,
            report.trending_entities,
            report.suggested_connections,
            report.orphan_notes,
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
    )

    # Duplicate detection
    from vaultmind.indexer.duplicate_detector import DuplicateDetector

    detector: DuplicateDetector | None = None
    if settings.duplicate_detection.enabled:
        detector = DuplicateDetector(settings.duplicate_detection, store)

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
    )

    tg_bot, dp = create_bot(settings.telegram.bot_token)
    register_handlers(handlers)

    # Wire up incremental watch mode
    from vaultmind.graph.maintenance import GraphMaintainer
    from vaultmind.vault.events import (
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
        event_bus.subscribe(NoteCreatedEvent, detector.on_note_changed)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, detector.on_note_changed)  # type: ignore[arg-type]
    if suggester is not None:
        event_bus.subscribe(NoteCreatedEvent, suggester.on_note_changed)  # type: ignore[arg-type]
        event_bus.subscribe(NoteModifiedEvent, suggester.on_note_changed)  # type: ignore[arg-type]

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

        async def _maturation_digest() -> None:
            assert maturation_pipeline is not None
            clusters = maturation_pipeline.discover()
            if clusters:
                logging.getLogger(__name__).info(
                    "Maturation digest: %d clusters ready", len(clusters)
                )
            maturation_pipeline.mark_run()

        maturation_job = ScheduledJob(
            name="maturation_digest",
            interval=timedelta(days=7),
            execute=_maturation_digest,
        )
        scheduler = SchedulerService(
            jobs=[maturation_job],
            state_path=Path.home() / ".vaultmind" / "data" / "scheduler_state.json",
        )

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


@cli.command("auto-tag")
@click.option("--apply", "do_apply", is_flag=True, help="Write tags to frontmatter")
@click.option("--approve-all", is_flag=True, help="Approve all quarantined tags")
@click.option("--show-quarantine", is_flag=True, help="Show quarantined tags and exit")
@click.pass_context
def auto_tag(
    ctx: click.Context,
    do_apply: bool,
    approve_all: bool,
    show_quarantine: bool,
) -> None:
    """Auto-tag vault notes using LLM classification.

    By default runs in dry-run mode — shows suggestions without writing.
    Use --apply to write tags to frontmatter.
    """
    from vaultmind.config import VAULTMIND_HOME, load_settings
    from vaultmind.indexer.auto_tagger import AutoTagger
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))
    _require_llm_key(settings)

    quarantine_path = VAULTMIND_HOME / "data" / "tag_quarantine.json"

    model = settings.auto_tag.tagging_model or settings.llm.fast_model
    llm_client = _create_llm_client(settings)
    tagger = AutoTagger(settings.auto_tag, llm_client, model, quarantine_path)

    if approve_all:
        tagger.quarantine.approve_all()
        tagger.save_quarantine()
        console.print("[green]✓[/green] All quarantined tags approved.")
        return

    if show_quarantine:
        q = tagger.quarantine
        if q.quarantined_tags:
            console.print("[bold]Quarantined tags (pending approval):[/bold]")
            for tag in sorted(q.quarantined_tags):
                console.print(f"  [yellow]{tag}[/yellow]")
        else:
            console.print("[green]No tags in quarantine.[/green]")
        if q.approved_tags:
            console.print(f"\n[bold]Approved tags:[/bold] {', '.join(sorted(q.approved_tags))}")
        return

    parser = VaultParser(settings.vault)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    vault_tags = tagger.collect_vault_tags(notes)
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
        tagger.save_quarantine()
        return

    for s in suggestions:
        tag_str = ", ".join(s.suggested_tags) if s.suggested_tags else "(none from vocab)"
        new_str = f" + new: {', '.join(s.new_tags)}" if s.new_tags else ""
        console.print(f"  [bold]{s.note_title}[/bold]: [cyan]{tag_str}[/cyan]{new_str}")

    console.print(f"\n[bold]Summary:[/bold] {len(suggestions)} notes with suggestions")

    q = tagger.quarantine
    if q.quarantined_tags:
        console.print(
            f"  [yellow]Quarantined tags:[/yellow] {', '.join(sorted(q.quarantined_tags))}"
        )
        console.print("  Run `auto-tag --approve-all` to approve, then `auto-tag --apply`")

    if do_apply:
        applied = 0
        for s in suggestions:
            if s.suggested_tags:
                note_path = settings.vault.path / s.note_path
                if note_path.exists():
                    tagger.apply_tags(note_path, s.suggested_tags)
                    applied += 1
        console.print(f"[green]✓[/green] Applied tags to {applied} notes")

    tagger.save_quarantine()


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
    from vaultmind.config import load_settings
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

    asyncio.run(_run())

    if cache is not None:
        cache.close()


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

    store = PreferenceStore(db_path)
    insights = analyze_preferences(store, days=days)
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


if __name__ == "__main__":
    cli()
