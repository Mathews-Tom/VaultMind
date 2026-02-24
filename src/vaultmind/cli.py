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

    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key)
    store = VaultStore(settings.chroma, embedder)

    with console.status("Parsing vault..."):
        notes = parser.iter_notes()

    with console.status(f"Indexing {len(notes)} notes..."):
        count = store.index_notes(notes, parser)

    console.print(f"[green]✓[/green] Indexed {count} chunks from {len(notes)} notes")


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
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, embed_key)
    store = VaultStore(settings.chroma, embedder)
    graph = KnowledgeGraph(settings.graph)

    from vaultmind.bot.session_store import SessionStore
    from vaultmind.config import VAULTMIND_HOME

    llm_client = _create_llm_client(settings)
    session_store = SessionStore(VAULTMIND_HOME / "data" / "sessions.db")
    thinking = ThinkingPartner(
        settings.llm,
        settings.telegram,
        llm_client,
        session_store=session_store,
    )

    handlers = CommandHandlers(
        settings=settings,
        store=store,
        graph=graph,
        parser=parser,
        thinking=thinking,
        llm_client=llm_client,
    )

    tg_bot, dp = create_bot(settings.telegram.bot_token)
    register_handlers(handlers)

    provider = settings.llm.provider
    model = settings.llm.thinking_model
    console.print(f"[green]✓[/green] Starting Telegram bot (LLM: {provider}/{model})...")
    asyncio.run(dp.start_polling(tg_bot))


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


@cli.command("mcp-serve")
@click.pass_context
def mcp_serve(ctx: click.Context) -> None:
    """Start the MCP server for agent integration."""
    from vaultmind.config import load_settings
    from vaultmind.graph import KnowledgeGraph
    from vaultmind.indexer import Embedder, VaultStore
    from vaultmind.mcp.server import create_mcp_server
    from vaultmind.vault import VaultParser

    settings = load_settings(ctx.obj.get("config_path"))

    is_openai = settings.embedding.provider == "openai"
    api_key = settings.openai_api_key if is_openai else settings.voyage_api_key
    parser = VaultParser(settings.vault)
    embedder = Embedder(settings.embedding, api_key)
    store = VaultStore(settings.chroma, embedder)
    graph = KnowledgeGraph(settings.graph)

    server = create_mcp_server(
        settings.vault.path,
        store,
        graph,
        parser,
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
@click.pass_context
def stats(ctx: click.Context) -> None:
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


if __name__ == "__main__":
    cli()
