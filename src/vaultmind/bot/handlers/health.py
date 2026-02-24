"""Health handler — system health status."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext


async def handle_health(ctx: HandlerContext, message: Message) -> None:
    """Report system health status."""
    if not _is_authorized(ctx, message):
        return

    checks: list[str] = []

    # Vector store
    try:
        chunk_count = ctx.store.count
        checks.append(f"\u2705 **Vector Store:** {chunk_count} chunks indexed")
    except Exception as e:
        checks.append(f"\u274c **Vector Store:** {e}")

    # Knowledge graph
    try:
        gs = ctx.graph.stats
        checks.append(f"\u2705 **Knowledge Graph:** {gs['nodes']} nodes, {gs['edges']} edges")
    except Exception as e:
        checks.append(f"\u274c **Knowledge Graph:** {e}")

    # Vault path
    vault_path = ctx.vault_root
    if vault_path.exists():
        md_count = sum(1 for _ in vault_path.rglob("*.md"))
        checks.append(f"\u2705 **Vault:** {md_count} markdown files at `{vault_path}`")
    else:
        checks.append(f"\u274c **Vault:** path not found: `{vault_path}`")

    # LLM provider
    provider = ctx.settings.llm.provider
    has_key = bool(ctx.settings.llm_api_key) or provider == "ollama"
    if has_key:
        checks.append(f"\u2705 **LLM:** {provider} (key configured)")
    else:
        checks.append(f"\u274c **LLM:** {provider} (no API key)")

    # Graph persistence
    graph_path = ctx.settings.graph.persist_path
    if graph_path.exists():
        size_kb = graph_path.stat().st_size / 1024
        checks.append(f"\u2705 **Graph file:** {size_kb:.1f} KB")
    else:
        checks.append("\u26a0\ufe0f **Graph file:** not yet created")

    await message.answer(
        "\U0001f3e5 **VaultMind Health**\n\n" + "\n".join(checks),
        parse_mode="Markdown",
    )
