"""Evolve handler -- belief evolution tracking commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaultmind.bot.handlers.utils import _is_authorized, _split_message

if TYPE_CHECKING:
    from aiogram.types import Message

    from vaultmind.bot.handlers.context import HandlerContext
    from vaultmind.graph.evolution import EvolutionDetector


async def handle_evolve(
    ctx: HandlerContext,
    message: Message,
    args: str,
    detector: EvolutionDetector,
) -> None:
    """Handle /evolve command -- scan for belief evolution signals."""
    if not _is_authorized(ctx, message):
        return

    parts = args.strip().split(maxsplit=1)
    subcommand = parts[0] if parts else ""

    if subcommand == "dismiss" and len(parts) > 1:
        await _handle_dismiss(message, parts[1].strip(), detector)
        return

    if subcommand == "detail" and len(parts) > 1:
        await _handle_detail(ctx, message, parts[1].strip(), detector)
        return

    # Default: run scan
    signals = detector.scan()

    if not signals:
        await message.answer("No belief evolution signals detected.")
        return

    lines = [f"Found {len(signals)} belief evolution signal(s):\n"]
    for i, sig in enumerate(signals[:10], 1):
        type_tag = {
            "confidence_drift": "drift",
            "relationship_shift": "shift",
            "stale_claim": "stale",
        }
        tag = type_tag.get(sig.signal_type, sig.signal_type)
        lines.append(f"{i}. [{tag}] {sig.entity_a} <-> {sig.entity_b}")
        lines.append(f"   {sig.detail}")
        if sig.source_notes:
            sources = ", ".join(sig.source_notes[:3])
            lines.append(f"   Sources: {sources}")
        lines.append(f"   ({sig.evolution_id[:8]})")
        lines.append("")

    lines.append("Use `/evolve dismiss <id>` to acknowledge")

    text = "\n".join(lines)
    for chunk in _split_message(text, max_len=4000):
        await message.answer(chunk)


async def _handle_dismiss(
    message: Message,
    id_prefix: str,
    detector: EvolutionDetector,
) -> None:
    """Dismiss a specific evolution signal."""
    found = detector.dismiss(id_prefix)
    if found:
        await message.answer(f"Dismissed evolution signal {id_prefix}.")
    else:
        await message.answer(f"No evolution signal found matching '{id_prefix}'.")


async def _handle_detail(
    ctx: HandlerContext,
    message: Message,
    id_prefix: str,
    detector: EvolutionDetector,
) -> None:
    """Show detail for a specific evolution signal."""
    signals = detector.scan()
    match = None
    for sig in signals:
        if sig.evolution_id.startswith(id_prefix):
            match = sig
            break

    if match is None:
        await message.answer(f"No evolution signal found matching '{id_prefix}'.")
        return

    lines = [
        f"**{match.signal_type}**: {match.entity_a} <-> {match.entity_b}",
        f"Detail: {match.detail}",
        f"Severity: {match.severity:.2f}",
        f"ID: {match.evolution_id}",
        "",
        "**Source notes:**",
    ]
    for note_path in match.source_notes:
        lines.append(f"- `{note_path}`")

    text = "\n".join(lines)
    for chunk in _split_message(text, max_len=4000):
        await message.answer(chunk, parse_mode="Markdown")
