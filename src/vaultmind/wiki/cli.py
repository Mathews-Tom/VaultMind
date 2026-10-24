"""Explicit, source-backed wiki commands; no background services or index access."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vaultmind.config import Settings
    from vaultmind.wiki.workspace import WikiWorkspace


@contextmanager
def _workspace(ctx: click.Context) -> Iterator[tuple[WikiWorkspace, Settings]]:
    # Loading and collaborator construction happen only for an invoked command.
    from vaultmind.config import load_settings
    from vaultmind.wiki.workspace import WikiError, WikiWorkspace

    try:
        settings = load_settings(ctx.obj.get("config_path"))
    except Exception as exc:
        raise click.ClickException(f"Wiki configuration failed ({type(exc).__name__}).") from None
    try:
        with WikiWorkspace(settings) as workspace:
            yield workspace, settings
    except WikiError as exc:
        raise click.ClickException(str(exc)) from None
    except Exception as exc:
        # Exception categories are useful; messages can contain private data.
        raise click.ClickException(
            f"Wiki operation failed ({type(exc).__name__}); no automatic retry was attempted."
        ) from None


@click.group("wiki")
def wiki_group() -> None:
    """Preserve sources and explicitly review single-page wiki changes."""


@wiki_group.command("add")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--id", "source_id", required=True, help="Stable ASCII source identifier.")
@click.option("--uri", default=None, help="Provenance URI only; never fetched.")
@click.pass_context
def add_source(ctx: click.Context, source_file: Path, source_id: str, uri: str | None) -> None:
    """Preserve FILE bytes without creating a page or calling a provider."""
    with _workspace(ctx) as (workspace, _settings):
        result = workspace.add_source(source_file, source_id, uri)
        click.echo(f"Source: {result['id']}")
        click.echo(f"Path: {result['path']}")
        click.echo(f"SHA256: {result['sha256']}")
        click.echo(f"Status: {result['status']}")


@wiki_group.command("propose")
@click.option("--page", required=True, help="Current page path: wiki/SLUG.md.")
@click.option("--source", "source_ids", multiple=True, required=True, help="Preserved source ID.")
@click.option("--draft-file", type=click.Path(path_type=Path), default=None)
@click.pass_context
def propose(
    ctx: click.Context, page: str, source_ids: tuple[str, ...], draft_file: Path | None
) -> None:
    """Propose a draft, or request one configured model completion when no draft is given."""
    with _workspace(ctx) as (workspace, settings):
        llm = None
        if draft_file is None:
            # Import at invocation to avoid a cycle with the root group's registration.
            from vaultmind.cli import _create_llm_client, _require_llm_key

            _require_llm_key(settings)
            llm = _create_llm_client(settings)
        proposal = workspace.propose(page, list(source_ids), draft_file=draft_file, llm=llm)
        click.echo(f"Proposal: {proposal.proposal_id}")


@wiki_group.command("review")
@click.argument("proposal_id")
@click.pass_context
def review(ctx: click.Context, proposal_id: str) -> None:
    """Print exact proposal identities and the plain before/after diff."""
    with _workspace(ctx) as (workspace, _settings):
        click.echo(workspace.review(proposal_id), nl=False)


@wiki_group.command("apply")
@click.argument("proposal_id")
@click.option("--expect", "expected", required=True, help="Exact SHA256 printed by review.")
@click.pass_context
def apply(ctx: click.Context, proposal_id: str, expected: str) -> None:
    """Apply only the reviewed payload, rejecting stale or incomplete operations."""
    with _workspace(ctx) as (workspace, _settings):
        click.echo(workspace.apply(proposal_id, expected))
