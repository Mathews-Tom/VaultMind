"""Explicit, source-backed wiki commands; no background services or index access."""

from __future__ import annotations

import json
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


@contextmanager
def _read_settings(ctx: click.Context) -> Iterator[Settings]:
    """Load settings without opening the default writable workspace."""
    from vaultmind.config import load_settings
    from vaultmind.wiki.workspace import WikiError

    try:
        settings = load_settings(ctx.obj.get("config_path"))
    except Exception as exc:
        raise click.ClickException(f"Wiki configuration failed ({type(exc).__name__}).") from None
    try:
        yield settings
    except WikiError as exc:
        raise click.ClickException(str(exc)) from None
    except Exception as exc:
        raise click.ClickException(
            f"Wiki operation failed ({type(exc).__name__}); no automatic retry was attempted."
        ) from None


@click.group("wiki")
def wiki_group() -> None:
    """Preserve, review and query explicitly selected source-backed wiki pages."""


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


@wiki_group.command("list")
@click.pass_context
def list_current_pages(ctx: click.Context) -> None:
    """List approved current pages without constructing a provider or writer."""
    from vaultmind.wiki.query import list_pages

    with _read_settings(ctx) as settings:
        click.echo(json.dumps(list_pages(settings), ensure_ascii=False, indent=2))


@wiki_group.command("query")
@click.argument("question")
@click.option(
    "--page", "pages", multiple=True, required=True, help="Approved current wiki/SLUG.md."
)
@click.option(
    "--evidence-only", is_flag=True, help="Inspect complete sections without a model call."
)
@click.pass_context
def query_current_pages(
    ctx: click.Context, question: str, pages: tuple[str, ...], evidence_only: bool
) -> None:
    """Answer QUESTION from only the explicitly selected approved current pages."""
    from vaultmind.wiki.query import query_pages

    with _read_settings(ctx) as settings:
        llm = None
        if not evidence_only:
            from vaultmind.cli import _create_llm_client, _require_llm_key

            _require_llm_key(settings)
            llm = _create_llm_client(settings)
        result = query_pages(settings, question, list(pages), evidence_only=evidence_only, llm=llm)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@wiki_group.command("lint")
@click.pass_context
def lint_current_pages(ctx: click.Context) -> None:
    """Diagnose current pages; exit 1 if any page is invalid."""
    from vaultmind.wiki.query import lint_pages

    with _read_settings(ctx) as settings:
        rows = lint_pages(settings)
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
    if any(row["status"] == "invalid" for row in rows):
        ctx.exit(1)
