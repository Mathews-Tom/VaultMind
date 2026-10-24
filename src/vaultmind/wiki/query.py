"""Wiki-owned listing and diagnostics for approved current pages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vaultmind.knowledge.sections import _page
from vaultmind.wiki.workspace import WikiError, WikiWorkspace

if TYPE_CHECKING:
    from vaultmind.config import Settings


def list_pages(settings: Settings) -> list[dict[str, Any]]:
    """List only approved current direct wiki pages; refuse invalid accepted state."""
    rows: list[dict[str, Any]] = []
    with WikiWorkspace(settings, read_only=True) as workspace, workspace.reading():
        for path in workspace.page_paths():
            payload = workspace.approved_page(path)
            title, _, unresolved = _page(payload)
            rows.append(
                dict(
                    path=path,
                    title=title,
                    sha256=payload["after_sha256"],
                    unresolved_conflict=unresolved,
                )
            )
    return rows


def lint_pages(settings: Settings) -> list[dict[str, Any]]:
    """Diagnose each current page without treating attributed disagreement as invalid."""
    rows: list[dict[str, Any]] = []
    with WikiWorkspace(settings, read_only=True) as workspace, workspace.reading():
        for path in workspace.page_paths():
            try:
                payload = workspace.approved_page(path)
                _, _, unresolved = _page(payload)
            except WikiError as exc:
                rows.append(dict(page=path, status="invalid", detail=str(exc)))
            else:
                rows.append(
                    dict(
                        page=path,
                        status="unresolved" if unresolved else "valid",
                        detail="Unresolved attributed conflict"
                        if unresolved
                        else "Approved current page and sources verified",
                    )
                )
    return rows
