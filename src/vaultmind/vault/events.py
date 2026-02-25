"""Typed vault events and async event bus for decoupled change propagation.

Downstream consumers (duplicate detection, note suggestions, daily digest)
subscribe to specific event types without coupling to the watch handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path
    from typing import Any

    from vaultmind.vault.models import Note

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VaultEvent:
    """Base event emitted by the watch handler."""

    path: Path
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True, slots=True)
class NoteCreatedEvent(VaultEvent):
    """A new note was created and indexed."""

    note: Note | None = None
    chunks_indexed: int = 0


@dataclass(frozen=True, slots=True)
class NoteModifiedEvent(VaultEvent):
    """An existing note was modified and re-indexed."""

    note: Note | None = None
    chunks_indexed: int = 0


@dataclass(frozen=True, slots=True)
class NoteDeletedEvent(VaultEvent):
    """A note was deleted and its chunks removed from the index."""


# Union of all subscribable event types
type AnyVaultEvent = NoteCreatedEvent | NoteModifiedEvent | NoteDeletedEvent

# Callback signature: async fn(event) -> None
type EventCallback = Callable[[AnyVaultEvent], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class VaultEventBus:
    """Async publish/subscribe bus for vault change events.

    Subscribers register for specific event types. Publishing dispatches
    to all matching subscribers concurrently via ``asyncio.gather``.
    Subscriber errors are logged and do not propagate.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type[AnyVaultEvent], list[EventCallback]] = {}

    def subscribe(
        self,
        event_type: type[AnyVaultEvent],
        callback: EventCallback,
    ) -> None:
        """Register *callback* for events of *event_type*."""
        self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug(
            "Subscriber registered: %s → %s",
            event_type.__name__,
            callback.__qualname__,
        )

    def unsubscribe(
        self,
        event_type: type[AnyVaultEvent],
        callback: EventCallback,
    ) -> None:
        """Remove *callback* from *event_type* subscribers."""
        subs = self._subscribers.get(event_type, [])
        with contextlib.suppress(ValueError):
            subs.remove(callback)

    async def publish(self, event: AnyVaultEvent) -> None:
        """Dispatch *event* to all registered subscribers for its type.

        Each subscriber runs as a shielded task so one failure doesn't
        block others. Errors are logged, not raised.
        """
        subs = self._subscribers.get(type(event), [])
        if not subs:
            return

        async def _safe_call(cb: EventCallback) -> None:
            try:
                await cb(event)
            except Exception:
                logger.exception(
                    "Subscriber %s failed on %s",
                    cb.__qualname__,
                    type(event).__name__,
                )

        await asyncio.gather(*[_safe_call(cb) for cb in subs])

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())
