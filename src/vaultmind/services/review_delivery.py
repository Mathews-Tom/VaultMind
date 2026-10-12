"""Delivery of newly-pending BLOCK-lane review proposals to a human.

`ReviewQueue` is a synchronous SQLite store and never performs network I/O.
Producers (CLI, duplicate subscriber, maturation/source schedulers, contradiction
detector) call `deliver_block()` *after* `ReviewQueue.propose()` has committed,
so a delivery failure can never roll back or corrupt queue state: an undelivered
BLOCK proposal simply stays PENDING and remains reachable through `/review`.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from vaultmind.services.review_queue import Lane, ProposalStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vaultmind.services.review_queue import ReviewProposal

logger = logging.getLogger(__name__)


class DeliveryStatus(StrEnum):
    """Outcome of attempting to put a BLOCK proposal in front of a human."""

    SENT = "sent"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


type BlockNotifier = Callable[[ReviewProposal], Awaitable[DeliveryStatus]]


async def deliver_block(
    proposal: ReviewProposal, notifier: BlockNotifier | None = None
) -> DeliveryStatus:
    """Deliver `proposal` once if it is a newly-created, still-PENDING BLOCK item.

    Deduplicated re-proposals (``occurrence_count > 1``), non-BLOCK lanes and
    already-resolved proposals report ``NOT_REQUIRED`` and never resend. With no
    notifier the delivery is ``UNAVAILABLE``; a notifier that raises is ``FAILED``.
    The proposal is unaffected in every case.
    """
    if (
        proposal.lane is not Lane.BLOCK
        or proposal.status is not ProposalStatus.PENDING
        or proposal.occurrence_count != 1
    ):
        return DeliveryStatus.NOT_REQUIRED

    if notifier is None:
        status = DeliveryStatus.UNAVAILABLE
    else:
        try:
            status = await notifier(proposal)
        except Exception:
            logger.exception("BLOCK delivery raised for proposal %s", proposal.proposal_id)
            status = DeliveryStatus.FAILED
    healthy = status in (DeliveryStatus.SENT, DeliveryStatus.DISABLED)
    logger.log(
        logging.INFO if healthy else logging.WARNING,
        "BLOCK proposal %s (%s) delivery: %s — remains pending",
        proposal.proposal_id,
        proposal.kind.value,
        status.value,
    )
    return status


__all__ = ["BlockNotifier", "DeliveryStatus", "deliver_block"]
