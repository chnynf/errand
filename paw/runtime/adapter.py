"""Runtime's contract for pluggable interface adapters.

Owned by runtime: ``PawApp`` (runtime/app.py) is the sole consumer that
dictates this shape. Each concrete adapter under ``paw/interfaces/``
(Discord, CLI, WeChat, Web) implements ``PawInterface`` and hands the
runtime ``UserMessage`` objects to plug in -- the dependency points from the
adapters to this module, never back, so this can live with its owner instead
of in a shared/neutral location.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


class ReplyTarget(Protocol):
    """Destination for an agent response."""

    async def send(self, message: str) -> None:
        """Send a response message."""

    async def send_progress(self, message: str) -> None:
        """Send an in-progress status update (non-final, may be overwritten)."""

    async def request_approval(
        self,
        *,
        title: str,
        details: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Ask the user to approve an action. Return True if approved."""


class PawInterface(Protocol):
    """Runtime adapter such as Discord or CLI."""

    name: str

    async def start(self) -> None:
        """Start listening until stopped."""

    async def stop(self) -> None:
        """Stop the interface."""


@dataclass(frozen=True)
class UserMessage:
    """Normalized user message emitted by any interface."""

    session_id: str
    text: str
    source: str
    reply_to: ReplyTarget
    metadata: dict[str, Any] = field(default_factory=dict)


class ScheduledDelivery(Protocol):
    """Interface capability for scheduled task delivery.

    Only Discord implements this today: it's the one interface that can
    receive a proactive push rather than merely reply to an inbound message.
    """

    async def deliver_scheduled_result(self, session_id: str, message: str) -> bool:
        """Create a fresh channel/session named ``session_id`` and post ``message``.

        Return True if handled.
        """


class FallbackDelivery(Protocol):
    """Interface capability for last-resort delivery of undeliverable messages.

    Used when a final message cannot be sent through its original interface
    (the origin is down, cannot receive proactive messages, etc.). An
    interface implementing this routes the message to a standing fallback
    destination so the user is not silently dropped.
    """

    async def deliver_fallback_message(
        self,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver a fallback message. Return True if handled."""
