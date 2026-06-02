"""Interface protocols shared between the runtime and interface adapters."""

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


class ErrandInterface(Protocol):
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
    """Interface capability for scheduled task delivery."""

    async def deliver_scheduled_result(
        self,
        task_session_id: str,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver a scheduled job result. Return True if handled."""


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
