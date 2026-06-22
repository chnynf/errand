"""Shared dataclasses and protocols across Paw components.

`contracts` is intentionally a leaf module: nothing inside it imports
from any other Paw component. Everything that needs to be shared
across `brain`, `tools`, `interfaces`, `sessions`, `agent_loop`, etc.
lives here.
"""

from paw.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from paw.contracts.interfaces import (
    ReplyTarget,
    ScheduledDelivery,
    FallbackDelivery,
    PawInterface,
    UserMessage,
)

__all__ = [
    "BrainDecision",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ReplyTarget",
    "ScheduledDelivery",
    "FallbackDelivery",
    "PawInterface",
    "UserMessage",
]
