"""Shared dataclasses and protocols across Errand components.

`contracts` is intentionally a leaf module: nothing inside it imports
from any other Errand component. Everything that needs to be shared
across `brain`, `tools`, `interfaces`, `sessions`, `agent_loop`, etc.
lives here.
"""

from errand.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from errand.contracts.interfaces import (
    ReplyTarget,
    ScheduledDelivery,
    FallbackDelivery,
    ErrandInterface,
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
    "ErrandInterface",
    "UserMessage",
]
