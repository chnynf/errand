"""Shared dataclasses and protocols across Swarm components.

`contracts` is intentionally a leaf module: nothing inside it imports
from any other Swarm component. Everything that needs to be shared
across `brain`, `tools`, `interfaces`, `sessions`, `agent_loop`, etc.
lives here.
"""

from swarm_agent.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from swarm_agent.contracts.interfaces import (
    ReplyTarget,
    ScheduledDelivery,
    SwarmInterface,
    UserMessage,
)

__all__ = [
    "BrainDecision",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ReplyTarget",
    "ScheduledDelivery",
    "SwarmInterface",
    "UserMessage",
]
