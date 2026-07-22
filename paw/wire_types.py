"""The tool-calling wire format shared by brain, tools, and agent_loop.

A leaf module: nothing here imports from any other Paw component, and it
carries no behavior of its own. It exists because these three components are
peers that must not depend on each other -- brain stays provider-and-tool-
implementation-agnostic, and tools stays swappable to a standalone MCP server
(see paw/tools/registry.py) -- so the shapes they exchange live here instead
of inside any one of them.

    tools.registry  --ToolDefinition-->  brain
    brain           --BrainDecision(tool_calls)-->  agent_loop
    agent_loop      --ToolCall/params-->  tools.registry
    tools.registry  --ToolResult-->  agent_loop
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolDefinition:
    """Standardized tool definition passed to providers."""

    name: str
    description: str
    parameters: dict  # JSON Schema format


@dataclass
class ToolCall:
    """A tool invocation requested by the AI."""

    id: str
    name: str
    params: dict

    @staticmethod
    def generate_id() -> str:
        return f"tc-{uuid.uuid4().hex[:12]}"


@dataclass
class ToolResult:
    """The outcome of executing a ToolCall."""

    tool_call_id: str
    name: str
    content: str


@dataclass
class BrainDecision:
    """Provider-agnostic decision returned to the agent loop."""

    tool_calls: List[ToolCall] = field(default_factory=list)
    text_response: Optional[str] = None
    context_summary: Optional[str] = None
    reasoning_content: Optional[str] = None
