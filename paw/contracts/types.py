"""Provider-agnostic types passed between components."""

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
