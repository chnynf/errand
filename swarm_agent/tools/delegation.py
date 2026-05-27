"""Delegate bounded tasks to internal Swarm agents."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from swarm_agent.config import load_swarm_config

MAX_DELEGATION_DEPTH = 2
DEFAULT_TIMEOUT_SECONDS = 900


async def invoke_agent(
    agent_id: str,
    task: str,
    context: str = "",
    _context: dict[str, Any] | None = None,
) -> str:
    """Delegate a bounded task to another configured Swarm agent.

    Use when a task benefits from an isolated specialist context, such as
    applied-science reasoning, experiment design, metrics analysis, or another
    configured specialist. Pass a clear task and only the context needed for
    that task. The delegated agent runs in its own session and returns a final
    summary for you to synthesize.
    """
    runtime_context = _context or {}
    caller_agent_id = str(runtime_context.get("agent_id") or "")
    parent_session_id = str(runtime_context.get("session_id") or "default")
    depth = int(runtime_context.get("delegation_depth") or 0)

    if depth >= MAX_DELEGATION_DEPTH:
        return "Delegation blocked: maximum delegation depth reached."

    config = load_swarm_config()
    caller = config.get_agent(caller_agent_id)
    target_id = (agent_id or "").strip()
    if not target_id:
        return "Delegation failed: agent_id is required."
    if target_id not in config.agents:
        return f"Delegation failed: agent '{target_id}' is not configured."
    if target_id not in caller.can_delegate:
        return (
            f"Delegation failed: agent '{caller.id}' is not allowed to delegate "
            f"to '{target_id}'."
        )

    task_text = (task or "").strip()
    if not task_text:
        return "Delegation failed: task is required."

    reply_to = runtime_context.get("reply_to")
    if reply_to:
        try:
            await reply_to.send_progress(f"Delegating to {target_id}...")
        except Exception:
            pass

    child_session_id = (
        f"{parent_session_id}::sub::{target_id}::{uuid.uuid4().hex[:8]}"
    )
    child_input = _build_child_input(task_text, context)

    # Import lazily to avoid a tool-registry import cycle.
    from swarm_agent.agent_loop import AgentLoop

    loop = AgentLoop(
        session_id=child_session_id,
        debug=bool(runtime_context.get("debug")),
        agent_id=target_id,
        config=config,
        delegation_depth=depth + 1,
    )
    try:
        result = await asyncio.wait_for(
            loop.process_input(
                child_input,
                metadata={
                    "is_subagent": True,
                    "suppress_usage_footer": True,
                    "parent_session_id": parent_session_id,
                    "parent_agent_id": caller.id,
                    "_reply_to": runtime_context.get("reply_to"),
                    "_source": runtime_context.get("source"),
                },
            ),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (
            f"Delegation to '{target_id}' timed out after "
            f"{DEFAULT_TIMEOUT_SECONDS} seconds."
        )
    finally:
        await loop.shutdown()

    return (
        f"Delegated agent: {target_id}\n"
        f"Child session: {child_session_id}\n\n"
        f"{result}"
    )


def _build_child_input(task: str, context: str) -> str:
    parts = [
        "You are being invoked as a delegated specialist inside Swarm.",
        "Complete the bounded task below and return a concise final result for the parent agent to synthesize.",
        "",
        f"TASK:\n{task}",
    ]
    if context.strip():
        parts.extend(["", f"CONTEXT FROM PARENT:\n{context.strip()}"])
    return "\n".join(parts)
