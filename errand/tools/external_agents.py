"""Delegate bounded tasks to configured external CLI agents."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from errand.config import load_errand_config


async def invoke_external_agent(
    agent: str,
    task: str,
    context: str = "",
    _context: dict[str, Any] | None = None,
) -> str:
    """Delegate a bounded task to a configured external CLI agent.

    Use for agents such as Cursor or Claude when their installed CLI harness is
    better suited to the task. The CLI receives a scoped prompt and Errand
    returns bounded stdout/stderr for synthesis.
    """
    runtime_context = _context or {}
    caller_agent_id = str(runtime_context.get("agent_id") or "")
    config = load_errand_config()
    caller = config.get_agent(caller_agent_id)

    agent_name = (agent or "").strip()
    if not agent_name:
        return "External delegation failed: agent is required."
    if agent_name not in config.external_agents:
        return f"External delegation failed: agent '{agent_name}' is not configured."
    if agent_name not in caller.can_delegate:
        return (
            f"External delegation failed: agent '{caller.id}' is not allowed "
            f"to delegate to '{agent_name}'."
        )

    spec = config.external_agents[agent_name]
    if not spec.command:
        return f"External delegation failed: agent '{agent_name}' has no command."
    if spec.prompt_mode not in {"stdin", "argument", "stream_json"}:
        return (
            f"External delegation failed: agent '{agent_name}' has invalid "
            f"prompt_mode {spec.prompt_mode!r}."
        )

    task_text = (task or "").strip()
    if not task_text:
        return "External delegation failed: task is required."

    reply_to = runtime_context.get("reply_to")
    if reply_to:
        try:
            await reply_to.send_progress(f"Delegating to {agent_name}...")
        except Exception:
            pass

    prompt = _build_external_prompt(task_text, context)
    if spec.prompt_mode == "stream_json":
        return await _run_stream_json_agent(agent_name, spec, prompt, runtime_context)

    command = list(spec.command)
    stdin = asyncio.subprocess.PIPE
    if spec.prompt_mode == "argument":
        command.append(prompt)
        stdin = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return (
            f"External delegation failed: command not found for '{agent_name}': "
            f"{spec.command[0]}"
        )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(
                prompt.encode("utf-8") if spec.prompt_mode != "argument" else None
            ),
            timeout=spec.timeout_seconds,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return (
            f"External delegation to '{agent_name}' timed out after "
            f"{spec.timeout_seconds} seconds."
        )

    stdout_text = _decode_and_truncate(stdout, spec.max_output_chars)
    stderr_text = _decode_and_truncate(stderr, min(spec.max_output_chars, 4000))

    parts = [
        f"External agent: {agent_name}",
        f"Command: {' '.join(command if spec.prompt_mode != 'argument' else spec.command)}",
        f"Exit code: {proc.returncode}",
        "",
        "STDOUT:",
        stdout_text or "(empty)",
    ]
    if stderr_text:
        parts.extend(["", "STDERR:", stderr_text])
    return "\n".join(parts)


async def _run_stream_json_agent(
    agent_name: str,
    spec,
    prompt: str,
    runtime_context: dict[str, Any],
) -> str:
    """Run an external agent using newline-delimited JSON I/O."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return (
            f"External delegation failed: command not found for '{agent_name}': "
            f"{spec.command[0]}"
        )

    assert proc.stdin is not None
    assert proc.stdout is not None
    user_message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": prompt,
        },
    }
    proc.stdin.write((json.dumps(user_message) + "\n").encode("utf-8"))
    await proc.stdin.drain()

    output_parts: list[str] = []
    events_seen = 0
    deadline = asyncio.get_event_loop().time() + spec.timeout_seconds
    _LINE_TIMEOUT = 60.0
    try:
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                proc.kill()
                await proc.communicate()
                return (
                    f"External delegation to '{agent_name}' timed out after "
                    f"{spec.timeout_seconds} seconds."
                )
            raw_line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=min(_LINE_TIMEOUT, remaining),
            )
            if not raw_line:
                break
            events_seen += 1
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                output_parts.append(line)
                continue

            if _is_permission_request(event):
                await _handle_permission_request(proc, event, runtime_context)
                continue

            text = _extract_stream_text(event)
            if text:
                output_parts.append(text)

            if event.get("type") == "result":
                if event.get("is_error"):
                    for error in event.get("errors") or []:
                        output_parts.append(f"Error: {error}")
                break
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return (
            f"External delegation to '{agent_name}' timed out after "
            f"{spec.timeout_seconds} seconds."
        )

    if proc.stdin and not proc.stdin.is_closing():
        proc.stdin.close()
    stdout_tail, stderr = await proc.communicate()
    if stdout_tail:
        output_parts.append(_decode_and_truncate(stdout_tail, spec.max_output_chars))
    stderr_text = _decode_and_truncate(stderr, min(spec.max_output_chars, 4000))
    body = "\n".join(part for part in output_parts if part).strip()
    body = _truncate_text(body, spec.max_output_chars)
    parts = [
        f"External agent: {agent_name}",
        f"Command: {' '.join(spec.command)}",
        f"Exit code: {proc.returncode}",
        f"Stream events: {events_seen}",
        "",
        "OUTPUT:",
        body or "(empty)",
    ]
    if stderr_text:
        parts.extend(["", "STDERR:", stderr_text])
    return "\n".join(parts)


def _is_permission_request(event: dict[str, Any]) -> bool:
    event_type = event.get("type")
    request = event.get("request") or event.get("control_request") or {}
    return event_type in {"control_request", "sdk_control_request"} or (
        isinstance(request, dict) and request.get("subtype") == "permission"
    )


async def _handle_permission_request(
    proc: asyncio.subprocess.Process,
    event: dict[str, Any],
    runtime_context: dict[str, Any],
) -> None:
    assert proc.stdin is not None
    request = event.get("request") or event.get("control_request") or event
    request_id = event.get("request_id") or request.get("request_id") or request.get("id")
    tool_name = request.get("tool_name") or request.get("tool") or "tool"
    tool_input = request.get("tool_input") or request.get("input") or {}
    approved = await _request_approval(
        runtime_context,
        title=f"Claude requests permission: {tool_name}",
        details=f"Tool input:\n```json\n{json.dumps(tool_input, indent=2)[:1500]}\n```",
    )
    response_type = (
        "sdk_control_response"
        if event.get("type") == "sdk_control_request"
        else "control_response"
    )
    if approved:
        response = {
            "type": response_type,
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {
                    "behavior": "allow",
                    "updatedInput": tool_input,
                },
            },
        }
    else:
        response = {
            "type": response_type,
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {
                    "behavior": "deny",
                    "message": "User denied this tool request through Errand.",
                },
            },
        }
    proc.stdin.write((json.dumps(response) + "\n").encode("utf-8"))
    await proc.stdin.drain()


def _extract_stream_text(event: dict[str, Any]) -> str:
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        chunks = []
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
        return "\n".join(chunks)
    if event.get("type") == "result" and event.get("result"):
        return str(event.get("result"))
    return ""


def _build_external_prompt(task: str, context: str) -> str:
    user_name = os.getenv("ERRAND_USER_NAME", "")
    user_email = os.getenv("ERRAND_USER_EMAIL", "")
    identity_lines = []
    if user_name:
        identity_lines.append(f"User name: {user_name}")
    if user_email:
        identity_lines.append(f"User email: {user_email}")

    parts = ["You are being invoked by Errand as an external delegated agent."]
    if identity_lines:
        parts.append("USER IDENTITY:\n" + "\n".join(identity_lines))
    parts += [
        "Complete the bounded task and return a concise final result.",
        "",
        f"TASK:\n{task}",
    ]
    if context.strip():
        parts.extend(["", f"CONTEXT FROM ERRAND:\n{context.strip()}"])
    return "\n".join(parts)


async def _request_approval(
    context: dict[str, Any],
    *,
    title: str,
    details: str,
) -> bool:
    reply_to = context.get("reply_to")
    if reply_to is None or not hasattr(reply_to, "request_approval"):
        return True
    return bool(
        await reply_to.request_approval(
            title=title,
            details=details,
            timeout_seconds=300,
        )
    )


def _decode_and_truncate(data: bytes, max_chars: int) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    return _truncate_text(text, max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
