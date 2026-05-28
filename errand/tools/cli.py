"""Generic local CLI execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 20000


async def run_cli(
    command: str,
    cwd: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    _context: dict[str, Any] | None = None,
) -> str:
    """Run a local shell command and return stdout/stderr.

    Use when the Swarm agent itself needs to run an installed CLI tool such as
    databricks, aws, git, uv, or python. For tasks the user explicitly wants a
    specific external agent (Claude, Cursor) to handle, use invoke_external_agent
    instead.

    Args:
        command: Shell command to run.
        cwd: Optional working directory. Defaults to the Swarm project root.
        timeout_seconds: Maximum runtime before killing the command.
        max_output_chars: Maximum stdout/stderr characters returned to the model.

    Returns:
        Command, working directory, exit code, stdout, and stderr.
    """
    command = (command or "").strip()
    if not command:
        return "CLI execution failed: command is required."

    timeout = max(1, min(int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), 3600))
    max_chars = max(1000, min(int(max_output_chars or DEFAULT_MAX_OUTPUT_CHARS), 100000))
    workdir = _resolve_cwd(cwd)

    approval = await _request_approval(
        _context or {},
        command=command,
        cwd=str(workdir),
        timeout_seconds=timeout,
    )
    if not approval:
        return (
            "CLI command was not approved, so it was not run.\n"
            f"Command: {command}\n"
            f"Working directory: {workdir}"
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return f"CLI execution failed to start: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return (
            f"CLI command timed out after {timeout}s.\n"
            f"Command: {command}\n"
            f"Working directory: {workdir}"
        )

    stdout_text = _decode_and_truncate(stdout, max_chars)
    stderr_text = _decode_and_truncate(stderr, min(max_chars, 20000))
    return "\n".join(
        [
            f"Command: {command}",
            f"Working directory: {workdir}",
            f"Exit code: {proc.returncode}",
            "",
            "STDOUT:",
            stdout_text or "(empty)",
            "",
            "STDERR:",
            stderr_text or "(empty)",
        ]
    )


def _resolve_cwd(cwd: str) -> Path:
    if cwd.strip():
        path = Path(cwd).expanduser()
    else:
        path = Path(__file__).resolve().parents[2]
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"cwd does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {path}")
    return path


async def _request_approval(
    context: dict[str, Any],
    *,
    command: str,
    cwd: str,
    timeout_seconds: int,
) -> bool:
    reply_to = context.get("reply_to")
    if reply_to is None or not hasattr(reply_to, "request_approval"):
        return True
    title = "Run local CLI command"
    details = (
        f"Agent: {context.get('agent_id', 'unknown')}\n"
        f"Command: `{command}`\n"
        f"Working directory: `{cwd}`\n"
        f"Timeout: {timeout_seconds}s"
    )
    return bool(
        await reply_to.request_approval(
            title=title,
            details=details,
            timeout_seconds=300,
        )
    )


def _decode_and_truncate(data: bytes, max_chars: int) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
