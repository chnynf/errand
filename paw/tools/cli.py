"""Generic local CLI execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 20000

# Commands redirected to the dedicated file tools. Only the leading token is
# checked, so pipelines and subcommands (`git grep`, `databricks fs ls`) pass.
_FILESYSTEM_COMMANDS = frozenset(
    {
        "cat", "head", "tail", "less", "more",
        "ls", "tree", "find", "grep", "rg",
        "sed", "awk", "echo", "printf", "tee",
        "touch", "cp", "mv", "rm", "mkdir",
    }
)

_FILESYSTEM_REDIRECT = (
    "Command not run: use the dedicated file tools for filesystem work "
    "(read_file, write_file, edit_file, append_file, delete_file, list_dir, "
    "grep_files, find_files) instead of `{name}`. "
    "run_cli is for external CLIs (databricks, aws, git, uv, python)."
)


async def run_cli(
    command: str,
    cwd: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    _context: dict[str, Any] | None = None,
) -> str:
    """Run a local shell command (databricks, aws, git, uv, python, ...).

    Never use this for file operations — reading, writing, listing, searching,
    or deleting files. Use the dedicated tools instead: read_file, write_file,
    edit_file, list_dir, grep_files, find_files. Commands like cat, ls, sed,
    or echo-redirects are rejected without running.
    For tasks requiring a specific external agent (Claude, Cursor), use invoke_external_agent.

    Args:
        command: Shell command to run.
        cwd: Working directory (optional; defaults to project root).
        timeout_seconds: Max runtime in seconds.
        max_output_chars: Max output characters returned.

    Returns: Command, cwd, exit code, stdout, stderr.
    """
    command = (command or "").strip()
    if not command:
        return "Error: command is required."

    blocked = _filesystem_command(command)
    if blocked:
        return _FILESYSTEM_REDIRECT.format(name=blocked)

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
        return "CLI command was not approved, so it was not run."

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return f"Error: failed to start: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Error: command timed out after {timeout}s."

    stdout_text = _decode_and_truncate(stdout, max_chars)
    stderr_text = _decode_and_truncate(stderr, min(max_chars, 20000))
    lines = [
        f"Exit code: {proc.returncode} (cwd: {workdir})",
        "STDOUT:",
        stdout_text or "(empty)",
    ]
    if stderr_text:
        lines += ["STDERR:", stderr_text]
    return "\n".join(lines)


def _filesystem_command(command: str) -> str | None:
    """The blocked leading command name, or ``None`` if the command may run.

    Skips ``sudo`` and leading ``VAR=value`` assignments, then compares the
    basename of the first real token (so ``/bin/cat`` is still caught).
    """
    for token in command.split():
        if token == "sudo" or ("=" in token and not token.startswith("=")):
            continue
        name = Path(token).name.lower()
        return name if name in _FILESYSTEM_COMMANDS else None
    return None


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
