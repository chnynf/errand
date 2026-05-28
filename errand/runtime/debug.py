"""Debug logging shared across runtime, brain, and agent loop.

Usage:
    from errand.runtime.debug import debug_log, set_debug
    debug_log("Loop -> AI", body, model="gemini-3-flash", extra="native tools")
    debug_log("! FAILED", body, model="gemini-3-flash")
"""

_enabled = False
_last_system_prompt: str | None = None
_last_session_id: str | None = None

WIDTH = 50


def set_debug(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def is_debug() -> bool:
    return _enabled


def debug_log(
    title: str,
    body: str = "",
    *,
    model: str | None = None,
    extra: str | None = None,
    level: str = "DEBUG",
    truncate: bool = True,
) -> None:
    """Print a debug block with a directional header.

    Args:
        title:  e.g. "User -> Loop", "AI -> Loop", "! FAILED"
        body:   multi-line content (each line will be prefixed with "| ")
        model:  model name shown in brackets
        extra:  additional tag in brackets (e.g. "native tools", "tool results")
        level:  log level ("DEBUG", "ERROR", etc.)
        truncate: whether to truncate the body to 300 chars (default True)
    """
    if not _enabled and level != "ERROR":
        return

    bracket_parts: list[str] = []
    if model:
        bracket_parts.append(model.split("/")[-1])
    if extra:
        bracket_parts.append(extra)
    bracket = f" [{', '.join(bracket_parts)}]" if bracket_parts else ""

    header = f"┌─ {title}{bracket} "
    header = header.ljust(WIDTH, "─")
    footer = "└" + "─" * (WIDTH - 1)

    final_body = body
    if truncate and len(body) > 300:
        final_body = body[:300] + "... (truncated)"

    lines = [f"\n{header}"]
    if final_body:
        for line in final_body.split("\n"):
            lines.append(f"│ {line}")
    lines.append(footer)

    print("\n".join(lines))


def debug_log_prompt(
    title: str,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    extra: str | None = None,
    tool_names: list[str] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    instruction: str | None = None,
) -> None:
    """Log an outgoing AI request, deduplicating repeated system prompts."""
    global _last_system_prompt, _last_session_id
    if not _enabled:
        return

    parts: list[str] = []
    if agent_id:
        parts.append(f"Agent: {agent_id}")

    if system_prompt == _last_system_prompt and session_id == _last_session_id:
        parts.append("System: [same as previous call]")
    else:
        _last_system_prompt = system_prompt
        _last_session_id = session_id
        token_est = len(system_prompt) // 4
        parts.append(f"System: ({token_est} tokens est.)")
        preview = system_prompt[:200].replace("\n", " ")
        if len(system_prompt) > 200:
            preview += "..."
        parts.append(f"  {preview}")

    prompt_preview = user_prompt[:300].replace("\n", " ")
    if len(user_prompt) > 300:
        prompt_preview += "..."
    parts.append(f"Prompt Context: {prompt_preview}")

    if instruction:
        parts.append(f"Instruction: {instruction}")

    if tool_names:
        parts.append(f"Tools:  {', '.join(tool_names)}")

    debug_log(title, "\n".join(parts), model=model, extra=extra, truncate=False)
