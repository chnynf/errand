"""Shared runtime control command helpers.

Control commands (``/new``, ``/reset``, ``archive``, ``/reload ...``) are
handled entirely by the runtime and never invoke the model. Each sends a short,
prefixed status line straight back to the user, so there is no wasted model turn
just to "acknowledge" a new session or a reload.
"""

# Prefix marking a message as a runtime control reply (not model output).
CONTROL_REPLY_PREFIX = "🐾 "

NEW_SESSION_MESSAGE = f"{CONTROL_REPLY_PREFIX}New session started."


def reload_message(target: str) -> str:
    """User-facing confirmation for a ``/reload`` command."""
    return f"{CONTROL_REPLY_PREFIX}Reloaded {target}."


def is_new_session_command(text: str) -> bool:
    return text.strip().lower() in {"/new", "/reset", "archive"}


def is_reload_command(text: str) -> bool:
    return text.strip().lower().startswith("/reload")


def is_control_command(text: str) -> bool:
    return is_new_session_command(text) or is_reload_command(text)
