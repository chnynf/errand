"""Shared runtime control command helpers."""

NEW_SESSION_PROMPT = "User started a new session. Acknowledge briefly."
RELOAD_PROMPT = "Runtime instructions were reloaded. Acknowledge briefly."


def is_new_session_command(text: str) -> bool:
    return text.strip().lower() in {"/new", "/reset", "archive"}


def is_reload_command(text: str) -> bool:
    return text.strip().lower().startswith("/reload")


def is_control_command(text: str) -> bool:
    return is_new_session_command(text) or is_reload_command(text)
