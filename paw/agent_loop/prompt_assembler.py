"""Assemble the message list for one model call.

Lives in ``agent_loop``: the running agent (the loop) is the only consumer of
prompt assembly, so it owns it. The brain just decides over the assembled
messages. The system prompt is intentionally small. It tells the agent:
- it is running inside Paw
- the shared soul and configured agent profile resolved from prompt includes
- how to use scoped file tools and the response format Paw expects

``agent.md`` may contain host-side include placeholders. The model only sees
the resolved prompt text, never the include directive.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from paw.config import FileAccessConfig

# The agent framing template is this component's own data; it lives next to the
# loop that emits it, not a separate top-level prompts folder.
_TEMPLATE_DIR = Path(__file__).resolve().parent
SHARED_SOUL_INCLUDE = "{{ include:SHARED_SOUL }}"
SHARED_NOTES_INDEX_INCLUDE = "{{ include:SHARED_NOTES_INDEX }}"
AGENT_PROFILE_INCLUDE = "{{ include:AGENT_PROFILE }}"

# Default user zone. Paw is UTC-internal; this is only the human-facing
# reference shown to the model alongside UTC, and the default zone the
# scheduler assumes when the user does not name one.
DEFAULT_TZ = ZoneInfo("America/New_York")


def _current_context() -> str:
    """Return a short header with the current time in US East and UTC."""
    utc = datetime.now(timezone.utc)
    east = utc.astimezone(DEFAULT_TZ)
    return (
        "CURRENT CONTEXT:\n"
        f"- US East: {east.strftime('%Y-%m-%d, %A, %H:%M %Z')}\n"
        f"- UTC: {utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )


class PromptAssembler:
    """Builds system and user prompts from runtime files and config."""

    def __init__(
        self,
        shared_soul: Optional[str] = None,
        agent_profile: Optional[str] = None,
        file_access: Optional[FileAccessConfig] = None,
        shared_notes_index: Optional[str] = None,
        tool_summary: Optional[str] = None,
    ):
        self._agent_template = self._load("agent.md")
        self._shared_soul = shared_soul
        self._shared_notes_index = shared_notes_index
        self._agent_profile = agent_profile
        self._file_access = file_access or FileAccessConfig()
        # Tool-component-authored cross-tool guidance, placed verbatim. The
        # assembler only positions it; the tools component owns the text.
        self._tool_summary = tool_summary
        # Cached rendered blocks — static for the process lifetime.
        self._soul_block: Optional[str] = None
        self._notes_block: Optional[str] = None
        self._profile_block: Optional[str] = None

    def reload_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        """Clear selected cached prompt resources; files reload on next prompt.

        The shared notes router is a shared resource like the soul, so it is
        cleared alongside the soul.
        """
        if soul:
            self._soul_block = None
            self._notes_block = None
        if profile:
            self._profile_block = None

    @staticmethod
    def _load(filename: str) -> str:
        path = _TEMPLATE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def build_system_prompt(self) -> str:
        """Return the full system prompt for native tool calling.

        Intentionally excludes time-varying content so the prefix is stable
        and eligible for provider-level prompt caching (Gemini, DeepSeek, …).
        Per-turn context such as current time is injected in the user message.
        """
        parts = [self._render_agent_template()]

        if self._tool_summary:
            parts.append(self._tool_summary)
        if self._file_access.scopes:
            parts.append(self._file_access_section())

        return "\n\n".join(parts)

    def build_context_messages(
        self,
        *,
        context_summary: str | None = None,
        session_note: str | None = None,
        instruction: str | None = None,
    ) -> list[dict]:
        """Build volatile per-turn context messages outside the cached system prompt."""
        parts = [_current_context()]
        if context_summary:
            parts.append(f"CONTEXT SUMMARY:\n{context_summary}")
        if session_note:
            parts.append(f"SESSION NOTE:\n{session_note}")
        if instruction:
            parts.append(f"INSTRUCTION:\n{instruction}")
        return [{"role": "user", "content": "\n\n".join(parts)}]

    def build_messages(
        self,
        history_messages: list[dict],
        *,
        context_summary: str | None = None,
        session_note: str | None = None,
        instruction: str | None = None,
    ) -> list[dict]:
        """Assemble the full message list for one model call.

        system prompt (cached prefix) -> per-turn context -> conversation history.
        """
        return [
            {"role": "system", "content": self.build_system_prompt()},
            *self.build_context_messages(
                context_summary=context_summary,
                session_note=session_note,
                instruction=instruction,
            ),
            *history_messages,
        ]

    def _file_access_section(self) -> str:
        lines = [
            "FILE TOOL SCOPES:",
            "Pass a scope name to the file tools. Each scope lists per-operation",
            "permissions: true=allow, false=block, ask=requires approval.",
            "Tools: read_file, list_dir, grep_files, find_files, write_file,",
            "append_file, edit_file, delete_file.",
            "Read files before quoting them. Use append_file to add new notes;",
            "edit_file for targeted updates; write_file only for new or replacement files.",
        ]
        if self._file_access.scopes:
            lines.append(f"- DEFAULT_FILE_SCOPE: {self._file_access.default_scope}")
            for name, scope in sorted(self._file_access.scopes.items()):
                roots = ", ".join(scope.roots)
                ops = {
                    "read":   scope.read,
                    "list":   scope.list,
                    "write":  scope.write,
                    "append": scope.append,
                    "edit":   scope.edit,
                    "delete": scope.delete,
                }
                perm_str = ", ".join(f"{op}={val}" for op, val in ops.items())
                lines.append(f"- FILE_SCOPE {name}: roots=[{roots}], ops=[{perm_str}]")
        return "\n".join(lines)

    def _render_agent_template(self) -> str:
        blocks = [
            ("_soul_block", "SHARED_SOUL", self._shared_soul, SHARED_SOUL_INCLUDE),
            ("_notes_block", "SHARED_NOTES_INDEX", self._shared_notes_index, SHARED_NOTES_INDEX_INCLUDE),
            ("_profile_block", "AGENT_PROFILE", self._agent_profile, AGENT_PROFILE_INCLUDE),
        ]
        rendered = self._agent_template
        for attr, name, path, placeholder in blocks:
            if getattr(self, attr) is None:
                setattr(self, attr, self._render_prompt_resource(name=name, resource_path=path, scope_hint="kb"))
            rendered = rendered.replace(placeholder, getattr(self, attr))
        return rendered

    @staticmethod
    def _load_prompt_resource(resource_path: Optional[str]) -> Optional[str]:
        if not resource_path:
            return None
        path = Path(resource_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Prompt resource not found: {path}")
        if not path.is_file():
            raise ValueError(f"Prompt resource is not a file: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _render_prompt_resource(
        self,
        *,
        name: str,
        resource_path: Optional[str],
        scope_hint: str = "kb",
    ) -> str:
        """Render a prompt resource with logical path + base path metadata.

        The goal is to let the model resolve relative references inside the
        loaded content without exposing absolute filesystem paths.
        """
        if not resource_path:
            return ""
        content = self._load_prompt_resource(resource_path) or ""
        logical_path, base_path = self._logical_path_and_base(
            resource_path, scope_hint=scope_hint
        )
        header = [
            f"--- BEGIN PROMPT RESOURCE: {name} ---",
            f"Logical path: {logical_path}",
            f"Base path: {base_path}",
            "",
        ]
        footer = ["", f"--- END PROMPT RESOURCE: {name} ---"]
        return "\n".join(header) + content + "\n".join(footer)

    def _logical_path_and_base(self, resource_path: str, *, scope_hint: str) -> tuple[str, str]:
        """Return (logical_path, base_path) for a prompt resource."""
        abs_path = Path(os.path.expanduser(str(resource_path))).resolve()
        roots = self._file_access.scopes.get(scope_hint).roots if self._file_access.scopes.get(scope_hint) else []
        roots_expanded = [Path(os.path.expanduser(str(r))).resolve() for r in roots]
        logical = abs_path.name
        for root in roots_expanded:
            try:
                logical = str(abs_path.relative_to(root)).replace("\\", "/")
                break
            except ValueError:
                continue
        base = "/".join(logical.split("/")[:-1])
        return logical, f"{base}/" if base else "./"
