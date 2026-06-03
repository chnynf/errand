"""Assemble system and user prompts for the model.

The system prompt is intentionally small. It tells the agent:
- it is running inside Errand
- the shared soul and configured agent profile resolved from runtime prompt includes
- how to use scoped file tools and the response format Errand expects

``runtime.md`` may contain host-side include placeholders. The model only
sees the resolved prompt text, never the include directive.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from errand.config import FileAccessConfig

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SHARED_SOUL_INCLUDE = "{{ include:SHARED_SOUL }}"
AGENT_PROFILE_INCLUDE = "{{ include:AGENT_PROFILE }}"


def _current_context() -> str:
    """Return a short header with current time, day, and location."""
    tz_str = os.getenv("TIMEZONE", "UTC")
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(tz_str)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now(timezone.utc)
        tz_str = "UTC"
    location = os.getenv("LOCATION", tz_str)
    utc_now = now.astimezone(timezone.utc)
    iso_utc = utc_now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        "CURRENT CONTEXT:\n"
        f"- Date and time: {now.strftime('%Y-%m-%d, %A, %H:%M %Z')} (UTC: {iso_utc})\n"
        f"- Location: {location}"
    )


class PromptAssembler:
    """Builds system and user prompts from runtime files and config."""

    def __init__(
        self,
        shared_soul: Optional[str] = None,
        agent_profile: Optional[str] = None,
        file_access: Optional[FileAccessConfig] = None,
    ):
        self._runtime_template = self._load("runtime.md")
        self._shared_soul = shared_soul
        self._agent_profile = agent_profile
        self._file_access = file_access or FileAccessConfig()
        # Cached rendered blocks — static for the process lifetime.
        self._soul_block: Optional[str] = None
        self._profile_block: Optional[str] = None

    def reload_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        """Clear selected cached prompt resources; files reload on next prompt."""
        if soul:
            self._soul_block = None
        if profile:
            self._profile_block = None

    @staticmethod
    def _load(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def build_system_prompt(self) -> str:
        """Return the full system prompt for native tool calling.

        Intentionally excludes time-varying content so the prefix is stable
        and eligible for provider-level prompt caching (Gemini, DeepSeek, …).
        Per-turn context such as current time is injected in the user message.
        """
        parts = [self._render_runtime()]

        if self._file_access.scopes:
            parts.append(self._file_access_section())

        return "\n\n".join(parts)

    def build_user_prompt(self, context_text: str, instruction: str) -> str:
        """Wrap session context and the per-turn instruction for the model."""
        return f"{_current_context()}\n\nSESSION CONTEXT:\n{context_text}\n\nINSTRUCTION:\n{instruction}"

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

    def _render_runtime(self) -> str:
        if self._soul_block is None:
            self._soul_block = self._render_prompt_resource(
                name="SHARED_SOUL",
                resource_path=self._shared_soul,
                scope_hint="kb",
            )
        if self._profile_block is None:
            self._profile_block = self._render_prompt_resource(
                name="AGENT_PROFILE",
                resource_path=self._agent_profile,
                scope_hint="kb",
            )
        runtime = self._runtime_template
        runtime = runtime.replace(SHARED_SOUL_INCLUDE, self._soul_block)
        runtime = runtime.replace(AGENT_PROFILE_INCLUDE, self._profile_block)
        return runtime

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
        base = "/".join(logical.split("/")[:-1]) + "/"
        if base == "/":
            base = "./"
        if base == "":
            base = "./"
        return logical, base
