"""Assemble the message list for one model call.

Lives in ``agent_loop``: the running agent (the loop) is the only consumer of
prompt assembly, so it owns it. The brain just decides over the assembled
messages. The system prompt is intentionally small. It tells the agent:
- it is running inside Paw
- the shared soul and configured agent profile resolved from prompt includes
- the response format Paw expects

The tool catalog and file-scope locations come from the tools component (passed
in as ``tool_summary``), not from this module.

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

# Model-facing marker for a KB-root-absolute path (mirrors files.KB_ROOT_MARKER).
# Preloaded resources and the links inside them are addressed this way so the
# model can pass any path verbatim to read_file.
KB_ROOT_MARKER = "[kb-root]/"

# Default user zone. Paw is UTC-internal; this is only the human-facing
# reference shown to the model alongside UTC, and the default zone the
# scheduler assumes when the user does not name one.
DEFAULT_TZ = ZoneInfo("America/New_York")


def _current_context() -> str:
    """One concise time line: US East (human/scheduling) plus UTC (canonical)."""
    utc = datetime.now(timezone.utc)
    east = utc.astimezone(DEFAULT_TZ)
    return f"Now: {east.strftime('%Y-%m-%d %H:%M %Z')} ({utc.strftime('%H:%M')} UTC)"


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

        return "\n\n".join(parts)

    def build_context_messages(
        self,
        *,
        context_summary: str | None = None,
        instruction: str | None = None,
        session_note: str | None = None,
    ) -> list[dict]:
        """Build volatile per-turn context messages outside the cached system prompt."""
        parts = [_current_context()]
        if session_note:
            parts.append(f"SESSION NOTE:\n{session_note}")
        if context_summary:
            parts.append(f"CONTEXT SUMMARY:\n{context_summary}")
        if instruction:
            parts.append(f"INSTRUCTION:\n{instruction}")
        return [{"role": "user", "content": "\n\n".join(parts)}]

    def build_prompt(
        self,
        history_messages: list[dict],
        current_exchange: list[dict],
        *,
        context_summary: str | None = None,
        instruction: str | None = None,
        session_note: str | None = None,
    ) -> list[dict]:
        """Assemble one model call as four explicit sections, in order:

        1. system            -- soul, indexes, tool catalog (static, cacheable)
        2. history           -- prior exchanges: distant (Q&A-condensed) then
                                recent (full folded), both from memory
        3. current_exchange  -- this exchange, full and uncompacted (live)
        4. context           -- time + session note + rolling summary +
                                instruction (volatile)

        The cacheable prefix is sections 1-3: it grows append-only within a turn
        (only ``current_exchange`` gains messages) and stays byte-identical
        across rounds, so provider prefix caching keeps re-sending it cheap. The
        volatile context is placed LAST so it can never break that prefix -- a
        time-varying block wedged earlier would act as a cache barrier and
        re-bill everything after it on every call.
        """
        return [
            {"role": "system", "content": self.build_system_prompt()},
            *history_messages,
            *current_exchange,
            *self.build_context_messages(
                context_summary=context_summary,
                instruction=instruction,
                session_note=session_note,
            ),
        ]

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
        """Render a prompt resource, labelled with its own ``[kb-root]/`` path.

        The model addresses every file the same way -- ``[kb-root]/<path>`` --
        so the links inside the loaded content are passed verbatim to read_file
        with no per-file base and no absolute filesystem paths exposed.
        """
        if not resource_path:
            return ""
        content = self._load_prompt_resource(resource_path) or ""
        logical_path = self._logical_path(resource_path, scope_hint=scope_hint)
        header = [
            f"--- BEGIN PROMPT RESOURCE: {name} ({KB_ROOT_MARKER}{logical_path}) ---",
            "Already in context -- do not read_file this. To open anything it",
            "lists, pass the `[kb-root]/...` path verbatim to read_file.",
            "",
        ]
        footer = ["", f"--- END PROMPT RESOURCE: {name} ---"]
        return "\n".join(header) + content + "\n".join(footer)

    def _logical_path(self, resource_path: str, *, scope_hint: str) -> str:
        """Return the KB-root-relative path for a prompt resource (no marker)."""
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
        return logical
