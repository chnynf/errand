"""Session persistence and conversation history.

Session JSON files live under ``errand/sessions/_data/``. The
``_data/`` folder is git-ignored runtime state, separate from the
Python modules in the same package.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import aiofiles

from errand.contracts.types import ToolCall, ToolResult

_SESSION_DIR = Path(__file__).resolve().parent / "_data"
_MAX_HISTORY_ENTRIES = 400
# Characters that are illegal in Windows filenames (e.g. ``:`` in WeChat and
# scheduled session ids). Mapped to ``_`` so files stay cross-platform safe.
_FILENAME_ILLEGAL = '<>:"/\\|?*'


def _safe_stem(session_id: str) -> str:
    return "".join("_" if c in _FILENAME_ILLEGAL else c for c in session_id)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    pricing: Optional[Dict[str, float]],
) -> float:
    """Estimate USD cost for one call from per-model pricing (per 1M tokens).

    ``input_tokens`` is the total prompt size and already includes any
    ``cache_read_tokens`` (cache hits). Cache hits are billed at the cheaper
    ``cache_read`` rate, so we split input into fresh vs cached and price each
    separately. Returns 0.0 when no pricing is configured for the model.
    """
    if not pricing:
        return 0.0
    in_rate = pricing.get("input", 0.0) or 0.0
    out_rate = pricing.get("output", 0.0) or 0.0
    cache_rate = pricing.get("cache_read", in_rate)
    if cache_rate is None:
        cache_rate = in_rate
    fresh_input = max(input_tokens - cache_read_tokens, 0)
    return (
        (fresh_input / 1_000_000) * in_rate
        + (cache_read_tokens / 1_000_000) * cache_rate
        + (output_tokens / 1_000_000) * out_rate
    )
IDLE_BOUNDARY_NOTE = (
    "There was a long idle gap in this conversation. Treat the current message "
    "as a new topic if it does not appear related to the last exchange."
)
RELOAD_BOUNDARY_NOTE = (
    "Runtime instructions were reloaded. Use the current system instructions "
    "going forward; prior conversation remains available as history but may "
    "reflect older instructions."
)


class Memory:
    """Persistent conversation history for one session.

    History roles:
        user    -- user's input
        ai      -- AI response (tool_calls or text), with model in metadata
        tool    -- tool execution results
        brain   -- legacy alias for ``ai`` (kept readable from old sessions)
        system  -- internal tool feedback messages
    """

    def __init__(self, session_id: str = "default", agent_id: str = "default"):
        self.session_id = session_id
        self.agent_id = agent_id
        self.session_dir = str(_SESSION_DIR)
        self._file_stem = _safe_stem(session_id)
        self.session_file = os.path.join(self.session_dir, f"{self._file_stem}.json")
        self._ensure_session_dir()
        self.data: Dict[str, Any] = self._load_session()
        self.data.setdefault("metadata", {})["agent_id"] = self.agent_id

    def _ensure_session_dir(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "archive"), exist_ok=True)

    def _load_session(self) -> Dict[str, Any]:
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_new_session()
        return self._create_new_session()

    def _create_new_session(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": time.time(),
            "history": [],
            "context_summary": None,
            "metadata": {},
            "token_summary": {
                "input_tokens": 0,
                "output_tokens": 0,
                # Subset of input_tokens that were served from a prompt cache
                # (billed at the much cheaper cache-hit rate).
                "cache_read_tokens": 0,
                "total_cost": 0.0,
            },
        }

    async def save_session(self) -> None:
        if len(self.data["history"]) > _MAX_HISTORY_ENTRIES:
            self.data["history"] = self.data["history"][-_MAX_HISTORY_ENTRIES:]
        async with aiofiles.open(self.session_file, "w") as f:
            await f.write(json.dumps(self.data, indent=2))

    def add_history(self, role: str, content: Any, metadata: Optional[Dict] = None) -> None:
        entry = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.data["history"].append(entry)

    def add_tool_results(
        self,
        tool_calls: Iterable[ToolCall],
        tool_results: Iterable[ToolResult],
    ) -> None:
        """Persist compact tool-result records without copying bulky contents."""
        calls_by_id = {call.id: call for call in tool_calls}
        entries = []
        for result in tool_results:
            call = calls_by_id.get(result.tool_call_id)
            entries.append(self._compact_tool_result(call, result))
        if entries:
            self.add_history("tool", entries)

    def update_token_usage(self, usage: Dict[str, Any]) -> None:
        """Accumulate one API call's usage into the session ``token_summary``.

        Token counts come straight from the provider's ``usage`` object (the
        formally billed values) -- nothing is estimated here. Cost is derived
        from the per-model ``pricing`` block carried on the usage dict; if a
        model has no pricing configured, its cost contributes 0.
        """
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cache_read_tokens = usage.get("cache_read_tokens", 0) or 0

        summary = self.data["token_summary"]
        summary["input_tokens"] += input_tokens
        summary["output_tokens"] += output_tokens
        summary.setdefault("cache_read_tokens", 0)
        summary["cache_read_tokens"] += cache_read_tokens
        summary["total_cost"] += estimate_cost(
            input_tokens, output_tokens, cache_read_tokens, usage.get("pricing")
        )

    def set_context_summary(self, summary: Optional[str]) -> None:
        self.data["context_summary"] = summary

    def last_activity_at(self) -> Optional[float]:
        history = self.data.get("history") or []
        if history:
            return float(history[-1].get("timestamp") or 0)
        return float(self.data.get("created_at") or 0) or None

    def add_session_note(self, note: str) -> None:
        self.data.setdefault("metadata", {})["session_note"] = note

    def clear_session_note(self) -> None:
        self.data.setdefault("metadata", {}).pop("session_note", None)

    def get_formatted_context(
        self,
        recent_n: int = 20,
        working_trace: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build prompt context from summary, visible conversation, and trace."""
        parts = []

        ctx_summary = self.data.get("context_summary")
        if ctx_summary:
            parts.append(f"CONTEXT SUMMARY:\n{ctx_summary}")

        session_note = self.data.get("metadata", {}).get("session_note")
        if session_note:
            parts.append(f"SESSION NOTE:\n{session_note}")

        history = self.data["history"]
        visible = [
            line
            for entry in history
            if (
                line := self._format_history_entry(
                    entry.get("role", "unknown"),
                    entry.get("content", ""),
                )
            )
        ]
        recent = visible[-recent_n:]

        if recent:
            parts.append("RECENT CONVERSATION:\n" + "\n".join(recent))

        if working_trace:
            parts.append(f"CURRENT TURN TRACE:\n{working_trace}")

        if working_trace:
            instruction = (
                "The previous action has completed. Analyze the result above. "
                "Decide whether to take another action or respond to the user."
            )
        else:
            instruction = "Analyze the user's input. Decide whether to call a tool or respond directly."

        return "\n\n".join(parts), instruction

    @staticmethod
    def _format_history_entry(role: str, content: Any) -> str:
        if role == "user":
            return f"USER: {content}"

        if role == "scheduled":
            return f"SCHEDULED TASK (firing now): {content}"

        def _format_ai_like_dict(payload: dict) -> str:
            tool_calls = payload.get("tool_calls") or []
            if tool_calls:
                descs = []
                for c in tool_calls:
                    params_str = ", ".join(
                        f"{k}={v!r}" for k, v in c.get("params", {}).items()
                    )
                    descs.append(f"{c.get('name', '?')}({params_str})")
                return "AI: [tool call] " + "; ".join(descs)
            text = payload.get("text_response") or payload.get("external_response")
            if text:
                return f"AI: {text}"
            return f"AI: {json.dumps(payload, ensure_ascii=False)}"

        if role == "ai":
            if isinstance(content, dict):
                if content.get("tool_calls"):
                    names = ", ".join(
                        c.get("name", "?") for c in content["tool_calls"]
                    )
                    return f"AI: [tools: {names}]"
                ctype = content.get("type", "")
                if ctype == "tool_calls":
                    return ""
                if ctype == "text":
                    return f"AI: {content.get('text', '')}"
                return _format_ai_like_dict(content)
            return f"AI: {content}"

        if role == "tool":
            return ""

        if role == "brain":
            if isinstance(content, dict):
                if content.get("type") == "tool_calls" or content.get("tool_calls"):
                    return ""
                return _format_ai_like_dict(content)
            return f"AI: {content}"
        if role == "system":
            return ""

        return ""

    @staticmethod
    def _compact_tool_result(
        call: Optional[ToolCall],
        result: ToolResult,
        preview_limit: int = 500,
    ) -> dict[str, Any]:
        params = call.params if call else {}
        content = result.content
        record: dict[str, Any] = {
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "params": params,
            "content_chars": len(content),
        }

        if result.name == "read_file":
            record["result_ref"] = {
                "type": "file",
                "scope": params.get("scope", "kb"),
                "path": params.get("path"),
            }
            if content.startswith("Error:"):
                record["error"] = content[:preview_limit]
            return record

        if result.name == "list_dir":
            record["result_ref"] = {
                "type": "directory",
                "scope": params.get("scope", "kb"),
                "path": params.get("path", ""),
            }
            record["entry_count"] = 0 if content.startswith("Error:") else len(
                [line for line in content.splitlines() if line.strip()]
            )
            if content.startswith("Error:"):
                record["error"] = content[:preview_limit]
            return record

        if result.name in ("write_file", "edit_file", "delete_file"):
            # Keep history small: do not persist large write payloads verbatim.
            record["params"] = {
                key: value
                for key, value in params.items()
                if key not in ("content", "old_string", "new_string")
            }
            record["result_ref"] = {
                "type": "file",
                "scope": params.get("scope", "kb"),
                "path": params.get("path"),
            }
            record["preview"] = content[:preview_limit]
            return record

        if result.name in ("grep_files", "find_files"):
            record["preview"] = content[:preview_limit]
            return record

        record["preview"] = (
            content
            if len(content) <= preview_limit
            else content[:preview_limit].rstrip() + "... [truncated]"
        )
        return record

    async def archive_session(self, start_new: bool = True) -> None:
        if os.path.exists(self.session_file):
            archive_dir = os.path.join(self.session_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)

            archive_path = os.path.join(
                archive_dir, f"{self._file_stem}_{int(time.time())}.json"
            )
            print(f"Archiving session {self.session_id} to {archive_path}")
            try:
                os.rename(self.session_file, archive_path)
                print(f"Successfully archived session {self.session_id}")
            except Exception as e:
                print(f"Failed to archive session {self.session_id}: {e}")
                raise

            if start_new:
                self.data = self._create_new_session()
                await self.save_session()
