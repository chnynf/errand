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
        scheduled -- scheduled task firing through the agent
        ai      -- AI response (tool_calls or text), with model in metadata
        tool    -- tool execution results
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
                with open(self.session_file, "r", encoding="utf-8") as f:
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
        async with aiofiles.open(self.session_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self.data, indent=2, ensure_ascii=False))

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

    def build_history_messages(self, recent_n: int = 20) -> list[dict]:
        """Build role-tagged chat messages from persisted compact history."""
        messages: list[dict] = []
        history = self.data["history"][-recent_n:]
        i = 0
        while i < len(history):
            entry = history[i]
            role = entry["role"]
            content = entry["content"]
            if role in ("user", "scheduled"):
                messages.append({"role": "user", "content": str(content)})
                i += 1
            elif role == "ai":
                msg = self._ai_message(content)
                if msg and not msg.get("tool_calls"):
                    messages.append(msg)
                elif msg and i + 1 < len(history) and history[i + 1]["role"] == "tool":
                    tool_messages = self._matching_tool_messages(
                        msg["tool_calls"],
                        history[i + 1]["content"],
                    )
                    if len(tool_messages) == len(msg["tool_calls"]):
                        messages.append(msg)
                        messages.extend(tool_messages)
                        i += 1
                i += 1
            elif role == "tool":
                # A dangling compact tool entry means the recent history slice
                # lost its assistant tool-call entry. Skip it rather than
                # sending an invalid OpenAI message sequence.
                i += 1
            else:
                i += 1
        return messages

    @staticmethod
    def _ai_message(content: Any) -> dict | None:
        if not isinstance(content, dict):
            return {"role": "assistant", "content": str(content)}
        tool_calls = content.get("tool_calls") or []
        if tool_calls:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("params", {})),
                        },
                    }
                    for call in tool_calls
                ],
            }
        text = content.get("text_response")
        return {"role": "assistant", "content": text} if text else None

    @classmethod
    def _matching_tool_messages(cls, tool_calls: list[dict], records: Any) -> list[dict]:
        expected_ids = {call["id"] for call in tool_calls}
        return [
            {
                "role": "tool",
                "tool_call_id": record["tool_call_id"],
                "content": cls._render_tool_record(record),
            }
            for record in records
            if record["tool_call_id"] in expected_ids
        ]

    @staticmethod
    def _render_tool_record(record: dict[str, Any]) -> str:
        if record.get("error"):
            return str(record["error"])
        if record.get("preview"):
            return str(record["preview"])
        if ref := record.get("result_ref"):
            scope = ref.get("scope", "kb")
            path = ref.get("path", "")
            chars = record.get("content_chars", 0)
            return f"{record['name']}: result stored as {ref['type']} ref {scope}:{path} ({chars} chars)"
        if "entry_count" in record:
            return f"{record['name']}: {record['entry_count']} entries"
        return f"{record['name']}: {record.get('content_chars', 0)} chars"

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
