"""Tool registry: auto-load Python tool plugins for native LiteLLM calls.

Each ``.py`` file under ``errand/tools/`` is auto-loaded as a tool
plugin, except for files that are either:

- the registry module itself (``registry.py``);
- private implementation modules (names starting with ``_``);
- ``__init__.py``.

Public module-level functions become tools. Schemas are generated from
the function signature; descriptions come from the function docstring.

A tool module may optionally export a top-level ``COMPACTORS`` dict
mapping tool name to a ``(result, params, preview_limit) -> dict``
function that returns the *tool-specific fields* of the compact memory
record. The registry merges those at load time and exposes
``compact_result`` for the agent loop to call before pushing tool
results into session memory. Tools without a compactor fall back to a
generic preview, so tool authors only opt in when they want a custom
history shape.

This is the local equivalent of an MCP tool server; the same Python
functions can later be wrapped in a stdio MCP server without changing
the tools themselves.
"""

import glob
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

from errand.contracts.types import ToolDefinition, ToolCall, ToolResult

TOOLS_DIR = Path(__file__).resolve().parent

_SKIP_MODULES = {"__init__", "registry"}

# Cross-tool decisioning guidance, authored by the tools component and surfaced
# as a dedicated block in the system prompt next to the file-tool scopes. These
# are nudges that apply across tools (not to any single one), so they live here
# rather than in individual tool descriptions or general runtime prose.
TOOL_USE_GUIDANCE = (
    "TOOL USE:\n"
    "- Think before calling. Reach for a tool only when it adds information or "
    "an effect you cannot produce yourself; otherwise reason and answer directly.\n"
    "- Route, don't search, for the knowledge base. Resolve the path from the "
    "inlined index and read_file it directly; use grep_files or list_dir only "
    "when the index has no pointer — not to rediscover what it already maps.\n"
    "- Batch independent work. When you need several independent things at once "
    "(multiple files, or a read plus a search), issue them as parallel tool calls "
    "in a single round; the runtime runs them concurrently. Sequence calls only "
    "when a later one genuinely depends on an earlier result."
)

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _annotation_name(annotation: Any, default: str) -> str:
    """Return a stable annotation name for real and postponed annotations."""
    if annotation is inspect.Parameter.empty:
        return default
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", default)


class ToolRegistry:
    """Registry of executable tools loaded from ``errand/tools/``."""

    def __init__(
        self,
        tools_dir: Path | None = None,
        can_delegate: list[str] | None = None,
    ):
        self._tools_dir = tools_dir or TOOLS_DIR
        self._tools: dict[str, Callable] = {}
        self._compactors: dict[str, Callable] = {}
        self._descriptions: list[dict[str, str]] = []
        self._load()
        if can_delegate is not None:
            self._patch_delegation_description(can_delegate)

    def _load(self) -> None:
        self._tools.clear()
        self._compactors.clear()
        self._descriptions.clear()

        for file_path in sorted(glob.glob(os.path.join(str(self._tools_dir), "*.py"))):
            module_name = os.path.basename(file_path)[:-3]
            if module_name in _SKIP_MODULES or module_name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if not inspect.isfunction(obj):
                    continue
                if name.startswith("_"):
                    continue
                if obj.__module__ != module_name:
                    continue
                self._tools[name] = obj
                self._descriptions.append(
                    {
                        "name": name,
                        "signature": str(inspect.signature(obj)),
                        "doc": (inspect.getdoc(obj) or "No description provided.").strip(),
                    }
                )

            module_compactors = getattr(module, "COMPACTORS", None)
            if isinstance(module_compactors, dict):
                self._compactors.update(module_compactors)

    def _patch_delegation_description(self, can_delegate: list[str]) -> None:
        """Append the caller's allowed delegate IDs to the invoke_agent description."""
        for desc in self._descriptions:
            if desc["name"] == "invoke_agent":
                if can_delegate:
                    ids = ", ".join(f'"{a}"' for a in can_delegate)
                    desc["doc"] += f"\n\nAllowed agent_id values: {ids}."
                else:
                    desc["doc"] += "\n\nNo sub-agents are configured for delegation."
                break

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Build native ToolDefinition list for LiteLLM."""
        definitions: list[ToolDefinition] = []
        for desc in self._descriptions:
            func = self._tools[desc["name"]]
            sig = inspect.signature(func)

            properties: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname.startswith("_"):
                    continue
                annotation = param.annotation
                py_type = _annotation_name(annotation, "string")
                properties[pname] = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)

            definitions.append(
                ToolDefinition(
                    name=desc["name"],
                    description=desc["doc"],
                    parameters={
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                )
            )
        return definitions

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a tool by name with the given parameters."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found.")
        func = self._tools[name]
        call_params = dict(params)
        if "_context" in inspect.signature(func).parameters:
            call_params["_context"] = context or {}
        if inspect.iscoroutinefunction(func):
            return await func(**call_params)
        return func(**call_params)

    def manifest(self) -> str:
        """Return a JSON manifest of tools (debug-only)."""
        entries: list[dict[str, Any]] = []
        for desc in self._descriptions:
            func = self._tools[desc["name"]]
            sig = inspect.signature(func)
            params: dict[str, dict[str, Any]] = {}
            for pname, param in sig.parameters.items():
                if pname.startswith("_"):
                    continue
                annotation = param.annotation
                type_str = _annotation_name(annotation, "any")
                params[pname] = {
                    "type": type_str,
                    "required": param.default is inspect.Parameter.empty,
                }
            return_type = sig.return_annotation
            entries.append(
                {
                    "name": desc["name"],
                    "description": desc["doc"].split("\n", 1)[0].strip(),
                    "parameters": params,
                    "returns": (
                        _annotation_name(return_type, "any")
                        if return_type is not inspect.Parameter.empty else None
                    ),
                }
            )
        return json.dumps(entries, indent=2)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def tool_summary(self) -> str:
        """Cross-tool decisioning guidance for the system prompt."""
        return TOOL_USE_GUIDANCE

    def compact_result(
        self,
        call: ToolCall | None,
        result: ToolResult,
        preview_limit: int = 500,
    ) -> dict[str, Any]:
        """Build a compact memory record for one tool result.

        Returns a dict shaped for ``Memory._render_tool_record``: always carries
        ``tool_call_id``, ``name``, ``params``, ``content_chars``; optionally
        carries ``preview``, ``error``, ``result_ref``, or ``entry_count``
        depending on the tool's registered compactor (or the generic default).
        """
        params = call.params if call else {}
        record: dict[str, Any] = {
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "params": params,
            "content_chars": len(result.content),
        }
        compactor = self._compactors.get(result.name, _default_compactor)
        record.update(compactor(result, params, preview_limit))
        return record


def _default_compactor(result: ToolResult, params: dict[str, Any], preview_limit: int) -> dict[str, Any]:
    """Generic compactor: truncated preview with an ellipsis marker when cut."""
    content = result.content
    preview = (
        content if len(content) <= preview_limit
        else content[:preview_limit].rstrip() + "... [truncated]"
    )
    return {"preview": preview}
