"""Tool registry: auto-load Python tool plugins for native LiteLLM calls.

Each ``.py`` file under ``paw/tools/`` is auto-loaded as a tool
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
from typing import Any, Callable, Iterable

from paw.contracts.types import ToolDefinition, ToolCall, ToolResult

TOOLS_DIR = Path(__file__).resolve().parent

_SKIP_MODULES = {"__init__", "registry"}

# Cross-tool decisioning guidance, authored by the tools component and surfaced
# as a dedicated block in the system prompt next to the file-tool scopes. These
# are nudges that apply across tools (not to any single one), so they live here
# rather than in individual tool descriptions or general runtime prose.
TOOL_USE_GUIDANCE = (
    "TOOL USE:\n"
    "- Think before calling. Reach for a tool only when it's definitely necessary.\n"
    "- Route, don't search, for the knowledge base. Resolve the path from the "
    "inlined index and read_file it directly; use grep_files or list_dir only "
    "when the index has no pointer — not to rediscover what it already maps.\n"
    "- Batch independent work. When you need several independent things at once "
    "(multiple files, or a read plus a search), issue them as parallel tool calls "
    "in a single round; the runtime runs them concurrently. Sequence calls only "
    "when a later one genuinely depends on an earlier result."
)

# Hot-path tools kept as real native tool-calls (full schema + provider-side
# arg validation). Every other tool is reached through the ``call_tool``
# dispatcher and described only in the lightweight catalog, so the always-loaded
# tool footprint stays small.
#
# Only direct reads (read_file/list_dir) are hot-path. Search (grep_files,
# find_files) is intentionally left in the catalog: routing it through
# call_tool gives it the same friction as every other tool, so the model
# reaches for the inlined index first instead of reflexively searching.
_DEFAULT_NATIVE = ("read_file", "list_dir")

_CALL_TOOL = "call_tool"
_TOOL_MANUAL = "tool_manual"

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

# Accepted Python types per JSON-schema type, for harness-side arg validation.
_JSON_TYPE_PY: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _json_type_ok(value: Any, json_type: str) -> bool:
    """Whether ``value`` matches a JSON-schema scalar type (bool is not int/number)."""
    py = _JSON_TYPE_PY.get(json_type)
    if py is None:
        return True  # unconstrained type -> accept
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass; don't accept it as a number
    return isinstance(value, py)


def _annotation_name(annotation: Any, default: str) -> str:
    """Return a stable annotation name for real and postponed annotations."""
    if annotation is inspect.Parameter.empty:
        return default
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", default)


class ToolRegistry:
    """Registry of executable tools loaded from ``paw/tools/``."""

    def __init__(
        self,
        tools_dir: Path | None = None,
        can_delegate: list[str] | None = None,
        native_names: Iterable[str] | None = None,
    ):
        self._tools_dir = tools_dir or TOOLS_DIR
        self._tools: dict[str, Callable] = {}
        self._compactors: dict[str, Callable] = {}
        self._descriptions: list[dict[str, str]] = []
        self._load()
        if can_delegate is not None:
            self._patch_delegation_description(can_delegate)
        # Only natives that actually loaded count; the rest go to the catalog.
        wanted = _DEFAULT_NATIVE if native_names is None else native_names
        self._native = [n for n in wanted if n in self._tools]

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

    def _param_schema(self, name: str) -> dict[str, Any]:
        """JSON-schema object for a tool's public parameters (type + required)."""
        sig = inspect.signature(self._tools[name])
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname.startswith("_"):
                continue
            py_type = _annotation_name(param.annotation, "string")
            properties[pname] = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": properties, "required": required}

    def _public_params(self, name: str) -> list[str]:
        return [
            p for p in inspect.signature(self._tools[name]).parameters
            if not p.startswith("_")
        ]

    def validate_args(self, name: str, args: dict[str, Any]) -> str | None:
        """Harness-side validation of a catalog call's args against the tool schema.

        The provider only validates the ``call_tool`` envelope, so we check the
        real tool's args ourselves: known tool, no missing required, no unknown
        params, basic types. Returns an ``Error: ...`` string for the model to
        self-correct on, or None if valid. Mirrors the provider-side check that
        native tools get for free.
        """
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Use a tool from the catalog."
        if not isinstance(args, dict):
            return f"Error: args for '{name}' must be an object."
        schema = self._param_schema(name)
        props, required = schema["properties"], schema["required"]

        missing = [r for r in required if r not in args]
        if missing:
            return f"Error: {name} is missing required argument(s): {', '.join(missing)}."
        unknown = [a for a in args if a not in props]
        if unknown:
            valid = ", ".join(props) or "(none)"
            return f"Error: {name} got unknown argument(s): {', '.join(unknown)}. Valid: {valid}."
        bad = [
            f"'{p}' must be {props[p]['type']}"
            for p, v in args.items()
            if not _json_type_ok(v, props[p]["type"])
        ]
        if bad:
            return f"Error: {name} argument type mismatch: {'; '.join(bad)}."
        return None

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Native tool defs for the hot-path tools, plus the two meta-tools.

        Hot-path file tools keep full schemas (provider validates their args).
        Every other tool is reached via ``call_tool`` and listed in the catalog
        (``tool_summary``), keeping the always-loaded footprint small.
        """
        definitions: list[ToolDefinition] = []
        for desc in self._descriptions:
            if desc["name"] not in self._native:
                continue
            definitions.append(
                ToolDefinition(
                    name=desc["name"],
                    description=desc["doc"].split("\n", 1)[0].strip(),  # one-liner
                    parameters=self._param_schema(desc["name"]),
                )
            )
        definitions.append(
            ToolDefinition(
                name=_CALL_TOOL,
                description=(
                    "Invoke a catalog tool. `name` is a tool from the catalog; "
                    "`args` is its argument object."
                ),
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
                    "required": ["name", "args"],
                },
            )
        )
        definitions.append(
            ToolDefinition(
                name=_TOOL_MANUAL,
                description="Return full usage docs for a catalog tool by name.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            )
        )
        return definitions

    def catalog(self) -> str:
        """One ``name(params): purpose`` line per non-native tool."""
        lines = []
        for desc in self._descriptions:
            name = desc["name"]
            if name in self._native:
                continue
            params = ", ".join(self._public_params(name))
            purpose = desc["doc"].split("\n", 1)[0].strip()
            lines.append(f"{name}({params}): {purpose}")
        return "\n".join(lines)

    def manual(self, name: str) -> str:
        """Full usage docs (signature + docstring) for one tool."""
        for desc in self._descriptions:
            if desc["name"] == name:
                return f"{name}{desc['signature']}\n\n{desc['doc']}"
        return f"Error: unknown tool '{name}'."

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a tool by name with the given parameters.

        Handles the two meta-tools: ``tool_manual`` returns a tool's docs, and
        ``call_tool`` dispatches to the named catalog tool.
        """
        if name == _TOOL_MANUAL:
            return self.manual(str(params.get("name", "")))
        if name == _CALL_TOOL:
            inner = str(params.get("name", ""))
            raw = params.get("args")
            args = raw if isinstance(raw, dict) else {}
            # execute is a trusting dispatcher: it never validates real-name calls
            # either. Arg validation lives solely in validate_args, which callers
            # (the agent loop) invoke as the single gate before dispatching here.
            return await self.execute(inner, args, context)
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
        """Tool catalog + cross-tool guidance, for the system prompt.

        The hot-path file tools are callable directly; every catalog tool below
        is invoked via ``call_tool(name, args)``, with ``tool_manual(name)`` for
        full usage.
        """
        return (
            "TOOLS:\n"
            "File read/search tools are callable directly. Every tool below is "
            "invoked with call_tool(name, args); call tool_manual(name) first if "
            "you are unsure how to use one.\n\n"
            "CATALOG (name(args): purpose):\n"
            f"{self.catalog()}\n\n"
            f"{TOOL_USE_GUIDANCE}"
        )

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
