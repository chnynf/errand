"""Tool registry: auto-load Python tool plugins for native LiteLLM calls.

Each ``.py`` file under ``paw/tools/`` is auto-loaded as a tool
plugin, except for files that are either:

- the registry module itself (``registry.py``);
- private implementation modules (names starting with ``_``);
- ``__init__.py``.

Public module-level functions become tools. Schemas are generated from
the function signature; descriptions come from the function docstring.
Tools are plain functions -- they know nothing about how their results
are stored in history.

Compaction is uniform and lives here, not in the tools: ``compact_interaction``
turns any tool call + result into one standard text (see
``_compact_tool_interaction``) that the agent loop uses when folding a
finished exchange into session memory. No per-tool customization.

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
# as a dedicated block in the system prompt. These are nudges that apply across
# tools (not to any single one), so they live here rather than in individual
# tool descriptions or general runtime prose.
TOOL_USE_GUIDANCE = (
    "TOOL USE:\n"
    "- Think before calling. Avoid unnecessary tool calls, such as file reads.\n"
    "- For file reads, pass the index's `[kb-root]/...` path verbatim to read_file; do not search or list files.\n"
    "- Batch work whenever possible; make independent calls in parallel. For example, combine edits into one run."
)

# Core tools kept as real native tool-calls (full schema + provider-side arg
# validation). Every other tool is reached through the ``call_tool`` dispatcher
# and described only in the lightweight catalog, so the always-loaded tool
# footprint stays small.
#
# The core set is the high-frequency work: reading (``read_file``) and
# delegation (``invoke_agent``, ``invoke_external_agent``). Everything else --
# search, writes, scheduling, email, etc. -- lives in the catalog and is reached
# via ``call_tool``, with ``tool_manual`` for full per-tool usage on demand.
_DEFAULT_NATIVE = ("invoke_agent", "invoke_external_agent", "read_file")

_CALL_TOOL = "call_tool"
_TOOL_MANUAL = "tool_manual"

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
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
    """Return a stable annotation name for real and postponed annotations.

    Postponed annotations arrive as strings ("list | None", "dict[str, Any]");
    strip the optional/generic parts down to the base type name.
    """
    if annotation is inspect.Parameter.empty:
        return default
    if isinstance(annotation, str):
        name = annotation
    else:
        name = getattr(annotation, "__name__", None) or str(annotation)
        if name in ("Union", "Optional"):
            # Evaluated unions (e.g. ``list | None``): their str() form
            # ("list | None") parses the same as a postponed annotation.
            name = str(annotation)
    return name.split("|", 1)[0].split("[", 1)[0].strip() or default


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
        self._descriptions: list[dict[str, str]] = []
        self._load()
        if can_delegate is not None:
            self._patch_delegation_description(can_delegate)
        # Only natives that actually loaded count; the rest go to the catalog.
        wanted = _DEFAULT_NATIVE if native_names is None else native_names
        self._native = [n for n in wanted if n in self._tools]

    def _load(self) -> None:
        self._tools.clear()
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

        The single contribution of the tools component to the system prompt: the
        core tools are exposed directly as functions (your tool list); every
        catalog tool below is invoked via ``call_tool(name, args)``, with
        ``tool_manual(name)`` for full usage on demand.
        """
        return (
            "TOOLS:\n"
            "The directly callable functions in your tool list are the core "
            "tools. Every other tool is in the catalog below and is invoked via "
            "call_tool(name, args). Call tool_manual(name) for a tool's full "
            "usage before using one you are unsure about.\n\n"
            "CATALOG (name(args): purpose):\n"
            f"{self.catalog()}\n\n"
            f"{TOOL_USE_GUIDANCE}"
        )

    def compact_interaction(self, call: ToolCall | None, result: ToolResult) -> str:
        """The standard compacted text for one tool call + result.

        Uniform across every tool -- no per-tool customization. The agent loop
        uses this when folding a finished exchange into history.
        """
        return _compact_tool_interaction(call, result)


# Uniform compaction limits: the tool-call arg values and the result body are
# each truncated so a large read/write can't bloat history.
_ARG_CHAR_CAP = 100
_RESULT_CHAR_CAP = 500


def _format_call(call: ToolCall | None) -> str:
    """One line: ``name(key=repr, ...)`` with each arg value capped."""
    if call is None:
        return "unknown()"
    parts = []
    for key, value in (call.params or {}).items():
        rendered = repr(value)
        if len(rendered) > _ARG_CHAR_CAP:
            rendered = rendered[:_ARG_CHAR_CAP] + "…"
        parts.append(f"{key}={rendered}")
    return f"{call.name}({', '.join(parts)})"


def _compact_tool_interaction(call: ToolCall | None, result: ToolResult) -> str:
    """Standard, tool-agnostic rendering of one tool call + its result:

        tool call: read_file(path='INDEX.md')
        tool result: <first 500 chars of the result>… [2300 chars total]

    The result is truncated to ``_RESULT_CHAR_CAP`` with the full size noted, so
    history keeps the gist without carrying the whole output.
    """
    total = len(result.content)
    body = result.content
    if total > _RESULT_CHAR_CAP:
        body = body[:_RESULT_CHAR_CAP] + "…"
    return (
        f"tool call: {_format_call(call)}\n"
        f"tool result: {body} [{total} chars total]"
    )
