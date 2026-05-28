"""Tool registry: auto-load Python tool plugins for native LiteLLM calls.

Each ``.py`` file under ``errand/tools/`` is auto-loaded as a tool
plugin, except for files that are either:

- the registry module itself (``registry.py``);
- private implementation modules (names starting with ``_``);
- ``__init__.py``.

Public module-level functions become tools. Schemas are generated from
the function signature; descriptions come from the function docstring.

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

from errand.contracts.types import ToolDefinition

TOOLS_DIR = Path(__file__).resolve().parent

_SKIP_MODULES = {"__init__", "registry"}

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

    def __init__(self, tools_dir: Path | None = None):
        self._tools_dir = tools_dir or TOOLS_DIR
        self._tools: dict[str, Callable] = {}
        self._descriptions: list[dict[str, str]] = []
        self._load()

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
