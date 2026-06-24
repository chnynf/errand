"""Tests for `ToolRegistry`.

Uses a temporary tools dir so tests are independent of which Python
tools currently ship in ``paw/tools/``.
"""

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from paw.contracts.types import ToolCall, ToolResult
from paw.sessions.memory import Memory
from paw.tools.registry import ToolRegistry


@pytest.fixture
def tools_dir(tmp_path: Path):
    d = tmp_path / "tools"
    d.mkdir()
    (d / "__init__.py").write_text("", encoding="utf-8")
    (d / "calc.py").write_text(
        dedent(
            '''
            def add(a: int, b: int = 0) -> int:
                """Add two numbers.

                Use for simple arithmetic.

                Args:
                    a: first
                    b: second
                """
                return a + b
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (d / "greet.py").write_text(
        dedent(
            '''
            async def greet(name: str) -> str:
                """Greet a person by name."""
                return f"hi {name}"
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (d / "contextual.py").write_text(
        dedent(
            '''
            def whoami(_context=None) -> str:
                """Return the current agent id from hidden runtime context."""
                return (_context or {}).get("agent_id", "unknown")
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (d / "_private_helper.py").write_text(
        dedent(
            '''
            def should_not_be_a_tool(x: int) -> int:
                """Private helper - registry must skip this module."""
                return x
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return d


def test_loads_only_public_functions_with_matching_module(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    assert sorted(registry.names) == ["add", "greet", "whoami"]


def test_skips_private_modules(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    assert "should_not_be_a_tool" not in registry.names


def test_get_tool_definitions_schema(tools_dir: Path):
    # Native tools get a full def (schema + one-line description).
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["add", "whoami"])
    defs = {d.name: d for d in registry.get_tool_definitions()}

    add_def = defs["add"]
    assert add_def.parameters["properties"]["a"]["type"] == "integer"
    assert add_def.parameters["properties"]["b"]["type"] == "integer"
    assert add_def.parameters["required"] == ["a"]
    assert add_def.description == "Add two numbers."  # one-liner, not full doc

    whoami_def = defs["whoami"]
    assert "_context" not in whoami_def.parameters["properties"]
    assert whoami_def.parameters["required"] == []


def test_hybrid_splits_native_defs_from_catalog(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["add"])
    names = {d.name for d in registry.get_tool_definitions()}
    # native tool + the two meta-tools are real defs; catalog tools are not
    assert "add" in names
    assert {"call_tool", "tool_manual"} <= names
    assert "greet" not in names and "whoami" not in names

    catalog = registry.catalog()
    assert "greet(name): Greet a person by name." in catalog
    assert "add(" not in catalog  # native tools stay out of the catalog


def test_call_tool_dispatches_to_inner_tool(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["add"])
    out = asyncio.run(registry.execute("call_tool", {"name": "greet", "args": {"name": "Bo"}}))
    assert out == "hi Bo"


def test_tool_manual_returns_full_docs(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["add"])
    manual = asyncio.run(registry.execute("tool_manual", {"name": "add"}))
    assert manual.startswith("add(")          # signature included
    assert "Add two numbers." in manual
    assert "simple arithmetic" in manual       # full docstring, not just first line


def test_tool_summary_lists_catalog_and_nudges(tools_dir: Path):
    summary = ToolRegistry(tools_dir=tools_dir, native_names=["add"]).tool_summary()
    assert "CATALOG" in summary
    assert "greet(name): Greet a person by name." in summary
    assert "Think before calling" in summary


def test_execute_sync_and_async_tools(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    assert asyncio.run(registry.execute("add", {"a": 2, "b": 3})) == 5
    assert asyncio.run(registry.execute("greet", {"name": "Alice"})) == "hi Alice"
    assert (
        asyncio.run(registry.execute("whoami", {}, context={"agent_id": "generalist"}))
        == "generalist"
    )


def test_execute_unknown_tool_raises(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    with pytest.raises(ValueError):
        asyncio.run(registry.execute("nope", {}))


# --- compact_result: the contract surface between tools and memory ---------

def _record_renders(record: dict) -> str:
    """A record produced by the registry must be renderable by Memory."""
    return Memory._render_tool_record(record)


def test_compact_result_default_compactor_truncates_long_content(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    call = ToolCall(id="tc-1", name="add", params={"a": 1, "b": 2})
    result = ToolResult(tool_call_id="tc-1", name="add", content="x" * 1200)

    record = registry.compact_result(call, result, preview_limit=500)

    assert record["tool_call_id"] == "tc-1"
    assert record["name"] == "add"
    assert record["params"] == {"a": 1, "b": 2}
    assert record["content_chars"] == 1200
    assert record["preview"].endswith("... [truncated]")
    # The record must also be renderable by Memory's history replay.
    assert "[truncated]" in _record_renders(record)


def test_compact_result_uses_registered_compactor_for_file_tools():
    # Uses the real paw/tools/ directory so the files.py COMPACTORS register.
    registry = ToolRegistry()
    call = ToolCall(
        id="tc-1",
        name="write_file",
        params={"path": "notes/big.md", "content": "y" * 5000},
    )
    result = ToolResult(
        tool_call_id="tc-1",
        name="write_file",
        content="Wrote 5000 bytes to /kb/notes/big.md.",
    )

    record = registry.compact_result(call, result)

    # Per-tool compactor strips the large payload from persisted params.
    assert "content" not in record["params"]
    assert record["params"]["path"] == "notes/big.md"
    assert record["result_ref"] == {"type": "file", "path": "notes/big.md"}
    assert record["content_chars"] == len(result.content)
    # Memory must be able to render this record without knowing the tool name.
    rendered = _record_renders(record)
    assert "Wrote 5000 bytes" in rendered


def test_compact_result_read_file_error_surfaces_in_record():
    registry = ToolRegistry()
    call = ToolCall(id="tc-2", name="read_file", params={"path": "missing.md"})
    result = ToolResult(
        tool_call_id="tc-2", name="read_file", content="Error: Not a file: /kb/missing.md"
    )

    record = registry.compact_result(call, result)

    assert record["result_ref"]["type"] == "file"
    assert record["error"].startswith("Error:")
    # Errors take priority in the renderer.
    assert _record_renders(record).startswith("Error:")


def test_compact_result_list_dir_counts_entries():
    registry = ToolRegistry()
    call = ToolCall(id="tc-3", name="list_dir", params={"path": ""})
    result = ToolResult(
        tool_call_id="tc-3", name="list_dir", content="a.md\nb.md\n\nc.md/\n"
    )

    record = registry.compact_result(call, result)

    assert record["entry_count"] == 3
    assert record["result_ref"] == {"type": "directory", "path": ""}
    # When result_ref is present the renderer prefers it (matches old behavior).
    assert "directory ref" in _record_renders(record)


def test_compact_result_tool_without_compactor_falls_back_to_default():
    # `calculate` ships without a COMPACTORS entry — the generic default applies.
    registry = ToolRegistry()
    call = ToolCall(id="tc-4", name="calculate", params={"expression": "2 + 2"})
    result = ToolResult(tool_call_id="tc-4", name="calculate", content="5")

    record = registry.compact_result(call, result)

    assert record["preview"] == "5"
    assert "result_ref" not in record
    assert "error" not in record


def test_tool_summary_carries_cross_tool_nudges():
    summary = ToolRegistry().tool_summary()
    assert "TOOL USE:" in summary
    # the three decisioning nudges
    assert "Think before calling" in summary
    assert "Route, don't search" in summary
    assert "Batch independent work" in summary


# validate_args is the single validation gate; the agent loop invokes it before
# dispatching a catalog call. execute itself is a trusting dispatcher.
def test_validate_args_missing_required(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    out = registry.validate_args("add", {})
    assert out.startswith("Error:") and "missing required" in out and "a" in out


def test_validate_args_unknown_arg(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    out = registry.validate_args("add", {"a": 1, "z": 9})
    assert out.startswith("Error:") and "unknown argument" in out and "z" in out


def test_validate_args_type(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    out = registry.validate_args("add", {"a": "two"})
    assert out.startswith("Error:") and "type mismatch" in out and "integer" in out


def test_validate_args_unknown_tool(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    out = registry.validate_args("nope", {})
    assert out.startswith("Error:") and "unknown tool" in out


def test_validate_args_valid_returns_none(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    assert registry.validate_args("add", {"a": 2, "b": 3}) is None


def test_call_tool_dispatches_without_validating(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    assert asyncio.run(registry.execute("call_tool", {"name": "add", "args": {"a": 2, "b": 3}})) == 5
