"""Tests for `ToolRegistry`.

Uses a temporary tools dir so tests are independent of which Python
tools currently ship in ``paw/tools/``.
"""

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from paw.wire_types import ToolCall, ToolResult
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


# --- compact_interaction: the standard folded text for one call + result ----

def test_compact_interaction_standard_format(tools_dir: Path):
    # Uniform across tools: "tool call: name(args)\ntool result: body [N total]".
    registry = ToolRegistry(tools_dir=tools_dir)
    call = ToolCall(id="tc-1", name="read_file", params={"path": "INDEX.md"})
    result = ToolResult(tool_call_id="tc-1", name="read_file", content="# Agent KB\nsecret soul")

    text = registry.compact_interaction(call, result)

    assert text.startswith("tool call: read_file(path='INDEX.md')")
    # Under the 500-char cap, the result is kept whole with the total noted.
    assert "tool result: # Agent KB\nsecret soul [22 chars total]" in text


def test_compact_interaction_truncates_long_result_body(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    call = ToolCall(id="tc-1", name="add", params={"a": 1, "b": 2})
    result = ToolResult(tool_call_id="tc-1", name="add", content="x" * 1200)

    text = registry.compact_interaction(call, result)

    assert "x" * 500 in text          # first 500 kept
    assert "x" * 501 not in text      # cut at 500
    assert "… [1200 chars total]" in text


def test_compact_interaction_caps_long_arg_values(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    call = ToolCall(
        id="tc-1", name="write_file",
        params={"path": "notes/big.md", "content": "y" * 5000},
    )
    result = ToolResult(tool_call_id="tc-1", name="write_file", content="Wrote 5000 bytes.")

    call_line = registry.compact_interaction(call, result).splitlines()[0]

    assert "path='notes/big.md'" in call_line
    assert "y" * 5000 not in call_line   # huge arg capped, not dumped
    assert "…" in call_line
    assert len(call_line) < 300


def test_compact_interaction_error_flows_through_as_result(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    call = ToolCall(id="tc-2", name="read_file", params={"path": "missing.md"})
    result = ToolResult(
        tool_call_id="tc-2", name="read_file", content="Error: Not a file: /kb/missing.md"
    )

    assert "tool result: Error: Not a file: /kb/missing.md" in registry.compact_interaction(call, result)


def test_compact_interaction_handles_missing_call(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir)
    result = ToolResult(tool_call_id="tc-9", name="add", content="5")

    text = registry.compact_interaction(None, result)

    assert text.startswith("tool call: unknown()")
    assert "tool result: 5 [1 chars total]" in text


def test_tool_summary_carries_cross_tool_nudges():
    summary = ToolRegistry().tool_summary()
    assert "TOOL USE:" in summary
    # the three decisioning nudges
    assert "Think before calling" in summary
    assert "pass the index's `[kb-root]/...` path verbatim" in summary
    assert "Batch independent calls" in summary


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


def test_optional_list_param_typed_as_array(tools_dir: Path):
    (tools_dir / "multi.py").write_text(
        'def multi(items: list | None = None) -> str:\n    """Take a list."""\n    return str(items)\n',
        encoding="utf-8",
    )
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    assert registry.validate_args("multi", {"items": [1, 2]}) is None
    out = registry.validate_args("multi", {"items": "not-a-list"})
    assert out.startswith("Error:") and "array" in out


def test_call_tool_dispatches_without_validating(tools_dir: Path):
    registry = ToolRegistry(tools_dir=tools_dir, native_names=["greet"])
    assert asyncio.run(registry.execute("call_tool", {"name": "add", "args": {"a": 2, "b": 3}})) == 5
