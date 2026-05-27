"""Tests for `ToolRegistry`.

Uses a temporary tools dir so tests are independent of which Python
tools currently ship in ``swarm_agent/tools/``.
"""

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from swarm_agent.tools.registry import ToolRegistry


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
    registry = ToolRegistry(tools_dir=tools_dir)
    defs = {d.name: d for d in registry.get_tool_definitions()}

    add_def = defs["add"]
    assert add_def.parameters["properties"]["a"]["type"] == "integer"
    assert add_def.parameters["properties"]["b"]["type"] == "integer"
    assert add_def.parameters["required"] == ["a"]
    assert add_def.description.startswith("Add two numbers.")

    whoami_def = defs["whoami"]
    assert "_context" not in whoami_def.parameters["properties"]
    assert whoami_def.parameters["required"] == []


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
