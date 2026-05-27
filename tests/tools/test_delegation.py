"""Tests for internal Swarm delegation tool."""

from dataclasses import dataclass, field

from swarm_agent.config import AgentSpec
from swarm_agent.tools import delegation


@dataclass
class FakeConfig:
    agents: dict[str, AgentSpec] = field(default_factory=dict)

    def get_agent(self, agent_id: str | None = None) -> AgentSpec:
        return self.agents[agent_id or "generalist"]


async def test_invoke_agent_rejects_disallowed_delegate(monkeypatch):
    config = FakeConfig(
        agents={
            "generalist": AgentSpec(id="generalist", can_delegate=[]),
            "applied-scientist": AgentSpec(id="applied-scientist"),
        }
    )
    monkeypatch.setattr(delegation, "load_swarm_config", lambda: config)

    result = await delegation.invoke_agent(
        "applied-scientist",
        "analyze this",
        _context={"agent_id": "generalist", "session_id": "s1"},
    )

    assert "is not allowed" in result


async def test_invoke_agent_blocks_recursive_depth():
    result = await delegation.invoke_agent(
        "applied-scientist",
        "analyze this",
        _context={
            "agent_id": "generalist",
            "session_id": "s1",
            "delegation_depth": delegation.MAX_DELEGATION_DEPTH,
        },
    )

    assert "maximum delegation depth" in result
