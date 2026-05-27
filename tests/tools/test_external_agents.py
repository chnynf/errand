"""Tests for external CLI delegation tool."""

import sys
from dataclasses import dataclass, field
from textwrap import dedent

from swarm_agent.config import AgentSpec, ExternalAgentSpec
from swarm_agent.tools import external_agents


@dataclass
class FakeConfig:
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    external_agents: dict[str, ExternalAgentSpec] = field(default_factory=dict)

    def get_agent(self, agent_id: str | None = None) -> AgentSpec:
        return self.agents[agent_id or "generalist"]


class FakeReplyTarget:
    def __init__(self, approved: bool):
        self.approved = approved
        self.requests = []

    async def request_approval(self, **kwargs):
        self.requests.append(kwargs)
        return self.approved


async def test_invoke_external_agent_argument_mode(monkeypatch):
    config = FakeConfig(
        agents={
            "generalist": AgentSpec(
                id="generalist",
                can_delegate=["echoer"],
            )
        },
        external_agents={
            "echoer": ExternalAgentSpec(
                id="echoer",
                command=[
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1])",
                ],
                prompt_mode="argument",
                timeout_seconds=5,
                max_output_chars=2000,
            )
        },
    )
    monkeypatch.setattr(external_agents, "load_swarm_config", lambda: config)

    result = await external_agents.invoke_external_agent(
        "echoer",
        "say hello",
        _context={"agent_id": "generalist"},
    )

    assert "External agent: echoer" in result
    assert "Exit code: 0" in result
    assert "say hello" in result


async def test_invoke_external_agent_does_not_request_launch_approval(monkeypatch):
    config = FakeConfig(
        agents={"generalist": AgentSpec(id="generalist", can_delegate=["echoer"])},
        external_agents={
            "echoer": ExternalAgentSpec(
                id="echoer",
                command=[sys.executable, "-c", "print('should not run')"],
                prompt_mode="stdin",
            )
        },
    )
    monkeypatch.setattr(external_agents, "load_swarm_config", lambda: config)
    reply_to = FakeReplyTarget(approved=False)

    result = await external_agents.invoke_external_agent(
        "echoer",
        "say hello",
        _context={"agent_id": "generalist", "reply_to": reply_to},
    )

    assert not reply_to.requests
    assert "should not run" in result


async def test_invoke_external_agent_stream_json_permission(monkeypatch):
    script = dedent(
        r'''
        import json
        import sys

        _ = sys.stdin.readline()
        print(json.dumps({
            "type": "control_request",
            "request_id": "req-1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "echo ok"}
            }
        }), flush=True)
        response = json.loads(sys.stdin.readline())
        assert response["response"]["request_id"] == "req-1"
        behavior = response["response"]["response"]["behavior"]
        print(json.dumps({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": f"permission={behavior}"}]
            }
        }), flush=True)
        print(json.dumps({"type": "result", "is_error": False}), flush=True)
        '''
    )
    config = FakeConfig(
        agents={"generalist": AgentSpec(id="generalist", can_delegate=["streamer"])},
        external_agents={
            "streamer": ExternalAgentSpec(
                id="streamer",
                command=[sys.executable, "-c", script],
                prompt_mode="stream_json",
                timeout_seconds=5,
            )
        },
    )
    monkeypatch.setattr(external_agents, "load_swarm_config", lambda: config)
    reply_to = FakeReplyTarget(approved=True)

    result = await external_agents.invoke_external_agent(
        "streamer",
        "say hello",
        _context={"agent_id": "generalist", "reply_to": reply_to},
    )

    assert len(reply_to.requests) == 1
    assert "permission=allow" in result


async def test_invoke_external_agent_rejects_disallowed_delegate(monkeypatch):
    config = FakeConfig(
        agents={"applied-scientist": AgentSpec(id="applied-scientist")},
        external_agents={
            "echoer": ExternalAgentSpec(
                id="echoer",
                command=["python", "-c", "print('ok')"],
            )
        },
    )
    monkeypatch.setattr(external_agents, "load_swarm_config", lambda: config)

    result = await external_agents.invoke_external_agent(
        "echoer",
        "say hello",
        _context={"agent_id": "applied-scientist"},
    )

    assert "is not allowed" in result
