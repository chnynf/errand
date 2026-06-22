"""In-loop compaction checkpoint behavior for AgentLoop.process_input."""

from dataclasses import dataclass

from paw.agent_loop.loop import (
    AgentLoop,
    COMPACTION_INSTRUCTION,
    COMPACTION_TRIGGER_TOKENS,
)
from paw.contracts.types import BrainDecision, ToolCall, ToolDefinition
from paw.sessions.memory import Memory


@dataclass
class _Spec:
    id: str = "test"
    max_tool_rounds: int = 8
    can_delegate: tuple = ()


class _FakeRegistry:
    """Returns one read tool that yields an oversized result."""

    def __init__(self, payload: str):
        self._payload = payload

    def get_tool_definitions(self):
        return [ToolDefinition(name="read_file", description="d", parameters={})]

    async def execute(self, name, params, context=None):
        return self._payload

    def compact_result(self, call, result, preview_limit=500):
        return {
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "params": {},
            "content_chars": len(result.content),
            "preview": "ref",
        }


class _FakePromptAssembler:
    def build_messages(self, history, *, context_summary=None, session_note=None, instruction=None):
        return [{"role": "system", "content": "sys"},
                {"role": "user", "content": f"CTX:{context_summary}"}, *history]


class _FakeBrain:
    """Scripts the loop: read -> (compaction) -> final text."""

    def __init__(self):
        self.compaction_calls = 0
        self.main_step = 0

    async def decide(self, messages, tool_definitions=None, session_id=None,
                     log_extra=None, usage_tracker=None, **kw):
        usage = {"input_tokens": 1, "output_tokens": 1, "model": "fake"}
        if usage_tracker is not None:
            usage_tracker.record(usage, agent_id="test")
        if log_extra == "compaction":
            self.compaction_calls += 1
            return {"decision": BrainDecision(text_response="SUMMARY: goal + key findings."),
                    "usage": usage}
        self.main_step += 1
        if self.main_step == 1:
            return {"decision": BrainDecision(
                tool_calls=[ToolCall(id="c1", name="read_file", params={"path": "big.md"})]),
                "usage": usage}
        return {"decision": BrainDecision(text_response="final answer"), "usage": usage}


def _make_loop(monkeypatch, tmp_path, payload):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    loop = object.__new__(AgentLoop)
    loop.debug = False
    loop.agent_id = "test"
    loop.agent_spec = _Spec()
    loop.delegation_depth = 0
    loop.memory = Memory("compaction-session", agent_id="test")
    loop.tool_registry = _FakeRegistry(payload)
    loop.prompt_assembler = _FakePromptAssembler()
    loop.brain = _FakeBrain()
    return loop


async def test_oversized_turn_triggers_compaction(monkeypatch, tmp_path):
    # Payload large enough that the live messages exceed the trigger after one read.
    payload = "x" * (COMPACTION_TRIGGER_TOKENS * 4 + 10_000)
    loop = _make_loop(monkeypatch, tmp_path, payload)

    result = await loop.process_input("Do a big task", metadata={"suppress_usage_footer": True})

    assert loop.brain.compaction_calls == 1            # condensed exactly once
    assert "final answer" in result                    # turn still produced one reply
    assert loop.memory.data["context_summary"] == "SUMMARY: goal + key findings."


async def test_small_turn_does_not_compact(monkeypatch, tmp_path):
    loop = _make_loop(monkeypatch, tmp_path, "small result")

    result = await loop.process_input("Do a small task", metadata={"suppress_usage_footer": True})

    assert loop.brain.compaction_calls == 0
    assert "final answer" in result


def test_compaction_instruction_is_loop_owned():
    # The orchestration instruction lives with the loop, not the brain.
    assert "summary" in COMPACTION_INSTRUCTION.lower()
