"""AgentLoop keeps the current exchange at full fidelity (no intra-exchange
compaction) and only forces a final response at the round cap or the
context-window overflow guard."""

from dataclasses import dataclass
from types import SimpleNamespace

from paw.agent_loop.loop import AgentLoop, CONTEXT_OVERFLOW_TOKENS
from paw.contracts.types import BrainDecision, ToolCall, ToolDefinition
from paw.sessions.memory import Memory


@dataclass
class _Spec:
    id: str = "test"
    max_tool_rounds: int = 8
    can_delegate: tuple = ()


class _FakeRegistry:
    """Returns one read tool that yields a configurable-size result."""

    def __init__(self, payload: str):
        self._payload = payload

    def get_tool_definitions(self):
        return [ToolDefinition(name="read_file", description="d", parameters={})]

    async def execute(self, name, params, context=None):
        return self._payload

    def compact_interaction(self, call, result):
        return f"tool call: {result.name}()\ntool result: ref [{len(result.content)} chars total]"


class _FakePromptAssembler:
    def build_prompt(self, history_messages, current_exchange, *, context_summary=None, instruction=None, session_note=None):
        return [
            {"role": "system", "content": "sys"},
            *history_messages,
            *current_exchange,
            {"role": "user", "content": f"CTX:{context_summary}"},
        ]


class _FakeBrain:
    """Scripts the loop. Records every decide call.

    ``always_tool`` keeps requesting a read on each round until the loop strips
    the tool list (the forced-response path), at which point it replies with
    text. Otherwise it reads once, then replies.
    """

    def __init__(self, always_tool: bool = False):
        self.always_tool = always_tool
        self.calls = 0

    async def decide(self, messages, tool_definitions=None, session_id=None,
                     log_extra=None, usage_tracker=None, **kw):
        usage = {"input_tokens": 1, "output_tokens": 1, "model": "fake"}
        if usage_tracker is not None:
            usage_tracker.record(usage, agent_id="test")
        self.calls += 1
        # The loop passes an empty tool list once it forces a final response.
        if not tool_definitions:
            return {"decision": BrainDecision(text_response="final answer"), "usage": usage}
        if self.always_tool or self.calls == 1:
            return {"decision": BrainDecision(
                tool_calls=[ToolCall(id=f"c{self.calls}", name="read_file", params={"path": "big.md"})]),
                "usage": usage}
        return {"decision": BrainDecision(text_response="final answer"), "usage": usage}


def _make_loop(monkeypatch, tmp_path, payload, *, always_tool=False):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    loop = object.__new__(AgentLoop)
    loop.debug = False
    loop.config = SimpleNamespace(models={})
    loop.agent_id = "test"
    loop.agent_spec = _Spec()
    loop.delegation_depth = 0
    loop.memory = Memory("compaction-session", agent_id="test")
    loop.tool_registry = _FakeRegistry(payload)
    loop.prompt_assembler = _FakePromptAssembler()
    loop.brain = _FakeBrain(always_tool=always_tool)
    return loop


async def test_oversized_read_is_not_compacted(monkeypatch, tmp_path):
    # A big read no longer spawns a separate compaction call: the loop makes
    # exactly two model calls (read-decision + answer), not three.
    payload = "x" * (CONTEXT_OVERFLOW_TOKENS * 4 // 4)  # well under the overflow guard
    loop = _make_loop(monkeypatch, tmp_path, payload)

    result = await loop.process_input("Do a big task", metadata={"suppress_usage_footer": True})

    assert "final answer" in result
    assert loop.brain.calls == 2                        # no extra summarization call
    assert loop.memory.data["context_summary"] is None  # nothing clobbers the rolling summary


async def test_overflow_guard_forces_final_response(monkeypatch, tmp_path):
    # When live context crosses the overflow threshold, the loop stops taking
    # tool calls and forces a reply -- long before the 8-round cap.
    monkeypatch.setattr("paw.agent_loop.loop.CONTEXT_OVERFLOW_TOKENS", 1_000)
    payload = "x" * 8_000  # ~2000 tokens: one read trips the guard
    loop = _make_loop(monkeypatch, tmp_path, payload, always_tool=True)

    result = await loop.process_input("Runaway", metadata={"suppress_usage_footer": True})

    assert "final answer" in result
    assert loop.brain.calls == 2  # one tool round, then forced answer (not 8+ rounds)


async def test_round_cap_forces_final_response(monkeypatch, tmp_path):
    # With overflow out of the way, the 8-round cap is what forces the reply.
    monkeypatch.setattr("paw.agent_loop.loop.CONTEXT_OVERFLOW_TOKENS", 10_000_000)
    loop = _make_loop(monkeypatch, tmp_path, "small result", always_tool=True)

    result = await loop.process_input("Loop forever", metadata={"suppress_usage_footer": True})

    assert "final answer" in result
    assert loop.brain.calls == 9  # 8 tool rounds + 1 forced answer


async def test_small_turn_answers_directly(monkeypatch, tmp_path):
    loop = _make_loop(monkeypatch, tmp_path, "small result")

    result = await loop.process_input("Do a small task", metadata={"suppress_usage_footer": True})

    assert "final answer" in result
    assert loop.brain.calls == 2
