"""Harness-side retries (validation catches, tool errors) are recorded on the
run's usage tracker and surfaced in the footer."""

from dataclasses import dataclass

from paw.agent_loop.loop import AgentLoop
from paw.contracts.types import BrainDecision, ToolCall, ToolDefinition
from paw.sessions.memory import Memory


@dataclass
class _Spec:
    id: str = "test"
    max_tool_rounds: int = 8
    can_delegate: tuple = ()


class _Registry:
    """A registry whose validation always rejects the catalog call."""

    def get_tool_definitions(self):
        return [ToolDefinition(
            name="call_tool", description="d",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
                        "required": ["name", "args"]})]

    def validate_args(self, name, args):
        return f"Error: {name} is missing required argument(s): subject."

    async def execute(self, name, params, context=None):  # pragma: no cover - not reached
        return "ok"

    def compact_result(self, call, result, preview_limit=500):
        return {"tool_call_id": result.tool_call_id, "name": result.name,
                "params": {}, "content_chars": len(result.content), "preview": result.content[:60]}


class _PromptAssembler:
    def build_messages(self, history, *, context_summary=None, session_note=None, instruction=None):
        return [{"role": "system", "content": "s"}, *history]


class _Brain:
    def __init__(self):
        self.step = 0

    async def decide(self, messages, tool_definitions=None, usage_tracker=None, **kw):
        usage = {"input_tokens": 1, "output_tokens": 1, "model": "fake"}
        if usage_tracker is not None:
            usage_tracker.record(usage, agent_id="test")
        self.step += 1
        if self.step == 1:
            return {"decision": BrainDecision(tool_calls=[
                ToolCall(id="c1", name="call_tool",
                         params={"name": "send_email", "args": {"to": "a@b.com"}})]),
                    "usage": usage}
        return {"decision": BrainDecision(text_response="done"), "usage": usage}


def _make_loop(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    loop = object.__new__(AgentLoop)
    loop.debug = False
    loop.agent_id = "test"
    loop.agent_spec = _Spec()
    loop.delegation_depth = 0
    loop.memory = Memory("validation-session", agent_id="test")
    loop.tool_registry = _Registry()
    loop.prompt_assembler = _PromptAssembler()
    loop.brain = _Brain()
    return loop


class _RaisingRegistry(_Registry):
    """Validation passes, but the tool raises during execution."""

    def validate_args(self, name, args):
        return None

    async def execute(self, name, params, context=None):
        raise RuntimeError("boom")


async def test_validation_catch_recorded_in_usage_footer(monkeypatch, tmp_path):
    loop = _make_loop(monkeypatch, tmp_path)

    result = await loop.process_input("send an email")  # footer not suppressed

    assert "done" in result                       # turn still completes
    assert "*Validation catches: 1*" in result    # the catch was tracked + surfaced


async def test_tool_error_recorded_in_usage_footer(monkeypatch, tmp_path):
    loop = _make_loop(monkeypatch, tmp_path)
    loop.tool_registry = _RaisingRegistry()

    result = await loop.process_input("send an email")  # footer not suppressed

    assert "done" in result                       # turn still completes
    assert "*Tool errors: 1*" in result           # the error was tracked + surfaced
