"""Harness-side retries (validation catches, tool errors) are recorded on the
run's usage tracker and surfaced in the footer."""

from dataclasses import dataclass
from types import SimpleNamespace

from paw.agent_loop.loop import AgentLoop
from paw.contracts.types import BrainDecision, ToolCall, ToolDefinition
from paw.runtime.run_context import RunContext
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
    def build_prefix_messages(self, history):
        return [{"role": "system", "content": "s"}, *history]

    def build_context_messages(self, *, context_summary=None, session_note=None, instruction=None):
        return [{"role": "user", "content": f"CTX:{instruction}"}]


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
    loop.config = SimpleNamespace(models={})
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


class _TextBrain:
    """Answers with text immediately; records the parent's own usage."""

    async def decide(self, messages, tool_definitions=None, usage_tracker=None, **kw):
        usage = {"input_tokens": 10, "output_tokens": 5, "model": "fake"}
        if usage_tracker is not None:
            usage_tracker.record(usage, agent_id="test")
        return {"decision": BrainDecision(text_response="done"), "usage": usage}


async def test_session_total_includes_subagent_tokens(monkeypatch, tmp_path):
    loop = _make_loop(monkeypatch, tmp_path)
    loop.brain = _TextBrain()

    # A delegated sub-agent recorded usage on the shared per-exchange tracker.
    rc = RunContext.root()
    rc.usage.record({"input_tokens": 100, "output_tokens": 40}, agent_id="notes-organizer")

    result = await loop.process_input("hi", metadata={"_run_context": rc})

    summary = loop.memory.data["token_summary"]
    # Parent's own call (10/5) plus the delegated sub-agent (100/40).
    assert summary["input_tokens"] == 110
    assert summary["output_tokens"] == 45
    assert "*Session total: 110 in, 45 out*" in result
