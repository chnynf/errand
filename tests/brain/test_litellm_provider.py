"""Tests for `LiteLLMProvider`.

LiteLLM is mocked end-to-end so the suite is hermetic - no env vars,
no network. We assert (a) the call kwargs we hand LiteLLM are shaped
correctly (model, api_base, api_key, tools, messages) and (b) the
response is parsed into our internal `BrainDecision` / usage shape.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from errand.brain.providers.litellm import LiteLLMProvider
from errand.contracts.types import ToolCall, ToolDefinition, ToolResult


def _mk_response(*, content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a minimal LiteLLM-shaped response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _mk_tool_call(call_id, name, args):
    """Build a LiteLLM-shaped tool_call entry."""
    function = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, function=function)


@pytest.fixture
def provider():
    return LiteLLMProvider()


@pytest.fixture
def calculator_tool():
    return ToolDefinition(
        name="calculate",
        description="Evaluate an expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )


async def test_text_response_parses_context_summary(provider):
    response = _mk_response(
        content="The answer is 42.\n---\nContext: User asked for the answer."
    )
    with patch(
        "errand.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, usage = await provider.generate_with_tools(
            model="gemini/gemini-3-flash-preview",
            system_prompt="soul",
            prompt="What is 42?",
            tool_definitions=[],
        )

    mock.assert_awaited_once()
    kwargs = mock.await_args.kwargs
    assert kwargs["model"] == "gemini/gemini-3-flash-preview"
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "soul"}
    assert kwargs["messages"][1] == {"role": "user", "content": "What is 42?"}
    assert kwargs["tools"] == []

    assert decision.tool_calls == []
    assert decision.text_response == "The answer is 42."
    assert decision.context_summary == "User asked for the answer."
    assert usage == {"input_tokens": 10, "output_tokens": 5}


async def test_tool_call_parsing(provider, calculator_tool):
    response = _mk_response(
        tool_calls=[_mk_tool_call("tc-abc", "calculate", {"expression": "1+1"})]
    )
    with patch(
        "errand.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, usage = await provider.generate_with_tools(
            model="deepseek/deepseek-chat",
            system_prompt="soul",
            prompt="Compute 1+1",
            tool_definitions=[calculator_tool],
        )

    kwargs = mock.await_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "calculate"
    assert kwargs["tools"][0]["function"]["parameters"]["required"] == ["expression"]

    assert len(decision.tool_calls) == 1
    tc = decision.tool_calls[0]
    assert tc.id == "tc-abc"
    assert tc.name == "calculate"
    assert tc.params == {"expression": "1+1"}
    assert decision.text_response is None
    assert usage["input_tokens"] == 10


async def test_api_base_and_key_passthrough_for_openai_compatible(provider, calculator_tool):
    response = _mk_response(content="ok")
    with patch(
        "errand.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate_with_tools(
            model="openai/deepseek-ai/DeepSeek-V3.2",
            system_prompt="soul",
            prompt="hello",
            tool_definitions=[calculator_tool],
            api_base="https://api.siliconflow.com/v1",
            api_key="sk-test-siliconflow",
        )

    kwargs = mock.await_args.kwargs
    assert kwargs["api_base"] == "https://api.siliconflow.com/v1"
    assert kwargs["api_key"] == "sk-test-siliconflow"


async def test_continue_with_tool_results_builds_messages(provider, calculator_tool):
    response = _mk_response(content="Final answer: 2")
    tool_calls = [ToolCall(id="tc-1", name="calculate", params={"expression": "1+1"})]
    tool_results = [ToolResult(tool_call_id="tc-1", name="calculate", content="2")]

    with patch(
        "errand.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, _ = await provider.continue_with_tool_results(
            model="gemini/gemini-3-flash-preview",
            system_prompt="soul",
            prompt="Compute 1+1",
            tool_calls=tool_calls,
            tool_results=tool_results,
            tool_definitions=[calculator_tool],
        )

    messages = mock.await_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "calculate"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"expression": "1+1"}
    )
    assert messages[3] == {"role": "tool", "tool_call_id": "tc-1", "content": "2"}

    assert decision.text_response == "Final answer: 2"


async def test_generate_decision_json_mode_strips_fences(provider):
    response = _mk_response(content='```json\n{"actions": [], "external_response": "hi"}\n```')
    schema = {"type": "object"}
    with patch(
        "errand.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        raw, usage = await provider.generate_decision(
            model="deepseek/deepseek-reasoner",
            system_prompt="soul",
            prompt="say hi",
            response_schema=schema,
        )

    kwargs = mock.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "OUTPUT FORMAT" in kwargs["messages"][0]["content"]

    assert raw == {"actions": [], "external_response": "hi"}
    assert usage == {"input_tokens": 10, "output_tokens": 5}


def test_is_retryable_classifies_known_litellm_errors(provider):
    from litellm.exceptions import APIConnectionError, RateLimitError, Timeout

    rate_limit = RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-4"
    )
    timeout = Timeout(message="timed out", model="gpt-4", llm_provider="openai")
    connection = APIConnectionError(
        message="conn refused", llm_provider="openai", model="gpt-4"
    )

    assert provider.is_retryable(rate_limit) is True
    assert provider.is_retryable(timeout) is True
    assert provider.is_retryable(connection) is True
    assert provider.is_retryable(ValueError("not a network error")) is False


def test_supports_native_tools(provider):
    assert provider.supports_native_tools() is True
