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

from paw.brain.providers.litellm import LiteLLMProvider
from paw.contracts.types import ToolDefinition


def _mk_response(
    *,
    content=None,
    tool_calls=None,
    prompt_tokens=10,
    completion_tokens=5,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
    cached_tokens=None,
):
    """Build a minimal LiteLLM-shaped response object.

    ``cached_tokens`` populates the OpenAI-style
    ``usage.prompt_tokens_details.cached_tokens`` that LiteLLM normalizes
    DeepSeek/OpenAI/Gemini cache hits into. ``cache_read_input_tokens`` is
    the Anthropic-specific fallback field.
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
    if cached_tokens is not None:
        usage.prompt_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)
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
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, usage = await provider.generate(
            model="gemini/gemini-3-flash-preview",
            messages=[
                {"role": "system", "content": "soul"},
                {"role": "user", "content": "What is 42?"},
            ],
            tool_definitions=[],
        )

    mock.assert_awaited_once()
    kwargs = mock.await_args.kwargs
    assert kwargs["model"] == "gemini/gemini-3-flash-preview"
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs
    sys_msg = kwargs["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"][0]["text"] == "soul"
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Rolling breakpoint: the last message also carries cache_control.
    last_msg = kwargs["messages"][1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["text"] == "What is 42?"
    assert last_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"] == []

    assert decision.tool_calls == []
    assert decision.text_response == "The answer is 42."
    assert decision.context_summary == "User asked for the answer."
    assert usage == {"input_tokens": 10, "output_tokens": 5, "cache_creation_tokens": 0, "cache_read_tokens": 0}


async def test_cache_hit_tokens_from_prompt_tokens_details(provider):
    """DeepSeek/OpenAI/Gemini cache hits arrive via prompt_tokens_details.cached_tokens."""
    response = _mk_response(content="ok", prompt_tokens=1000, cached_tokens=800)
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        _, usage = await provider.generate(
            model="openai/deepseek-ai/DeepSeek-V4-Flash",
            messages=[{"role": "system", "content": "soul"}],
            tool_definitions=[],
        )

    assert usage["input_tokens"] == 1000
    assert usage["cache_read_tokens"] == 800
    assert usage["cache_creation_tokens"] == 0


async def test_cache_hit_tokens_anthropic_fallback(provider):
    """When prompt_tokens_details is absent, fall back to the Anthropic field."""
    response = _mk_response(
        content="ok",
        prompt_tokens=1000,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=200,
    )
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        _, usage = await provider.generate(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "system", "content": "soul"}],
            tool_definitions=[],
        )

    assert usage["cache_read_tokens"] == 600
    assert usage["cache_creation_tokens"] == 200


async def test_tool_call_parsing(provider, calculator_tool):
    response = _mk_response(
        tool_calls=[_mk_tool_call("tc-abc", "calculate", {"expression": "1+1"})]
    )
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, usage = await provider.generate(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "soul"},
                {"role": "user", "content": "Compute 1+1"},
            ],
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
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate(
            model="openai/deepseek-ai/DeepSeek-V3.2",
            messages=[
                {"role": "system", "content": "soul"},
                {"role": "user", "content": "hello"},
            ],
            tool_definitions=[calculator_tool],
            api_base="https://api.siliconflow.com/v1",
            api_key="sk-test-siliconflow",
        )

    kwargs = mock.await_args.kwargs
    assert kwargs["api_base"] == "https://api.siliconflow.com/v1"
    assert kwargs["api_key"] == "sk-test-siliconflow"


async def test_reasoning_effort_passed_with_drop_params(provider, calculator_tool):
    """``reasoning_effort`` is forwarded to LiteLLM with ``drop_params`` so it
    can span a fallback chain that mixes reasoning and non-reasoning models."""
    response = _mk_response(content="ok")
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate(
            model="gemini/gemini-3.5-flash",
            messages=[{"role": "system", "content": "soul"}],
            tool_definitions=[calculator_tool],
            reasoning_effort="high",
        )

    kwargs = mock.await_args.kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["drop_params"] is True


async def test_reasoning_effort_omitted_when_unset(provider, calculator_tool):
    response = _mk_response(content="ok")
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate(
            model="gemini/gemini-3.5-flash",
            messages=[{"role": "system", "content": "soul"}],
            tool_definitions=[calculator_tool],
        )

    kwargs = mock.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "drop_params" not in kwargs


async def test_generate_accepts_tool_result_messages(provider, calculator_tool):
    response = _mk_response(content="Final answer: 2")
    request_messages = [
        {"role": "system", "content": "soul"},
        {"role": "user", "content": "Compute 1+1"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": json.dumps({"expression": "1+1"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": "2"},
    ]

    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        decision, _ = await provider.generate(
            model="gemini/gemini-3-flash-preview",
            messages=request_messages,
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
    # The last message (the tool result) gets the rolling cache breakpoint, so
    # its content is wrapped into a block list carrying cache_control.
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tc-1"
    assert tool_msg["content"][0]["text"] == "2"
    assert tool_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    assert decision.text_response == "Final answer: 2"


async def test_cache_breakpoints_mark_system_and_tail_only(provider, calculator_tool):
    """Static breakpoint on the system prompt + rolling breakpoint on the tail;
    interior messages are left untouched so the prefix stays byte-stable."""
    response = _mk_response(content="ok")
    messages = [
        {"role": "system", "content": "soul"},
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": "interim result"},
    ]
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate(
            model="anthropic/claude-3-5-sonnet",
            messages=messages,
            tool_definitions=[calculator_tool],
        )

    sent = mock.await_args.kwargs["messages"]
    # System: wrapped with cache_control.
    assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Interior user + assistant messages: unchanged (no cache_control).
    assert sent[1] == {"role": "user", "content": "do the thing"}
    assert "cache_control" not in json.dumps(sent[2])
    # Tail (the tool result): rolling breakpoint.
    assert sent[3]["content"][0]["text"] == "interim result"
    assert sent[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # The original input list is not mutated.
    assert messages[3] == {"role": "tool", "tool_call_id": "tc-1", "content": "interim result"}


async def test_single_system_message_gets_no_rolling_breakpoint(provider):
    """With only a system message there is no separate tail to mark."""
    response = _mk_response(content="ok")
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ) as mock:
        await provider.generate(
            model="gemini/gemini-3.5-flash",
            messages=[{"role": "system", "content": "soul"}],
            tool_definitions=[],
        )

    sent = mock.await_args.kwargs["messages"]
    assert len(sent) == 1
    assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


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
