"""Tests for `Brain`.

Covers:
- `_resolve_model` reads `litellm_model`, `api_base`, and resolves
  `api_key_env` from environment.
- `decide` falls through `model_strategy` on retryable errors.
- `decide` can resume after a failed model key for continuation-style retries.

LiteLLM is mocked at the module level used by the provider so no
network is touched. Config is patched onto the `Brain` instance.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from paw.brain import Brain
from paw.wire_types import ToolDefinition


def _mk_response(*, content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4)
    return SimpleNamespace(choices=[choice], usage=usage)


def _mk_tool_call(call_id, name, args):
    function = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, function=function)


CONFIG_FIXTURE = {
    "gemini-flash": {
        "provider": "gemini",
        "litellm_model": "gemini/gemini-3-flash-preview",
        "native_tools": True,
    },
    "siliconflow-v3": {
        "provider": "siliconflow",
        "litellm_model": "openai/deepseek-ai/DeepSeek-V3",
        "api_base": "https://api.siliconflow.com/v1",
        "api_key_env": "FAKE_SILICONFLOW_KEY",
        "native_tools": True,
    },
}


@pytest.fixture
def brain(monkeypatch):
    monkeypatch.setenv("FAKE_SILICONFLOW_KEY", "sk-fake")
    b = Brain(debug=False)
    b.models_config = CONFIG_FIXTURE
    b.model_strategy = ["gemini-flash", "siliconflow-v3"]
    return b


def test_resolve_model_reads_litellm_fields(brain):
    provider_name, model, native_flag, overrides = brain._resolve_model("gemini-flash")
    assert provider_name == "gemini"
    assert model == "gemini/gemini-3-flash-preview"
    assert native_flag is True
    assert overrides == {}


def test_resolve_model_resolves_api_key_env(brain):
    provider_name, model, _, overrides = brain._resolve_model("siliconflow-v3")
    assert provider_name == "siliconflow"
    assert model == "openai/deepseek-ai/DeepSeek-V3"
    assert overrides == {
        "api_base": "https://api.siliconflow.com/v1",
        "api_key": "sk-fake",
    }


def test_resolve_model_reads_extra_body(brain):
    brain.models_config["siliconflow-v4"] = {
        "provider": "siliconflow",
        "litellm_model": "openai/deepseek-ai/DeepSeek-V4-Flash",
        "api_base": "https://api.siliconflow.com/v1",
        "api_key_env": "FAKE_SILICONFLOW_KEY",
        "native_tools": True,
        "extra_body": {"enable_thinking": False},
    }

    _, _, _, overrides = brain._resolve_model("siliconflow-v4")
    assert overrides["extra_body"] == {"enable_thinking": False}


def test_resolve_model_unknown_key_falls_back_to_passthrough(brain):
    provider_name, model, native_flag, overrides = brain._resolve_model("ad-hoc-model")
    assert provider_name == "ad-hoc-model"
    assert model == "ad-hoc-model"
    assert native_flag is None
    assert overrides == {}


def test_resolve_model_includes_agent_reasoning_effort(brain):
    brain.reasoning_effort = "high"
    _, _, _, overrides = brain._resolve_model("gemini-flash")
    assert overrides["reasoning_effort"] == "high"


def test_resolve_model_no_effort_omits_override(brain):
    """Unset effort sends no param so the provider's own default applies."""
    _, _, _, overrides = brain._resolve_model("gemini-flash")
    assert "reasoning_effort" not in overrides


async def test_decide_forwards_agent_reasoning_effort(brain):
    """Agent-level effort flows through ``decide`` into the LiteLLM call."""
    brain.reasoning_effort = "high"
    response = _mk_response(content="ok\n---\nContext: cs")
    mock_acompletion = AsyncMock(return_value=response)
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=mock_acompletion,
    ):
        await brain.decide(
            messages=[{"role": "system", "content": "soul"}, {"role": "user", "content": "hi"}],
        )

    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["drop_params"] is True


async def test_decide_falls_through_to_next_model_on_retryable_error(brain):
    """First strategy entry raises a retryable error; second succeeds."""
    from litellm.exceptions import RateLimitError

    tool_def = ToolDefinition(
        name="calculate",
        description="x",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    rate_limit = RateLimitError(
        message="too many", llm_provider="gemini", model="gemini-3-flash-preview"
    )
    success_response = _mk_response(
        content="The answer is 42.\n---\nContext: cs"
    )

    mock_acompletion = AsyncMock(side_effect=[rate_limit, success_response])
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=mock_acompletion,
    ), patch("paw.brain.brain.asyncio.sleep", new=AsyncMock()):
        result = await brain.decide(
            messages=[{"role": "system", "content": "soul"}, {"role": "user", "content": "hi"}],
            tool_definitions=[tool_def],
        )

    decision = result["decision"]
    usage = result["usage"]

    assert mock_acompletion.await_count == 2
    first_kwargs = mock_acompletion.await_args_list[0].kwargs
    second_kwargs = mock_acompletion.await_args_list[1].kwargs
    assert first_kwargs["model"] == "gemini/gemini-3-flash-preview"
    assert "api_base" not in first_kwargs
    assert second_kwargs["model"] == "openai/deepseek-ai/DeepSeek-V3"
    assert second_kwargs["api_base"] == "https://api.siliconflow.com/v1"
    assert second_kwargs["api_key"] == "sk-fake"

    assert decision.text_response == "The answer is 42."
    assert decision.context_summary == "cs"
    assert usage["model"] == "openai/deepseek-ai/DeepSeek-V3"


async def test_decide_records_usage_into_tracker_under_agent_id(brain):
    """Brain is the emitter: a successful call records into the passed tracker.

    The retryable first model must NOT record (it failed); only the model that
    actually returned a response contributes usage.
    """
    from litellm.exceptions import RateLimitError

    from paw.runtime.run_context import UsageTracker

    brain.agent_id = "notes-organizer"
    tracker = UsageTracker()

    rate_limit = RateLimitError(
        message="too many", llm_provider="gemini", model="gemini-3-flash-preview"
    )
    success_response = _mk_response(content="ok\n---\nContext: cs")
    mock_acompletion = AsyncMock(side_effect=[rate_limit, success_response])
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=mock_acompletion,
    ), patch("paw.brain.brain.asyncio.sleep", new=AsyncMock()):
        await brain.decide(
            messages=[{"role": "user", "content": "hi"}],
            usage_tracker=tracker,
        )

    # prompt_tokens=3 / completion_tokens=4 come from _mk_response; only the
    # one successful call is recorded, attributed to this brain's agent_id.
    assert tracker.calls == 1
    assert tracker.input_tokens == 3
    assert tracker.output_tokens == 4
    assert "notes-organizer" in tracker.per_agent()


async def test_decide_can_resume_after_failed_model_key(brain):
    """Continuation fallback can skip a failed model key and resume strategy order."""
    tool_def = ToolDefinition(
        name="calculate",
        description="x",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    messages = [
        {"role": "system", "content": "soul"},
        {"role": "user", "content": "calc"},
        {"role": "assistant", "tool_calls": []},
        {"role": "tool", "tool_call_id": "tc-1", "content": "42"},
    ]

    final_response = _mk_response(content="done\n---\nContext: cs")

    mock_acompletion = AsyncMock(return_value=final_response)
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=mock_acompletion,
    ), patch("paw.brain.brain.asyncio.sleep", new=AsyncMock()):
        result = await brain.decide(
            messages=messages,
            tool_definitions=[tool_def],
            skip_model_keys=["gemini-flash"],
            start_after_model_key="gemini-flash",
        )

    decision = result["decision"]
    assert mock_acompletion.await_count == 1
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs["model"] == "openai/deepseek-ai/DeepSeek-V3"
    assert kwargs["messages"][2]["role"] == "assistant"
    assert kwargs["messages"][3]["role"] == "tool"
    assert decision.text_response == "done"


async def test_decide_start_after_failed_model_not_provider(brain):
    """A failed continuation should still try later models from the same provider."""
    brain.models_config = {
        "sf-v4-flash": {
            "provider": "siliconflow",
            "litellm_model": "openai/deepseek-ai/DeepSeek-V4-Flash",
            "api_base": "https://api.siliconflow.com/v1",
            "api_key_env": "FAKE_SILICONFLOW_KEY",
            "native_tools": True,
        },
        "sf-v4-pro": {
            "provider": "siliconflow",
            "litellm_model": "openai/deepseek-ai/DeepSeek-V4-Pro",
            "api_base": "https://api.siliconflow.com/v1",
            "api_key_env": "FAKE_SILICONFLOW_KEY",
            "native_tools": True,
        },
        "gemini-flash": CONFIG_FIXTURE["gemini-flash"],
    }
    brain.model_strategy = ["gemini-flash", "sf-v4-flash", "sf-v4-pro"]
    tool_def = ToolDefinition(
        name="calculate",
        description="x",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    final_response = _mk_response(content="done\n---\nContext: cs")

    mock_acompletion = AsyncMock(return_value=final_response)
    with patch(
        "paw.brain.providers.litellm.litellm.acompletion",
        new=mock_acompletion,
    ), patch("paw.brain.brain.asyncio.sleep", new=AsyncMock()):
        result = await brain.decide(
            messages=[
                {"role": "system", "content": "soul"},
                {"role": "user", "content": "calc"},
            ],
            tool_definitions=[tool_def],
            skip_model_keys=["sf-v4-flash"],
            start_after_model_key="sf-v4-flash",
        )

    second_kwargs = mock_acompletion.await_args.kwargs
    assert second_kwargs["model"] == "openai/deepseek-ai/DeepSeek-V4-Pro"
    assert result["decision"].text_response == "done"
