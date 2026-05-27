"""LiteLLM-backed provider.

A single provider class that handles every model in ``config.json`` by
routing the call through LiteLLM. Replaces per-vendor Gemini /
DeepSeek / SiliconFlow provider classes.

Model selection happens at the call site: ``brain.brain`` iterates over
``model_strategy``, looks up the per-model entry in ``config.json``, and
passes ``model``, ``api_base``, and ``api_key`` into this provider.

Env vars LiteLLM recognises automatically:
    - GEMINI_API_KEY    (for ``gemini/`` prefix)
    - DEEPSEEK_API_KEY  (for ``deepseek/`` prefix)

For any OpenAI-compatible custom endpoint (e.g. SiliconFlow), pass
``api_base`` and ``api_key`` explicitly; the model string should use
the ``openai/`` prefix.
"""

import json
from typing import List, Optional, Tuple

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from swarm_agent.brain.providers.base import LLMProvider, ProviderRegistry
from swarm_agent.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def _tool_defs_to_openai(tool_definitions: List[ToolDefinition]) -> list:
    """Convert internal ToolDefinition list to the OpenAI-style schema LiteLLM expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
        for td in tool_definitions
    ]


def _parse_tool_calls(message) -> List[ToolCall]:
    """Extract ToolCall list from a LiteLLM response message."""
    calls: List[ToolCall] = []
    raw_calls = getattr(message, "tool_calls", None) or []
    for tc in raw_calls:
        args = tc.function.arguments
        params = json.loads(args) if isinstance(args, str) and args else (args or {})
        calls.append(ToolCall(id=tc.id, name=tc.function.name, params=params))
    return calls


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_text_for_context_summary(text: str) -> BrainDecision:
    """Extract an optional ``\\n---\\nContext: ...`` trailer from AI text."""
    context_summary: Optional[str] = None
    response_text = text

    marker = "\n---\nContext:"
    idx = text.find(marker)
    if idx != -1:
        response_text = text[:idx].rstrip()
        context_summary = text[idx + len(marker):].strip()

    return BrainDecision(text_response=response_text, context_summary=context_summary)


def _build_call_kwargs(
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    extra_body: Optional[dict] = None,
) -> dict:
    kwargs = {"model": model}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


@ProviderRegistry.register("litellm")
class LiteLLMProvider(LLMProvider):
    """Universal provider; every model in config.json is routed through this."""

    def supports_native_tools(self) -> bool:
        return True

    async def generate_with_tools(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        tool_definitions: List[ToolDefinition],
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[BrainDecision, dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await litellm.acompletion(
            **_build_call_kwargs(model, api_base, api_key, extra_body),
            messages=messages,
            tools=_tool_defs_to_openai(tool_definitions),
        )
        msg = response.choices[0].message
        usage = _extract_usage(response)

        if getattr(msg, "tool_calls", None):
            return BrainDecision(tool_calls=_parse_tool_calls(msg)), usage

        text = (msg.content or "").strip()
        return _parse_text_for_context_summary(text), usage

    async def continue_with_tool_results(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        tool_calls: List[ToolCall],
        tool_results: List[ToolResult],
        tool_definitions: List[ToolDefinition],
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[BrainDecision, dict]:
        assistant_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.params)},
            }
            for tc in tool_calls
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_calls": assistant_tool_calls},
        ]
        for tr in tool_results:
            messages.append(
                {"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content}
            )

        response = await litellm.acompletion(
            **_build_call_kwargs(model, api_base, api_key, extra_body),
            messages=messages,
            tools=_tool_defs_to_openai(tool_definitions),
        )
        msg = response.choices[0].message
        usage = _extract_usage(response)

        if getattr(msg, "tool_calls", None):
            return BrainDecision(tool_calls=_parse_tool_calls(msg)), usage

        text = (msg.content or "").strip()
        return _parse_text_for_context_summary(text), usage

    async def generate_decision(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        response_schema: dict,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """JSON-mode fallback for models without native tool calling.

        Embeds the schema in the system prompt to keep behaviour identical
        across providers. LiteLLM's ``response_format={"type": "json_object"}``
        works for OpenAI-compatible providers; Gemini also honours it.
        """
        schema_instruction = (
            "\n\nOUTPUT FORMAT: You MUST respond with ONLY a raw JSON object "
            "(no markdown, no commentary). The JSON object MUST conform to "
            "this schema:\n"
            f"{json.dumps(response_schema, indent=2)}\n\n"
            "No extra keys are allowed."
        )
        messages = [
            {"role": "system", "content": system_prompt + schema_instruction},
            {"role": "user", "content": prompt},
        ]
        response = await litellm.acompletion(
            **_build_call_kwargs(model, api_base, api_key, extra_body),
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = _strip_markdown_fences(response.choices[0].message.content or "")
        return json.loads(text), _extract_usage(response)

    def is_retryable(self, exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                APIConnectionError,
                Timeout,
                RateLimitError,
                ServiceUnavailableError,
                InternalServerError,
                APIError,
            ),
        )
