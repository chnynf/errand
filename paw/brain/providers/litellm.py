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

from paw.brain.providers.base import LLMProvider, ProviderRegistry
from paw.wire_types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
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


def _cached_read_tokens(usage) -> int:
    """Number of input tokens served from a prompt cache (a "cache hit").

    Providers expose this differently and LiteLLM does not unify it:
      - OpenAI-compatible / DeepSeek / Gemini: normalized into
        ``usage.prompt_tokens_details.cached_tokens``. (DeepSeek's native
        ``prompt_cache_hit_tokens`` is mapped here by LiteLLM.)
      - Anthropic: a dedicated ``usage.cache_read_input_tokens`` field.

    We check the OpenAI-style location first (it covers every model in our
    config today) and fall back to the Anthropic field.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    if not cached:
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    return cached or 0


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
    return {
        # ``prompt_tokens`` is the TOTAL input, including any cache-hit tokens.
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        # Anthropic-only: tokens written into the cache on this call.
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        # Input tokens served from cache (DeepSeek/OpenAI/Gemini/Anthropic).
        "cache_read_tokens": _cached_read_tokens(usage),
    }


def _with_cache_breakpoint(message: dict) -> dict:
    """Return a copy of ``message`` with a cache_control breakpoint on its content.

    Anthropic-style explicit breakpoint: LiteLLM maps it to Anthropic
    ``cache_control`` blocks. It is a NO-OP for Gemini and DeepSeek -- they use
    *implicit* (automatic, prefix-based) caching that needs no flag, so the
    cache hits we see on those providers come from implicit caching, not from
    this breakpoint. Providers ignore the field harmlessly.

    Handles both string content (wrapped into a single text block) and an
    existing list of content blocks (breakpoint added to the last block).
    Messages with no usable content (e.g. an assistant message carrying only
    ``tool_calls``) are returned unchanged.
    """
    content = message.get("content")
    if isinstance(content, str) and content:
        return {
            **message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        if isinstance(blocks[-1], dict):
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            return {**message, "content": blocks}
    return message


def _apply_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """Place cache_control breakpoints to maximize prefix-cache reuse.

    Two breakpoints (within Anthropic's limit of four):

    - **Static**: the system prompt, hot across every call in the session.
    - **Rolling**: the LAST message of this request. Each call in a tool loop
      (and each new turn) extends the previous request, so marking the tail
      lets a breakpoint-based cache (Anthropic) grow with the conversation
      instead of being pinned to the system prompt. The cache is incremental,
      so a breakpoint further along reuses the already-cached prefix and only
      writes the new suffix.
    """
    if not messages:
        return messages
    out = list(messages)
    if out[0].get("role") == "system":
        out[0] = _with_cache_breakpoint(out[0])
    # Rolling breakpoint on the tail, unless the tail *is* the system message.
    if len(out) > 1:
        out[-1] = _with_cache_breakpoint(out[-1])
    return out


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
    reasoning_effort: Optional[str] = None,
) -> dict:
    kwargs = {"model": model}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    if reasoning_effort:
        # LiteLLM normalizes ``reasoning_effort`` into each provider's native
        # thinking/reasoning controls. ``drop_params`` lets it silently skip
        # the param for models that don't support reasoning, so a single
        # effort setting can span the whole fallback chain without crashing.
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["drop_params"] = True
    return kwargs


@ProviderRegistry.register("litellm")
class LiteLLMProvider(LLMProvider):
    """Universal provider; every model in config.json is routed through this."""

    def supports_native_tools(self) -> bool:
        return True

    async def generate(
        self,
        model: str,
        messages: list[dict],
        tool_definitions: List[ToolDefinition],
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[BrainDecision, dict]:
        request_messages = _apply_cache_breakpoints(messages)
        response = await litellm.acompletion(
            **_build_call_kwargs(model, api_base, api_key, extra_body, reasoning_effort),
            messages=request_messages,
            tools=_tool_defs_to_openai(tool_definitions),
        )
        msg = response.choices[0].message
        usage = _extract_usage(response)

        if getattr(msg, "tool_calls", None):
            reasoning = getattr(msg, "reasoning_content", None) or None
            return BrainDecision(tool_calls=_parse_tool_calls(msg), reasoning_content=reasoning), usage

        text = (msg.content or "").strip()
        return _parse_text_for_context_summary(text), usage
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
