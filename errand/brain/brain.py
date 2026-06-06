"""Brain: model routing + native tool call orchestration.

Brain is intentionally provider-agnostic. It iterates through the
configured ``model_strategy``, asks each provider for a decision, and
falls back to the next model on retryable errors.

Native tool calling is the only supported mode. JSON-mode fallback was
removed when LiteLLM's native tool support became universal across our
configured providers.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from errand.brain.prompt_assembler import PromptAssembler
from errand.brain.providers import ProviderRegistry
from errand.config import AgentSpec, FileAccessConfig, ErrandConfig, load_raw_config
from errand.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from errand.runtime.debug import debug_log, debug_log_prompt, set_debug

load_dotenv()
logger = logging.getLogger(__name__)


class Brain:
    """Routes turns through a provider chain with retry/fallback."""

    def __init__(
        self,
        debug: bool = False,
        *,
        agent_spec: AgentSpec | None = None,
        config: ErrandConfig | None = None,
    ):
        set_debug(debug)

        raw_config = load_raw_config() if config is None else {
            "models": config.models,
            "model_strategy": config.model_strategy,
            "retry": config.retry,
            "shared_soul": config.shared_soul,
            "shared_notes_index": config.shared_notes_index,
            "agent_profile": config.agent_profile,
            "file_access": {
                "default_scope": config.file_access.default_scope,
                "scopes": {
                    name: {
                        "roots": scope.roots,
                        "read": scope.read,
                        "list": scope.list,
                    }
                    for name, scope in config.file_access.scopes.items()
                },
            },
        }
        self.models_config = raw_config.get("models", {})
        self.agent_id = agent_spec.id if agent_spec else "default"
        self.model_strategy: List[str] = (
            list(agent_spec.model_strategy) if agent_spec and agent_spec.model_strategy
            else raw_config.get("model_strategy", [])
        )
        self.base_delay: int = raw_config.get("retry", {}).get("base_delay_seconds", 2)

        if not self.model_strategy:
            raise ValueError(
                "model_strategy in config.json must contain at least one model."
            )

        self.prompt_assembler = PromptAssembler(
            shared_soul=(
                agent_spec.shared_soul
                if agent_spec and agent_spec.shared_soul is not None
                else raw_config.get("shared_soul")
            ),
            agent_profile=(
                agent_spec.agent_profile
                if agent_spec and agent_spec.agent_profile is not None
                else raw_config.get("agent_profile")
            ),
            shared_notes_index=raw_config.get("shared_notes_index"),
            file_access=(
                config.file_access
                if config is not None
                else FileAccessConfig.from_dict(raw_config.get("file_access", {}))
            ),
        )

        self._last_provider_name: Optional[str] = None
        self._last_model_key: Optional[str] = None
        self._last_model: Optional[str] = None
        self._last_call_overrides: Dict[str, Any] = {}

    def reload_prompt_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        self.prompt_assembler.reload_resources(soul=soul, profile=profile)

    def _resolve_model(
        self, model_key: str
    ) -> Tuple[str, str, Optional[bool], Dict[str, Any]]:
        """Return (provider_name, litellm_model, native_tools flag, overrides).

        ``overrides`` carries optional per-model ``api_base`` / ``api_key``
        that LiteLLM cannot infer from env vars (e.g. OpenAI-compatible
        endpoints like SiliconFlow).
        """
        model_info = self.models_config.get(model_key) or {}
        provider_name = model_info.get("provider", model_key)
        actual_model = model_info.get("litellm_model") or model_info.get("model", model_key)
        native_flag = model_info.get("native_tools")

        overrides: Dict[str, Any] = {}
        if model_info.get("api_base"):
            overrides["api_base"] = model_info["api_base"]
        api_key_env = model_info.get("api_key_env")
        if api_key_env:
            overrides["api_key"] = os.getenv(api_key_env)
        if model_info.get("extra_body"):
            overrides["extra_body"] = dict(model_info["extra_body"])

        return provider_name, actual_model, native_flag, overrides

    async def decide(
        self,
        context_text: str,
        instruction: str,
        tool_definitions: Optional[List[ToolDefinition]] = None,
        skip_providers: Optional[List[str]] = None,
        skip_model_keys: Optional[List[str]] = None,
        start_after_model_key: Optional[str] = None,
        session_id: Optional[str] = None,
        log_extra: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the Brain to decide the next action.

        Returns a dict with keys: decision (BrainDecision), usage (dict),
        and optionally error (bool) / error_message (str).
        """
        max_attempts = len(self.model_strategy)
        last_exception: Optional[Exception] = None

        start_index = 0
        if start_after_model_key in self.model_strategy:
            start_index = self.model_strategy.index(start_after_model_key) + 1

        for attempt in range(start_index, max_attempts):
            model_key = self.model_strategy[attempt]
            provider_name, actual_model, native_flag, overrides = self._resolve_model(model_key)

            if skip_model_keys and model_key in skip_model_keys:
                continue
            if skip_providers and provider_name in skip_providers:
                continue
            if native_flag is False:
                # JSON-mode fallback is no longer supported; skip non-native models.
                continue

            provider = ProviderRegistry.get_provider(provider_name)
            if not provider.supports_native_tools():
                continue

            try:
                system_prompt = self.prompt_assembler.build_system_prompt()
                user_prompt = self.prompt_assembler.build_user_prompt(context_text, instruction)
                tool_names = [td.name for td in tool_definitions] if tool_definitions else []

                debug_log_prompt(
                    f"Loop({self.agent_id}) -> AI", system_prompt, user_prompt,
                    model=actual_model, extra=log_extra or "native tools",
                    tool_names=tool_names,
                    session_id=session_id,
                    agent_id=self.agent_id,
                    instruction=instruction,
                )

                decision, usage_data = await provider.generate_with_tools(
                    model=actual_model,
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    tool_definitions=tool_definitions or [],
                    **overrides,
                )

                if not decision.tool_calls and not (
                    decision.text_response and decision.text_response.strip()
                ):
                    raise ValueError(
                        "AI returned empty response (no tool calls and no text response)."
                    )

                usage_data["model"] = actual_model
                usage_data["model_key"] = model_key
                usage_data["pricing"] = (self.models_config.get(model_key) or {}).get("pricing")

                if decision.tool_calls:
                    self._last_provider_name = provider_name
                    self._last_model_key = model_key
                    self._last_model = actual_model
                    self._last_call_overrides = overrides
                    tc_desc = "; ".join(
                        f"{tc.name}({', '.join(f'{k}={v!r}' for k, v in tc.params.items())})"
                        for tc in decision.tool_calls
                    )
                    debug_log(
                        f"AI -> Loop({self.agent_id})",
                        f"Tool call: {tc_desc}",
                        model=actual_model,
                    )
                else:
                    self._last_provider_name = None
                    preview = (decision.text_response or "")[:300]
                    body = f"Text: {preview}"
                    if decision.context_summary:
                        body += f"\nContext: {decision.context_summary}"
                    debug_log(f"AI -> Loop({self.agent_id})", body, model=actual_model)

                return {"decision": decision, "usage": usage_data}

            except json.JSONDecodeError as e:
                debug_log("! FAILED", f"JSON parse error: {e}", model=actual_model)
                return {
                    "decision": BrainDecision(
                        text_response="I encountered an internal error (JSON Parse)."
                    ),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "model": actual_model},
                    "error": True,
                    "error_message": str(e),
                }

            except Exception as e:
                last_exception = e
                if not provider.is_retryable(e) or attempt == max_attempts - 1:
                    debug_log("! FAILED", str(e), model=actual_model)
                    break

                delay = self.base_delay
                retry_msg = (
                    f"Retryable error (attempt {attempt + 1}/{max_attempts}), "
                    f"retrying in {delay}s: {e}"
                )
                debug_log("! FAILED", retry_msg, model=actual_model, level="ERROR")
                await asyncio.sleep(delay)

        return {
            "decision": BrainDecision(
                text_response=f"I encountered a connection error with my brain: {last_exception}"
            ),
            "usage": {
                "input_tokens": 0, "output_tokens": 0,
                "model": self.model_strategy[0] if self.model_strategy else "unknown",
            },
            "error": True,
            "error_message": str(last_exception),
        }

    async def submit_tool_results(
        self,
        tool_calls: List[ToolCall],
        tool_results: List[ToolResult],
        tool_definitions: List[ToolDefinition],
        fallback_context: str,
        fallback_instruction: str,
        prompt_context: Optional[str] = None,
        prompt_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send tool results back to the AI.

        Tries the provider that issued the tool calls (native continuation).
        On failure, falls back to decide() with the failed vendor skipped.
        """
        if self._last_provider_name and self._last_model_key:
            provider_name = self._last_provider_name
            provider = ProviderRegistry.get_provider(provider_name)
            _, actual_model, _, overrides = self._resolve_model(self._last_model_key)

            tr_desc = "; ".join(f"{tr.name} -> {tr.content[:80]}" for tr in tool_results)
            debug_log(
                f"Loop({self.agent_id}) -> AI", tr_desc,
                model=actual_model, extra="tool results",
            )

            try:
                system_prompt = self.prompt_assembler.build_system_prompt()

                p_ctx = prompt_context if prompt_context is not None else fallback_context
                p_inst = prompt_instruction if prompt_instruction is not None else fallback_instruction
                user_prompt = self.prompt_assembler.build_user_prompt(p_ctx, p_inst)

                decision, usage_data = await provider.continue_with_tool_results(
                    model=actual_model,
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    tool_definitions=tool_definitions,
                    **overrides,
                )

                if not decision.tool_calls and not (
                    decision.text_response and decision.text_response.strip()
                ):
                    raise ValueError(
                        "AI returned empty response (no tool calls and no text response)."
                    )

                usage_data["model"] = actual_model
                usage_data["model_key"] = self._last_model_key
                usage_data["pricing"] = (
                    self.models_config.get(self._last_model_key) or {}
                ).get("pricing")

                if decision.tool_calls:
                    self._last_provider_name = provider_name
                    tc_desc = "; ".join(
                        f"{tc.name}({', '.join(f'{k}={v!r}' for k, v in tc.params.items())})"
                        for tc in decision.tool_calls
                    )
                    debug_log(
                        f"AI -> Loop({self.agent_id})",
                        f"Tool call: {tc_desc}",
                        model=actual_model,
                    )
                else:
                    self._last_provider_name = None
                    preview = (decision.text_response or "")[:300]
                    body = f"Text: {preview}"
                    if decision.context_summary:
                        body += f"\nContext: {decision.context_summary}"
                    debug_log(f"AI -> Loop({self.agent_id})", body, model=actual_model)

                return {"decision": decision, "usage": usage_data}

            except Exception as e:
                debug_log(
                    "! FAILED",
                    f"{e}\nRolling back -> retrying with next model...",
                    model=actual_model,
                )
                failed_model_key = self._last_model_key
                self._last_provider_name = None
                self._last_model_key = None

                return await self.decide(
                    fallback_context,
                    fallback_instruction,
                    tool_definitions=tool_definitions,
                    skip_model_keys=[failed_model_key] if failed_model_key else None,
                    start_after_model_key=failed_model_key,
                    log_extra="tool results (fallback)",
                )

        return await self.decide(
            fallback_context,
            fallback_instruction,
            tool_definitions=tool_definitions,
            log_extra="tool results (fallback)",
        )
