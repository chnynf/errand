"""Agent loop: orchestrates one user turn through brain + tools + memory.

Given a session's persisted memory and a fresh user input, the loop:

1. Logs the input in memory.
2. Asks ``Brain.decide`` for the next step.
3. If the brain asks for tool calls, runs them through ``ToolRegistry``
   and appends the results to the running message list.
4. Repeats until the brain returns final text or a safety cap fires.
5. Persists memory and returns the final response.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import List, Optional

from errand.brain import Brain
from errand.config import ErrandConfig, load_errand_config
from errand.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from errand.runtime.debug import debug_log
from errand.sessions.memory import Memory
from errand.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 8
MAX_RESPONSE_RETRIES = 3

# Tools a job is forbidden from calling while it is itself executing, so a
# scheduled run can never (re)schedule and spin into an infinite loop.
_SCHEDULED_RUN_BLOCKED_TOOLS = {"schedule_message", "delete_scheduled_job"}


class AgentLoop:
    """The think/act loop for one persisted session."""

    def __init__(
        self,
        session_id: str = "default",
        debug: bool = False,
        *,
        agent_id: str | None = None,
        config: ErrandConfig | None = None,
        delegation_depth: int = 0,
    ):
        self.debug = debug
        self.config = config or load_errand_config()
        self.agent_spec = self.config.get_agent(agent_id)
        self.agent_id = self.agent_spec.id
        self.delegation_depth = delegation_depth
        self.memory = Memory(session_id, agent_id=self.agent_id)
        self.tool_registry = ToolRegistry(can_delegate=self.agent_spec.can_delegate)
        self.brain = Brain(debug=debug, agent_spec=self.agent_spec, config=self.config)

    def add_session_note(self, note: str) -> None:
        self.memory.add_session_note(note)

    def last_activity_at(self) -> Optional[float]:
        return self.memory.last_activity_at()

    def reload_prompt_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        self.brain.reload_prompt_resources(soul=soul, profile=profile)

    async def process_input(
        self, user_input: str, metadata: Optional[dict] = None
    ) -> str:
        """Process a single user interaction and return the final response."""
        is_subagent = bool(metadata and metadata.get("is_subagent"))
        is_scheduled = bool(metadata and metadata.get("is_scheduled_task"))
        input_title = "Parent Agent -> Loop" if is_subagent else "User -> Loop"
        debug_log(input_title, user_input, extra=f"agent={self.agent_id}", truncate=False)
        self.memory.add_history("scheduled" if is_scheduled else "user", user_input)

        reply_to = (metadata or {}).get("_reply_to")

        async def _progress(message: str) -> None:
            if reply_to and not is_subagent:
                try:
                    await reply_to.send_progress(message)
                except Exception:
                    pass

        final_response = ""

        tool_definitions = self.tool_registry.get_tool_definitions()
        if is_scheduled or is_subagent:
            tool_definitions = [
                t for t in tool_definitions if t.name not in _SCHEDULED_RUN_BLOCKED_TOOLS
            ]
        action_count = 0
        force_respond = False
        hit_tool_cap = False
        suppress_usage_footer = bool(metadata and metadata.get("suppress_usage_footer"))

        round_input_tokens = 0
        round_output_tokens = 0
        round_cache_creation_tokens = 0
        round_cache_read_tokens = 0
        round_tools_used: list[str] = []
        round_models_used: list[str] = []
        last_tool_results: list[ToolResult] = []
        scheduled_messages: list[str] = []
        next_log_extra: Optional[str] = None
        messages = self.brain.build_messages(
            self.memory.build_history_messages(),
            context_summary=self.memory.data.get("context_summary"),
            session_note=self.memory.data.get("metadata", {}).get("session_note"),
            instruction="Analyze the user's input. Decide whether to call a tool or respond directly.",
        )

        while True:
            await _progress(f"Thinking... [{self.agent_id}]")
            brain_output = await self.brain.decide(
                messages,
                tool_definitions=[] if force_respond else tool_definitions,
                session_id=self.memory.session_id,
                log_extra=next_log_extra,
            )
            next_log_extra = None

            decision: BrainDecision = brain_output["decision"]
            usage = brain_output["usage"]

            if not brain_output.get("error"):
                round_input_tokens += usage.get("input_tokens", 0)
                round_output_tokens += usage.get("output_tokens", 0)
                round_cache_creation_tokens += usage.get("cache_creation_tokens", 0)
                round_cache_read_tokens += usage.get("cache_read_tokens", 0)
                if usage.get("model"):
                    round_models_used.append(usage.get("model").split("/")[-1])

                self.memory.update_token_usage(usage)

            if brain_output.get("error"):
                final_response = decision.text_response or "I encountered an internal error."
                break

            self.memory.add_history("ai", asdict(decision), metadata={"usage": usage})

            ctx_summary = decision.context_summary
            if ctx_summary:
                self.memory.set_context_summary(ctx_summary)

            if force_respond:
                debug_log(
                    "Forced Response",
                    "Tool round cap was reached; accepting text response without further tools.",
                    extra=f"agent={self.agent_id}",
                )
                final_response = (
                    decision.text_response
                    or "I wasn't able to complete that action after several attempts."
                )
                break

            tool_calls: List[ToolCall] = decision.tool_calls

            if tool_calls:
                action_count += 1
                tool_names = ", ".join(call.name for call in tool_calls)
                await _progress(f"Using tools: {tool_names}")
                debug_log(
                    "Tool Round",
                    f"Round {action_count}/{MAX_TOOL_ROUNDS}: {tool_names}",
                    extra=f"agent={self.agent_id}",
                )
                async def _run_tool(tc: ToolCall) -> ToolResult:
                    debug_log(
                        "Loop Execution",
                        f"Executing tool: {tc.name}({tc.params})",
                        extra=f"agent={self.agent_id}",
                    )
                    try:
                        result = await self.tool_registry.execute(
                            tc.name,
                            tc.params,
                            context={
                                "agent_id": self.agent_id,
                                "session_id": self.memory.session_id,
                                "delegation_depth": self.delegation_depth,
                                "debug": self.debug,
                                "reply_to": (metadata or {}).get("_reply_to"),
                                "source": (metadata or {}).get("_source"),
                            },
                        )
                        return ToolResult(tool_call_id=tc.id, name=tc.name, content=str(result))
                    except Exception as e:
                        return ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=f"Tool '{tc.name}' execution failed: {e}",
                        )

                round_tools_used.extend(tc.name for tc in tool_calls if tc.name)
                last_tool_results = await asyncio.gather(*(_run_tool(tc) for tc in tool_calls))
                last_tool_results = list(last_tool_results)
                for tr in last_tool_results:
                    if tr.name == "schedule_message" and not tr.content.startswith("Tool '"):
                        scheduled_messages.append(tr.content)

                if last_tool_results:
                    self.memory.add_tool_results(tool_calls, last_tool_results)
                messages.append(self._assistant_tool_message(tool_calls, decision.reasoning_content))
                messages.extend(self._tool_result_messages(last_tool_results))

                max_rounds = self.agent_spec.max_tool_rounds
                if action_count >= max_rounds:
                    debug_log(
                        "Action Cap",
                        (
                            f"Reached {action_count}/{max_rounds} tool rounds. "
                            "Requesting a final text response with no more tool calls."
                        ),
                        extra=f"agent={self.agent_id}",
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have reached the maximum number of action attempts. "
                                "Do not call any more tools. Respond now with a summary "
                                "of what happened."
                            ),
                        }
                    )
                    force_respond = True
                    hit_tool_cap = True
                    next_log_extra = "forced response after tool cap"

                continue

            external_response = decision.text_response
            if external_response:
                final_response = external_response
                break

            final_response = "I'm not sure what to do. (No action or response provided)"
            break

        if not final_response:
            final_response = self._summarize_tool_results(last_tool_results)

        if hit_tool_cap and not suppress_usage_footer:
            final_response = (
                f"[Reached action limit after {action_count} tool rounds — "
                f"response may be incomplete]\n\n{final_response}"
            )

        unique_tools = list(dict.fromkeys(round_tools_used))
        tools_str = ", ".join(unique_tools) if unique_tools else "None"

        unique_models = list(dict.fromkeys(round_models_used))
        models_str = ", ".join(unique_models) if unique_models else "None"

        # cache_read is a subset of input_tokens (a "cache hit"); cache_write
        # is Anthropic-only and stays 0 for DeepSeek/Gemini, so hide it then.
        cache_bits = []
        if round_cache_read_tokens:
            cache_bits.append(f"{round_cache_read_tokens} cached")
        if round_cache_creation_tokens:
            cache_bits.append(f"{round_cache_creation_tokens} cache-write")
        cache_str = f" ({', '.join(cache_bits)})" if cache_bits else ""
        usage_msg = (
            f"\n\n---\n*Models used: {models_str}*\n"
            f"*Tokens: {round_input_tokens} in{cache_str}, {round_output_tokens} out*\n"
            f"*Tools used: {tools_str}*"
        )
        if metadata and metadata.get("is_scheduled_task"):
            job_name = metadata.get("job_name", "Scheduled task")
            final_response = f"[{job_name}]\n\n" + final_response
        elif scheduled_messages:
            details = "\n".join(f"- {msg}" for msg in scheduled_messages)
            if details not in final_response:
                final_response = f"Scheduling details:\n{details}\n\n{final_response}"
        if not suppress_usage_footer:
            final_response += usage_msg

        self.memory.clear_session_note()
        output_title = "Loop -> Parent Agent" if is_subagent else "Loop -> User"
        debug_log(
            output_title,
            final_response,
            extra=f"agent={self.agent_id}",
            truncate=False,
        )
        await self.memory.save_session()

        return final_response

    @staticmethod
    def _summarize_tool_results(results: List[ToolResult]) -> str:
        if not results:
            return "Done."
        parts = []
        for r in results:
            parts.append(f"{r.name}: {r.content}")
        return "\n".join(parts)

    @staticmethod
    def _assistant_tool_message(tool_calls: List[ToolCall], reasoning_content: Optional[str] = None) -> dict:
        msg: dict = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.params),
                    },
                }
                for call in tool_calls
            ],
        }
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        return msg

    @staticmethod
    def _tool_result_messages(tool_results: List[ToolResult]) -> list[dict]:
        return [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            }
            for result in tool_results
        ]

    async def shutdown(self) -> None:
        await self.memory.save_session()
