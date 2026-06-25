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

from paw.agent_loop.prompt_assembler import PromptAssembler
from paw.brain import Brain
from paw.config import PawConfig, load_paw_config
from paw.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from paw.runtime.debug import debug_log
from paw.runtime.run_context import RunContext
from paw.sessions.memory import Memory
from paw.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 8
MAX_RESPONSE_RETRIES = 3

# When a single turn's live context grows past this estimate (~chars/4), the
# loop condenses everything gathered so far into a summary and continues as a
# fresh internal segment -- same turn, same single reply to the interface.
# Caching covers the cheap case; this only fires when one turn reads a lot.
COMPACTION_TRIGGER_TOKENS = 20_000
MAX_COMPACTIONS = 3

# Orchestration instruction the loop injects to condense an oversized turn
# before continuing (see _compact_segment). Like the other instruction strings
# in this file, it drives the loop's control flow, so it lives with the loop.
COMPACTION_INSTRUCTION = (
    "Your working context for this turn has grown large and is about to be "
    "condensed so you can keep going with a clean slate. Write a summary that "
    "lets you continue WITHOUT the detailed tool history below.\n\n"
    "Anchor everything to the user's original request (shown at the top of this "
    "conversation). Keep only what is relevant to fulfilling it; drop tangential "
    "detail.\n\n"
    "Include, concisely:\n"
    "1. Goal -- restate the user's request in one or two sentences.\n"
    "2. Findings -- the concrete facts, file contents, search results, or data "
    "gathered from tool calls so far that matter for the goal. Quote exact values "
    "(paths, numbers, names) you will need.\n"
    "3. Decisions -- anything you have already concluded or chosen.\n"
    "4. Remaining -- what still needs to be done to finish the task.\n\n"
    "Do not call any tools. Respond with the summary text only -- no preamble. The "
    "detailed history is being discarded, so anything you omit is gone. Capture "
    "every path, value, and file detail you will need so you can finish from this "
    "summary ALONE, without re-reading anything you have already read."
)

# Tools a job is forbidden from calling while it is itself executing, so a
# scheduled run can never (re)schedule and spin into an infinite loop.
_SCHEDULED_RUN_BLOCKED_TOOLS = {"schedule_message", "cancel_scheduled_job"}


def _effective_call(tc: ToolCall) -> tuple[str, dict]:
    """Resolve a tool call to its real (name, params).

    Catalog tools arrive wrapped as ``call_tool(name, args)``; unwrap them so
    history, compaction, logging, and the scheduled-run block all see the real
    tool name. Native/meta tools pass through unchanged.
    """
    if tc.name == "call_tool" and isinstance(tc.params, dict):
        args = tc.params.get("args")
        return str(tc.params.get("name") or ""), args if isinstance(args, dict) else {}
    return tc.name, tc.params


class AgentLoop:
    """The think/act loop for one persisted session."""

    def __init__(
        self,
        session_id: str = "default",
        debug: bool = False,
        *,
        agent_id: str | None = None,
        config: PawConfig | None = None,
        delegation_depth: int = 0,
    ):
        self.debug = debug
        self.config = config or load_paw_config()
        self.agent_spec = self.config.get_agent(agent_id)
        self.agent_id = self.agent_spec.id
        self.delegation_depth = delegation_depth
        self.memory = Memory(session_id, agent_id=self.agent_id)
        self.tool_registry = ToolRegistry(can_delegate=self.agent_spec.can_delegate)
        # The loop is the running agent instance and the sole consumer of prompt
        # assembly, so it owns the assembler. Brain just decides over messages.
        self.prompt_assembler = PromptAssembler(
            shared_soul=self.agent_spec.shared_soul or self.config.shared_soul,
            agent_profile=self.agent_spec.agent_profile or self.config.agent_profile,
            shared_notes_index=self.config.shared_notes_index,
            file_access=self.config.file_access,
            tool_summary=self.tool_registry.tool_summary(),
        )
        self.brain = Brain(debug=debug, agent_spec=self.agent_spec, config=self.config)

    def add_session_note(self, note: str) -> None:
        self.memory.add_session_note(note)

    def last_activity_at(self) -> Optional[float]:
        return self.memory.last_activity_at()

    def reload_prompt_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        self.prompt_assembler.reload_resources(soul=soul, profile=profile)

    async def process_input(
        self, user_input: str, metadata: Optional[dict] = None
    ) -> str:
        """Process a single user interaction and return the final response."""
        m = metadata or {}
        is_subagent = bool(m.get("is_subagent"))
        is_scheduled = bool(m.get("is_scheduled_task"))
        input_title = "Parent Agent -> Loop" if is_subagent else "User -> Loop"
        debug_log(input_title, user_input, extra=f"agent={self.agent_id}", truncate=False)
        self.memory.add_history("scheduled" if is_scheduled else "user", user_input)

        # Root exchanges create the RunContext (+ its UsageTracker); delegated
        # child loops reuse the parent's so usage rolls up across agents.
        run_context: RunContext = m.get("_run_context") or RunContext.root(
            reply_to=m.get("_reply_to"),
            source=m.get("_source"),
        )
        reply_to = run_context.reply_to

        async def _progress(message: str) -> None:
            if reply_to and not is_subagent:
                try:
                    await reply_to.send_progress(message)
                except Exception:
                    pass

        final_response = ""

        # Blocked tools are no longer in the tool list (they live in the catalog,
        # reached via call_tool), so enforcement moves to execution time below.
        tool_definitions = self.tool_registry.get_tool_definitions()
        action_count = 0
        compaction_count = 0
        force_respond = False
        hit_tool_cap = False
        suppress_usage_footer = bool(m.get("suppress_usage_footer"))

        round_tools_used: list[str] = []
        last_tool_results: list[ToolResult] = []
        scheduled_messages: list[str] = []
        next_log_extra: Optional[str] = None

        # ``prefix`` is the stable, append-only part: system prompt + history +
        # the tool rounds accumulated this turn. The volatile per-turn context
        # (current time, rolling summary, instruction) is NOT baked in here -- it
        # is appended as the LAST message on every model call so the prefix grows
        # monotonically and stays prefix-cacheable across rounds and turns. A
        # volatile block wedged into the prefix would break the cache for
        # everything after it on the next, differing request.
        prefix = self.prompt_assembler.build_prefix_messages(
            self.memory.build_history_messages()
        )
        context_summary = self.memory.data.get("context_summary")
        session_note = self.memory.data.get("metadata", {}).get("session_note")
        instruction = "Analyze the user's input. Decide whether to call a tool or respond directly."

        while True:
            await _progress(f"Thinking... [{self.agent_id}]")
            messages = prefix + self.prompt_assembler.build_context_messages(
                context_summary=context_summary,
                session_note=session_note,
                instruction=instruction,
            )
            brain_output = await self.brain.decide(
                messages,
                tool_definitions=[] if force_respond else tool_definitions,
                session_id=self.memory.session_id,
                log_extra=next_log_extra,
                usage_tracker=run_context.usage,
            )
            next_log_extra = None

            decision: BrainDecision = brain_output["decision"]
            usage = brain_output["usage"]

            if not brain_output.get("error"):
                # Per-exchange totals (incl. sub-agents) live on the tracker via
                # the Brain; here we only persist the per-session lifetime total.
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
                eff = {tc.id: _effective_call(tc) for tc in tool_calls}

                async def _run_tool(tc: ToolCall) -> ToolResult:
                    eff_name, eff_params = eff[tc.id]
                    debug_log(
                        "Loop Execution",
                        f"Executing tool: {eff_name}({eff_params})",
                        extra=f"agent={self.agent_id}",
                    )
                    if (is_scheduled or is_subagent) and eff_name in _SCHEDULED_RUN_BLOCKED_TOOLS:
                        return ToolResult(
                            tool_call_id=tc.id,
                            name=eff_name,
                            content=f"Tool '{eff_name}' is not available during a scheduled or delegated run.",
                        )
                    # Catalog tools (reached via call_tool) get harness-side arg
                    # validation; the provider only checked the envelope. A catch
                    # is recorded as a quality signal -- it forces a retry.
                    if tc.name == "call_tool":
                        verror = self.tool_registry.validate_args(eff_name, eff_params)
                        if verror:
                            run_context.usage.record_validation_catch()
                            return ToolResult(tool_call_id=tc.id, name=eff_name, content=verror)
                    try:
                        result = await self.tool_registry.execute(
                            eff_name,
                            eff_params,
                            context={
                                "agent_id": self.agent_id,
                                "session_id": self.memory.session_id,
                                "delegation_depth": self.delegation_depth,
                                "debug": self.debug,
                                "reply_to": run_context.reply_to,
                                "source": run_context.source,
                                "run_context": run_context,
                            },
                        )
                        return ToolResult(tool_call_id=tc.id, name=eff_name, content=str(result))
                    except Exception as e:
                        # The tool raised; hand the error back as a result. Like a
                        # validation catch, this forces the model to retry, so it's
                        # tracked as a quality signal.
                        run_context.usage.record_tool_error()
                        return ToolResult(
                            tool_call_id=tc.id,
                            name=eff_name,
                            content=f"Tool '{eff_name}' execution failed: {e}",
                        )

                round_tools_used.extend(eff[tc.id][0] for tc in tool_calls if eff[tc.id][0])
                last_tool_results = await asyncio.gather(*(_run_tool(tc) for tc in tool_calls))
                last_tool_results = list(last_tool_results)
                for tr in last_tool_results:
                    if tr.name == "schedule_message" and not tr.content.startswith("Tool '"):
                        scheduled_messages.append(tr.content)

                if last_tool_results:
                    calls_by_id = {
                        tc.id: ToolCall(id=tc.id, name=eff[tc.id][0], params=eff[tc.id][1])
                        for tc in tool_calls
                    }
                    records = [
                        self.tool_registry.compact_result(calls_by_id.get(tr.tool_call_id), tr)
                        for tr in last_tool_results
                    ]
                    self.memory.add_tool_results(records)
                # Whether to replay this turn's chain-of-thought to the model on
                # the next tool round. Off by default: the model does not need its
                # own prior reasoning as context, and replaying it re-bills those
                # tokens every subsequent round (and persists them in live
                # context). Only providers that REQUIRE same-turn thinking blocks
                # for tool-call validation (e.g. Anthropic extended thinking) need
                # this; set ``replay_reasoning: true`` on that model in config.
                model_cfg = self.config.models.get(usage.get("model_key")) or {}
                replay_reasoning = bool(model_cfg.get("replay_reasoning", False))
                prefix.append(
                    self._assistant_tool_message(
                        tool_calls,
                        decision.reasoning_content if replay_reasoning else None,
                    )
                )
                prefix.extend(self._tool_result_messages(last_tool_results))

                # Threshold-triggered compaction. Caching makes re-sending full
                # tool content cheap, so we don't prune per round; only when one
                # turn's live context grows large do we condense it into a
                # summary and continue as a fresh internal segment. Takes
                # precedence over the round cap while compactions remain.
                est_tokens = len(json.dumps(prefix, ensure_ascii=False, default=str)) // 4
                if (
                    not force_respond
                    and compaction_count < MAX_COMPACTIONS
                    and est_tokens >= COMPACTION_TRIGGER_TOKENS
                ):
                    await _progress(f"Condensing context... [{self.agent_id}]")
                    summary = await self._compact_segment(prefix, run_context, user_input)
                    if summary:
                        compaction_count += 1
                        self.memory.set_context_summary(summary)
                        debug_log(
                            "Compaction",
                            f"Segment {compaction_count}/{MAX_COMPACTIONS}: "
                            f"condensed ~{est_tokens} tok of live context.",
                            extra=f"agent={self.agent_id}",
                        )
                        prefix = self.prompt_assembler.build_prefix_messages(
                            [{"role": "user", "content": user_input}]
                        )
                        context_summary = summary
                        instruction = (
                            "The earlier tool history was condensed into the context "
                            "summary above, which you wrote to be self-sufficient. "
                            "Continue working toward the user's request using that "
                            "summary; only re-read a file if you genuinely failed to "
                            "capture something you need from it."
                        )
                        action_count = 0
                        next_log_extra = "post-compaction segment"
                        continue

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
                    instruction = (
                        "You have reached the maximum number of action attempts. "
                        "Do not call any more tools. Respond now with a summary "
                        "of what happened."
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

        tools_str = ", ".join(dict.fromkeys(round_tools_used)) or "None"

        # The tracker holds this whole exchange's usage (this agent + any
        # sub-agents it delegated to). Tools stay loop-local. cache_read is a
        # subset of input_tokens; cache-write is hidden when 0 (non-Anthropic).
        # The session total is this conversation's running lifetime, persisted
        # per-session in memory (this agent only; sub-agents keep their own).
        session = self.memory.data.get("token_summary", {})
        session_line = (
            f"*Session total: {session.get('input_tokens', 0)} in, "
            f"{session.get('output_tokens', 0)} out*"
        )
        usage_msg = (
            f"\n\n---\n{run_context.usage.render_footer()}\n"
            f"{session_line}\n"
            f"*Tools used: {tools_str}*"
        )
        if is_scheduled:
            job_name = m.get("job_name", "Scheduled task")
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

    async def _compact_segment(
        self, messages: list[dict], run_context: RunContext, user_input: str
    ) -> Optional[str]:
        """Summarize the current live context so the turn can continue clean.

        One forced-text model call (no tools). Returns the summary text, or
        None if the call failed or produced nothing usable. The call's usage is
        tracked, but it is never persisted as conversation history -- only the
        resulting summary is kept, via ``set_context_summary``.
        """
        probe = messages + [{"role": "user", "content": COMPACTION_INSTRUCTION}]
        out = await self.brain.decide(
            probe,
            tool_definitions=[],
            session_id=self.memory.session_id,
            log_extra="compaction",
            usage_tracker=run_context.usage,
        )
        if out.get("error"):
            return None
        return (out["decision"].text_response or "").strip() or None

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
