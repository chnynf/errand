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
import time
from typing import List, Optional

from paw.agent_loop.prompt_assembler import PromptAssembler
from paw.brain import Brain
from paw.config import PawConfig, load_paw_config
from paw.wire_types import (
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

# The current exchange is kept at full fidelity, entry by entry, with no
# intra-turn compaction: full tool results stay in the append-only
# ``current_exchange`` section so it grows monotonically and the
# system+history+current_exchange prefix stays cacheable. Trimming is deferred to
# step-out (tool results persist as compact refs; old exchanges leave the
# history window). The only intra-exchange guard is an overflow stop: if the
# live context approaches the model's context window, the loop forces a final
# response rather than taking more tool calls, so a runaway turn can't exceed
# the window. Set below the smallest supported window, leaving headroom for the
# forced final call's own output.
CONTEXT_OVERFLOW_TOKENS = 150_000

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
        force_respond = False
        hit_tool_cap = False
        suppress_usage_footer = bool(m.get("suppress_usage_footer"))

        round_tools_used: list[str] = []
        last_tool_results: list[ToolResult] = []
        scheduled_messages: list[str] = []
        next_log_extra: Optional[str] = None
        # Per-round (effective calls, results), collected for the step-out fold.
        tool_rounds: list[tuple[list[ToolCall], list[ToolResult]]] = []

        # History = prior exchanges, compacted (from memory). The CURRENT
        # exchange is assembled live at full fidelity in ``current_exchange``,
        # kept separate from history: it starts with the user's input and gains
        # one assistant message + its tool results per round (appended below).
        # Each model call is ``system + history + current_exchange + context``
        # (see PromptAssembler.build_prompt); the first three form the stable,
        # append-only prefix that stays prefix-cacheable across rounds, while the
        # volatile context (time, rolling summary, instruction) is re-appended
        # last so it never breaks that prefix.
        # Roll old exchanges into the distant (Q&A-compacted) section before
        # building history: a >=2h silence gap ends the previous conversation
        # (everything rolls), and token-budget overflow migrates the oldest
        # group (see Memory.roll_distant). A gap roll adds a session note to
        # the volatile context so the model treats this as a fresh
        # conversation; the note stays constant for the whole exchange.
        now = time.time()
        gap_seconds = now - (self.memory.last_activity_at() or now)
        session_note: Optional[str] = None
        if self.memory.roll_distant(now):
            gap_hours = gap_seconds / 3600
            session_note = (
                f"New conversation: previous activity was ~{gap_hours:.1f}h ago. "
                "Earlier chats appear above in condensed form; treat this "
                "message as the start of a fresh conversation."
            )
        history_messages = self.memory.build_history_messages()
        current_exchange: list[dict] = [self._user_message(user_input)]
        context_summary = self.memory.data.get("context_summary")
        # No default instruction: the system prompt already states the model's
        # job and response format. Only real control directives (the forced
        # final response below) set this, keeping the volatile block minimal.
        instruction: Optional[str] = None

        # Session-total baseline, captured before this exchange adds anything.
        # At step-out the root rebuilds the session total as baseline + this
        # exchange's full cross-agent usage (see below), so it always includes
        # delegated work and can never read lower than this exchange's footer.
        base_ts = self.memory.data.get("token_summary") or {}
        base_input = base_ts.get("input_tokens", 0) or 0
        base_output = base_ts.get("output_tokens", 0) or 0
        base_cache_read = base_ts.get("cache_read_tokens", 0) or 0

        while True:
            await _progress(f"Thinking... [{self.agent_id}]")
            messages = self.prompt_assembler.build_prompt(
                history_messages,
                current_exchange,
                context_summary=context_summary,
                instruction=instruction,
                session_note=session_note,
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
                    # Effective (unwrapped) calls, kept for the step-out fold.
                    effective_calls = [
                        ToolCall(id=tc.id, name=eff[tc.id][0], params=eff[tc.id][1])
                        for tc in tool_calls
                    ]
                    tool_rounds.append((effective_calls, last_tool_results))
                # Whether to replay this turn's chain-of-thought to the model on
                # the next tool round. Off by default: the model does not need its
                # own prior reasoning as context, and replaying it re-bills those
                # tokens every subsequent round (and persists them in live
                # context). Only providers that REQUIRE same-turn thinking blocks
                # for tool-call validation (e.g. Anthropic extended thinking) need
                # this; set ``replay_reasoning: true`` on that model in config.
                model_cfg = self.config.models.get(usage.get("model_key")) or {}
                replay_reasoning = bool(model_cfg.get("replay_reasoning", False))
                # Append this round to the live current-exchange section (full
                # fidelity). Nothing is persisted mid-exchange -- the whole
                # exchange is folded once at step-out (see _fold_exchange).
                current_exchange.append(
                    self._assistant_message(
                        tool_calls,
                        decision.reasoning_content if replay_reasoning else None,
                    )
                )
                current_exchange.extend(self._tool_messages(last_tool_results))

                # No intra-exchange compaction: the current exchange grows at
                # full fidelity and prefix caching keeps re-sending it cheap.
                # Force a final response when the round cap is reached, or as an
                # overflow guard when live context nears the context window (so a
                # runaway turn can't exceed it -- the threshold leaves headroom
                # for the forced final call's own input+output).
                est_tokens = (
                    len(json.dumps(history_messages, ensure_ascii=False, default=str))
                    + len(json.dumps(current_exchange, ensure_ascii=False, default=str))
                ) // 4
                overflow = est_tokens >= CONTEXT_OVERFLOW_TOKENS
                max_rounds = self.agent_spec.max_tool_rounds
                if overflow or action_count >= max_rounds:
                    reason = (
                        f"live context ~{est_tokens} tok near the window limit"
                        if overflow
                        else f"reached {action_count}/{max_rounds} tool rounds"
                    )
                    debug_log(
                        "Action Cap",
                        f"{reason}; requesting a final text response with no more tool calls.",
                        extra=f"agent={self.agent_id}",
                    )
                    instruction = (
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

        # Step-out: fold the finished exchange (compacted) into the durable log.
        # Uses the core response text, before display banners/footers are added.
        self.memory.add_exchange(
            self._fold_exchange(user_input, tool_rounds, final_response)
        )

        if hit_tool_cap and not suppress_usage_footer:
            final_response = (
                f"[Reached action limit after {action_count} tool rounds — "
                f"response may be incomplete]\n\n{final_response}"
            )

        # Count each tool's invocations, preserving first-seen order, so the
        # footer shows how many times each tool ran (e.g. "read_file (3)").
        tool_counts: dict[str, int] = {}
        for name in round_tools_used:
            tool_counts[name] = tool_counts.get(name, 0) + 1
        tools_str = ", ".join(f"{name} ({n})" for name, n in tool_counts.items()) or "None"

        # Rebuild this session's lifetime total as the pre-exchange baseline plus
        # this exchange's FULL cross-agent usage (the same total the footer
        # reports). The per-call updates during the loop only captured this
        # agent's own direct calls; a delegated sub-agent's tokens live in the
        # shared tracker but never hit this session via those updates -- and when
        # a sub-agent runs under the SAME agent_id as its parent they can't be
        # told apart by id at all. Setting baseline + tracker total sidesteps
        # that: it counts every agent exactly once, so the session total can
        # never read lower than the exchange footer. Only the user-facing root
        # does this; sub-agent loops keep their own per-session totals.
        if not is_subagent:
            self.memory.set_token_usage(
                input_tokens=base_input + run_context.usage.input_tokens,
                output_tokens=base_output + run_context.usage.output_tokens,
                cache_read_tokens=base_cache_read + run_context.usage.cache_read_tokens,
            )

        # The tracker holds this whole exchange's usage (this agent + any
        # sub-agents it delegated to). Tools stay loop-local. cache_read is a
        # subset of input_tokens; cache-write is hidden when 0 (non-Anthropic).
        # The session total is this conversation's running lifetime, persisted
        # per-session in memory, and now includes delegated sub-agent tokens.
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

    def _fold_exchange(
        self,
        user_input: str,
        tool_rounds: list[tuple[list[ToolCall], list[ToolResult]]],
        final_text: str,
    ) -> list[dict]:
        """Fold the finished exchange into compact chat messages for the log.

        Shape per exchange: the user message; then per round one skeleton
        assistant anchor (real tool names, synthetic ids, empty args -- just
        enough to keep every tool message legally paired) followed by its
        compacted tool results; then the final assistant text. The real call
        args live inside each tool message's compact text (capped), not in the
        anchor, so a large write payload can't bloat history.
        """
        messages = [self._user_message(user_input)]
        seq = 0
        for calls, results in tool_rounds:
            fold_ids: dict[str, str] = {}
            anchor: list[dict] = []
            for call in calls:
                seq += 1
                fold_ids[call.id] = f"h{seq}"
                anchor.append(
                    {
                        "id": f"h{seq}",
                        "type": "function",
                        "function": {"name": call.name, "arguments": "{}"},
                    }
                )
            messages.append({"role": "assistant", "tool_calls": anchor})
            calls_by_id = {c.id: c for c in calls}
            for result in results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": fold_ids[result.tool_call_id],
                        "content": self.tool_registry.compact_interaction(
                            calls_by_id.get(result.tool_call_id), result
                        ),
                    }
                )
        messages.append({"role": "assistant", "content": final_text})
        return messages

    # -- Current-exchange message builders ---------------------------------
    # These produce the live, full-fidelity chat messages for the current
    # exchange, in OpenAI chat-role vocabulary (system/user/assistant/tool) --
    # the same vocabulary the durable log stores, so there is no translation
    # anywhere: the log IS the replay format (see _fold_exchange above).

    @staticmethod
    def _user_message(text: str) -> dict:
        return {"role": "user", "content": str(text)}

    @staticmethod
    def _assistant_message(tool_calls: List[ToolCall], reasoning_content: Optional[str] = None) -> dict:
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
    def _tool_messages(tool_results: List[ToolResult]) -> list[dict]:
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
