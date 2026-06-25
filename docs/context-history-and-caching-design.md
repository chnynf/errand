# Context, History & Caching Design

Durable record of how Paw assembles per-call context, what it persists as
history, and how those choices interact with provider prompt caching. Read this
before changing `prompt_assembler.py`, the message-building / tool-round loop in
`agent_loop/loop.py`, `sessions/memory.py` (history + tool-result compaction),
or `process_scheduled_job` in `runtime/app.py`. The *why* here is easy to
accidentally undo, and several of these choices are about token cost, which is
invisible in normal testing.

Provider context: Paw runs primarily on **Gemini** (`gemini-3.5-flash`), which
uses **implicit** (automatic, prefix-based) prompt caching — there is no
`cache_control` breakpoint to set. The Anthropic-style explicit breakpoints in
`providers/litellm.py` are a **no-op for Gemini** and only matter if an
Anthropic model is in the strategy. Design for implicit prefix caching first.

---

## The mental model: one turn, its rounds, and the message tokens

A **turn** = one `process_input` call = one human/scheduled message plus every
assistant/tool round it triggers. Notation used throughout this doc and in
`loop.py` comments:

- **mₖ** — a message from the human/scheduler side (turn k's input).
- **Aₖ** — an assistant message (may carry a tool call *or* be the final text).
- **Tₖ** — the tool-result message answering Aₖ (exists only for tool calls).
- **CTX** — the volatile per-turn context block: a `user`-role message holding
  current time + rolling context summary + session note + instruction. Built
  once per turn.

Each tool round is **one model call**, and each call re-sends the whole message
list. So input tokens are cumulative across rounds and turns — a "simple" task
that takes 6 rounds pays the growing prefix 6 times.

---

## Caching invariant: the stable prefix must grow monotonically

Implicit prefix caching only reuses a prefix the provider has already seen
**byte-identical**. So the golden rule:

> Keep `system prompt + conversation history` as a stable, append-only prefix,
> and keep the single volatile block (CTX) at the **literal tail** of every call.

Any volatile content placed *inside* the prefix becomes a cache barrier:
everything after it is re-billed on the next, differing request.

### Why CTX-at-the-tail matters (the m1/A1/T1 example)

`build_messages` orders messages `system → history → CTX`. The subtle bug we
fixed: the loop built that list **once** and then **appended** each tool round's
`Aₖ, Tₖ` *after* CTX — so from round 2 on, CTX was buried in the middle, not at
the tail.

Buried-CTX request sequences within a turn (the old behavior):

```
R1: [sys, m1, CTX1]
R2: [sys, m1, CTX1, A1, T1]
R3: [sys, m1, CTX1, A1, T1, A2, T2]
```

Within the turn this caches fine (CTX1 sits at a fixed index; rounds append
after it). The damage shows up on the **next** turn, which rebuilds from
persisted history (CTX is never persisted) and wedges a *fresh* CTX2 in:

```
Turn 2 R1: [sys, m1, A1, T1, A2, T2, A3, m2, CTX2]
```

Turn 1's cached sequences all read `[sys, m1, CTX1, …]`; turn 2 reads
`[sys, m1, A1, …]`. They diverge **right after m1** — purely because CTX1 sits
between m1 and A1 in the cached copy. So even though `m1, A1, T1, A2, T2` are
byte-identical across turns, turn 2 can reuse only `[sys, m1]`.

General rule under buried-CTX: **turn N can reuse the cache only up to where
turn N−1's CTX landed** (= end of history when turn N−1 began). Everything turn
N−1 appended, plus turn N's new content, is re-billed.

### The fix

`prompt_assembler.build_prefix_messages(history)` returns just
`[system, *history]`. The loop keeps that as an append-only `prefix`, appends
each `Aₖ, Tₖ` to it, and re-appends a fresh CTX as the last message on **every**
call. Now the cached sequences are `[sys, m1]`, `[sys, m1, A1, T1]`,
`[sys, m1, A1, T1, A2, T2]`, … with CTX always trailing. Turn 2 reuses all of
`[sys … A3]`; only the small trailing CTX is ever re-billed.

Trade-off accepted: CTX (a few hundred tokens) is re-billed once per round
instead of being cached within a turn. That is far cheaper than re-billing the
entire history across turns. `build_messages` is retained (it still composes
`prefix + CTX`) so callers/tests that want the full list keep working.

### Honest limit observed

Even with a clean growing prefix *within* a turn, Gemini held `cache_read` flat
at ~the system-prompt size (~2.3k tokens) and did **not** extend the cache to
the small incremental history — likely an implicit-cache minimum/granularity,
not propagation latency (gaps of 30s+ between rounds still didn't help). So:

- The CTX-at-tail fix is correct and free, but its payoff is **modest for short
  runs** and grows for **long** conversations where the stable suffix past the
  system prompt is itself large.
- Across a multi-hour gap (e.g. nightly jobs) the implicit cache is **cold**
  regardless of message ordering — see "Scheduled jobs" below.

---

## History persistence ≠ what the model is fed

`memory.data["history"]` is **both** the on-disk audit record and the source
`build_history_messages` replays to the model — there is no separate log. But
two compactions already keep replay cheap, which shapes what is and isn't worth
pruning:

1. **Tool results are stored compacted, not raw.** `ToolRegistry.compact_result`
   → `Memory._render_tool_record` persists a one-line preview / `result_ref`
   pointer + `content_chars`, **never the full output**. The full bytes exist
   only in-memory during the live turn (where the model needs them to act).
   - Example: a `read_file` of a 351-char file persists as
     `read_file: result stored as file ref <path> (351 chars)` (~15 tokens), not
     the 351 chars. On replay the model sees the pointer and re-reads on demand
     (now I/O-cached) if it actually needs the content.
2. **Reasoning content is not replayed across turns.** `_ai_message` rebuilds
   assistant messages from `tool_calls` / `text_response` only — it never
   re-emits `reasoning_content`. (It *is* still written to disk; that bloats the
   session file but costs no model tokens.)

**Consequence — decided, do not "optimize" away:** because completed-turn tool
rounds are already ~15-token pointers on replay, **dropping them entirely buys
~200 tokens/turn and reintroduces correctness risk** (lost detail a later turn
might need; reliance on the user-facing final text being self-sufficient). So we
**keep them**. The lever that *does* matter is caching (above), not pruning.

Rejected alternative: an extra model call to summarize already-compacted rounds
into one synthetic "tool context." It adds a call to compress ~15-token entries
and is redundant with the existing oversized-turn compaction. Net-negative.

### History window

`build_history_messages(recent_n=20, chunk=8)` windows to the last ~20–27
entries, with the start **quantized to `chunk`** so it holds steady for several
turns instead of sliding one entry per turn (a per-turn slide would shift the
whole prefix and defeat prefix caching). The on-disk file still keeps everything.

### Oversized-turn compaction

When one turn's live context passes `COMPACTION_TRIGGER_TOKENS` (20k), the loop
does a single forced-text model call to condense the turn into a self-sufficient
summary, then continues as a fresh internal segment (still one reply to the
user). The summary is written to be usable **alone** — the post-compaction
instruction tells the model to continue from it and only re-read if it genuinely
missed something (avoiding a re-read loop).

---

## Reasoning content: drop from replay by default, conditionally

The model does not need its own prior chain-of-thought as context to continue a
tool loop, and replaying it re-bills those tokens every subsequent round (and
inflates later-round input — observed: a 4k-token reasoning blob carried forward
turned a follow-up call's input from ~5.6k to ~9.7k).

**Decision:** in `loop.py`, the assistant tool message carries
`reasoning_content` **only when the active model's config sets
`replay_reasoning: true`** (looked up via `self.config.models[model_key]`).
Default is **drop**.

**Caveat encoded in the flag:** Anthropic extended-thinking-with-tools *requires*
the same-turn thinking blocks to be passed back or tool-call signature
validation fails. Such models must set `replay_reasoning: true`. Gemini /
DeepSeek-style reasoning needs no replay, so the default is safe for them.

Separate lever (config, not code): for mechanical agents (e.g. a notes triage
SOP) the **output** reasoning is the bigger cost — set `reasoning_effort:
"minimal"`/`"low"` on that agent. `reasoning_effort` is a per-agent knob
normalized across providers by LiteLLM.

---

## Scheduled jobs run in a fresh session per fire

A scheduled job's session id is stable: `scheduled:{job_id}` (`scheduler/store.py`).
So without intervention, every nightly fire **reuses the same session** and
replays prior fires' history — on *every* call of the new run, and **uncached**
(the implicit cache is cold after a multi-hour gap). The history window caps this
at ~2–3 nights, but the session file also grows unbounded.

**Decision:** `process_scheduled_job` (`runtime/app.py`) archives-and-starts-new
**before** each fire — the internal equivalent of `/new` then the task — reusing
the same `session_manager.archive(session_id, start_new=True, agent_id=...)` call
the `/new` control command uses.

Rationale and properties:

- A scheduled job's **durable state lives in the files it edits**
  (`inbox.md`, `reviews/`, subject files), *not* in the chat transcript. Carrying
  the transcript forward adds little and costs tokens every fire.
- **Same session id / same channel** — delivery is unaffected (it routes via
  `context_id`/`delivery_session_id`).
- **Interactive follow-ups** the user sends after a fire (e.g. "Please
  implement") land in the same session and continue normally; they're archived at
  the *next* fire (~24h later), not mid-conversation.
- **Audit preserved** — prior runs move to `_data/archive/`, exactly like `/new`.
- First-ever fire is a safe no-op archive, then a fresh run.

This reset is scoped to scheduled jobs; interactive sessions are untouched.

---

## Quick checklist before touching this area

- Adding per-turn context? It goes in `build_context_messages` (the CTX tail),
  never wedged into the prefix.
- Appending to the live message list mid-turn? Append to `prefix`, not to a list
  that already has CTX at the end.
- Tempted to prune tool rounds from history to save tokens? They're already
  ~15-token pointers; fix caching instead.
- Adding a thinking/Anthropic model? Set `replay_reasoning: true` for it.
- A new kind of stateless recurring job? Confirm it should reset per fire like
  scheduled jobs (state in files, not transcript).
