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
assistant/tool round it triggers. (The code calls this an **exchange** —
`current_exchange`, `add_exchange`, `_fold_exchange`; same thing.) Notation
used throughout this doc and in `loop.py` comments:

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

`prompt_assembler.build_prompt` assembles every call as four explicit sections,
in order: `system → history → current_exchange → CTX`. The loop appends each
round's `Aₖ, Tₖ` to `current_exchange`, and `build_prompt` re-appends a fresh
CTX as the last message on **every** call. Now the cached sequences are
`[sys, m1]`, `[sys, m1, A1, T1]`, `[sys, m1, A1, T1, A2, T2]`, … with CTX
always trailing. Turn 2 reuses all of `[sys … A3]`; only the small trailing CTX
is ever re-billed.

Trade-off accepted: CTX (a few hundred tokens) is re-billed once per round
instead of being cached within a turn. That is far cheaper than re-billing the
entire history across turns.

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

`memory.data` is **both** the on-disk audit record and the source
`build_history_messages` replays to the model — there is no separate log, and
the log stores provider-ready chat messages, so replay is a plain slice with no
translation layer. The turn lives at full fidelity only while it runs; it is
**folded once at step-out** (`AgentLoop._fold_exchange` → `Memory.add_exchange`):

1. **Tool rounds fold to skeleton anchors + compacted results.** Per round, one
   assistant anchor (real tool names, synthetic ids, empty args — just enough
   to keep every tool message legally paired) followed by one compacted tool
   message per result (`ToolRegistry.compact_interaction`: the call with capped
   arg values + a capped result preview and total size). The full bytes exist
   only in the live `current_exchange` during the turn, **never on disk**.
   - Example: a `read_file` of a 2300-char file persists as
     `tool call: read_file(path='…')\ntool result: <first 500 chars>… [2300
     chars total]`, not the 2300 chars. On replay the model re-reads on demand
     if it actually needs the content — files themselves are the durable memory.
2. **Reasoning content is not persisted or replayed across turns.**
   `_fold_exchange` keeps only tool anchors and the final text. (Within a turn,
   see the `replay_reasoning` flag below.)

**Consequence — decided, do not "optimize" away:** because folded tool rounds
are already compact on replay, **dropping them entirely buys little and
reintroduces correctness risk** (lost detail a later turn might need; reliance
on the user-facing final text being self-sufficient). So recent history keeps
them. The lever that *does* matter is caching (above) — and, for *old*
exchanges, the distant roll below.

Rejected alternative: an extra model call to summarize already-compacted rounds
into one synthetic "tool context." It adds a call to compress already-small
entries; the distant section (below) gets the same effect for free by keeping
only what the fold already stores verbatim. Net-negative.

### History window: two resolutions, distant + recent

**Recent** replay is a token-budgeted slice of the log
(`_HISTORY_TOKEN_BUDGET`, 10K): newest whole exchanges, cut only on exchange
boundaries so anchor/tool pairs stay intact. When over budget, the oldest
exchanges leave in **groups of 3** (`_HISTORY_DROP_EXCHANGES`), always keeping
at least one — the group drop keeps the window start steady for several turns
instead of sliding every turn (a per-turn slide would shift the whole prefix
and defeat prefix caching).

**Distant** is where those exchanges go instead of vanishing: compacted to just
the user text + final assistant text (tool anchors and results drop together,
keeping every pair a legal message sequence), replayed *before* recent history.
`Memory.roll_distant`, called at exchange start, migrates on whichever trigger
fires first:

- **Silence gap ≥ 2h** (`_DISTANT_GAP_SECONDS`) — the previous conversation is
  over: *everything* rolls to distant, and the loop puts a `SESSION NOTE` in
  CTX telling the model to treat the message as a fresh conversation. This is
  the forever-session fix (WeChat): yesterday's thread stops replaying as if
  mid-flight, without losing what was said.
- **Token budget overflow** — the group-drop above, routed into distant.

Distant is capped at 2K chars of content (`_DISTANT_MAX_CHARS`); when
exceeded, the oldest whole Q&A pairs drop down to 1K (`_DISTANT_KEEP_CHARS`) —
trimming past the cap in one step keeps the section byte-stable for many
exchanges afterward, the same stepping philosophy as the group drop.

Cache accounting for the roll: distant sits early in the prefix, so changing it
invalidates everything after — but the gap roll happens only after ≥2h of
silence, when the implicit cache is cold anyway (free), and the budget roll
moves distant's young edge and recent's old edge in the **same event** (one
invalidation, not two).

### Oversized-turn guard

There is **no intra-turn compaction**: the current exchange grows append-only
at full fidelity, and prefix caching keeps re-sending it cheap. Two guards
bound a runaway turn, both forcing a final text response with no more tools:
the per-agent tool-round cap, and an overflow stop when the live context nears
the model's window (`CONTEXT_OVERFLOW_TOKENS`, 150k — below the smallest
supported window, leaving headroom for the forced final call itself).

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

## Scheduled jobs: a fresh session and channel per fire, delivered via Discord

Earlier design: a scheduled job's session id was stable (`scheduled:{job_id}`),
delivered back to whichever channel/interface a stored `delivery_session_id`
pointed at. Two problems fell out of that in practice:

- `delivery_session_id` was captured from whatever the *current* session
  happened to be when `schedule_message` was called — if that session was
  itself a scheduled job's ephemeral session (e.g. scheduling one job from
  inside another job's delivery thread), the new job inherited a
  non-routable id and silently fell through to fallback.
- Reusing the same session id across fires meant an in-between interactive
  reply (e.g. approving a proposal) could be archived out from under the user
  by the *next* fire before they got to it, and Discord had to reverse-scan a
  channel↔job mapping to route replies back correctly.
- Not every interface can even receive a proactive push — WeChat, for
  instance, can only reply to an inbound message, so a job scheduled from
  WeChat had no delivery path at all.

**Decision:** `jobs.json` stores only the originating `interface` name (for a
support check), no session/channel identity at all. Each fire
(`SchedulerService.run_tick`) mints a brand-new session id,
`scheduled-{job_id}-{epoch}`, and runs the job under it — there is nothing to
archive, since the id has never been used before. Delivery
(`PawApp.deliver_scheduled_result`) always goes through Discord: it creates a
new channel named after that same session id and binds `channel_id ->
session_id` directly (`DiscordInterface._bind_channel_to_session`), so a
later reply in that channel resolves straight back to the run that produced
it via `_session_id_for_channel` — a plain forward lookup, not a scan. If the
job's origin `interface` isn't `"discord"`, or Discord delivery fails, the
result routes through `FallbackDelivery` instead, with a note that the origin
interface wasn't supported. If a job produces no text response at all, a
fixed `"Scheduled job {name} finished."` is delivered instead of nothing.

This keeps a job's durable state where it already lived — the files it
edits — while making delivery unconditional and routing self-correcting
(every fire gets an unambiguous, never-reused destination).

---

## Quick checklist before touching this area

- Adding per-turn context? It goes in `build_context_messages` (the CTX tail),
  never wedged into the prefix. The gap-roll `SESSION NOTE` follows this rule.
- Appending to the live message list mid-turn? Append to `current_exchange` and
  let `build_prompt` re-append CTX last — never to a list that already has CTX
  at the end.
- Tempted to prune tool rounds from recent history to save tokens? They're
  already folded compact; old exchanges already roll into the distant Q&A
  section. Fix caching instead.
- Touching `roll_distant` / the window constants? Keep drops in groups and trims
  past the cap — per-turn slides defeat prefix caching.
- Adding a thinking/Anthropic model? Set `replay_reasoning: true` for it.
- A new kind of stateless recurring job? Confirm it should reset per fire like
  scheduled jobs (state in files, not transcript).
