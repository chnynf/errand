# Prompt & Tooling Design Decisions

Durable record of the owner's preferences and the design choices they imply for
how Paw assembles system prompts and exposes tools. Read this before changing
`agent.md`, the prompt assembler, the tool registry, or the file tools — the
*why* here is easy to accidentally undo.

> Companion doc: [`context-history-and-caching-design.md`](./context-history-and-caching-design.md)
> covers per-call context assembly, history persistence/compaction, prompt
> caching (the CTX-at-tail rule), reasoning-content replay, and scheduled-job
> session resets.

## Principles

- **Token efficiency is a first-class concern.** Almost every token of a "simple
  hello" is fixed overhead (system prompt + tool schemas), so trimming the
  always-loaded prompt has outsized leverage. Remove redundancy aggressively;
  prefer on-demand loading over always-loaded text.
- **Each component owns its own prompt contribution.** The runtime frame, the
  knowledge base, and the tools component each own their slice. Don't render one
  component's content from another.
- **No redundancy across surfaces.** If the tool catalog, a docstring, the kb, or
  a runtime contract already states something, don't restate it elsewhere.

## Prompt layering & ownership

The system prompt is assembled from layers with distinct owners:

| Layer | Owner | Lives in |
| --- | --- | --- |
| Runtime frame (identity, response-format contract) | runtime | `paw/agent_loop/agent.md` |
| Shared soul (cross-agent behavior/principles) | knowledge base | `~/Documents/kb/SOUL.md` |
| Notes router | knowledge base | `~/Documents/kb/notes/INDEX.md` |
| Agent profile (role, delegation, SOPs) | knowledge base | `~/Documents/kb/<agent>/INDEX.md` |
| Tool catalog + cross-tool guidance | tools component | `paw/tools/registry.py` (`tool_summary`) |
| Per-tool manual + file scopes | tools component | tool docstrings, surfaced via `tool_manual` |

### `agent.md` is a minimal frame, not a settings file

- **Behavior, principles, and role-specific guidance belong in the kb** (SOUL /
  profile), **not** in `agent.md`. `agent.md` holds only: the "running inside
  Paw" intro, the three include placeholders (`SHARED_SOUL`,
  `SHARED_NOTES_INDEX`, `AGENT_PROFILE` — structurally required), and the
  response-format contract.
- The **response-format `\n---\nContext:` trailer stays in `agent.md`** because
  it is a runtime contract parsed by `paw/brain/providers/litellm.py`
  (`_parse_text_for_context_summary`), not a personal preference. Do not move it
  into the kb — that would couple the personal kb to Paw's parser.
- **Delegation guidance is per-agent**, so it lives in the agent profile (e.g.
  `generalist/INDEX.md`), not the shared frame. Only agents with `can_delegate`
  need it. Caveat: the shared frame no longer provides a delegation fallback, so
  a new delegating agent needs its own delegation lines in its profile.

## Tool exposure architecture

Two surfaces, both owned by the tools component:

1. **`tool_summary` (always loaded)** = a high-level catalog (one
   `name(args): purpose` line per non-core tool) + cross-tool guidance. Keep the
   **full** catalog — at the current tool count the dump is cheap (~hundreds of
   tokens) and lets the model plan in one shot. A search-on-demand index is only
   worth building when the catalog gets large (~50+ tools).
2. **`tool_manual(name)` (on demand)** = full signature + docstring for one tool.
   This *is* the "search tool usage on demand" mechanism. Its result returns as a
   `role:"tool"` message the model reads on its next turn.

`call_tool(name, args)` dispatches to any catalog tool.

### Default (native, full-schema) tools

The core always-callable set is the high-frequency work — **`invoke_agent`,
`invoke_external_agent`, `read_file`** — plus the meta-tools `call_tool` and
`tool_manual`. Everything else (search, writes, scheduling, email, …) is in the
catalog and reached via `call_tool`. Configured in `registry.py:_DEFAULT_NATIVE`.

### Tools are plain Python + docstrings

- **No prompt API on tools.** A per-tool prompt hook is rejected as a maintenance
  burden. Each tool is a plain function; its **docstring is the single source**
  for both its catalog one-liner (first line) and its full manual.
- The **tools component (registry) takes over** from there: it extracts the
  catalog and serves manuals. Anything config-derived (e.g. file scopes) is
  rendered by the tools component, not by individual tools and not by the
  agent-loop.

## File scopes

- **Keep scopes.** They are a blast-radius limiter for an LLM-driven,
  externally-reachable agent (Discord/WeChat/web + web research = injection
  vectors; it holds `send_email`/`delete_file`). The escape hatch for new
  locations is a deliberate, human edit to `config.json` — not the LLM deciding.
- **Surface scopes as locations only**, in the file tool's **manual** (not the
  high-level summary). `registry.manual()` appends the scope locations whenever a
  tool takes a `scope` argument. Rendering lives in `paw/tools/files.py`
  (`_scope_locations`), which the file tools own.
- **Do not expose the permission matrix** (`true`/`ask`/`false`) to the model.
  The harness enforces run / approve / deny at call time, so pre-announcing
  permissions is dead weight. (`false` is per-(scope×operation), so it can't be
  expressed as "hide the tool"; in practice nothing is `false` and a blocked op
  just returns an error.)

## Open follow-ups

- **`run_cli` undercuts the file sandbox.** It is effectively unrestricted shell,
  so the file-scope confinement is partly theater. To make scopes meaningful,
  constrain `run_cli` (allowlist or working-dir jail). This is the higher-value
  hardening, not more prompt trimming.
- **Pre-existing linter nits in `registry.py`** (left untouched): unused `params`
  in `_default_compactor`, an unreachable defensive `isinstance` in
  `validate_args`, and `manual()` showing the internal `_context` param in the
  signature.
