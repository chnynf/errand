"""Per-exchange run context and cross-agent usage tracking.

Ownership is one-directional, so the dependency line never bends back:

    AgentLoop (root)  ──creates──▶  RunContext { UsageTracker }
                                          ▲           ▲
                          records ────────┘           │
                          (Brain, parent + children)  │
                          reuses ─────────────────────┘
                          (delegated child AgentLoops)

- The root ``AgentLoop`` creates one ``RunContext`` per exchange (one user
  message -> final response) and owns its lifecycle.
- The ``Brain`` is a pure emitter: it calls ``usage.record(...)`` on every
  successful API call. It never creates or aggregates.
- Delegated child loops reuse the parent's ``RunContext`` unchanged, so one
  ``UsageTracker`` aggregates the parent agent and every sub-agent it spawns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from errand.sessions.memory import estimate_cost


@dataclass
class AgentUsage:
    """Accumulated model usage for one agent within a single exchange."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost: float = 0.0
    calls: int = 0
    models: List[str] = field(default_factory=list)


class UsageTracker:
    """Aggregates per-call model usage across one exchange.

    Fed by the Brain (one ``record()`` per successful API call) and owned by
    the root ``AgentLoop``. Keeps a per-agent breakdown so a delegated
    sub-agent's tokens roll up into the same total as the parent that spawned
    it -- fixing the gap where child-session usage was tracked separately and
    never surfaced in the parent's report.
    """

    def __init__(self) -> None:
        self._by_agent: Dict[str, AgentUsage] = {}

    def record(self, usage: Dict[str, Any], *, agent_id: str) -> None:
        """Accumulate one successful API call's usage under ``agent_id``."""
        bucket = self._by_agent.setdefault(agent_id, AgentUsage())
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_tokens", 0) or 0)

        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.cache_read_tokens += cache_read
        bucket.cache_creation_tokens += cache_creation
        bucket.cost += estimate_cost(
            input_tokens, output_tokens, cache_read, usage.get("pricing")
        )
        bucket.calls += 1

        model = usage.get("model")
        if model:
            short = model.split("/")[-1]
            if short not in bucket.models:
                bucket.models.append(short)

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self._by_agent.values())

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self._by_agent.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(a.cache_read_tokens for a in self._by_agent.values())

    @property
    def cache_creation_tokens(self) -> int:
        return sum(a.cache_creation_tokens for a in self._by_agent.values())

    @property
    def cost(self) -> float:
        return sum(a.cost for a in self._by_agent.values())

    @property
    def calls(self) -> int:
        return sum(a.calls for a in self._by_agent.values())

    def models(self) -> List[str]:
        seen: List[str] = []
        for bucket in self._by_agent.values():
            for model in bucket.models:
                if model not in seen:
                    seen.append(model)
        return seen

    def per_agent(self) -> Dict[str, AgentUsage]:
        return dict(self._by_agent)

    def render_footer(self) -> str:
        """Render the per-exchange usage footer (markdown, no leading rule).

        Lists the cross-agent total; when more than one agent contributed
        (i.e. delegation happened), adds an indented per-agent breakdown.
        """
        models_str = ", ".join(self.models()) or "None"
        cache_bits: List[str] = []
        if self.cache_read_tokens:
            cache_bits.append(f"{self.cache_read_tokens} cached")
        if self.cache_creation_tokens:
            cache_bits.append(f"{self.cache_creation_tokens} cache-write")
        cache_str = f" ({', '.join(cache_bits)})" if cache_bits else ""

        lines = [
            f"*Models used: {models_str}*",
            f"*Tokens: {self.input_tokens} in{cache_str}, {self.output_tokens} out*",
        ]
        if len(self._by_agent) > 1:
            for agent_id, bucket in self._by_agent.items():
                lines.append(
                    f"*  └ {agent_id}: {bucket.input_tokens} in, "
                    f"{bucket.output_tokens} out*"
                )
        if self.cost:
            lines.append(f"*Est. cost: ${self.cost:.4f}*")
        return "\n".join(lines)


@dataclass
class RunContext:
    """Cross-cutting state for one exchange, flowing root -> delegated children.

    Created by the root ``AgentLoop``; passed unchanged into child loops via
    delegation. Carries only state that is constant for the whole exchange:
    the shared usage tracker and the origin/destination of the turn.
    (Per-loop state like ``delegation_depth`` is NOT here -- it changes per
    loop and stays on the loop.)
    """

    usage: UsageTracker
    reply_to: Any = None
    source: Optional[str] = None

    @classmethod
    def root(cls, *, reply_to: Any = None, source: Optional[str] = None) -> "RunContext":
        """Create the root context for a new exchange with a fresh tracker."""
        return cls(usage=UsageTracker(), reply_to=reply_to, source=source)
