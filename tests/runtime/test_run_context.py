"""Tests for the per-exchange UsageTracker / RunContext.

Covers the core fix: a single tracker aggregates usage across the parent
agent and any sub-agents, with a per-agent breakdown, so delegated work no
longer hides in a separate per-session total.
"""

from errand.runtime.run_context import RunContext, UsageTracker

GEMINI_PRICING = {"input": 0.30, "output": 2.50, "cache_read": 0.075}


def _usage(inp, out, *, cache_read=0, model="gemini/gemini-3.5-flash", pricing=GEMINI_PRICING):
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": 0,
        "model": model,
        "pricing": pricing,
    }


def test_record_accumulates_totals_for_one_agent():
    t = UsageTracker()
    t.record(_usage(100, 10), agent_id="generalist")
    t.record(_usage(200, 20, cache_read=50), agent_id="generalist")

    assert t.input_tokens == 300
    assert t.output_tokens == 30
    assert t.cache_read_tokens == 50
    assert t.calls == 2
    assert t.models() == ["gemini-3.5-flash"]


def test_rollup_across_parent_and_subagent():
    """The bug fix: a delegated sub-agent's tokens roll into the same total."""
    t = UsageTracker()
    t.record(_usage(347_000, 800), agent_id="generalist")
    t.record(_usage(151_000, 1_200), agent_id="notes-organizer")

    assert t.input_tokens == 498_000  # not just the generalist's 347k
    assert t.output_tokens == 2_000
    breakdown = t.per_agent()
    assert breakdown["generalist"].input_tokens == 347_000
    assert breakdown["notes-organizer"].input_tokens == 151_000


def test_cost_splits_cached_and_fresh_input():
    t = UsageTracker()
    t.record(_usage(1_000, 100, cache_read=400), agent_id="generalist")
    # fresh 600 @0.30 + cached 400 @0.075 + out 100 @2.50, all per-1M
    expected = (600 / 1e6) * 0.30 + (400 / 1e6) * 0.075 + (100 / 1e6) * 2.50
    assert abs(t.cost - expected) < 1e-12


def test_footer_shows_breakdown_only_when_multiple_agents():
    solo = UsageTracker()
    solo.record(_usage(100, 10), agent_id="generalist")
    assert "└" not in solo.render_footer()

    multi = UsageTracker()
    multi.record(_usage(100, 10), agent_id="generalist")
    multi.record(_usage(50, 5), agent_id="notes-organizer")
    footer = multi.render_footer()
    assert "notes-organizer" in footer
    assert "└" in footer


def test_missing_pricing_contributes_zero_cost():
    t = UsageTracker()
    t.record(_usage(1_000, 100, pricing=None), agent_id="generalist")
    assert t.cost == 0.0
    assert t.input_tokens == 1_000


def test_root_factory_creates_fresh_tracker():
    ctx = RunContext.root(reply_to="rt", source="discord")
    assert isinstance(ctx.usage, UsageTracker)
    assert ctx.reply_to == "rt"
    assert ctx.source == "discord"
    assert ctx.usage.input_tokens == 0
