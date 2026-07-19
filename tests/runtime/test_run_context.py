"""Tests for the per-exchange UsageTracker / RunContext.

Covers the core fix: a single tracker aggregates usage across the parent
agent and any sub-agents, with a per-agent breakdown, so delegated work no
longer hides in a separate per-session total.
"""

from paw.runtime.run_context import RunContext, UsageTracker

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


def test_footer_reports_ai_call_count():
    t = UsageTracker()
    t.record(_usage(100, 10), agent_id="generalist")
    t.record(_usage(200, 20), agent_id="generalist")
    assert "*AI calls: 2*" in t.render_footer()


def test_footer_breakdown_includes_per_agent_calls():
    t = UsageTracker()
    t.record(_usage(100, 10), agent_id="generalist")
    t.record(_usage(50, 5), agent_id="notes-organizer")
    t.record(_usage(30, 3), agent_id="notes-organizer")
    footer = t.render_footer()
    assert "notes-organizer: 80 in, 8 out, 2 calls" in footer


def test_cache_creation_tokens_counted_without_cost():
    """Anthropic-style cache-write tokens are still COUNTED (model-agnostic),
    even though Paw no longer prices them."""
    t = UsageTracker()
    t.record(
        {
            "input_tokens": 1_000,
            "output_tokens": 200,
            "cache_read_tokens": 400,
            "cache_creation_tokens": 500,
            "model": "anthropic/claude-x",
        },
        agent_id="generalist",
    )
    assert t.cache_creation_tokens == 500
    assert t.cache_read_tokens == 400
    assert t.input_tokens == 1_000


def test_root_factory_creates_fresh_tracker():
    ctx = RunContext.root(reply_to="rt", source="discord")
    assert isinstance(ctx.usage, UsageTracker)
    assert ctx.reply_to == "rt"
    assert ctx.source == "discord"
    assert ctx.usage.input_tokens == 0


def test_validation_catches_tracked_and_in_footer():
    t = UsageTracker()
    t.record(_usage(100, 10), agent_id="generalist")
    assert t.validation_catches == 0
    assert "Validation catches" not in t.render_footer()  # hidden when zero

    t.record_validation_catch()
    t.record_validation_catch()
    assert t.validation_catches == 2
    assert "*Validation catches: 2*" in t.render_footer()
