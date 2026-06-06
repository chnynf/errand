from errand.contracts.types import ToolCall, ToolResult
from errand.sessions.memory import (
    IDLE_BOUNDARY_NOTE,
    Memory,
    _safe_stem,
    estimate_cost,
)


def test_prompt_context_uses_visible_conversation_not_persisted_file_contents(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    memory.add_history("user", "Read the profile.")
    memory.add_history(
        "ai",
        {
            "tool_calls": [
                {
                    "id": "tc-1",
                    "name": "read_file",
                    "params": {"path": "INDEX.md", "scope": "kb"},
                }
            ],
            "text_response": None,
        },
    )
    memory.add_tool_results(
        [
            ToolCall(
                id="tc-1",
                name="read_file",
                params={"path": "INDEX.md", "scope": "kb"},
            )
        ],
        [
            ToolResult(
                tool_call_id="tc-1",
                name="read_file",
                content="# Agent KB\nsecret soul",
            )
        ],
    )
    memory.add_history(
        "ai",
        {
            "tool_calls": [],
            "text_response": "I loaded the profile.",
            "context_summary": "Profile loaded from INDEX.md.",
        },
    )
    memory.set_context_summary("Profile loaded from INDEX.md.")

    context, instruction = memory.get_formatted_context()

    assert "CONTEXT SUMMARY:\nProfile loaded from INDEX.md." in context
    assert "USER: Read the profile." in context
    assert "AI: I loaded the profile." in context
    assert "# Agent KB" not in context
    assert "secret soul" not in context
    assert "TOOL RESULT" not in context
    assert instruction == (
        "Analyze the user's input. Decide whether to call a tool or respond directly."
    )

    tool_entry = next(entry for entry in memory.data["history"] if entry["role"] == "tool")
    stored_result = tool_entry["content"][0]
    assert stored_result["result_ref"] == {
        "type": "file",
        "scope": "kb",
        "path": "INDEX.md",
    }
    assert "secret soul" not in str(stored_result)


def test_working_trace_is_available_for_current_turn_only(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")
    memory.add_history("user", "Read the profile.")

    context, instruction = memory.get_formatted_context(
        working_trace="Tool result: read_file(path='INDEX.md')\n# Agent KB\nsecret soul"
    )

    assert "CURRENT TURN TRACE:" in context
    assert "# Agent KB" in context
    assert "secret soul" in context
    assert instruction.startswith("The previous action has completed.")


def test_session_note_is_sent_as_runtime_context(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    memory.add_session_note(IDLE_BOUNDARY_NOTE)
    context, _ = memory.get_formatted_context()

    assert f"SESSION NOTE:\n{IDLE_BOUNDARY_NOTE}" in context
    assert "USER:" not in context


def test_session_id_with_illegal_filename_chars_is_archivable(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    session_id = "wechat:o9cq80wyNyZ@im.wechat"
    memory = Memory(session_id)
    assert ":" not in memory._file_stem
    assert memory.session_id == session_id


async def test_archive_session_with_illegal_chars(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("wechat:o9cq80wyNyZ@im.wechat")
    memory.add_history("user", "hi")
    await memory.save_session()

    await memory.archive_session(start_new=True)

    assert memory.data["history"] == []
    archives = list((tmp_path / "archive").iterdir())
    assert len(archives) == 1
    assert ":" not in archives[0].name


def test_safe_stem_replaces_windows_illegal_chars():
    assert _safe_stem('a:b/c\\d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_estimate_cost_prices_cache_hits_cheaper():
    pricing = {"input": 0.14, "output": 0.28, "cache_read": 0.0028}
    # 1,000,000 input of which 800,000 are cache hits, 500,000 output.
    cost = estimate_cost(1_000_000, 500_000, 800_000, pricing)
    expected = (
        (200_000 / 1_000_000) * 0.14   # fresh input
        + (800_000 / 1_000_000) * 0.0028  # cached input
        + (500_000 / 1_000_000) * 0.28    # output
    )
    assert cost == expected


def test_estimate_cost_zero_when_no_pricing():
    assert estimate_cost(1000, 1000, 0, None) == 0.0


def test_estimate_cost_defaults_cache_rate_to_input_when_absent():
    pricing = {"input": 1.0, "output": 2.0}
    # No cache_read configured: cached tokens fall back to the input rate.
    cost = estimate_cost(1_000_000, 0, 400_000, pricing)
    assert cost == 1.0


def test_update_token_usage_accumulates_counts_and_cost(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("cost-session")

    memory.update_token_usage(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 500_000,
            "cache_read_tokens": 800_000,
            "pricing": {"input": 0.14, "output": 0.28, "cache_read": 0.0028},
        }
    )
    memory.update_token_usage(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "pricing": None,  # unpriced model contributes 0 cost
        }
    )

    summary = memory.data["token_summary"]
    assert summary["input_tokens"] == 1_000_100
    assert summary["output_tokens"] == 500_050
    assert summary["cache_read_tokens"] == 800_000
    expected_cost = estimate_cost(
        1_000_000, 500_000, 800_000, {"input": 0.14, "output": 0.28, "cache_read": 0.0028}
    )
    assert summary["total_cost"] == expected_cost
