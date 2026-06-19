from errand.contracts.types import ToolCall, ToolResult
from errand.sessions.memory import (
    IDLE_BOUNDARY_NOTE,
    Memory,
    _safe_stem,
)
from errand.tools.registry import ToolRegistry


def test_history_messages_use_roles_without_persisted_file_contents(
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
    registry = ToolRegistry()
    call = ToolCall(id="tc-1", name="read_file", params={"path": "INDEX.md", "scope": "kb"})
    result = ToolResult(
        tool_call_id="tc-1", name="read_file", content="# Agent KB\nsecret soul"
    )
    memory.add_tool_results([registry.compact_result(call, result)])
    memory.add_history(
        "ai",
        {
            "tool_calls": [],
            "text_response": "I loaded the profile.",
            "context_summary": "Profile loaded from INDEX.md.",
        },
    )
    memory.set_context_summary("Profile loaded from INDEX.md.")

    messages = memory.build_history_messages()

    assert messages[0] == {"role": "user", "content": "Read the profile."}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "tc-1"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "tc-1"
    assert "kb:INDEX.md" in messages[2]["content"]
    assert "# Agent KB" not in messages[2]["content"]
    assert "secret soul" not in messages[2]["content"]
    assert messages[3] == {"role": "assistant", "content": "I loaded the profile."}

    tool_entry = next(entry for entry in memory.data["history"] if entry["role"] == "tool")
    stored_result = tool_entry["content"][0]
    assert stored_result["result_ref"] == {
        "type": "file",
        "scope": "kb",
        "path": "INDEX.md",
    }
    assert "secret soul" not in str(stored_result)


def test_scheduled_history_renders_as_user_message(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")
    memory.add_history("scheduled", "A scheduled task is firing now.\nTASK: water plants")

    messages = memory.build_history_messages()

    assert messages == [
        {
            "role": "user",
            "content": "A scheduled task is firing now.\nTASK: water plants",
        }
    ]


def test_session_note_is_sent_as_runtime_context(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    memory.add_session_note(IDLE_BOUNDARY_NOTE)

    assert memory.data["metadata"]["session_note"] == IDLE_BOUNDARY_NOTE
    assert memory.build_history_messages() == []


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


def test_update_token_usage_accumulates_counts(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("count-session")

    memory.update_token_usage(
        {"input_tokens": 1_000_000, "output_tokens": 500_000, "cache_read_tokens": 800_000}
    )
    memory.update_token_usage(
        {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0}
    )

    summary = memory.data["token_summary"]
    assert summary["input_tokens"] == 1_000_100
    assert summary["output_tokens"] == 500_050
    assert summary["cache_read_tokens"] == 800_000
    assert "total_cost" not in summary
