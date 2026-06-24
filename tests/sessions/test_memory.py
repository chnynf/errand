from paw.contracts.types import ToolCall, ToolResult
from paw.sessions.memory import (
    IDLE_BOUNDARY_NOTE,
    Memory,
    _safe_stem,
    strip_tool_call_signature,
)
from paw.tools.registry import ToolRegistry

# A Gemini-style id with a thought signature smuggled in by LiteLLM.
_THOUGHT_ID = "call_82d77cbf__thought__EoCaAQr8mQEBDDnWxxktVVmd" * 4


def test_history_messages_use_roles_without_persisted_file_contents(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    memory.add_history("user", "Read the profile.")
    memory.add_history(
        "ai",
        {
            "tool_calls": [
                {
                    "id": "tc-1",
                    "name": "read_file",
                    "params": {"path": "INDEX.md"},
                }
            ],
            "text_response": None,
        },
    )
    registry = ToolRegistry()
    call = ToolCall(id="tc-1", name="read_file", params={"path": "INDEX.md"})
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
    assert "ref INDEX.md" in messages[2]["content"]
    assert "# Agent KB" not in messages[2]["content"]
    assert "secret soul" not in messages[2]["content"]
    assert messages[3] == {"role": "assistant", "content": "I loaded the profile."}

    tool_entry = next(entry for entry in memory.data["history"] if entry["role"] == "tool")
    stored_result = tool_entry["content"][0]
    assert stored_result["result_ref"] == {
        "type": "file",
        "path": "INDEX.md",
    }
    assert "secret soul" not in str(stored_result)


def test_strip_tool_call_signature():
    # Gemini thought signature is removed, leaving the stable handle.
    assert strip_tool_call_signature(_THOUGHT_ID) == "call_82d77cbf"
    # Plain ids from other providers are untouched (no marker -> no-op).
    assert strip_tool_call_signature("call_abc123") == "call_abc123"
    assert strip_tool_call_signature("tc-1") == "tc-1"
    assert strip_tool_call_signature("") == ""


def test_thought_signature_stripped_on_persist_and_replay(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("sig-session")

    memory.add_history("user", "What's for dinner?")
    memory.add_history(
        "ai",
        {
            "tool_calls": [
                {"id": _THOUGHT_ID, "name": "read_file",
                 "params": {"path": "meal.md", "scope": "notes"}}
            ],
            "text_response": None,
        },
    )
    registry = ToolRegistry()
    call = ToolCall(id=_THOUGHT_ID, name="read_file", params={"path": "meal.md", "scope": "notes"})
    result = ToolResult(tool_call_id=_THOUGHT_ID, name="read_file", content="pasta")
    memory.add_tool_results([registry.compact_result(call, result)])

    # Persisted to disk without the blob.
    ai_entry = next(e for e in memory.data["history"] if e["role"] == "ai")
    assert ai_entry["content"]["tool_calls"][0]["id"] == "call_82d77cbf"
    tool_entry = next(e for e in memory.data["history"] if e["role"] == "tool")
    assert tool_entry["content"][0]["tool_call_id"] == "call_82d77cbf"
    assert "__thought__" not in str(memory.data["history"])

    # Replayed history keeps assistant/tool ids matched and blob-free.
    messages = memory.build_history_messages()
    assert messages[1]["tool_calls"][0]["id"] == "call_82d77cbf"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_82d77cbf"
    assert "__thought__" not in str(messages)


def test_legacy_session_with_blob_ids_is_stripped_on_read(monkeypatch, tmp_path):
    # Simulate a session persisted before the fix: full blob ids on disk.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("legacy-session")
    memory.data["history"] = [
        {"role": "user", "content": "hi", "metadata": {}},
        {"role": "ai", "content": {
            "tool_calls": [{"id": _THOUGHT_ID, "name": "read_file",
                            "params": {"path": "x.md", "scope": "kb"}}],
            "text_response": None}, "metadata": {}},
        {"role": "tool", "content": [{
            "tool_call_id": _THOUGHT_ID, "name": "read_file",
            "params": {"path": "x.md", "scope": "kb"}, "content_chars": 5,
            "result_ref": {"type": "file", "scope": "kb", "path": "x.md"}}],
         "metadata": {}},
    ]

    messages = memory.build_history_messages()

    # The assistant + tool pair still resolves (ids matched after stripping).
    assert messages[1]["tool_calls"][0]["id"] == "call_82d77cbf"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_82d77cbf"
    assert "__thought__" not in str(messages)


def test_scheduled_history_renders_as_user_message(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
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
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    memory.add_session_note(IDLE_BOUNDARY_NOTE)

    assert memory.data["metadata"]["session_note"] == IDLE_BOUNDARY_NOTE
    assert memory.build_history_messages() == []


def test_history_window_start_is_cache_stable(monkeypatch, tmp_path):
    # The window start must hold steady across turns (so the message prefix
    # stays cache-eligible) and only advance in coarse `chunk`-sized jumps,
    # rather than sliding one entry per turn.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("window-session")
    for i in range(40):
        memory.add_history("user", f"msg {i}")

    def first_content(n_entries: int) -> str:
        memory.data["history"] = [
            {"role": "user", "content": f"msg {i}", "metadata": {}}
            for i in range(n_entries)
        ]
        return memory.build_history_messages(recent_n=20, chunk=8)[0]["content"]

    # Under the floor: nothing dropped, window starts at the very first entry.
    assert first_content(20) == "msg 0"
    # Past the floor but within the same chunk: start holds steady at 0.
    assert first_content(21) == "msg 0"
    assert first_content(27) == "msg 0"
    # Crossing the chunk boundary advances the floor by exactly `chunk` (8).
    assert first_content(28) == "msg 8"
    assert first_content(35) == "msg 8"
    assert first_content(36) == "msg 16"


def test_session_id_with_illegal_filename_chars_is_archivable(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    session_id = "wechat:o9cq80wyNyZ@im.wechat"
    memory = Memory(session_id)
    assert ":" not in memory._file_stem
    assert memory.session_id == session_id


async def test_archive_session_with_illegal_chars(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
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
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
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
