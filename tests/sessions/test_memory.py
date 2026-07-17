from paw.sessions.memory import Memory, _safe_stem


def _exchange(question: str, answer: str, *, tool_note: str | None = None) -> list[dict]:
    """A folded exchange in the unified-log shape the loop produces."""
    messages: list[dict] = [{"role": "user", "content": question}]
    if tool_note is not None:
        messages.append(
            {"role": "assistant", "tool_calls": [
                {"id": "h1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]}
        )
        messages.append({"role": "tool", "tool_call_id": "h1", "content": tool_note})
    messages.append({"role": "assistant", "content": answer})
    return messages


def test_log_stores_and_replays_chat_messages_verbatim(monkeypatch, tmp_path):
    # The log IS the replay format: add_exchange appends chat messages and
    # build_history_messages returns them unchanged (no translation layer).
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("test-session")

    folded = _exchange(
        "Read the profile.",
        "I loaded the profile.",
        tool_note=(
            "tool call: read_file(path='INDEX.md')\n"
            "tool result: # Agent KB\nsecret soul [22 chars total]"
        ),
    )
    memory.add_exchange(folded)

    assert memory.build_history_messages() == folded
    # Every persisted entry wraps its message with a timestamp.
    assert all(e["timestamp"] > 0 and e["message"] for e in memory.data["history"])


def test_history_window_is_token_budgeted_and_cache_stable(monkeypatch, tmp_path):
    # Budgeted by tokens, cut on exchange boundaries; oldest exchanges drop in
    # groups of 3 so the window start holds steady (prefix stays cacheable) and
    # advances only in coarse jumps.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    monkeypatch.setattr("paw.sessions.memory._HISTORY_TOKEN_BUDGET", 500)
    memory = Memory("window-session")

    def first_marker(n_exchanges: int) -> str:
        memory.data["history"] = []
        for i in range(n_exchanges):
            # Each exchange ~100 tokens, marked "e{i}".
            memory.add_exchange(
                [{"role": "user", "content": f"e{i}".ljust(400, "x")}]
            )
        return memory.build_history_messages()[0]["content"].split("x", 1)[0]

    # Under budget: all exchanges shown, starting at e0.
    assert first_marker(4) == "e0"
    # Over budget: oldest drop in groups of 3; start holds...
    assert first_marker(5) == "e3"
    assert first_marker(7) == "e3"
    # ...then advances by exactly one group.
    assert first_marker(8) == "e6"
    assert first_marker(10) == "e6"
    assert first_marker(11) == "e9"


def test_history_window_keeps_at_least_one_exchange(monkeypatch, tmp_path):
    # A single oversized exchange must survive whole rather than be dropped.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    monkeypatch.setattr("paw.sessions.memory._HISTORY_TOKEN_BUDGET", 100)
    memory = Memory("floor-session")
    memory.add_exchange(_exchange("q1", "y" * 8000))  # ~2000 tokens

    msgs = memory.build_history_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "q1"


def test_empty_history_replays_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    assert Memory("empty-session").build_history_messages() == []


async def test_save_trims_on_exchange_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    monkeypatch.setattr("paw.sessions.memory._MAX_HISTORY_ENTRIES", 7)
    memory = Memory("trim-session")
    for i in range(4):  # 4 exchanges x 2 entries = 8 > 7
        memory.add_exchange(_exchange(f"q{i}", f"a{i}"))

    await memory.save_session()

    # Trimmed to whole exchanges: the oldest exchange dropped entirely, and the
    # remaining log still starts at a user message.
    history = memory.data["history"]
    assert len(history) == 6
    assert history[0]["message"] == {"role": "user", "content": "q1"}


def test_old_format_sessions_are_discarded_on_load(monkeypatch, tmp_path):
    # Pre-unified-log entries (no "message" key) are dropped, not translated.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    import json
    (tmp_path / "old-session.json").write_text(
        json.dumps({
            "session_id": "old-session",
            "created_at": 1.0,
            "history": [
                {"timestamp": 1.0, "role": "user", "content": "old entry"},
                {"timestamp": 2.0, "role": "ai", "content": {"text_response": "old"}},
            ],
            "context_summary": "kept",
            "metadata": {},
            "token_summary": {"input_tokens": 5, "output_tokens": 5},
        }),
        encoding="utf-8",
    )

    memory = Memory("old-session")

    assert memory.data["history"] == []
    assert memory.data["context_summary"] == "kept"  # non-history state survives
    assert memory.build_history_messages() == []


def test_session_id_with_illegal_filename_chars_is_archivable(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    session_id = "wechat:o9cq80wyNyZ@im.wechat"
    memory = Memory(session_id)
    assert ":" not in memory._file_stem
    assert memory.session_id == session_id


async def test_archive_session_with_illegal_chars(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("wechat:o9cq80wyNyZ@im.wechat")
    memory.add_exchange(_exchange("hi", "hello"))
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
