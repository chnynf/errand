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


# -- Distant history (two-tier replay) --------------------------------------


def test_gap_roll_moves_all_history_to_distant_qa(monkeypatch, tmp_path):
    # A >=2h silence gap ends the conversation: every recent exchange rolls
    # into distant, compacted to [user, assistant] pairs (no anchors/tools).
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("gap-session")
    memory.add_exchange(_exchange("q0", "a0", tool_note="tool stuff"))
    memory.add_exchange(_exchange("q1", "a1"))
    last = memory.last_activity_at()

    assert memory.roll_distant(last + 2 * 3600) is True

    assert memory.data["history"] == []
    msgs = memory.build_history_messages()
    assert msgs == [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert not any("tool_calls" in m or m["role"] == "tool" for m in msgs)


def test_no_roll_under_gap_or_budget(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("nogap-session")
    memory.add_exchange(_exchange("q0", "a0", tool_note="keep me"))
    folded = list(memory.data["history"])
    last = memory.last_activity_at()

    assert memory.roll_distant(last + 3600) is False  # only 1h of silence

    assert memory.data["history"] == folded
    assert memory.data["distant"] == []


def test_token_overflow_migrates_oldest_group_to_distant(monkeypatch, tmp_path):
    # The budget group-drop now routes exchanges into distant instead of
    # silently windowing them out; the drop cadence (groups of 3) is unchanged.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    monkeypatch.setattr("paw.sessions.memory._HISTORY_TOKEN_BUDGET", 500)
    monkeypatch.setattr("paw.sessions.memory._DISTANT_MAX_CHARS", 100_000)
    memory = Memory("budget-session")
    for i in range(5):  # each exchange ~100 tokens -> over the 500 budget
        memory.add_exchange(
            [
                {"role": "user", "content": f"q{i}".ljust(200, "x")},
                {"role": "assistant", "content": f"a{i}".ljust(200, "y")},
            ]
        )
    last = memory.last_activity_at()

    assert memory.roll_distant(last + 60) is False  # budget, not gap

    # Oldest 3 exchanges moved (as Q&A) into distant; recent starts at q3.
    distant_users = [
        e["message"]["content"][:2]
        for e in memory.data["distant"]
        if e["message"]["role"] == "user"
    ]
    assert distant_users == ["q0", "q1", "q2"]
    assert memory.data["history"][0]["message"]["content"].startswith("q3")
    # Replay is distant + recent, in order.
    msgs = memory.build_history_messages()
    assert msgs[0]["content"][:2] == "q0"
    assert msgs[-1]["content"].startswith("a4")


def test_distant_cap_drops_oldest_whole_pairs(monkeypatch, tmp_path):
    # Over the char cap, distant trims down to the keep target, oldest whole
    # Q&A pairs first, never splitting a pair.
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    monkeypatch.setattr("paw.sessions.memory._DISTANT_MAX_CHARS", 500)
    monkeypatch.setattr("paw.sessions.memory._DISTANT_KEEP_CHARS", 250)
    memory = Memory("cap-session")
    for i in range(6):  # each pair ~200 chars -> 1200 total, cap 500
        memory.add_exchange(
            [
                {"role": "user", "content": f"q{i}".ljust(100, "x")},
                {"role": "assistant", "content": f"a{i}".ljust(100, "y")},
            ]
        )
    last = memory.last_activity_at()
    memory.roll_distant(last + 3 * 3600)  # gap roll: all 6 pairs -> distant

    distant = memory.data["distant"]
    # Trimmed to <=250 chars: only the newest pair survives, intact.
    assert [e["message"]["role"] for e in distant] == ["user", "assistant"]
    assert distant[0]["message"]["content"].startswith("q5")
    assert distant[1]["message"]["content"].startswith("a5")


def test_last_activity_falls_back_to_distant(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("fallback-session")
    memory.add_exchange(_exchange("q0", "a0"))
    last = memory.last_activity_at()
    memory.roll_distant(last + 2 * 3600)

    assert memory.data["history"] == []
    assert memory.last_activity_at() == last  # distant keeps the timestamp


async def test_distant_persists_and_legacy_sessions_load(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    memory = Memory("persist-session")
    memory.add_exchange(_exchange("q0", "a0"))
    memory.roll_distant(memory.last_activity_at() + 2 * 3600)
    await memory.save_session()

    reloaded = Memory("persist-session")
    assert reloaded.data["distant"] == memory.data["distant"]
    assert reloaded.build_history_messages() == [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
    ]

    # A pre-distant session file loads with an empty distant section.
    import json
    (tmp_path / "legacy-session.json").write_text(
        json.dumps({
            "session_id": "legacy-session",
            "created_at": 1.0,
            "history": [],
            "context_summary": None,
            "metadata": {},
            "token_summary": {"input_tokens": 0, "output_tokens": 0},
        }),
        encoding="utf-8",
    )
    assert Memory("legacy-session").data["distant"] == []


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
