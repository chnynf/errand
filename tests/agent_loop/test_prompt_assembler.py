"""Tests for `PromptAssembler`.

We assert the structural shape (sections present, profile/scopes
surfaced), not the exact wording of the static prompt fragments.
"""

from paw.agent_loop.prompt_assembler import PromptAssembler
from paw.config import FileAccessConfig, FileScope


def test_system_prompt_stable_no_current_context():
    assembler = PromptAssembler()
    prompt = assembler.build_system_prompt()
    # CURRENT CONTEXT must NOT be in the system prompt — it is time-varying
    # and would break prompt caching by changing the cached prefix every minute.
    assert "CURRENT CONTEXT:" not in prompt
    assert "Paw Agent" in prompt


def test_tool_summary_is_placed_verbatim_when_provided():
    # The tools component authors the summary; the assembler only positions it.
    assembler = PromptAssembler(tool_summary="TOOL USE:\n- batch in parallel.")
    prompt = assembler.build_system_prompt()
    assert "TOOL USE:" in prompt
    assert "batch in parallel." in prompt


def test_tool_summary_absent_when_not_provided():
    assert "TOOL USE:" not in PromptAssembler().build_system_prompt()


def test_context_messages_include_current_context():
    assembler = PromptAssembler()
    messages = assembler.build_context_messages()
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("Now:")
    assert "UTC)" in messages[0]["content"]


def test_system_prompt_includes_agent_profile_contents(tmp_path):
    soul = tmp_path / "SOUL.md"
    profile = tmp_path / "INDEX.md"
    soul.write_text("# Shared Soul\n\nBe grounded.", encoding="utf-8")
    profile.write_text("# Test Profile\n\nBe concise.", encoding="utf-8")

    assembler = PromptAssembler(
        shared_soul=str(soul),
        agent_profile=str(profile),
        file_access=FileAccessConfig(
            default_scope="kb",
            scopes={
                "kb": FileScope(roots=[str(tmp_path)]),
            },
        ),
    )
    prompt = assembler.build_system_prompt()
    assert "--- BEGIN PROMPT RESOURCE: SHARED_SOUL ---" in prompt
    assert "Logical path: SOUL.md" in prompt
    assert "Base path: ./" in prompt
    assert "--- BEGIN PROMPT RESOURCE: AGENT_PROFILE ---" in prompt
    assert "Logical path: INDEX.md" in prompt
    assert "Base path: ./" in prompt
    assert "# Shared Soul\n\nBe grounded." in prompt
    assert "# Test Profile\n\nBe concise." in prompt
    assert "{{ include:SHARED_SOUL }}" not in prompt
    assert "{{ include:AGENT_PROFILE }}" not in prompt
    assert str(soul) not in prompt
    assert str(profile) not in prompt
    # File scopes are owned by the tools component now, not the assembler.
    assert "FILE TOOL SCOPES:" not in prompt
    assert "Start by reading AGENT_PROFILE" not in prompt


def test_system_prompt_includes_shared_notes_index(tmp_path):
    soul = tmp_path / "SOUL.md"
    notes = tmp_path / "NOTES.md"
    profile = tmp_path / "INDEX.md"
    soul.write_text("# Soul", encoding="utf-8")
    notes.write_text("# Notes Router\n\nLook in notes/inbox.md.", encoding="utf-8")
    profile.write_text("# Profile", encoding="utf-8")

    assembler = PromptAssembler(
        shared_soul=str(soul),
        agent_profile=str(profile),
        shared_notes_index=str(notes),
        file_access=FileAccessConfig(
            default_scope="kb",
            scopes={"kb": FileScope(roots=[str(tmp_path)])},
        ),
    )
    prompt = assembler.build_system_prompt()
    assert "--- BEGIN PROMPT RESOURCE: SHARED_NOTES_INDEX ---" in prompt
    assert "Logical path: NOTES.md" in prompt
    assert "Look in notes/inbox.md." in prompt
    assert "{{ include:SHARED_NOTES_INDEX }}" not in prompt
    # Order: soul, then notes, then agent profile.
    assert (
        prompt.index("SHARED_SOUL")
        < prompt.index("SHARED_NOTES_INDEX")
        < prompt.index("AGENT_PROFILE")
    )


def test_no_notes_index_leaves_no_placeholder(tmp_path):
    assembler = PromptAssembler()
    prompt = assembler.build_system_prompt()
    assert "{{ include:SHARED_NOTES_INDEX }}" not in prompt


def test_notes_block_reloads_with_soul(tmp_path):
    notes = tmp_path / "NOTES.md"
    notes.write_text("v1", encoding="utf-8")
    assembler = PromptAssembler(shared_notes_index=str(notes))
    assembler.build_system_prompt()
    assert assembler._notes_block is not None
    notes.write_text("v2", encoding="utf-8")
    assembler.reload_resources(soul=True, profile=False)
    assert assembler._notes_block is None
    prompt = assembler.build_system_prompt()
    assert "v2" in prompt


def test_soul_and_profile_blocks_are_cached(tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("Soul content.", encoding="utf-8")
    assembler = PromptAssembler(shared_soul=str(soul))
    assembler.build_system_prompt()
    assert assembler._soul_block is not None
    cached = assembler._soul_block
    # Mutate the file on disk — cached value must not change.
    soul.write_text("Changed.", encoding="utf-8")
    assembler.build_system_prompt()
    assert assembler._soul_block == cached


def test_build_prompt_orders_the_four_sections():
    """Caching invariant: stable prefix is system + history + current_exchange;
    volatile per-turn context (time/summary/instruction) goes last so it can't
    break the prefix."""
    assembler = PromptAssembler()
    history = [
        {"role": "user", "content": "apply the plan"},
        {"role": "assistant", "content": "done"},
    ]
    current_exchange = [
        {"role": "user", "content": "now do the next thing"},
    ]
    messages = assembler.build_prompt(
        history,
        current_exchange,
        context_summary="rolling summary",
        instruction="Decide whether to call a tool or respond directly.",
    )

    assert messages[0]["role"] == "system"
    # History sits immediately after the system prompt (cacheable prefix).
    assert messages[1] == {"role": "user", "content": "apply the plan"}
    assert messages[2] == {"role": "assistant", "content": "done"}
    # The current exchange follows history, before the volatile context.
    assert messages[3] == {"role": "user", "content": "now do the next thing"}
    # The time-varying context is the final message.
    last = messages[-1]
    assert last["role"] == "user"
    assert "Now:" in last["content"]
    assert "CONTEXT SUMMARY:\nrolling summary" in last["content"]
    assert "INSTRUCTION:\nDecide whether" in last["content"]


def test_context_messages_include_summary_and_instruction():
    assembler = PromptAssembler()
    messages = assembler.build_context_messages(
        context_summary="ctx body",
        instruction="do thing",
    )
    content = messages[0]["content"]
    assert "Now:" in content
    assert "CONTEXT SUMMARY:\nctx body" in content
    assert "INSTRUCTION:\ndo thing" in content
    assert content.index("Now:") < content.index("CONTEXT SUMMARY:")
