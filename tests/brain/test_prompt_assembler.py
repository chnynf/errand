"""Tests for `PromptAssembler`.

We assert the structural shape (sections present, profile/scopes
surfaced), not the exact wording of the static prompt fragments.
"""

from errand.brain.prompt_assembler import PromptAssembler
from errand.config import FileAccessConfig, FileScope


def test_system_prompt_includes_runtime_and_context_header():
    assembler = PromptAssembler()
    prompt = assembler.build_system_prompt()
    assert "CURRENT CONTEXT:" in prompt
    assert "Errand Runtime" in prompt


def test_system_prompt_includes_agent_profile_contents_and_scope(tmp_path):
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
    assert "FILE TOOL SCOPES:" in prompt
    assert "Use these scope names when calling read_file or list_dir." in prompt
    assert "DEFAULT_FILE_SCOPE: kb" in prompt
    assert f"FILE_SCOPE kb: roots=[{tmp_path}]" in prompt
    assert "read_file" in prompt
    assert "Start by reading AGENT_PROFILE" not in prompt


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


def test_user_prompt_layout():
    assembler = PromptAssembler()
    prompt = assembler.build_user_prompt("ctx body", "do thing")
    assert "SESSION CONTEXT:\nctx body" in prompt
    assert "INSTRUCTION:\ndo thing" in prompt
