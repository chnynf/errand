"""Tests for the scoped file-system tools in ``errand.tools.files``.

Covers scope/permission enforcement, every file operation, the per-operation
approval policy, and session-memory compaction of large write payloads.
"""

from pathlib import Path

import pytest

from errand.config import ErrandConfig, FileAccessConfig, FileScope
from errand.contracts.types import ToolCall, ToolResult
from errand.sessions.memory import Memory
from errand.tools import files as ft


def _patch_scope(monkeypatch, root: Path, **scope_kwargs) -> None:
    """Patch the file-access config with a single 'kb' scope.

    Defaults to fully permissive (all operations True). Pass keyword arguments
    matching ``FileScope`` fields to override individual permissions.
    """
    scope_kwargs.setdefault("read",   True)
    scope_kwargs.setdefault("list",   True)
    scope_kwargs.setdefault("write",  True)
    scope_kwargs.setdefault("append", True)
    scope_kwargs.setdefault("edit",   True)
    scope_kwargs.setdefault("delete", True)
    config = ErrandConfig(
        file_access=FileAccessConfig(
            default_scope="kb",
            scopes={"kb": FileScope(roots=[str(root)], **scope_kwargs)},
        )
    )
    monkeypatch.setattr(ft, "load_errand_config", lambda: config)


@pytest.fixture
def kb(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("# Index\nsee notes/\n", encoding="utf-8")
    (root / ".hidden").write_text("secret", encoding="utf-8")
    (root / "notes").mkdir()
    (root / "notes" / "a.md").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    _patch_scope(monkeypatch, root)
    return root


class _Approver:
    """Fake reply_to channel that records and answers approval requests."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.calls: list[dict] = []

    async def request_approval(self, *, title: str, details: str, timeout_seconds: int) -> bool:
        self.calls.append({"title": title, "details": details})
        return self.answer


# --- access / permission checks ------------------------------------------------

def test_read_rejects_parent_traversal(kb: Path):
    assert ft.read_file("../secret.md").startswith("Error:")


def test_read_rejects_absolute_path_outside_root(kb: Path, tmp_path: Path):
    outside = tmp_path / "secret.md"
    outside.write_text("nope", encoding="utf-8")
    assert ft.read_file(str(outside)).startswith("Error:")


def test_unknown_scope_is_reported(kb: Path):
    assert "Unknown file scope" in ft.read_file("INDEX.md", scope="nope")


def test_read_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("x", encoding="utf-8")
    _patch_scope(monkeypatch, root, read=False)
    assert "not permitted" in ft.read_file("INDEX.md")


def test_list_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, list=False)
    assert "not permitted" in ft.list_dir("")


async def test_write_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, write=False)
    assert "not permitted" in await ft.write_file("x.md", "hi")


async def test_append_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, append=False)
    assert "not permitted" in await ft.append_file("x.md", "hi")


async def test_edit_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("hello", encoding="utf-8")
    _patch_scope(monkeypatch, root, edit=False)
    assert "not permitted" in await ft.edit_file("x.md", "hello", "world")


async def test_delete_disabled_scope_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("bye", encoding="utf-8")
    _patch_scope(monkeypatch, root, delete=False)
    assert "not permitted" in await ft.delete_file("x.md")
    assert (root / "x.md").exists()


async def test_write_cannot_escape_root(kb: Path):
    assert (await ft.write_file("../escape.md", "x")).startswith("Error:")


# --- read / list ---------------------------------------------------------------

def test_read_file_returns_content(kb: Path):
    assert ft.read_file("INDEX.md").startswith("# Index")


def test_read_file_with_base_path(kb: Path):
    assert ft.read_file("a.md", base_path="notes").startswith("alpha")


def test_list_dir_marks_dirs_and_hides_dotfiles(kb: Path):
    entries = ft.list_dir("").splitlines()
    assert "INDEX.md" in entries
    assert "notes/" in entries
    assert ".hidden" not in entries


# --- append --------------------------------------------------------------------

async def test_append_creates_file(kb: Path):
    out = await ft.append_file("notes/new.md", "first line\n")
    assert out.startswith("Appended")
    assert (kb / "notes" / "new.md").read_text(encoding="utf-8") == "first line\n"


async def test_append_adds_to_existing_without_overwriting(kb: Path):
    (kb / "notes" / "log.md").write_text("original\n", encoding="utf-8")
    out = await ft.append_file("notes/log.md", "added\n")
    assert out.startswith("Appended")
    content = (kb / "notes" / "log.md").read_text(encoding="utf-8")
    assert content == "original\nadded\n"


async def test_append_cannot_escape_root(kb: Path):
    assert (await ft.append_file("../escape.md", "x")).startswith("Error:")


# --- write / edit / delete -----------------------------------------------------

async def test_write_creates_then_overwrites(kb: Path):
    out = await ft.write_file("notes/new.md", "first")
    assert out.startswith("Wrote")
    assert (kb / "notes" / "new.md").read_text(encoding="utf-8") == "first"
    await ft.write_file("notes/new.md", "second")
    assert (kb / "notes" / "new.md").read_text(encoding="utf-8") == "second"


async def test_write_auto_creates_parent_dirs(kb: Path):
    await ft.write_file("memories/2026/today.md", "noted")
    assert (kb / "memories" / "2026" / "today.md").read_text(encoding="utf-8") == "noted"


async def test_edit_unique_match(kb: Path):
    out = await ft.edit_file("INDEX.md", "see notes/", "see notes folder")
    assert "Replaced 1 occurrence" in out
    assert "see notes folder" in (kb / "INDEX.md").read_text(encoding="utf-8")


async def test_edit_ambiguous_match_errors(kb: Path):
    out = await ft.edit_file("notes/a.md", "alpha", "ALPHA")
    assert out.startswith("Error:") and "not unique" in out
    assert (kb / "notes" / "a.md").read_text(encoding="utf-8").count("alpha") == 2


async def test_edit_replace_all(kb: Path):
    out = await ft.edit_file("notes/a.md", "alpha", "ALPHA", replace_all=True)
    assert "Replaced 2 occurrences" in out
    assert "alpha" not in (kb / "notes" / "a.md").read_text(encoding="utf-8")


async def test_edit_empty_old_string_errors(kb: Path):
    assert (await ft.edit_file("INDEX.md", "", "x")).startswith("Error:")


async def test_edit_missing_match_errors(kb: Path):
    assert "not found" in await ft.edit_file("INDEX.md", "absent", "x")


async def test_delete_file(kb: Path):
    out = await ft.delete_file("notes/a.md")
    assert out.startswith("Deleted file")
    assert not (kb / "notes" / "a.md").exists()


async def test_delete_empty_dir(kb: Path):
    (kb / "empty").mkdir()
    out = await ft.delete_file("empty")
    assert out.startswith("Deleted directory")
    assert not (kb / "empty").exists()


async def test_delete_non_empty_dir_rejected(kb: Path):
    assert "not empty" in await ft.delete_file("notes")
    assert (kb / "notes").exists()


async def test_delete_scope_root_rejected(kb: Path):
    assert "scope root" in await ft.delete_file("")


# --- search --------------------------------------------------------------------

def test_grep_finds_matches(kb: Path):
    out = ft.grep_files("beta")
    assert "notes/a.md:2:beta" in out


def test_grep_glob_filter_excludes_non_matches(kb: Path):
    (kb / "notes" / "a.txt").write_text("beta", encoding="utf-8")
    out = ft.grep_files("beta", glob="*.md")
    assert "a.txt" not in out
    assert "notes/a.md" in out


def test_grep_invalid_regex_reported(kb: Path):
    assert "invalid regex" in ft.grep_files("[unclosed")


def test_find_files_by_glob(kb: Path):
    out = ft.find_files("*.md").splitlines()
    assert "INDEX.md" in out
    assert "notes/a.md" in out


# --- approval policy -----------------------------------------------------------

async def test_ask_write_blocks_without_channel(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, write="ask")
    out = await ft.write_file("x.md", "hi", _context={})
    assert "requires approval" in out
    assert not (root / "x.md").exists()


async def test_ask_write_proceeds_when_approved(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, write="ask")
    approver = _Approver(answer=True)
    out = await ft.write_file("x.md", "hi", _context={"reply_to": approver})
    assert out.startswith("Wrote")
    assert (root / "x.md").read_text(encoding="utf-8") == "hi"
    assert approver.calls


async def test_ask_write_denied_leaves_file_untouched(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, write="ask")
    approver = _Approver(answer=False)
    out = await ft.write_file("x.md", "hi", _context={"reply_to": approver})
    assert "not approved" in out
    assert not (root / "x.md").exists()


async def test_ask_append_blocks_without_channel(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, append="ask")
    out = await ft.append_file("x.md", "hi", _context={})
    assert "requires approval" in out
    assert not (root / "x.md").exists()


async def test_ask_append_proceeds_when_approved(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    _patch_scope(monkeypatch, root, append="ask")
    approver = _Approver(answer=True)
    out = await ft.append_file("x.md", "hi", _context={"reply_to": approver})
    assert out.startswith("Appended")
    assert (root / "x.md").read_text(encoding="utf-8") == "hi"


async def test_ask_edit_blocks_without_channel(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("hello", encoding="utf-8")
    _patch_scope(monkeypatch, root, edit="ask")
    out = await ft.edit_file("x.md", "hello", "world", _context={})
    assert "requires approval" in out
    assert (root / "x.md").read_text(encoding="utf-8") == "hello"


async def test_ask_edit_proceeds_when_approved(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("hello", encoding="utf-8")
    _patch_scope(monkeypatch, root, edit="ask")
    approver = _Approver(answer=True)
    out = await ft.edit_file("x.md", "hello", "world", _context={"reply_to": approver})
    assert "Replaced 1 occurrence" in out
    assert (root / "x.md").read_text(encoding="utf-8") == "world"
    assert approver.calls


async def test_ask_delete_blocks_without_channel(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("bye", encoding="utf-8")
    _patch_scope(monkeypatch, root, delete="ask")
    out = await ft.delete_file("x.md", _context={})
    assert "requires approval" in out
    assert (root / "x.md").exists()


async def test_ask_delete_proceeds_when_approved(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("bye", encoding="utf-8")
    _patch_scope(monkeypatch, root, delete="ask")
    approver = _Approver(answer=True)
    out = await ft.delete_file("x.md", _context={"reply_to": approver})
    assert out.startswith("Deleted file")
    assert not (root / "x.md").exists()


async def test_per_op_permissions_are_independent(tmp_path: Path, monkeypatch):
    """append:True should not grant write or edit."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "x.md").write_text("original", encoding="utf-8")
    _patch_scope(monkeypatch, root, write=False, append=True, edit=False)
    # append allowed
    out = await ft.append_file("x.md", " appended")
    assert out.startswith("Appended")
    # write blocked
    assert "not permitted" in await ft.write_file("x.md", "replaced")
    # edit blocked
    assert "not permitted" in await ft.edit_file("x.md", "original", "changed")
    # file content unchanged from write/edit attempts
    content = (root / "x.md").read_text(encoding="utf-8")
    assert content == "original appended"


# --- session memory compaction -------------------------------------------------

def test_memory_strips_large_write_payload():
    call = ToolCall(
        id="tc-1",
        name="write_file",
        params={"path": "notes/big.md", "scope": "kb", "content": "x" * 10000},
    )
    result = ToolResult(
        tool_call_id="tc-1",
        name="write_file",
        content="Wrote 10000 bytes to /kb/notes/big.md.",
    )
    record = Memory._compact_tool_result(call, result)
    assert "content" not in record["params"]
    assert record["result_ref"] == {"type": "file", "scope": "kb", "path": "notes/big.md"}
    assert record["content_chars"] == len(result.content)
