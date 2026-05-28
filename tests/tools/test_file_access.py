"""Tests for the private scoped filesystem helper used by the file tools."""

from pathlib import Path

import pytest

from errand.config import FileAccessConfig, FileScope
from errand.tools._file_access import is_within_roots, list_dir, read_file


def _config(root: Path, **scope_kwargs) -> FileAccessConfig:
    return FileAccessConfig(
        default_scope="kb",
        scopes={"kb": FileScope(roots=[str(root)], **scope_kwargs)},
    )


def test_read_file_inside_scope_with_absolute_path(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("hello", encoding="utf-8")

    content = read_file(str(root / "INDEX.md"), _config(root), scope="kb")
    assert content == "hello"


def test_read_file_inside_scope_with_relative_path(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("hello", encoding="utf-8")

    content = read_file("INDEX.md", _config(root), scope="kb")
    assert content == "hello"


def test_read_file_resolves_relative_against_base_path(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "agent").mkdir()
    (root / "agent" / "INDEX.md").write_text("see sops", encoding="utf-8")
    (root / "agent" / "sops").mkdir()
    (root / "agent" / "sops" / "x.md").write_text("x", encoding="utf-8")

    content = read_file(
        "sops/x.md",
        _config(root),
        scope="kb",
        base_path="agent/INDEX.md",
    )
    assert content == "x"


def test_read_file_base_path_rejects_escape(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "agent").mkdir()
    (root / "agent" / "INDEX.md").write_text("index", encoding="utf-8")
    (tmp_path / "secret.md").write_text("nope", encoding="utf-8")

    # base_path resolves inside root/agent/; ../../ escapes root entirely.
    with pytest.raises(PermissionError):
        read_file(
            "../../secret.md",
            _config(root),
            scope="kb",
            base_path="agent/INDEX.md",
        )

def test_read_file_rejects_outside_scope(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError):
        read_file(str(outside), _config(root), scope="kb")


def test_read_file_rejects_traversal(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError):
        read_file("../secret.md", _config(root), scope="kb")


def test_read_file_rejects_missing_file(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()

    with pytest.raises(FileNotFoundError):
        read_file("nope.md", _config(root), scope="kb")


def test_read_file_respects_size_limit(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "big.md").write_text("x" * 1000, encoding="utf-8")

    with pytest.raises(ValueError):
        read_file("big.md", _config(root), scope="kb", max_bytes=10)


def test_list_dir_filters_hidden_and_marks_dirs(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("x", encoding="utf-8")
    (root / "sops").mkdir()
    (root / ".hidden").write_text("x", encoding="utf-8")

    entries = list_dir("", _config(root), scope="kb")
    assert entries == ["INDEX.md", "sops/"]


def test_unknown_scope_fails_clearly(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()

    with pytest.raises(ValueError, match="Unknown file scope"):
        read_file("INDEX.md", _config(root), scope="workspace")


def test_disabled_read_scope_rejects_read(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        read_file("INDEX.md", _config(root, read=False), scope="kb")


def test_is_within_roots_multiple_roots(tmp_path: Path):
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()
    file_b = root_b / "x"
    file_b.write_text("x", encoding="utf-8")

    assert is_within_roots(file_b, [root_a, root_b])
    assert not is_within_roots(tmp_path / "outside", [root_a, root_b])
