"""Tests for the model-facing `read_file` and `list_dir` tools."""

from pathlib import Path

import pytest

from errand.config import FileAccessConfig, FileScope, ErrandConfig
from errand.tools import files as file_tools


def _fake_errand_config(root: Path) -> ErrandConfig:
    return ErrandConfig(
        file_access=FileAccessConfig(
            default_scope="kb",
            scopes={"kb": FileScope(roots=[str(root)], read=True, list=True)},
        )
    )


@pytest.fixture
def fake_file_scope(tmp_path: Path, monkeypatch):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "INDEX.md").write_text("# Hi\nSee sops/.", encoding="utf-8")
    (root / "sops").mkdir()
    (root / "sops" / "ad-hoc.md").write_text("ad hoc", encoding="utf-8")

    monkeypatch.setattr(
        file_tools,
        "load_errand_config",
        lambda: _fake_errand_config(root),
    )
    return root


def test_read_file_returns_content(fake_file_scope: Path):
    out = file_tools.read_file("INDEX.md", scope="kb")
    assert out.startswith("# Hi")


def test_read_file_supports_base_path(fake_file_scope: Path):
    out = file_tools.read_file("ad-hoc.md", scope="kb", base_path="sops")
    assert out.strip() == "ad hoc"


def test_read_file_returns_error_for_outside_path(fake_file_scope: Path, tmp_path: Path):
    out = file_tools.read_file(str(tmp_path / "secret.md"), scope="kb")
    assert out.startswith("Error:")


def test_list_dir_lists_entries(fake_file_scope: Path):
    out = file_tools.list_dir("", scope="kb")
    assert "INDEX.md" in out
    assert "sops/" in out


def test_list_dir_can_list_relative_subdirectory(fake_file_scope: Path):
    out = file_tools.list_dir("sops", scope="kb")
    assert "ad-hoc.md" in out


def test_read_file_no_config(monkeypatch):
    monkeypatch.setattr(
        file_tools,
        "load_errand_config",
        lambda: ErrandConfig(file_access=FileAccessConfig()),
    )
    assert file_tools.read_file("anything").startswith("Error:")
    assert file_tools.list_dir().startswith("Error:")
