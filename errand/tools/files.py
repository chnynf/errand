"""Scoped file access tools.

Use these tools to read or list files from runtime-configured scopes.
The model chooses the scope by name, but scope roots and permissions
are defined in ``config.json`` under ``file_access``.
"""

from errand.config import load_errand_config
from errand.tools._file_access import list_dir as _scoped_list_dir
from errand.tools._file_access import read_file as _scoped_read_file


def _file_access_config():
    return load_errand_config().file_access


def read_file(path: str, scope: str = "kb", base_path: str = "") -> str:
    """Read a text file from a configured file scope.

    Use for reading agent knowledge, profiles, SOPs, skills, templates, or
    other files in scopes granted by the runtime. For knowledge-base work,
    use the default ``kb`` scope and start with the configured agent profile.

    Args:
        path: File path inside the chosen scope. Relative paths are resolved
            inside the scope root unless ``base_path`` is provided. When
            ``base_path`` is set and ``path`` is relative, the path is resolved
            relative to the directory containing ``base_path`` (still confined
            to the scope roots).
        scope: Configured file scope to use. Defaults to ``kb``.
        base_path: Optional base file or directory path (within the same scope)
            used to resolve relative paths.

    Returns: File contents as UTF-8 text, or an ``Error: ...`` message.

    Example: read_file("INDEX.md", scope="kb")
    """
    try:
        return _scoped_read_file(
            path, _file_access_config(), scope=scope, base_path=base_path or None
        )
    except (FileNotFoundError, PermissionError, ValueError) as e:
        return f"Error: {e}"


def list_dir(path: str = "", scope: str = "kb", base_path: str = "") -> str:
    """List entries in a directory from a configured file scope.

    Use to discover files before reading them. For the default ``kb`` scope,
    an empty path lists the scope root.

    Args:
        path: Directory path inside the chosen scope. If empty, lists the
            selected scope's first configured root.
        scope: Configured file scope to use. Defaults to ``kb``.
        base_path: Optional base file or directory path (within the same scope)
            used to resolve relative paths.

    Returns: Newline-separated entries. Directories have a trailing ``/``.

    Example: list_dir("as-agent-kb", scope="kb")
    """
    try:
        entries = _scoped_list_dir(
            path, _file_access_config(), scope=scope, base_path=base_path or None
        )
        if not entries:
            shown_path = path or "."
            return f"(empty directory: {shown_path})"
        return "\n".join(entries)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as e:
        return f"Error: {e}"
