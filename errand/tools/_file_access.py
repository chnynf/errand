"""Scoped, safe filesystem access used by ``tools/files.py``.

This is a private helper for the file tools; the tool discovery
registry skips it because its name starts with ``_``.

Scope policies (roots, permissions) come from ``errand.config``.
The model picks a scope by name when it calls ``read_file`` /
``list_dir`` but never controls which roots that scope contains.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from errand.config import FileAccessConfig, FileScope

# Cap reads to keep prompts manageable; larger files should be split/excerpted.
MAX_READ_BYTES = 1_000_000


def expand(path: str | Path) -> Path:
    """Expand ``~``, resolve symlinks, and return an absolute Path."""
    return Path(os.path.expanduser(str(path))).resolve()


def is_within_roots(target: Path, roots: Iterable[Path]) -> bool:
    """Return True if ``target`` is inside any of the allowed ``roots``."""
    target_resolved = target.resolve()
    for root in roots:
        try:
            target_resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _scope_or_raise(
    config: FileAccessConfig,
    scope: str | None,
) -> tuple[str, FileScope, list[Path]]:
    scope_name = scope or config.default_scope
    file_scope = config.scopes.get(scope_name)
    if file_scope is None:
        available = ", ".join(sorted(config.scopes)) or "(none)"
        raise ValueError(
            f"Unknown file scope '{scope_name}'. Available scopes: {available}"
        )
    roots = [expand(root) for root in file_scope.roots]
    if not roots:
        raise ValueError(f"File scope '{scope_name}' has no configured roots.")
    return scope_name, file_scope, roots


def _resolve_target(path: str, roots: list[Path], *, base_path: str | None = None) -> Path:
    raw = Path(os.path.expanduser(path))
    if raw.is_absolute():
        return raw.resolve()
    if base_path:
        base_raw = Path(os.path.expanduser(base_path))
        base_target = (roots[0] / base_raw).resolve() if not base_raw.is_absolute() else base_raw.resolve()
        base_dir = base_target if base_target.is_dir() else base_target.parent
        return (base_dir / raw).resolve()
    # Relative paths resolve inside the scope's first root.
    return (roots[0] / raw).resolve()


def read_file(
    path: str,
    config: FileAccessConfig,
    *,
    scope: str | None = None,
    base_path: str | None = None,
    max_bytes: int = MAX_READ_BYTES,
) -> str:
    """Read a UTF-8 text file constrained to the selected file scope."""
    scope_name, file_scope, roots = _scope_or_raise(config, scope)
    if not file_scope.read:
        raise PermissionError(f"Read is disabled for file scope '{scope_name}'.")

    target = _resolve_target(path, roots, base_path=base_path)
    if not is_within_roots(target, roots):
        raise PermissionError(f"Path not within file scope '{scope_name}': {target}")
    if not target.is_file():
        raise FileNotFoundError(f"Not a file: {target}")
    size = target.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large ({size} bytes > {max_bytes}): {target}")
    return target.read_text(encoding="utf-8", errors="replace")


def list_dir(
    path: str,
    config: FileAccessConfig,
    *,
    scope: str | None = None,
    base_path: str | None = None,
) -> list[str]:
    """List directory entries constrained to the selected file scope."""
    scope_name, file_scope, roots = _scope_or_raise(config, scope)
    if not file_scope.list:
        raise PermissionError(f"List is disabled for file scope '{scope_name}'.")

    target = _resolve_target(path, roots, base_path=base_path) if path else roots[0]
    if not is_within_roots(target, roots):
        raise PermissionError(f"Path not within file scope '{scope_name}': {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    entries: list[str] = []
    for entry in sorted(target.iterdir(), key=lambda p: p.name):
        if entry.name.startswith("."):
            continue
        suffix = "/" if entry.is_dir() else ""
        entries.append(f"{entry.name}{suffix}")
    return entries
