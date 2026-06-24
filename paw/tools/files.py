"""Scoped file-system tools for reading and maintaining the knowledge base.

This single module is the local equivalent of an MCP file server. Public
module-level functions are auto-discovered as tools; private ``_``-prefixed
helpers handle scope confinement and operation authorization.

Security model
--------------
The model selects a *scope* by name. Each scope's roots and per-operation
permissions live in ``config.json`` under ``file_access`` and cannot be
expanded by the model.

Each operation has its own permission field (``True`` / ``False`` / ``"ask"``):

- ``read``   gates ``read_file`` and ``grep_files``.
- ``list``   gates ``list_dir`` and ``find_files``.
- ``write``  gates ``write_file`` (full create or overwrite).
- ``append`` gates ``append_file`` (add-only, never overwrites).
- ``edit``   gates ``edit_file`` (surgical string replace).
- ``delete`` gates ``delete_file``.

``True``  — always allow, no prompt.
``False`` — always block, returns an error.
``"ask"`` — require human approval through the ``reply_to`` channel; if no
            channel is available the operation is blocked rather than applied.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Iterable

from paw.config import FileScope, load_paw_config

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000
MAX_GREP_MATCHES = 200
MAX_FIND_RESULTS = 500

_FS_ERRORS = (OSError, ValueError)


def _expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def _within_roots(target: Path, roots: Iterable[Path]) -> bool:
    target = target.resolve()
    for root in roots:
        try:
            target.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _scope(scope: str | None) -> tuple[str, FileScope, list[Path]]:
    config = load_paw_config().file_access
    name = scope or config.default_scope
    file_scope = config.scopes.get(name)
    if file_scope is None:
        available = ", ".join(sorted(config.scopes)) or "(none)"
        raise ValueError(f"Unknown file scope '{name}'. Available scopes: {available}")
    roots = [_expand(root) for root in file_scope.roots]
    if not roots:
        raise ValueError(f"File scope '{name}' has no configured roots.")
    return name, file_scope, roots


def _resolve(path: str, roots: list[Path], *, base_path: str | None = None) -> Path:
    raw = Path(os.path.expanduser(path))
    if raw.is_absolute():
        return raw.resolve()
    if base_path:
        base_raw = Path(os.path.expanduser(base_path))
        base_target = (
            base_raw.resolve()
            if base_raw.is_absolute()
            else (roots[0] / base_raw).resolve()
        )
        base_dir = base_target if base_target.is_dir() else base_target.parent
        return (base_dir / raw).resolve()
    return (roots[0] / raw).resolve()


def _located(path: str, roots: list[Path], scope_name: str, *, base_path: str | None = None) -> Path:
    target = _resolve(path, roots, base_path=base_path)
    if not _within_roots(target, roots):
        raise PermissionError(f"Path not within file scope '{scope_name}': {target}")
    return target


def _preview(text: str, n: int = 200) -> str:
    return text[:n].replace("\n", "↵")


def _suggest_paths(name: str, roots: list[Path], limit: int = 5) -> str:
    """Find files anywhere under ``roots`` whose basename matches ``name``.

    Used to turn a not-found ``read_file`` into a self-correcting error: the
    model often guesses the wrong relative prefix or scope (e.g. ``sops/x.md``
    when the file lives at ``generalist/sops/x.md``), so we point it straight at
    the real scope-relative path instead of forcing a separate ``find_files``
    round-trip. Returns a comma-separated list of scope-relative paths, or "".
    """
    if not name:
        return ""
    found: list[str] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for entry in root.rglob(name):
            rel = entry.relative_to(root)
            if entry.is_file() and not any(part.startswith(".") for part in rel.parts):
                found.append(str(rel).replace("\\", "/"))
                if len(found) >= limit:
                    return ", ".join(found)
    return ", ".join(found)


async def _authorize(
    file_scope: FileScope,
    scope_name: str,
    context: dict[str, Any] | None,
    *,
    op_key: str,
    operation: str,
    detail: str,
) -> str | None:
    """Return a user-facing block message, or ``None`` if the operation is allowed.

    Looks up ``file_scope.<op_key>`` (with optional per-agent override),
    then: ``True`` → allow; ``False`` → block; ``"ask"`` → prompt via the
    reply channel.
    """
    perm: bool | str = getattr(file_scope, op_key)
    agent_id = (context or {}).get("agent_id")
    if agent_id and file_scope.agent_overrides:
        override = file_scope.agent_overrides.get(agent_id, {}).get(op_key)
        if override is not None:
            perm = override

    if perm is True:
        return None
    if perm is False:
        return f"Error: '{operation}' is not permitted in file scope '{scope_name}'."

    reply_to = (context or {}).get("reply_to")
    if reply_to is None or not hasattr(reply_to, "request_approval"):
        return (
            f"Error: scope '{scope_name}' requires approval to {operation}, "
            "but no approval channel is available."
        )
    approved = await reply_to.request_approval(
        title=f"File {operation} in scope '{scope_name}'",
        details=detail,
        timeout_seconds=300,
    )
    if not approved:
        return f"The {operation} was not approved, so no changes were made.\n{detail}"
    return None


def _scope_locations() -> str:
    """Render the file scopes as a compact location list for the system prompt.

    A scope is a named set of directories the file tools may touch; the model
    passes a scope name to each file tool. Only names and roots are surfaced --
    a routing hint so the model can pick the right scope. Per-operation
    permissions are intentionally omitted: the harness enforces run/approve/deny
    at call time, so the model never needs them in advance. Leading ``_`` keeps
    this out of the auto-discovered tool set.
    """
    file_access = load_paw_config().file_access
    if not file_access.scopes:
        return ""
    lines = [
        "FILE SCOPES (locations the file tools can access; pass `scope`, "
        f"default {file_access.default_scope}):"
    ]
    for name, scope in sorted(file_access.scopes.items()):
        lines.append(f"- {name}: {', '.join(scope.roots)}")
    return "\n".join(lines)


def read_file(path: str, scope: str = "kb", base_path: str = "") -> str:
    """Read a known file path. Reads agent knowledge, profiles, SOPs, notes.

    Prefer this once you know where something lives (from an index or a prior
    search) instead of browsing for it. Reads one file; when you need several,
    issue multiple read_file calls in a single round (the runtime runs them
    concurrently). For knowledge-base work use the default ``kb`` scope. If a
    read comes back not-found, the error suggests the real scope-relative path
    -- read that directly instead of issuing a separate search.

    Args:
        path: File path inside the scope. Relative paths resolve against the
            scope root, or against ``base_path``'s directory when provided.
        scope: Configured file scope. Defaults to ``kb``.
        base_path: Optional base file/dir (same scope) for resolving relatives.

    Returns: File contents as text, or an ``Error: ...`` message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        if file_scope.read is False:
            raise PermissionError(f"'read' is not permitted in file scope '{name}'.")
        target = _located(path, roots, name, base_path=base_path or None)
        if not target.is_file():
            suggestion = _suggest_paths(Path(path).name, roots)
            hint = f" Did you mean (scope '{name}'): {suggestion}?" if suggestion else ""
            return f"Error: Not a file: {target}.{hint}"
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"File too large ({size} bytes > {MAX_READ_BYTES}): {target}")
        return target.read_text(encoding="utf-8", errors="replace")
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


def list_dir(path: str = "", scope: str = "kb", base_path: str = "", depth: int = 1) -> str:
    """List directory entries from a file scope; one level, or a deeper tree.

    Use to orient within a scope. To see a whole subtree, call once with a
    larger ``depth`` rather than many single-level lists. To locate files by
    name use ``find_files``; to find content use ``grep_files``. Directories
    are suffixed with ``/``; dotfiles are hidden.

    Args:
        path: Directory path inside the scope. Empty lists the scope root.
        scope: Configured file scope. Defaults to ``kb``.
        base_path: Optional base file/dir (same scope) for resolving relatives.
        depth: Levels to descend. 1 (default) lists only the immediate entries;
            higher values return an indented tree of nested entries.

    Returns: Entries (indented tree when depth > 1), or an ``Error: ...`` message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        if file_scope.list is False:
            raise PermissionError(f"'list' is not permitted in file scope '{name}'.")
        target = _located(path, roots, name, base_path=base_path or None) if path else roots[0]
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")

        if depth <= 1:
            entries = [
                f"{entry.name}{'/' if entry.is_dir() else ''}"
                for entry in sorted(target.iterdir(), key=lambda p: p.name)
                if not entry.name.startswith(".")
            ]
            if not entries:
                return f"(empty directory: {path or '.'})"
            return "\n".join(entries)

        lines: list[str] = []
        truncated = False
        for entry in sorted(target.rglob("*"), key=lambda p: str(p.relative_to(target))):
            rel = entry.relative_to(target)
            if len(rel.parts) > depth or any(part.startswith(".") for part in rel.parts):
                continue
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}{entry.name}{'/' if entry.is_dir() else ''}")
            if len(lines) >= MAX_FIND_RESULTS:
                truncated = True
                break
        if not lines:
            return f"(empty directory: {path or '.'})"
        out = "\n".join(lines)
        if truncated:
            out += f"\n... [truncated at {MAX_FIND_RESULTS} entries]"
        return out
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def write_file(
    path: str,
    content: str,
    scope: str = "kb",
    _context: dict[str, Any] | None = None,
) -> str:
    """Create or overwrite a text file.

    Parent directories are created as needed. Prefer ``edit_file`` for partial
    changes and ``append_file`` to add at the end; use this for new files or
    full replacement only.
    Call at most once per file per turn; put all content in a single write.
    The returned status line confirms the write -- do not re-read to verify.

    Args:
        path: File path inside the scope.
        content: Full UTF-8 text to write.
        scope: Configured file scope. Defaults to ``kb``.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        target = _located(path, roots, name)
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ValueError(f"Content too large ({len(data)} bytes > {MAX_WRITE_BYTES}).")
        action = "overwrite" if target.exists() else "create"
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="write",
            operation=f"{action} file",
            detail=f"Scope: {name}\nPath: {target}\nBytes: {len(data)}\nPreview: {_preview(content)}",
        )
        if blocked:
            return blocked
        if not _within_roots(target.parent, roots):
            raise PermissionError(f"Parent directory not within file scope '{name}': {target.parent}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(data)} bytes to {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def append_file(
    path: str,
    content: str,
    scope: str = "kb",
    _context: dict[str, Any] | None = None,
) -> str:
    """Append text to a file; creates the file if missing. Never overwrites.

    Use for new notes, memories, or log entries. Prefer ``edit_file`` for in-place
    changes and ``write_file`` to replace a file wholesale.
    Call at most once per file per turn; combine what you add into one append.
    The returned status line confirms the bytes were written -- do not re-read
    the file afterward to verify.

    Args:
        path: File path inside the scope.
        content: UTF-8 text to append.
        scope: Configured file scope. Defaults to ``kb``.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        target = _located(path, roots, name)
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ValueError(f"Content too large ({len(data)} bytes > {MAX_WRITE_BYTES}).")
        action = "append to" if target.exists() else "create and append to"
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="append",
            operation="append to file",
            detail=f"Scope: {name}\nPath: {target}\nBytes: {len(data)}\nPreview: {_preview(content)}",
        )
        if blocked:
            return blocked
        if not _within_roots(target.parent, roots):
            raise PermissionError(f"Parent directory not within file scope '{name}': {target.parent}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write("\n" + content)
        return f"Appended {len(data)} bytes to {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    scope: str = "kb",
    replace_all: bool = False,
    _context: dict[str, Any] | None = None,
) -> str:
    """Replace text in an existing file.

    ``old_string`` must match exactly once unless ``replace_all`` is true;
    include enough surrounding context to make the match unique.
    Prefer this over ``write_file`` for partial changes.
    Call at most once per file per turn; combine multiple edits into one call.
    The returned status line confirms the change -- do not re-read to verify.

    Args:
        path: File path inside the scope.
        old_string: Exact text to find.
        new_string: Replacement text.
        scope: Configured file scope. Defaults to ``kb``.
        replace_all: Replace every occurrence instead of requiring uniqueness.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        if not old_string:
            return "Error: old_string must not be empty."
        name, file_scope, roots = _scope(scope)
        target = _located(path, roots, name)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {target}")
        original = target.read_text(encoding="utf-8", errors="replace")
        count = original.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {target}."
        if count > 1 and not replace_all:
            return (
                f"Error: old_string is not unique in {target} ({count} matches). "
                "Add surrounding context or set replace_all=True."
            )
        replacements = count if replace_all else 1
        updated = original.replace(old_string, new_string, -1 if replace_all else 1)
        old_snippet = _preview(old_string)
        new_snippet = _preview(new_string)
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="edit",
            operation="edit file",
            detail=(
                f"Scope: {name}\nPath: {target}\nReplacements: {replacements}\n"
                f"From: {old_snippet}\nTo:   {new_snippet}"
            ),
        )
        if blocked:
            return blocked
        target.write_text(updated, encoding="utf-8")
        suffix = "s" if replacements != 1 else ""
        return f"Replaced {replacements} occurrence{suffix} in {target}: {old_snippet[:60]!r} → {new_snippet[:60]!r}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def delete_file(
    path: str,
    scope: str = "kb",
    _context: dict[str, Any] | None = None,
) -> str:
    """Delete a file or an empty directory.

    Non-empty directories and scope roots are refused.

    Args:
        path: File or empty-directory path inside the scope.
        scope: Configured file scope. Defaults to ``kb``.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        target = _located(path, roots, name)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        if any(target == root for root in roots):
            raise PermissionError(f"Refusing to delete a scope root: {target}")
        is_dir = target.is_dir()
        if is_dir and any(target.iterdir()):
            raise OSError(f"Directory not empty: {target}")
        kind = "directory" if is_dir else "file"
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="delete",
            operation=f"delete {kind}",
            detail=f"Scope: {name}\nPath: {target}",
        )
        if blocked:
            return blocked
        target.rmdir() if is_dir else target.unlink()
        return f"Deleted {kind}: {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


def grep_files(pattern: str, scope: str = "kb", path: str = "", glob: str = "*") -> str:
    """Search file contents by regex within a scope (recursive).

    Use to find where something is recorded (a past memory, a decision). One
    well-chosen pattern usually locates it in a single call -- prefer that over
    browsing directories with repeated list_dir calls.

    Args:
        pattern: Python regular expression to search for.
        scope: Configured file scope. Defaults to ``kb``.
        path: Subdirectory to search under. Empty searches the scope root.
        glob: Filename glob to restrict which files are scanned (e.g. ``*.md``).

    Returns: Matching ``relpath:line:text`` lines, or an ``Error: ...`` message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        if file_scope.read is False:
            raise PermissionError(f"'read' is not permitted in file scope '{name}'.")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"
        base = _located(path, roots, name) if path else roots[0]
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        matches: list[str] = []
        truncated = False
        for file in sorted(base.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(base)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                if file.stat().st_size > MAX_READ_BYTES:
                    continue
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        truncated = True
                        break
            if truncated:
                break

        if not matches:
            return f"No matches for /{pattern}/ under {base}."
        out = "\n".join(matches)
        if truncated:
            out += f"\n... [truncated at {MAX_GREP_MATCHES} matches]"
        return out
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


def find_files(glob_pattern: str, scope: str = "kb", path: str = "") -> str:
    """Find files and directories by name glob within a scope (recursive).

    Use to locate files by name, or to see a subtree's layout, in one call
    instead of repeated single-level list_dir calls.

    Args:
        glob_pattern: Glob to match against paths, e.g. ``*.md`` or ``**/*.py``.
        scope: Configured file scope. Defaults to ``kb``.
        path: Subdirectory to search under. Empty searches the scope root.

    Returns: Newline-separated relative paths, or an ``Error: ...`` message.
    """
    try:
        name, file_scope, roots = _scope(scope)
        if file_scope.list is False:
            raise PermissionError(f"'list' is not permitted in file scope '{name}'.")
        base = _located(path, roots, name) if path else roots[0]
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        results: list[str] = []
        truncated = False
        for entry in sorted(base.rglob(glob_pattern)):
            rel = entry.relative_to(base)
            if any(part.startswith(".") for part in rel.parts):
                continue
            results.append(f"{rel}{'/' if entry.is_dir() else ''}")
            if len(results) >= MAX_FIND_RESULTS:
                truncated = True
                break

        if not results:
            return f"No entries matching '{glob_pattern}' under {base}."
        out = "\n".join(results)
        if truncated:
            out += f"\n... [truncated at {MAX_FIND_RESULTS} results]"
        return out
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Memory compaction
#
# Tool authors who want a smaller, more readable history record for their
# tool can register a compactor here. The registry adds the base fields
# (tool_call_id, name, params, content_chars); each function below returns
# only the *additive* fields. See paw/sessions/memory.py for the schema
# the renderer expects: ``preview``, ``error``, ``result_ref``,
# ``entry_count`` are the recognised optional keys.
# ---------------------------------------------------------------------------

_WRITE_PAYLOAD_KEYS = ("content", "old_string", "new_string")


def _file_ref(params: dict, kind: str, default_path: Any = None) -> dict:
    return {"type": kind, "scope": params.get("scope", "kb"), "path": params.get("path", default_path)}


def _is_error(content: str) -> bool:
    return content.startswith("Error:")


def _compact_read(result, params, limit):
    out: dict[str, Any] = {"result_ref": _file_ref(params, "file")}
    if _is_error(result.content):
        out["error"] = result.content[:limit]
    return out


def _compact_list(result, params, limit):
    content = result.content
    out: dict[str, Any] = {
        "result_ref": _file_ref(params, "directory", default_path=""),
        "entry_count": 0 if _is_error(content) else sum(1 for line in content.splitlines() if line.strip()),
    }
    if _is_error(content):
        out["error"] = content[:limit]
    return out


def _compact_write(result, params, limit):
    # Override params to strip large write payloads from persisted history.
    return {
        "params": {k: v for k, v in params.items() if k not in _WRITE_PAYLOAD_KEYS},
        "result_ref": _file_ref(params, "file"),
        "preview": result.content[:limit],
    }


def _compact_search(result, params, limit):
    return {"preview": result.content[:limit]}


COMPACTORS = {
    "read_file": _compact_read,
    "list_dir": _compact_list,
    "write_file": _compact_write,
    "edit_file": _compact_write,
    "delete_file": _compact_write,
    "grep_files": _compact_search,
    "find_files": _compact_search,
}
