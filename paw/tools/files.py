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
from typing import Any

from paw.config import FileScope, load_paw_config

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000
MAX_GREP_MATCHES = 200
MAX_FIND_RESULTS = 500

_FS_ERRORS = (OSError, ValueError)


def _expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def _scope_roots() -> list[tuple[str, FileScope, Path]]:
    """All configured permission zones as ``(name, scope, resolved_root)``.

    Scopes are concentric (e.g. ``review`` ⊂ ``notes`` ⊂ ``kb``). Sorted by
    root-path length descending so the FIRST entry containing a target is the
    *most specific* zone -- the one whose per-operation permissions govern it.
    """
    config = load_paw_config().file_access
    entries: list[tuple[str, FileScope, Path]] = []
    for name, scope in config.scopes.items():
        for root in scope.roots:
            entries.append((name, scope, _expand(root)))
    entries.sort(key=lambda e: len(str(e[2])), reverse=True)
    return entries


def _base_root() -> Path:
    """Root that bare relative paths resolve against (the default scope's root).

    This is the broad outer zone; everything the model can touch lives under it,
    so the model only ever needs paths relative to this single root and never
    names a scope.
    """
    config = load_paw_config().file_access
    default = config.scopes.get(config.default_scope)
    if default is None or not default.roots:
        raise ValueError("No default file scope with a root is configured.")
    return _expand(default.roots[0])


def _resolve(path: str, *, base_path: str | None = None) -> Path:
    base_root = _base_root()
    raw = Path(os.path.expanduser(path))
    if raw.is_absolute():
        return raw.resolve()
    if base_path:
        base_raw = Path(os.path.expanduser(base_path))
        base_target = (
            base_raw.resolve()
            if base_raw.is_absolute()
            else (base_root / base_raw).resolve()
        )
        base_dir = base_target if base_target.is_dir() else base_target.parent
        return (base_dir / raw).resolve()
    return (base_root / raw).resolve()


def _locate(path: str, *, base_path: str | None = None) -> tuple[Path, str, FileScope]:
    """Resolve ``path`` and return ``(target, scope_name, governing_scope)``.

    The governing scope is the most specific configured zone containing the
    target; its per-operation permissions are what the caller enforces. A path
    outside every zone is rejected here, so the model can never reach -- or even
    perceive -- anything beyond the allowed roots.
    """
    target = _resolve(path, base_path=base_path)
    for name, scope, root in _scope_roots():
        try:
            target.relative_to(root)
            return target, name, scope
        except ValueError:
            continue
    raise PermissionError(f"Path is outside all allowed locations: {target}")


def _read_cache(_context: dict[str, Any] | None) -> Any | None:
    """The per-exchange file read cache, or ``None`` when unavailable.

    Duck-typed off the ``run_context`` the loop injects, so this module keeps no
    import dependency on the runtime. Absent (direct calls, tests) means caching
    is simply skipped and every read hits disk.
    """
    return getattr((_context or {}).get("run_context"), "read_cache", None)


def _invalidate(_context: dict[str, Any] | None, target: Path) -> None:
    """Drop ``target`` from the read cache after a successful mutation."""
    cache = _read_cache(_context)
    if cache is not None:
        cache.invalidate(str(target))


def _preview(text: str, n: int = 200) -> str:
    return text[:n].replace("\n", "↵")


def _suggest_paths(name: str, limit: int = 5) -> str:
    """Find files under the base root whose basename matches ``name``.

    Turns a not-found ``read_file`` into a self-correcting error: the model
    often guesses the wrong relative prefix (e.g. ``sops/x.md`` when the file is
    at ``generalist/sops/x.md``), so we point it straight at the real path
    instead of forcing a separate ``find_files`` round-trip. Returns a
    comma-separated list of base-root-relative paths, or "".
    """
    if not name:
        return ""
    root = _base_root()
    if not root.is_dir():
        return ""
    found: list[str] = []
    for entry in root.rglob(name):
        rel = entry.relative_to(root)
        if entry.is_file() and not any(part.startswith(".") for part in rel.parts):
            found.append(str(rel).replace("\\", "/"))
            if len(found) >= limit:
                break
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
        return f"Error: '{operation}' is not permitted for this path."

    reply_to = (context or {}).get("reply_to")
    if reply_to is None or not hasattr(reply_to, "request_approval"):
        return (
            f"Error: '{operation}' requires approval, "
            "but no approval channel is available."
        )
    # The approval prompt is human-facing, so the internal zone name is fine here.
    approved = await reply_to.request_approval(
        title=f"File {operation} (zone '{scope_name}')",
        details=detail,
        timeout_seconds=300,
    )
    if not approved:
        return f"The {operation} was not approved, so no changes were made.\n{detail}"
    return None


def read_file(path: str, base_path: str = "", _context: dict[str, Any] | None = None) -> str:
    """Read a known file path. Reads agent knowledge, profiles, SOPs, notes.

    Prefer this once you know where something lives (from an index or a prior
    search) instead of browsing for it. Reads one file; when you need several,
    issue multiple read_file calls in a single round (the runtime runs them
    concurrently). If a read comes back not-found, the error suggests the real
    path -- read that directly instead of issuing a separate search.

    Args:
        path: The file path to read -- absolute, or relative. A relative path
            resolves against ``base_path`` when given, otherwise against the
            knowledge-base root.
        base_path: What a relative ``path`` is relative to. Index files list
            their entries relative to the index's own location, so when ``path``
            comes from an index, pass that index's path here. Omit it only for a
            path that is already relative to the knowledge-base root.

    Returns: File contents as text, or an ``Error: ...`` message.
    """
    try:
        target, _, file_scope = _locate(path, base_path=base_path or None)
        if file_scope.read is False:
            raise PermissionError("Reading is not permitted for this path.")
        # A file read once this exchange is served from the per-exchange cache,
        # so a re-read costs no disk I/O. Only successful reads are cached
        # (below), so a not-found here can still succeed after a later create.
        cache = _read_cache(_context)
        key = str(target)
        if cache is not None and (hit := cache.get(key)) is not None:
            return hit
        if not target.is_file():
            suggestion = _suggest_paths(Path(path).name)
            hint = f" Did you mean: {suggestion}?" if suggestion else ""
            return f"Error: Not a file: {target}.{hint}"
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"File too large ({size} bytes > {MAX_READ_BYTES}): {target}")
        content = target.read_text(encoding="utf-8", errors="replace")
        if cache is not None:
            cache.put(key, content)
        return content
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


def list_dir(path: str = "", base_path: str = "", depth: int = 1) -> str:
    """List directory entries; one level, or a deeper tree.

    Use to orient yourself. To see a whole subtree, call once with a larger
    ``depth`` rather than many single-level lists. To locate files by name use
    ``find_files``; to find content use ``grep_files``. Directories are suffixed
    with ``/``; dotfiles are hidden.

    Args:
        path: Directory path. Empty lists the knowledge-base root.
        base_path: Optional base file/dir for resolving a relative ``path``.
        depth: Levels to descend. 1 (default) lists only the immediate entries;
            higher values return an indented tree of nested entries.

    Returns: Entries (indented tree when depth > 1), or an ``Error: ...`` message.
    """
    try:
        target, _, file_scope = _locate(path or ".", base_path=base_path or None)
        if file_scope.list is False:
            raise PermissionError("Listing is not permitted for this path.")
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
    _context: dict[str, Any] | None = None,
) -> str:
    """Create or overwrite a text file.

    Parent directories are created as needed. Prefer ``edit_file`` for partial
    changes and ``append_file`` to add at the end; use this for new files or
    full replacement only.
    Call at most once per file per turn; put all content in a single write.
    The status returned confirms success; 
    Do not re-read the file to verify it or to retrieve its contents unless necessary.

    Args:
        path: File path. Relative paths resolve against the knowledge-base root.
        content: Full UTF-8 text to write.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        target, name, file_scope = _locate(path)
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
            detail=f"Path: {target}\nBytes: {len(data)}\nPreview: {_preview(content)}",
        )
        if blocked:
            return blocked
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _invalidate(_context, target)
        return f"Wrote {len(data)} bytes to {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def append_file(
    path: str,
    content: str,
    _context: dict[str, Any] | None = None,
) -> str:
    """Append text to a file; creates the file if missing. Never overwrites.

    Use for new notes, memories, or log entries. Prefer ``edit_file`` for in-place
    changes and ``write_file`` to replace a file wholesale.
    Call at most once per file per turn; combine what you add into one append.
    The status returned confirms success;
    Do not re-read the file to verify it or to retrieve its contents unless necessary.

    Args:
        path: File path. Relative paths resolve against the knowledge-base root.
        content: UTF-8 text to append.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        target, name, file_scope = _locate(path)
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ValueError(f"Content too large ({len(data)} bytes > {MAX_WRITE_BYTES}).")
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="append",
            operation="append to file",
            detail=f"Path: {target}\nBytes: {len(data)}\nPreview: {_preview(content)}",
        )
        if blocked:
            return blocked
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write("\n" + content)
        _invalidate(_context, target)
        return f"Appended {len(data)} bytes to {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    _context: dict[str, Any] | None = None,
) -> str:
    """Replace text in an existing file.

    ``old_string`` must match exactly once unless ``replace_all`` is true;
    include enough surrounding context to make the match unique.
    Prefer this over ``write_file`` for partial changes.
    Call at most once per file per turn; combine multiple edits into one call.
    The status returned confirms success;
    Do not re-read the file to verify it or to retrieve its contents unless necessary.

    Args:
        path: File path. Relative paths resolve against the knowledge-base root.
        old_string: Exact text to find.
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of requiring uniqueness.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        if not old_string:
            return "Error: old_string must not be empty."
        target, name, file_scope = _locate(path)
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
                f"Path: {target}\nReplacements: {replacements}\n"
                f"From: {old_snippet}\nTo:   {new_snippet}"
            ),
        )
        if blocked:
            return blocked
        target.write_text(updated, encoding="utf-8")
        _invalidate(_context, target)
        suffix = "s" if replacements != 1 else ""
        return f"Replaced {replacements} occurrence{suffix} in {target}: {old_snippet[:60]!r} → {new_snippet[:60]!r}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


async def delete_file(
    path: str,
    _context: dict[str, Any] | None = None,
) -> str:
    """Delete a file or an empty directory.

    Non-empty directories and the knowledge-base roots themselves are refused.

    Args:
        path: File or empty-directory path. Relative paths resolve against the
            knowledge-base root.

    Returns: A status line, or an ``Error: ...`` / not-approved message.
    """
    try:
        target, name, file_scope = _locate(path)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        if any(target == root for _, _, root in _scope_roots()):
            raise PermissionError(f"Refusing to delete a protected root: {target}")
        is_dir = target.is_dir()
        if is_dir and any(target.iterdir()):
            raise OSError(f"Directory not empty: {target}")
        kind = "directory" if is_dir else "file"
        blocked = await _authorize(
            file_scope, name, _context,
            op_key="delete",
            operation=f"delete {kind}",
            detail=f"Path: {target}",
        )
        if blocked:
            return blocked
        target.rmdir() if is_dir else target.unlink()
        _invalidate(_context, target)
        return f"Deleted {kind}: {target}."
    except _FS_ERRORS as exc:
        return f"Error: {exc}"


def grep_files(pattern: str, path: str = "", glob: str = "*") -> str:
    """Search file contents by regex (recursive).

    Use to find where something is recorded (a past memory, a decision). One
    well-chosen pattern usually locates it in a single call -- prefer that over
    browsing directories with repeated list_dir calls.

    Args:
        pattern: Python regular expression to search for.
        path: Subdirectory to search under. Empty searches the knowledge-base root.
        glob: Filename glob to restrict which files are scanned (e.g. ``*.md``).

    Returns: Matching ``relpath:line:text`` lines, or an ``Error: ...`` message.
    """
    try:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"
        base, _, file_scope = _locate(path or ".")
        if file_scope.read is False:
            raise PermissionError("Reading is not permitted for this path.")
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


def find_files(glob_pattern: str, path: str = "") -> str:
    """Find files and directories by name glob (recursive).

    Use to locate files by name, or to see a subtree's layout, in one call
    instead of repeated single-level list_dir calls.

    Args:
        glob_pattern: Glob to match against paths, e.g. ``*.md`` or ``**/*.py``.
        path: Subdirectory to search under. Empty searches the knowledge-base root.

    Returns: Newline-separated relative paths, or an ``Error: ...`` message.
    """
    try:
        base, _, file_scope = _locate(path or ".")
        if file_scope.list is False:
            raise PermissionError("Listing is not permitted for this path.")
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
    return {"type": kind, "path": params.get("path", default_path)}


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
