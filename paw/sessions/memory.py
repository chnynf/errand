"""Session persistence and conversation history.

Session JSON files live under ``paw/sessions/_data/``. The
``_data/`` folder is git-ignored runtime state, separate from the
Python modules in the same package.

The history is a single unified log of provider-ready chat messages
(``system``/``user``/``assistant``/``tool`` vocabulary -- the same format the
model API consumes). Each finished exchange is folded ONCE at step-out by the
agent loop (skeleton tool-call anchors + compacted tool results, see
``AgentLoop._fold_exchange``) and appended here via ``add_exchange``. Replay is
therefore a plain windowed slice: no role translation, no per-record rendering.
The current, in-flight exchange never lives here -- the loop carries it live at
full fidelity until it finishes.

History replays in two resolutions. Exchanges leave the recent (full-folded)
log for the ``distant`` section -- compacted to just the user text and the
final assistant text -- when either a >=2h silence gap ends the conversation
or the recent token budget pushes them out (see ``roll_distant``). Replay is
``distant + recent``, both plain chat messages.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

_SESSION_DIR = Path(__file__).resolve().parent / "_data"
_MAX_HISTORY_ENTRIES = 400
# History window sent to the model, budgeted in tokens (~chars/4). Exchanges
# vary in length, so we cap by tokens rather than a fixed entry count.
_HISTORY_TOKEN_BUDGET = 10_000
# When over budget, drop the oldest exchanges this many at a time. Dropping in
# groups (not one-by-one) keeps the window start stable across several
# exchanges, so the system+history prefix stays byte-identical and provider
# prefix caching survives. We only drop while >3 exchanges remain, so at least
# one exchange always survives (4 - 3 = 1).
_HISTORY_DROP_EXCHANGES = 3
_HISTORY_MIN_EXCHANGES = 1
# A silence gap this long ends a "conversation": on the next message every
# recent exchange rolls into the distant section (compacted to Q&A). The
# provider prefix cache is long cold after such a gap, so the roll is free.
_DISTANT_GAP_SECONDS = 2 * 3600
# Distant-section budget in content chars. When the cap is exceeded, the oldest
# whole Q&A pairs are dropped down to the keep target (not just below the cap),
# so the section stays byte-stable across many exchanges for prefix caching --
# the same stepping philosophy as _HISTORY_DROP_EXCHANGES above.
_DISTANT_MAX_CHARS = 2_000
_DISTANT_KEEP_CHARS = 1_000
# Characters that are illegal in Windows filenames (e.g. ``:`` in WeChat and
# scheduled session ids). Mapped to ``_`` so files stay cross-platform safe.
_FILENAME_ILLEGAL = '<>:"/\\|?*'


def _safe_stem(session_id: str) -> str:
    return "".join("_" if c in _FILENAME_ILLEGAL else c for c in session_id)


class Memory:
    """Persistent conversation history for one session.

    Each history entry wraps one provider-ready chat message:
    ``{"timestamp": <epoch>, "message": {"role": ..., ...}}``. An exchange is a
    ``user`` message followed by its assistant/tool messages, ending at the next
    ``user`` message; exchanges are appended whole via ``add_exchange``.
    """

    def __init__(self, session_id: str = "default", agent_id: str = "default"):
        self.session_id = session_id
        self.agent_id = agent_id
        self.session_dir = str(_SESSION_DIR)
        self._file_stem = _safe_stem(session_id)
        self.session_file = os.path.join(self.session_dir, f"{self._file_stem}.json")
        self._ensure_session_dir()
        self.data: Dict[str, Any] = self._load_session()
        self.data.setdefault("metadata", {})["agent_id"] = self.agent_id

    def _ensure_session_dir(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "archive"), exist_ok=True)

    def _load_session(self) -> Dict[str, Any]:
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                return self._create_new_session()
            # Pre-unified-log entries (old ai/tool record schema) are not
            # replayable as chat messages; drop them rather than translate.
            data["history"] = [
                e for e in data.get("history", []) if isinstance(e.get("message"), dict)
            ]
            # Sessions saved before the distant section existed load with an
            # empty one.
            data.setdefault("distant", [])
            return data
        return self._create_new_session()

    def _create_new_session(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": time.time(),
            "history": [],
            "distant": [],
            "context_summary": None,
            "metadata": {},
            "token_summary": {
                "input_tokens": 0,
                "output_tokens": 0,
                # Subset of input_tokens that were served from a prompt cache.
                "cache_read_tokens": 0,
            },
        }

    async def save_session(self) -> None:
        history = self.data["history"]
        if len(history) > _MAX_HISTORY_ENTRIES:
            # Trim on an exchange boundary so no tool message loses its anchor.
            cut = len(history) - _MAX_HISTORY_ENTRIES
            starts = self._exchange_starts()
            boundary = next((s for s in starts if s >= cut), None)
            if boundary is not None:
                self.data["history"] = history[boundary:]
        async with aiofiles.open(self.session_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self.data, indent=2, ensure_ascii=False))

    def add_exchange(self, messages: list[dict]) -> None:
        """Append one folded, finished exchange to the log.

        ``messages`` are provider-ready chat messages, starting with the
        exchange's ``user`` message (the loop folds them at step-out).
        """
        now = time.time()
        self.data["history"].extend(
            {"timestamp": now, "message": message} for message in messages
        )

    def update_token_usage(self, usage: Dict[str, Any]) -> None:
        """Accumulate one API call's token counts into ``token_summary``.

        Counts come straight from the provider's ``usage`` object. Paw does
        not track cost (rates vary too much across providers to maintain).
        """
        summary = self.data["token_summary"]
        summary["input_tokens"] += usage.get("input_tokens", 0) or 0
        summary["output_tokens"] += usage.get("output_tokens", 0) or 0
        summary.setdefault("cache_read_tokens", 0)
        summary["cache_read_tokens"] += usage.get("cache_read_tokens", 0) or 0

    def set_token_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
    ) -> None:
        """Set ``token_summary`` to absolute lifetime totals.

        Used at exchange step-out to rebuild the session total as the
        pre-exchange baseline plus this exchange's full cross-agent usage,
        rather than accumulating per call. This is what keeps a session total
        from ever reading lower than the exchange it just reported.
        """
        summary = self.data.setdefault(
            "token_summary",
            {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
        )
        summary["input_tokens"] = input_tokens
        summary["output_tokens"] = output_tokens
        summary["cache_read_tokens"] = cache_read_tokens

    def set_context_summary(self, summary: Optional[str]) -> None:
        self.data["context_summary"] = summary

    def last_activity_at(self) -> Optional[float]:
        history = self.data.get("history") or []
        if history:
            return float(history[-1].get("timestamp") or 0)
        # After a gap roll the recent log is empty but the session is not new;
        # the distant section still carries the last activity time.
        distant = self.data.get("distant") or []
        if distant:
            return float(distant[-1].get("timestamp") or 0)
        return float(self.data.get("created_at") or 0) or None

    @staticmethod
    def _entry_tokens(entry: dict) -> int:
        """Rough token estimate (~chars/4) for one persisted history entry."""
        return len(json.dumps(entry["message"], ensure_ascii=False, default=str)) // 4

    def _exchange_starts(self) -> list[int]:
        """Entry indices where each exchange begins (its ``user`` message).

        An exchange spans one such entry up to (but not including) the next, so
        its anchor/tool messages never get split across the window edge.
        """
        return [
            i
            for i, e in enumerate(self.data["history"])
            if e["message"].get("role") == "user"
        ]

    def _history_window_start(self, starts: list[int]) -> int:
        """First entry index of the token-budgeted history window.

        Keep the newest whole exchanges; while over budget and more than
        ``_HISTORY_DROP_EXCHANGES + _HISTORY_MIN_EXCHANGES`` remain, drop the
        oldest ``_HISTORY_DROP_EXCHANGES``. Dropping in groups keeps the start
        stable across several exchanges so the prefix stays prefix-cacheable;
        the floor keeps at least ``_HISTORY_MIN_EXCHANGES`` even if one is
        oversized.
        """
        full = self.data["history"]
        first = 0
        window_tokens = sum(self._entry_tokens(e) for e in full[starts[first]:])
        min_keep = _HISTORY_DROP_EXCHANGES + _HISTORY_MIN_EXCHANGES
        while window_tokens > _HISTORY_TOKEN_BUDGET and len(starts) - first > min_keep:
            dropped = full[starts[first]:starts[first + _HISTORY_DROP_EXCHANGES]]
            window_tokens -= sum(self._entry_tokens(e) for e in dropped)
            first += _HISTORY_DROP_EXCHANGES
        return starts[first]

    def roll_distant(self, now: float) -> bool:
        """Migrate old exchanges into the distant section; True on a gap roll.

        Called by the loop at exchange start, before history is built. Two
        triggers, whichever fires:

        - Silence gap: ``now`` is >= ``_DISTANT_GAP_SECONDS`` past the last
          activity, so the previous conversation is over -- every recent
          exchange rolls into distant. The provider prefix cache is long cold
          after such a gap, so this invalidation costs nothing.
        - Token budget: the recent log exceeds ``_HISTORY_TOKEN_BUDGET``, so
          the same group-drop that used to silently window old exchanges out
          (see ``_history_window_start``) now routes them into distant. The
          drop cadence is unchanged, so distant's young edge and recent's old
          edge move in the same single cache-invalidation event.

        Either way the migrated exchanges are compacted to Q&A pairs and the
        distant section is re-trimmed to its char budget.
        """
        history = self.data["history"]
        distant = self.data.setdefault("distant", [])
        last = self.last_activity_at()
        gap_roll = bool(history) and last is not None and (
            now - last >= _DISTANT_GAP_SECONDS
        )
        if gap_roll:
            distant.extend(self._compact_exchanges(history))
            self.data["history"] = []
        else:
            starts = self._exchange_starts()
            if starts:
                start = self._history_window_start(starts)
                if start > 0:
                    distant.extend(self._compact_exchanges(history[:start]))
                    self.data["history"] = history[start:]
        self._trim_distant()
        return gap_roll

    @staticmethod
    def _compact_exchanges(entries: list[dict]) -> list[dict]:
        """Compact whole exchanges to distant Q&A entries.

        Per exchange: the ``user`` message plus the last assistant message that
        has text content. Tool anchors and tool messages are dropped together
        (an anchor without its results would be an illegal message), so every
        compacted exchange replays as a legal ``[user, assistant]`` pair.
        """
        exchanges: list[list[dict]] = []
        for entry in entries:
            if entry["message"].get("role") == "user" or not exchanges:
                exchanges.append([])
            exchanges[-1].append(entry)

        compacted: list[dict] = []
        for exchange in exchanges:
            user = next(
                (e for e in exchange if e["message"].get("role") == "user"), None
            )
            final = next(
                (
                    e
                    for e in reversed(exchange)
                    if e["message"].get("role") == "assistant"
                    and e["message"].get("content")
                ),
                None,
            )
            if not (user and final):
                continue
            compacted.append(
                {
                    "timestamp": user["timestamp"],
                    "message": {
                        "role": "user",
                        "content": str(user["message"].get("content") or ""),
                    },
                }
            )
            compacted.append(
                {
                    "timestamp": final["timestamp"],
                    "message": {
                        "role": "assistant",
                        "content": str(final["message"]["content"]),
                    },
                }
            )
        return compacted

    @staticmethod
    def _distant_chars(entries: list[dict]) -> int:
        return sum(len(str(e["message"].get("content") or "")) for e in entries)

    def _trim_distant(self) -> None:
        """Drop oldest whole Q&A pairs once distant exceeds its char cap.

        Trims down to ``_DISTANT_KEEP_CHARS`` (not just below the cap), so the
        section start then holds steady across many exchanges -- the same
        stepping that keeps the recent window prefix-cacheable. Never splits a
        pair; a single oversized pair survives intact.
        """
        distant = self.data.get("distant") or []
        if self._distant_chars(distant) <= _DISTANT_MAX_CHARS:
            return
        starts = [
            i for i, e in enumerate(distant) if e["message"].get("role") == "user"
        ]
        first = 0
        remaining = self._distant_chars(distant)
        while remaining > _DISTANT_KEEP_CHARS and first + 1 < len(starts):
            dropped = distant[starts[first]:starts[first + 1]]
            remaining -= self._distant_chars(dropped)
            first += 1
        self.data["distant"] = distant[starts[first]:] if starts else []

    def build_history_messages(self) -> list[dict]:
        """Return distant + recent history as provider-ready chat messages.

        Both sections already store chat messages, so this is concatenation of
        two plain slices: the whole distant section (char-budgeted at roll
        time), then the token-budgeted recent window (see
        ``_history_window_start``), cut only on exchange boundaries so
        skeleton-anchor/tool pairs stay intact and the window start holds
        steady across exchanges for prefix caching. ``roll_distant`` normally
        keeps recent within budget already; the windowing here is the backstop.
        """
        distant = [e["message"] for e in self.data.get("distant") or []]
        starts = self._exchange_starts()
        if not starts:
            return distant
        start = self._history_window_start(starts)
        return distant + [e["message"] for e in self.data["history"][start:]]

    async def archive_session(self, start_new: bool = True) -> None:
        if os.path.exists(self.session_file):
            archive_dir = os.path.join(self.session_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)

            archive_path = os.path.join(
                archive_dir, f"{self._file_stem}_{int(time.time())}.json"
            )
            print(f"Archiving session {self.session_id} to {archive_path}")
            try:
                os.rename(self.session_file, archive_path)
                print(f"Successfully archived session {self.session_id}")
            except Exception as e:
                print(f"Failed to archive session {self.session_id}: {e}")
                raise

            if start_new:
                self.data = self._create_new_session()
                await self.save_session()
