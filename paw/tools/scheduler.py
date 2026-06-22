"""Scheduling tools. The AI supplies pre-parsed schedule values."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paw.scheduler.store import JobStore
from paw.scheduler.schedule import (
    format_iso,
    next_run_at,
    parse_every,
    validate_at,
    validate_cron,
)

_store = JobStore()

# Zone assumed when the user names a clock time but no timezone is given.
DEFAULT_TZ_NAME = "America/New_York"


def _resolve_tz(tz: str) -> tuple[str, ZoneInfo | None, str]:
    """Return (tz_name, ZoneInfo|None, error_msg). On failure, ZoneInfo is None."""
    tz_name = (tz or "").strip() or DEFAULT_TZ_NAME
    try:
        return tz_name, ZoneInfo(tz_name), ""
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return tz_name, None, f"Unknown timezone: {tz!r}. Use an IANA name like America/New_York."


def _build_schedule(kind: str, value: str, end_at: str, tz: str = "") -> tuple[dict | None, str]:
    """Validate inputs and return (schedule, error); schedule is None on error."""
    if kind == "at":
        if end_at:
            return None, "end_at is only supported for recurring schedules ('every' or 'cron')."
        value = value.strip()
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None, (
                f"Invalid time for 'at': {value!r}. Provide a local date+time like "
                f"2026-06-10T11:00 (with an optional timezone), or an absolute UTC "
                f"time like 2026-06-10T15:00:00Z."
            )
        # A naive wall-clock time is interpreted in `tz` (default US East) so the
        # model never does timezone math; a value that already carries Z/offset
        # is absolute and converts straight to UTC.
        if dt.tzinfo is None:
            tz_name, zone, err = _resolve_tz(tz)
            if err:
                return None, err
            dt = dt.replace(tzinfo=zone)
        else:
            tz_name = "UTC"
        return {"kind": "at", "at": format_iso(dt), "tz": tz_name}, ""
    if kind == "every":
        interval = parse_every(value)
        if not interval:
            return None, f"Invalid interval for 'every': {value!r}. Use seconds (3600) or '1h', '30m', '1d'."
        return {"kind": "every", "interval_seconds": interval}, ""
    if kind == "cron":
        expr = value.strip()
        if not validate_cron(expr):
            return None, f"Invalid cron expression: {expr!r}. Use 5-field format (e.g. '0 7 * * *')."
        tz_name, _, err = _resolve_tz(tz)
        if err:
            return None, err
        return {"kind": "cron", "expr": expr, "tz": tz_name}, ""
    return None, f"schedule_kind must be 'at', 'every', or 'cron'. Got: {kind!r}"


def _describe(schedule: dict) -> str:
    """Render a human-readable summary of a schedule, with timezone."""
    kind = schedule["kind"]
    tz = schedule.get("tz", "UTC")
    if kind == "at":
        text = f"at {schedule['at']} ({tz})"
    elif kind == "every":
        text = f"every {schedule['interval_seconds']}s"
    else:  # cron
        text = f"cron '{schedule['expr']}' ({tz})"
    end_at = schedule.get("end_at")
    return f"{text} until {end_at}" if end_at else text


def _preview(text: str, limit: int = 120) -> str:
    """One-line, length-capped preview of a stored message."""
    s = (text or "").strip().replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def schedule_message(
    message: str,
    schedule_kind: str,
    schedule_value: str,
    name: str = "",
    intent: str = "execute",
    end_at: str = "",
    timezone: str = "",
    _context: dict[str, Any] | None = None,
) -> str:
    """Schedule a future or recurring task/message.

    Create a future or recurring job. Use for user requests to schedule
    reminders, messages, or actions. The job fires later on its own.

    Args:
        message: What should happen when the job fires (NOT the user's scheduling
            request verbatim). Resolve the payload now:
            - intent="execute": the action to perform, phrased as an instruction to
              yourself, e.g. "Summarize today's unread emails and send them to the user."
              Never store the trigger phrasing like "remind me in 10 minutes" as the task.
            - intent="say": the exact text to deliver to the user, verbatim.
        schedule_kind: "at" (one-shot), "every" (recurring), or "cron".
        schedule_value: For "at", EITHER a local wall-clock time with NO offset
            (e.g. "2026-06-10T11:00") which is interpreted in `timezone`, OR an
            absolute UTC time ending in Z (e.g. "2026-06-10T15:00:00Z").
            For a relative request ("in 3 hours"), compute now (UTC is shown in
            context) + the offset and pass that as an absolute "...Z" value.
            "every"=seconds or "1h"/"30m"/"1d"; "cron"=5-field expr.
        name: Optional job name.
        intent: "execute" (run as instruction) or "say" (deliver text verbatim).
        end_at: ISO 8601 UTC end time for recurring schedules.
        timezone: IANA zone name, e.g. "America/New_York" (美东) or
            "America/Los_Angeles" (美西). Defaults to America/New_York if omitted.
            For "at": a naive wall-clock value is interpreted in this zone; an
            absolute "...Z" value ignores it. For "cron": the expression fields
            are evaluated in this zone (e.g. "30 9 * * 5" fires at 09:30 local
            time, not 09:30 UTC). Do NOT convert timezones yourself.

    Returns: Confirmation with job ID and next run time.
    """
    session_id = str((_context or {}).get("session_id") or "").strip()
    if not session_id:
        return "Scheduling is unavailable: no active session context."

    kind = (schedule_kind or "").strip().lower()
    intent = (intent or "execute").strip().lower()
    if intent not in ("execute", "say"):
        return f"intent must be 'execute' or 'say'. Got: {intent!r}"

    end_at = str(end_at or "").strip()
    if end_at and not validate_at(end_at):
        return f"Invalid ISO 8601 timestamp for 'end_at': {end_at!r}. Use e.g. 2025-02-24T09:00:00Z."

    schedule, error = _build_schedule(kind, str(schedule_value), end_at, str(timezone or ""))
    if error:
        return error
    if end_at:
        schedule["end_at"] = end_at

    next_run = next_run_at(schedule)
    if not next_run:
        return "No valid run can be scheduled" + (f" before {end_at}." if end_at else ".")

    job = _store.add(
        name=(name or "").strip() or "Scheduled task",
        schedule=schedule,
        message=message,
        delivery_session_id=session_id,
        next_run_at=next_run,
        intent=intent,
    )
    return (
        f"Scheduled: {job['name']} (id: {job['id']}). "
        f"Schedule: {_describe(schedule)}. Next run: {next_run}. "
        f"Intent: {intent}. Message: {_preview(message)!r}."
    )


def list_scheduled_jobs() -> str:
    """List enabled scheduled jobs.

    Use when the user asks what is scheduled, wants to inspect reminders, or
    needs job IDs before cancelling.
    """
    jobs = _store.list_enabled()
    if not jobs:
        return "No scheduled jobs."
    lines = [f"Scheduled jobs ({len(jobs)}):"]
    for j in jobs:
        lines.append(f"  - {j['name']} (id: {j['id']})")
        lines.append(f"    Schedule: {_describe(j['schedule'])}")
        lines.append(f"    Next run: {j['next_run_at']}")
        lines.append(f"    Intent: {j.get('intent', 'execute')}. Message: {_preview(j.get('message', ''), 80)!r}")
    return "\n".join(lines)


def cancel_scheduled_job(job_id: str) -> str:
    """Cancel a scheduled job by ID.

    Use when the user asks to cancel/delete/disable a scheduled reminder or
    recurring task. If the job ID is unknown, call list_scheduled_jobs first.
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return "job_id is required."
    cancelled = _store.update(job_id, enabled=False)
    if cancelled is None:
        return f"Job {job_id} not found."
    return (
        f"Cancelled: {cancelled['name']} (id: {cancelled['id']}). "
        f"Was: {_describe(cancelled['schedule'])}. "
        f"Message: {_preview(cancelled.get('message', ''))!r}."
    )
