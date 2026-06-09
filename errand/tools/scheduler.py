"""Scheduling tools. The AI supplies pre-parsed schedule values."""

from typing import Any

from errand.scheduler.store import JobStore
from errand.scheduler.schedule import (
    next_run_at,
    parse_every,
    validate_at,
    validate_cron,
)

_store = JobStore()


def _build_schedule(kind: str, value: str, end_at: str) -> tuple[dict | None, str]:
    """Validate inputs and return (schedule, error); schedule is None on error."""
    if kind == "at":
        if end_at:
            return None, "end_at is only supported for recurring schedules ('every' or 'cron')."
        if not validate_at(value):
            return None, f"Invalid ISO 8601 timestamp for 'at': {value!r}. Use e.g. 2025-02-24T09:00:00Z."
        return {"kind": "at", "at": value.strip()}, ""
    if kind == "every":
        interval = parse_every(value)
        if not interval:
            return None, f"Invalid interval for 'every': {value!r}. Use seconds (3600) or '1h', '30m', '1d'."
        return {"kind": "every", "interval_seconds": interval}, ""
    if kind == "cron":
        expr = value.strip()
        if not validate_cron(expr):
            return None, f"Invalid cron expression: {expr!r}. Use 5-field format (e.g. '0 7 * * *')."
        return {"kind": "cron", "expr": expr}, ""
    return None, f"schedule_kind must be 'at', 'every', or 'cron'. Got: {kind!r}"


def _describe(schedule: dict) -> str:
    """Render a human-readable summary of a schedule."""
    kind = schedule["kind"]
    if kind == "at":
        text = f"at {schedule['at']}"
    elif kind == "every":
        text = f"every {schedule['interval_seconds']}s"
    else:  # cron
        text = f"cron '{schedule['expr']}'"
    end_at = schedule.get("end_at")
    return f"{text} until {end_at}" if end_at else text


def schedule_message(
    message: str,
    schedule_kind: str,
    schedule_value: str,
    name: str = "",
    intent: str = "execute",
    end_at: str = "",
    _context: dict[str, Any] | None = None,
) -> str:
    """Schedule a future or recurring task/message.

    Creates the job only; do NOT run the task now. The job fires later on its own.

    Args:
        message: What should happen when the job fires (NOT the user's scheduling
            request verbatim). Resolve the payload now:
            - intent="execute": the action to perform, phrased as an instruction to
              yourself, e.g. "Summarize today's unread emails and send them to the user."
              Never store the trigger phrasing like "remind me in 10 minutes" as the task.
            - intent="say": the exact text to deliver to the user, verbatim.
        schedule_kind: "at" (one-shot), "every" (recurring), or "cron".
        schedule_value: "at"=ISO 8601 UTC; "every"=seconds or "1h"/"30m"/"1d"; "cron"=5-field expr.
        name: Optional job name.
        intent: "execute" (run as instruction) or "say" (deliver text verbatim).
        end_at: ISO 8601 UTC end time for recurring schedules.

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

    schedule, error = _build_schedule(kind, str(schedule_value), end_at)
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
        f"Schedule: {_describe(schedule)}. Next run: {next_run}. Intent: {intent}."
    )


def list_scheduled_jobs() -> str:
    """List enabled scheduled jobs.

    Use when the user asks what is scheduled, wants to inspect reminders, or
    needs job IDs before cancelling.
    """
    jobs = _store.list_enabled()
    if not jobs:
        return "No scheduled jobs."
    lines = ["Scheduled jobs:"]
    for j in jobs:
        lines.append(f"  - {j['name']} (id: {j['id']})")
        lines.append(f"    Schedule: {_describe(j['schedule'])}")
        lines.append(f"    Next run: {j['next_run_at']}")
    return "\n".join(lines)


def cancel_scheduled_job(job_id: str) -> str:
    """Cancel a scheduled job by ID.

    Use when the user asks to cancel/delete/disable a scheduled reminder or
    recurring task. If the job ID is unknown, call list_scheduled_jobs first.
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return "job_id is required."
    if _store.disable(job_id):
        return f"Job {job_id} has been cancelled."
    return f"Job {job_id} not found."
