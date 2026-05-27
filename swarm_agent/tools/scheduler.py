"""Scheduler tools. AI provides pre-parsed schedule values."""

from typing import Optional

from swarm_agent.scheduler.store import JobStore
from swarm_agent.scheduler.schedule import (
    next_run_at,
    parse_every,
    validate_at,
    validate_cron,
)

_store = JobStore()


def schedule_message(
    message: str,
    schedule_kind: str,
    schedule_value: str,
    name: str = "",
    session_id: str = "",
    intent: str = "execute",
    end_at: str = "",
) -> str:
    """
    Schedule a future or recurring task/message.

    Use when the user explicitly asks to remind, schedule, run later, repeat,
    or check something on a cadence. This tool only creates the scheduled job;
    do NOT execute the task immediately unless the user separately asks for
    immediate execution.

    For simple reminders that should just deliver text, use intent="say".
    For tasks that should be run by the agent at trigger time, use
    intent="execute".

    Args:
        message: The task to schedule. You MUST format the message using this exact template: "I am the user. I want you to [insert task description here]. Please execute now." Do not just copy the user's input; wrap it in this template.
        schedule_kind: "at" (one-shot), "every" (recurring), or "cron" (cron expression).
        schedule_value: Pre-parsed by AI. For "at" = ISO 8601 (e.g. "2025-02-24T09:00:00Z");
            for "every" = seconds or "1h"/"1d"/"30m"; for "cron" = "0 7 * * *".
        name: Optional name for the job.
        session_id: Current session ID (from session context; required). Used for delivery.
        intent: "execute" (default) to run as an instruction, or "say" to just deliver the message text to the user without AI processing.
        end_at: Optional ISO 8601 UTC timestamp to stop recurring jobs at/after this time.
            Only supported for "every" and "cron" schedules.

    Returns: Confirmation with job ID and next run time.
    """
    if not session_id:
        return (
            "session_id is required for scheduling. "
            "It should be provided by the current session context."
        )

    kind = (schedule_kind or "").strip().lower()
    if kind not in ("at", "every", "cron"):
        return f"schedule_kind must be 'at', 'every', or 'cron'. Got: {schedule_kind!r}"

    schedule: dict
    next_run: Optional[str] = None
    end_at_iso = str(end_at or "").strip()
    if end_at_iso and not validate_at(end_at_iso):
        return (
            f"Invalid ISO 8601 timestamp for 'end_at': {end_at!r}. "
            "Use format like 2025-02-24T09:00:00Z."
        )

    if kind == "at":
        if end_at_iso:
            return "end_at is only supported for recurring schedules ('every' or 'cron')."
        if not validate_at(str(schedule_value)):
            return f"Invalid ISO 8601 timestamp for 'at': {schedule_value!r}. Use format like 2025-02-24T09:00:00Z."
        at_iso = str(schedule_value).strip()
        schedule = {"kind": "at", "at": at_iso}
        next_run = at_iso

    elif kind == "every":
        interval = parse_every(schedule_value)
        if not interval:
            return (
                f"Invalid interval for 'every': {schedule_value!r}. "
                "Use seconds (e.g. 3600) or '1h', '1d', '30m'."
            )
        schedule = {"kind": "every", "interval_seconds": interval}
        if end_at_iso:
            schedule["end_at"] = end_at_iso
        next_run = next_run_at(schedule)
        if not next_run:
            if end_at_iso:
                return "No valid run can be scheduled before end_at."
            return "Failed to compute next run time for interval."

    elif kind == "cron":
        expr = str(schedule_value).strip()
        if not validate_cron(expr):
            return f"Invalid cron expression: {expr!r}. Use 5-field format (e.g. '0 7 * * *')."
        schedule = {"kind": "cron", "expr": expr}
        if end_at_iso:
            schedule["end_at"] = end_at_iso
        next_run = next_run_at(schedule)
        if not next_run:
            if end_at_iso:
                return "No valid run can be scheduled before end_at."
            return "Failed to compute next run time for cron expression."

    intent = (intent or "execute").strip().lower()
    if intent not in ("execute", "say"):
        return f"intent must be 'execute' or 'say'. Got: {intent!r}"

    job_name = (name or "").strip() or "Scheduled task"
    job = _store.add(
        name=job_name,
        schedule=schedule,
        message=message,
        delivery_session_id=session_id,
        next_run_at=next_run,
        intent=intent,
    )
    if kind == "at":
        schedule_details = f"at {schedule.get('at')}"
    elif kind == "every":
        schedule_details = f"every {schedule.get('interval_seconds')}s"
    else:
        schedule_details = f"cron '{schedule.get('expr')}'"
    if schedule.get("end_at"):
        schedule_details = f"{schedule_details} until {schedule['end_at']}"

    return (
        f"Scheduled: {job_name} (id: {job['id']}). "
        f"Schedule: {schedule_details}. "
        f"Next run: {next_run}. "
        f"Intent: {intent}."
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
        name = j.get("name", "?")
        job_id = j.get("id", "?")
        schedule = j.get("schedule", {})
        kind = schedule.get("kind", "?")
        if kind == "at":
            sched_str = schedule.get("at", "?")
        elif kind == "every":
            sec = schedule.get("interval_seconds", "?")
            sched_str = f"every {sec}s"
        else:
            sched_str = schedule.get("expr", "?")
        end_at = schedule.get("end_at")
        if end_at:
            sched_str = f"{sched_str} (until {end_at})"
        next_run = j.get("next_run_at", "?")
        lines.append(f"  - {name} (id: {job_id})")
        lines.append(f"    Schedule: {sched_str}")
        lines.append(f"    Next run: {next_run}")
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
