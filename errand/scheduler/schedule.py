"""Schedule validation and next-run computation. AI provides pre-parsed values."""

import re
from datetime import datetime, timezone
from typing import Optional


def validate_at(value: str) -> bool:
    """Validate ISO 8601 timestamp. AI provides pre-parsed value."""
    if not value or not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def parse_every(value: str | int) -> Optional[int]:
    """Parse interval to seconds. Accepts 3600, '3600', '1h', '1d', '30m'."""
    if isinstance(value, int):
        return value if value > 0 else None
    s = str(value).strip().lower()
    if not s:
        return None
    try:
        n = int(s)
        return n if n > 0 else None
    except ValueError:
        pass
    m = re.match(r"^(\d+)\s*(s|m|h|d)$", s)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit == "s":
            return num if num > 0 else None
        if unit == "m":
            return num * 60 if num > 0 else None
        if unit == "h":
            return num * 3600 if num > 0 else None
        if unit == "d":
            return num * 86400 if num > 0 else None
    return None


def validate_cron(expr: str) -> bool:
    """Validate 5-field cron expression."""
    try:
        from croniter import croniter

        croniter(expr, datetime.now(timezone.utc))
        return True
    except Exception:
        return False


def next_run_at(schedule: dict, from_ts: Optional[float] = None) -> Optional[str]:
    """Compute next run time as ISO 8601 UTC string."""
    now = datetime.now(timezone.utc)
    if from_ts is not None:
        now = datetime.fromtimestamp(from_ts, tz=timezone.utc)

    kind = schedule.get("kind")
    if not kind:
        return None
    end_at = schedule.get("end_at")
    end_at_dt = None
    if end_at:
        try:
            end_at_dt = datetime.fromisoformat(str(end_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    if kind == "at":
        at = schedule.get("at")
        if not at:
            return None
        if end_at_dt is None:
            return at
        try:
            at_dt = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        return at if at_dt <= end_at_dt else None

    if kind == "every":
        interval = schedule.get("interval_seconds")
        if not interval or interval <= 0:
            return None
        base_ts = from_ts or now.timestamp()
        next_ts = base_ts + interval
        next_dt = datetime.fromtimestamp(next_ts, tz=timezone.utc)
        if end_at_dt is not None and next_dt > end_at_dt:
            return None
        return next_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if kind == "cron":
        expr = schedule.get("expr")
        if not expr:
            return None
        tz_str = schedule.get("tz")
        try:
            from croniter import croniter

            if tz_str:
                import zoneinfo

                tz = zoneinfo.ZoneInfo(tz_str)
                base = now.astimezone(tz)
                cron = croniter(expr, base)
            else:
                cron = croniter(expr, now)
            next_dt = cron.get_next(datetime)
            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(tzinfo=timezone.utc)
            next_dt_utc = next_dt.astimezone(timezone.utc)
            if end_at_dt is not None and next_dt_utc > end_at_dt:
                return None
            return next_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    return None


def next_run_after_trigger(schedule: dict, last_run_at: str) -> Optional[str]:
    """Compute next_run_at for recurring jobs after a trigger."""
    try:
        dt = datetime.fromisoformat(last_run_at.replace("Z", "+00:00"))
        return next_run_at(schedule, from_ts=dt.timestamp())
    except (ValueError, TypeError):
        return None
