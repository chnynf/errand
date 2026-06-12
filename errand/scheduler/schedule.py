"""Schedule validation and next-run computation. The AI supplies pre-parsed values."""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_iso(value: object) -> Optional[datetime]:
    """Parse an ISO 8601 string into an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def format_iso(dt: datetime) -> str:
    """Format a datetime as an ISO 8601 UTC string (e.g. 2025-02-24T09:00:00Z)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return format_iso(datetime.now(timezone.utc))


def _next_cron(expr: object, after: datetime, tz_name: str = "UTC") -> Optional[datetime]:
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        from croniter import croniter
        from zoneinfo import ZoneInfo

        after_local = after.astimezone(ZoneInfo(tz_name))
        nxt_ts = croniter(expr, after_local).get_next(float)
        return datetime.fromtimestamp(nxt_ts, tz=timezone.utc)
    except Exception:
        return None


def validate_at(value: str) -> bool:
    """Return True if value is a parseable ISO 8601 timestamp."""
    return parse_iso(value) is not None


def parse_every(value: str | int) -> Optional[int]:
    """Parse an interval to positive seconds. Accepts 3600, '3600', '1h', '30m', '1d'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    m = re.fullmatch(r"(\d+)\s*([smhd]?)", str(value).strip().lower())
    if not m:
        return None
    return int(m.group(1)) * _UNITS[m.group(2) or "s"] or None


def validate_cron(expr: str) -> bool:
    """Return True if expr is a valid 5-field cron expression."""
    return _next_cron(expr, datetime.now(timezone.utc)) is not None


def next_run_at(schedule: dict, from_ts: Optional[float] = None) -> Optional[str]:
    """Compute the next run time as an ISO 8601 UTC string, honoring end_at."""
    now = (
        datetime.fromtimestamp(from_ts, timezone.utc)
        if from_ts is not None
        else datetime.now(timezone.utc)
    )
    end_raw = schedule.get("end_at")
    end_dt = parse_iso(end_raw) if end_raw else None
    if end_raw and end_dt is None:
        return None

    kind = schedule.get("kind")
    if kind == "at":
        nxt = parse_iso(schedule.get("at"))
    elif kind == "every":
        seconds = schedule.get("interval_seconds") or 0
        nxt = now + timedelta(seconds=seconds) if seconds > 0 else None
    elif kind == "cron":
        nxt = _next_cron(schedule.get("expr"), now, schedule.get("tz", "UTC"))
    else:
        nxt = None

    if nxt is None or (end_dt is not None and nxt > end_dt):
        return None
    return format_iso(nxt)


def next_run_after_trigger(schedule: dict, last_run_at: str) -> Optional[str]:
    """Compute next_run_at for a recurring job after a trigger fired."""
    dt = parse_iso(last_run_at)
    return next_run_at(schedule, from_ts=dt.timestamp()) if dt else None
