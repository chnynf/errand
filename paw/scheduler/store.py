"""Job store for scheduled tasks. Persists jobs to JSON."""

import json
import uuid
from pathlib import Path
from typing import Optional

from paw.scheduler.schedule import now_iso, parse_iso

_JOBS_FILE = Path(__file__).resolve().parent / "jobs.json"


def _default_data() -> dict:
    return {"jobs": []}


class JobStore:
    """Manages persistence and CRUD for scheduled jobs."""

    def __init__(self, jobs_path: Optional[Path] = None):
        self._path = jobs_path or _JOBS_FILE

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return _default_data()
        return _default_data()

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add(
        self,
        *,
        name: str,
        schedule: dict,
        message: str,
        interface: str,
        next_run_at: str,
        intent: str = "execute",
    ) -> dict:
        """Add a new job. ``interface`` is the origin interface name (e.g. "discord");
        delivery always happens through Discord regardless of origin -- see
        ``PawApp.deliver_scheduled_result``. No session is stored: each fire gets
        its own freshly-named session (see ``SchedulerService.run_tick``)."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = now_iso()
        job = {
            "id": job_id,
            "name": name,
            "schedule": schedule,
            "message": message,
            "interface": interface,
            "enabled": True,
            "created_at": now,
            "last_run_at": None,
            "next_run_at": next_run_at,
            "intent": intent,
        }
        data = self._load()
        data["jobs"].append(job)
        self._save(data)
        return job

    def list_enabled(self) -> list[dict]:
        return [j for j in self._load()["jobs"] if j.get("enabled", True)]

    def get_due(self, now_ts: float) -> list[dict]:
        """Get enabled jobs whose next_run_at <= now."""
        due = []
        for j in self.list_enabled():
            dt = parse_iso(j.get("next_run_at"))
            if dt and dt.timestamp() <= now_ts:
                due.append(j.copy())
        return due

    def update(
        self,
        job_id: str,
        *,
        last_run_at: Optional[str] = None,
        next_run_at: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[dict]:
        data = self._load()
        for job in data["jobs"]:
            if job["id"] == job_id:
                if last_run_at is not None:
                    job["last_run_at"] = last_run_at
                if next_run_at is not None:
                    job["next_run_at"] = next_run_at
                if enabled is not None:
                    job["enabled"] = enabled
                self._save(data)
                return job.copy()
        return None

    def disable(self, job_id: str) -> bool:
        return self.update(job_id, enabled=False) is not None

    def purge_done(self) -> int:
        """Remove jobs that are done (enabled=False). Returns count purged."""
        data = self._load()
        original = len(data["jobs"])
        data["jobs"] = [j for j in data["jobs"] if j.get("enabled", True)]
        purged = original - len(data["jobs"])
        if purged > 0:
            self._save(data)
        return purged
