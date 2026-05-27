"""Scheduler service owned by the Swarm runtime."""

import asyncio
from datetime import datetime, timezone
from typing import Protocol

from swarm_agent.scheduler.schedule import next_run_after_trigger
from swarm_agent.scheduler.store import JobStore


class ScheduledApp(Protocol):
    """Subset of SwarmApp needed by SchedulerService."""

    async def process_scheduled_job(self, session_id: str, text: str, name: str) -> str:
        """Run a scheduled prompt through an agent session."""

    async def deliver_scheduled_result(
        self,
        task_session_id: str,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver scheduled output through an available interface."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SchedulerService:
    """Polls due jobs, executes them, and delivers results."""

    def __init__(
        self,
        app: ScheduledApp,
        store: JobStore | None = None,
        poll_seconds: int = 10,
    ):
        self._app = app
        self._store = store or JobStore()
        self._poll_seconds = poll_seconds
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Run scheduler ticks until stopped."""
        while True:
            try:
                await self.run_tick()
                await asyncio.sleep(self._poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Scheduler tick error: {e}")
                await asyncio.sleep(self._poll_seconds)

    async def stop(self) -> None:
        """Stop a running scheduler task if one is attached externally."""
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def run_tick(self) -> None:
        """Check for due jobs, mark them triggered, run each, deliver results."""
        self._store.purge_done()
        now_ts = datetime.now(timezone.utc).timestamp()
        now_iso = _iso_now()
        due = self._store.get_due(now_ts)
        if not due:
            return

        for job in due:
            job_id = job["id"]
            task_session_id = job["session_id"]
            delivery_session_id = job.get("delivery_session_id") or job.get(
                "delivery_target", ""
            ).split(":")[-1]
            message = job["message"]
            name = job.get("name", "Scheduled task")
            schedule = job["schedule"]
            kind = schedule.get("kind")
            intent = job.get("intent", "execute")

            next_run = next_run_after_trigger(schedule, now_iso)
            if kind == "at":
                self._store.update(job_id, last_run_at=now_iso, enabled=False)
            elif next_run:
                self._store.update(job_id, last_run_at=now_iso, next_run_at=next_run)
            else:
                self._store.update(job_id, last_run_at=now_iso, enabled=False)

            if intent == "say":
                response = message
            else:
                created_at = job.get("created_at", "earlier")
                # PROMPT: scheduled-task trigger — provides context for a due scheduled job
                prompt = (
                    f'On {created_at}, I said "{message}", now is the time to execute, '
                    "please respond. (Do not reschedule!)"
                )
                try:
                    response = await self._app.process_scheduled_job(
                        task_session_id,
                        prompt,
                        name=name,
                    )
                except Exception as e:
                    response = f"Error running scheduled task: {e}"

            delivered = await self._app.deliver_scheduled_result(
                task_session_id,
                response,
                context_id=delivery_session_id,
            )
            if not delivered:
                print(f"Scheduler delivery failed: no interface handled {task_session_id}")
