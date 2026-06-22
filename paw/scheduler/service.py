"""Scheduler service owned by the Paw runtime."""

import asyncio
import time
from typing import Protocol

from paw.scheduler.schedule import next_run_after_trigger, now_iso
from paw.scheduler.store import JobStore


class ScheduledApp(Protocol):
    """Subset of PawApp needed by SchedulerService."""

    async def process_scheduled_job(self, session_id: str, text: str, name: str) -> str:
        """Run a scheduled prompt through an agent session."""

    async def deliver_scheduled_result(
        self,
        task_session_id: str,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver scheduled output through an available interface."""


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
        self._task = asyncio.current_task()
        try:
            while True:
                try:
                    await self.run_tick()
                    await asyncio.sleep(self._poll_seconds)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"Scheduler tick error: {e}")
                    await asyncio.sleep(self._poll_seconds)
        finally:
            if self._task is asyncio.current_task():
                self._task = None

    async def stop(self) -> None:
        """Stop a running scheduler task if one is attached externally."""
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def run_tick(self) -> None:
        """Check for due jobs, mark them triggered, run each, deliver results."""
        self._store.purge_done()
        now = now_iso()
        due = self._store.get_due(time.time())
        if not due:
            return

        for job in due:
            schedule = job["schedule"]
            task_session_id = job["session_id"]

            # Recurring jobs advance to their next fire; one-shots and expired
            # recurrences are retired.
            next_run = (
                None if schedule["kind"] == "at"
                else next_run_after_trigger(schedule, now)
            )
            self._store.update(job["id"], last_run_at=now, next_run_at=next_run, enabled=None if next_run else False)

            if job["intent"] == "say":
                response = job["message"]
            else:
                # PROMPT: scheduled-task trigger — a due job firing now, framed as a
                # task to carry out (never as the user speaking, never as a request to schedule).
                prompt = (
                    f"A scheduled task you set on {job['created_at']} is firing now.\n"
                    f"TASK: {job['message']}\n"
                    "Carry it out now and report the result to the user. "
                    "This is not a request to schedule anything; do not create, modify, or repeat any schedule."
                )
                try:
                    response = await self._app.process_scheduled_job(
                        task_session_id, prompt, name=job["name"],
                    )
                except Exception as e:
                    response = f"Error running scheduled task: {e}"

            delivered = await self._app.deliver_scheduled_result(
                task_session_id, response, context_id=job["delivery_session_id"],
            )
            if not delivered:
                print(f"Scheduler delivery failed: no interface handled {task_session_id}")
