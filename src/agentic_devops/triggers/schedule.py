"""Scheduled maintenance runs — the third trigger.

Interval-based rather than cron, deliberately: a cron expression needs a
dependency and a timezone policy, and "every 6 hours" covers the maintenance
sweeps this is for. A task that must fire at a wall-clock time belongs in a
Kubernetes CronJob posting to the webhook endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .runs import RunManager, RunRequest

log = logging.getLogger("devops-agent.schedule")


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    name: str
    goal: str
    interval: timedelta
    #: Spread identical schedules across replicas and restarts.
    jitter: timedelta = timedelta(seconds=30)
    #: Run once at startup as well as on the interval.
    run_at_start: bool = False


@dataclass
class Scheduler:
    manager: RunManager
    tasks: tuple[ScheduledTask, ...] = ()
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False, repr=False)

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._loop(task), name=f"schedule-{task.name}")
            for task in self.tasks
        ]
        if self._tasks:
            log.info("scheduler started with %d task(s)", len(self._tasks))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _loop(self, task: ScheduledTask) -> None:
        if not task.run_at_start:
            await self._sleep(task)
        while True:
            await self._fire(task)
            await self._sleep(task)

    async def _sleep(self, task: ScheduledTask) -> None:
        jitter = random.uniform(0, task.jitter.total_seconds())
        await asyncio.sleep(task.interval.total_seconds() + jitter)

    async def _fire(self, task: ScheduledTask) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        request = RunRequest(
            event_key=f"schedule:{task.name}:{stamp}",
            thread_id=f"maintenance-{task.name}-{stamp}",
            goal=task.goal,
            kind="scheduled",
            source="scheduler",
        )
        submission = await self.manager.submit(request)
        if not submission.accepted:
            # Skipping a maintenance sweep under load is correct: incidents
            # outrank housekeeping for the worker pool.
            log.info("scheduled task %s skipped: %s", task.name, submission.reason)
