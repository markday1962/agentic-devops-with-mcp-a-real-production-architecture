"""Accepting work, exactly once, at a rate the system can survive.

``bg.add_task(run_agent, ...)`` — the article's trigger mechanism — has three
properties that only show up in production:

*It runs the same incident twice.* PagerDuty and GitHub both retry deliveries
they don't get a timely 2xx for, and a retried delivery starts a second agent
investigating the same incident, racing the first and asking for its own
approvals.

*It has no ceiling.* An alert storm is exactly the moment a hundred webhooks
arrive at once, and exactly the moment you least want a hundred concurrent
agent runs competing for rate limit and posting a hundred approval requests.

*It forgets.* A pod restart drops every in-flight task with nothing recorded,
so nobody can tell afterwards whether the agent looked at an incident.

This module is a claim-first ledger, a bounded queue, and a fixed worker pool.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

log = logging.getLogger("devops-agent.triggers")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    HALTED = "halted"
    FAILED = "failed"
    REJECTED = "rejected"
    #: The process died while this run was in flight.
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RunRequest:
    """A unit of work a trigger wants done.

    ``event_key`` is the sender's own delivery identifier — PagerDuty's event
    id, GitHub's ``X-GitHub-Delivery``. It is the deduplication key, so it must
    identify the *delivery*, not the incident: a genuinely new alert on the
    same service should be a new key.
    """

    event_key: str
    thread_id: str
    goal: str
    kind: str
    source: str
    #: Set when this is a restart picking a killed run back up, rather than
    #: new work. The worker resumes the transcript instead of starting over.
    resume_of: str | None = None


@dataclass
class RunRecord:
    event_key: str
    run_id: str
    thread_id: str
    kind: str
    source: str
    status: RunStatus
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    turns: int = 0
    findings: int = 0
    summary: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Submission:
    accepted: bool
    event_key: str
    run_id: str | None = None
    reason: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    event_key   TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    source      TEXT NOT NULL,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL,
    queued_at   TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    turns       INTEGER NOT NULL DEFAULT 0,
    findings    INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status, queued_at);
CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs (thread_id);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class RunLedger:
    """Durable record of every run this service has been asked to do."""

    def __init__(self, path: str | Path = "runs.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def claim(self, request: RunRequest) -> RunRecord | None:
        """Take ownership of a delivery, or return None if it is a duplicate.

        The INSERT is the claim: ``event_key`` is the primary key, so two
        concurrent deliveries of the same event cannot both succeed regardless
        of how the workers are scheduled.
        """
        run_id = uuid.uuid4().hex[:12]
        queued_at = utcnow()
        with self._lock:
            changed = self._conn.execute(
                """
                INSERT OR IGNORE INTO runs (
                    event_key, run_id, thread_id, kind, source, goal, status, queued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.event_key,
                    run_id,
                    request.thread_id,
                    request.kind,
                    request.source,
                    request.goal,
                    RunStatus.QUEUED.value,
                    _iso(queued_at),
                ),
            ).rowcount
        if not changed:
            return None
        return RunRecord(
            event_key=request.event_key,
            run_id=run_id,
            thread_id=request.thread_id,
            kind=request.kind,
            source=request.source,
            status=RunStatus.QUEUED,
            queued_at=queued_at,
        )

    def _update(self, event_key: str, **columns: Any) -> None:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {assignments} WHERE event_key = ?",
                (*columns.values(), event_key),
            )

    def mark_running(self, event_key: str) -> None:
        self._update(
            event_key, status=RunStatus.RUNNING.value, started_at=_iso(utcnow())
        )

    def mark_finished(
        self,
        event_key: str,
        *,
        status: RunStatus,
        turns: int = 0,
        findings: int = 0,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        self._update(
            event_key,
            status=status.value,
            finished_at=_iso(utcnow()),
            turns=turns,
            findings=findings,
            summary=summary,
            error=error,
        )

    def mark_rejected(self, event_key: str, reason: str) -> None:
        self._update(
            event_key,
            status=RunStatus.REJECTED.value,
            finished_at=_iso(utcnow()),
            error=reason,
        )

    def reclaim_interrupted(self) -> int:
        """Called at startup: anything still queued or running belongs to a
        process that no longer exists. Recorded honestly rather than silently
        left looking in-flight forever."""
        with self._lock:
            return self._conn.execute(
                """
                UPDATE runs SET status = ?, finished_at = ?, error = ?
                 WHERE status IN (?, ?)
                """,
                (
                    RunStatus.INTERRUPTED.value,
                    _iso(utcnow()),
                    "process restarted while this run was in flight",
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                ),
            ).rowcount

    def _row(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            event_key=row["event_key"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            kind=row["kind"],
            source=row["source"],
            status=RunStatus(row["status"]),
            queued_at=_parse(row["queued_at"]),  # type: ignore[arg-type]
            started_at=_parse(row["started_at"]),
            finished_at=_parse(row["finished_at"]),
            turns=row["turns"],
            findings=row["findings"],
            summary=row["summary"],
            error=row["error"],
        )

    def get(self, event_key: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE event_key = ?", (event_key,)
            ).fetchone()
        return self._row(row) if row else None

    def recent(self, limit: int = 50) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY queued_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]


class Runner(Protocol):
    """What the manager needs from layer 1."""

    async def run(self, goal: str, *, thread_id: str) -> Any: ...

    async def resume(self, thread_id: str) -> Any: ...


#: Called after each run with (record, run-or-None, exception-or-None).
Reporter = Callable[[RunRecord, Any, BaseException | None], Awaitable[None]]


@dataclass
class RunManager:
    """A bounded worker pool over the run ledger."""

    agent: Runner
    ledger: RunLedger
    workers: int = 2
    queue_size: int = 32
    reporter: Reporter | None = None
    _queue: asyncio.Queue = field(init=False, repr=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False, repr=False)
    _active: dict[str, RunRecord] = field(default_factory=dict, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.queue_size)

    async def start(self) -> None:
        reclaimed = self.ledger.reclaim_interrupted()
        if reclaimed:
            log.warning("marked %d run(s) interrupted by a previous restart", reclaimed)
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"agent-worker-{i}")
            for i in range(self.workers)
        ]
        self._started = True
        log.info("run manager started with %d workers", self.workers)

    @property
    def running(self) -> bool:
        return self._started

    @property
    def active(self) -> list[RunRecord]:
        return list(self._active.values())

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def submit(self, request: RunRequest) -> Submission:
        """Claim, then enqueue. Never blocks the caller — a webhook handler has
        seconds, and the sender retries what it does not get a 2xx for."""
        record = self.ledger.claim(request)
        if record is None:
            existing = self.ledger.get(request.event_key)
            log.info(
                "ignoring duplicate delivery %s (already %s)",
                request.event_key,
                existing.status.value if existing else "claimed",
            )
            return Submission(
                accepted=False,
                event_key=request.event_key,
                run_id=existing.run_id if existing else None,
                reason="duplicate delivery",
            )

        try:
            self._queue.put_nowait((request, record))
        except asyncio.QueueFull:
            # Backpressure is a 503, not a silent drop: the sender retries,
            # and the retry is a fresh chance once the storm passes.
            self.ledger.mark_rejected(request.event_key, "queue full")
            log.warning("rejected %s: queue full at %d", request.event_key, self.queue_size)
            return Submission(
                accepted=False,
                event_key=request.event_key,
                run_id=record.run_id,
                reason="at capacity",
            )

        return Submission(accepted=True, event_key=request.event_key, run_id=record.run_id)

    async def _worker(self, index: int) -> None:
        while True:
            request, record = await self._queue.get()
            self._active[record.event_key] = record
            try:
                await self._execute(request, record)
            finally:
                self._active.pop(record.event_key, None)
                self._queue.task_done()

    async def _execute(self, request: RunRequest, record: RunRecord) -> None:
        self.ledger.mark_running(record.event_key)
        record.status = RunStatus.RUNNING
        log.info("run %s starting (%s/%s)", record.run_id, request.kind, request.thread_id)

        run: Any = None
        failure: BaseException | None = None
        try:
            if request.resume_of:
                run = await self.agent.resume(request.resume_of)
            else:
                run = await self.agent.run(request.goal, thread_id=request.thread_id)
        except asyncio.CancelledError:
            self.ledger.mark_finished(
                record.event_key,
                status=RunStatus.INTERRUPTED,
                error="cancelled during shutdown",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the worker
            failure = exc
            log.exception("run %s failed", record.run_id)
            self.ledger.mark_finished(
                record.event_key,
                status=RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            halted = getattr(run, "halted", None)
            self.ledger.mark_finished(
                record.event_key,
                status=RunStatus.HALTED if halted else RunStatus.COMPLETED,
                turns=getattr(run, "turns", 0),
                findings=len(getattr(run, "findings", ())),
                summary=getattr(run, "text", None),
                error=halted,
            )

        if self.reporter is not None:
            try:
                await self.reporter(record, run, failure)
            except Exception:  # noqa: BLE001
                log.exception("reporter failed for run %s", record.run_id)

    async def drain(self, timeout: float = 30.0) -> None:
        """Stop accepting, let in-flight runs finish, then cancel the workers.

        An agent killed mid-run may have already spent an approval, so finishing
        is strictly better than being cut off — but not forever, because
        Kubernetes will SIGKILL us when the grace period ends.
        """
        self._started = False
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("drain timed out after %.0fs with %d queued", timeout, self.depth)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        log.info("run manager stopped")


def findings_summary(run: Any, *, limit: int = 8) -> str:
    """A compact text digest of a run, for Slack."""
    if run is None:
        return "_The run did not produce a result._"
    lines: list[str] = []
    if getattr(run, "halted", None):
        lines.append(f":warning: Run halted: {run.halted}")
    for finding in list(getattr(run, "findings", ()))[:limit]:
        lines.append(f"• *{finding.significance}* — {finding.summary}\n   _{finding.evidence}_")
    denied = getattr(run, "denied_writes", ())
    if denied:
        names = ", ".join(sorted({i.name for i in denied}))
        lines.append(f":no_entry: Not executed (no approval): {names}")
    if getattr(run, "text", None):
        lines.append(f"\n{run.text}")
    return "\n".join(lines) or "_No findings recorded._"


def as_json(record: RunRecord) -> dict[str, Any]:
    return {
        "event_key": record.event_key,
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "kind": record.kind,
        "source": record.source,
        "status": record.status.value,
        "queued_at": _iso(record.queued_at),
        "started_at": _iso(record.started_at),
        "finished_at": _iso(record.finished_at),
        "turns": record.turns,
        "findings": record.findings,
        "summary": record.summary,
        "error": record.error,
    }
