"""The FastAPI trigger server.

Three inbound paths: alerts from PagerDuty, deployment events from GitHub, and
Slack button clicks resolving the approvals layer 2 is waiting on. Plus a
scheduler for maintenance sweeps.

Every handler does the same four things and returns: authenticate the sender,
translate the payload, claim the delivery, enqueue. Nothing slow happens in a
request — PagerDuty and GitHub both give you seconds before they call it a
failure and retry.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncContextManager, AsyncIterator, Callable

from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from ..approval.slack import (
    SlackSignatureError,
    payload_from_form_body,
    resolve_from_payload,
    verify_slack_signature,
)
from ..orchestration.agent import AgentConfig, DevOpsAgent, build_agent
from .events import github_request, pagerduty_request
from .runs import (
    RunLedger,
    RunManager,
    RunRequest,
    Submission,
    as_json,
    findings_summary,
)
from .schedule import ScheduledTask, Scheduler
from .signatures import (
    SignatureError,
    verify_github_signature,
    verify_pagerduty_signature,
)

log = logging.getLogger("devops-agent.triggers")


class MissingSecret(RuntimeError):
    """A sender is enabled but has no secret to authenticate it with."""


@dataclass
class TriggerSettings:
    agent_config: AgentConfig
    github_secret: str | None = None
    pagerduty_secret: str | None = None
    slack_signing_secret: str | None = None
    runs_db: str = "runs.db"
    workers: int = 2
    queue_size: int = 32
    deployment_watch_minutes: int = 10
    scheduled_tasks: tuple[ScheduledTask, ...] = ()
    drain_timeout: float = 30.0
    #: Pick up runs a previous process was killed in the middle of.
    resume_interrupted: bool = True
    #: Only resume recent ones. Resuming an incident from last Tuesday is not
    #: helping anybody, and its cluster state is long gone.
    resume_max_age: timedelta = timedelta(hours=1)
    #: Only ever set False on a laptop. Without signatures, anyone who can
    #: reach the port can make the agent investigate a fabricated incident and
    #: put approval requests in front of a human.
    require_signatures: bool = True

    @classmethod
    def from_env(cls, agent_config: AgentConfig, **overrides: Any) -> "TriggerSettings":
        return cls(
            agent_config=agent_config,
            github_secret=os.environ.get("GITHUB_WEBHOOK_SECRET"),
            pagerduty_secret=os.environ.get("PAGERDUTY_WEBHOOK_SECRET"),
            slack_signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
            runs_db=os.environ.get("RUNS_DB", "runs.db"),
            workers=int(os.environ.get("AGENT_WORKERS", "2")),
            **overrides,
        )

    def validate(self) -> None:
        if not self.require_signatures:
            log.warning(
                "SIGNATURE VERIFICATION IS DISABLED — every webhook endpoint on "
                "this process is unauthenticated. Never do this outside local "
                "development."
            )
            return
        missing = [
            name
            for name, value in (
                ("GITHUB_WEBHOOK_SECRET", self.github_secret),
                ("PAGERDUTY_WEBHOOK_SECRET", self.pagerduty_secret),
                ("SLACK_SIGNING_SECRET", self.slack_signing_secret),
            )
            if not value
        ]
        if missing:
            raise MissingSecret(
                "refusing to start without " + ", ".join(missing) +
                " (set require_signatures=False only for local development)"
            )


async def resubmit_interrupted(
    agent: DevOpsAgent, manager: RunManager, *, max_age: timedelta
) -> int:
    """Re-queue runs a previous process died in the middle of."""
    if agent.transcripts is None:
        return 0

    cutoff = datetime.now(timezone.utc) - max_age
    resumed = 0
    for checkpoint in agent.transcripts.resumable(limit=50):
        if checkpoint.updated_at < cutoff:
            log.info(
                "not resuming %s: last checkpoint %s is too old",
                checkpoint.thread_id,
                checkpoint.updated_at.isoformat(),
            )
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        submission = await manager.submit(
            RunRequest(
                event_key=f"resume:{checkpoint.thread_id}:{stamp}",
                thread_id=checkpoint.thread_id,
                goal=checkpoint.goal,
                kind="resume",
                source="startup",
                resume_of=checkpoint.thread_id,
            )
        )
        if submission.accepted:
            resumed += 1
    if resumed:
        log.info("resumed %d interrupted run(s)", resumed)
    return resumed


def _accepted(submission: Submission) -> JSONResponse:
    """202 for work we took, 409 for a duplicate, 503 when full.

    The status codes matter: a 503 makes the sender retry later, while a 409
    tells it to stop — which is what we want for a delivery we have already
    claimed.
    """
    if submission.accepted:
        return JSONResponse(
            {"status": "accepted", "run_id": submission.run_id},
            status_code=status.HTTP_202_ACCEPTED,
        )
    if submission.reason == "duplicate delivery":
        return JSONResponse(
            {"status": "duplicate", "run_id": submission.run_id},
            status_code=status.HTTP_409_CONFLICT,
        )
    return JSONResponse(
        {"status": "rejected", "reason": submission.reason},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "60"},
    )


def build_router(settings: TriggerSettings, metrics: Any | None = None) -> APIRouter:
    router = APIRouter()

    def seen(source: str, disposition: str) -> None:
        if metrics is not None:
            metrics.record_webhook(source, disposition)

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(request: Request) -> Response:
        manager: RunManager | None = getattr(request.app.state, "manager", None)
        if manager is None or not manager.running:
            return JSONResponse(
                {"status": "starting"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        return JSONResponse(
            {
                "status": "ready",
                "queue_depth": manager.depth,
                "active_runs": [r.run_id for r in manager.active],
                "tools": request.app.state.agent.tool_names,
            }
        )

    @router.post("/webhooks/pagerduty")
    async def pagerduty(request: Request) -> Response:
        body = await request.body()
        if settings.require_signatures:
            try:
                verify_pagerduty_signature(
                    settings.pagerduty_secret or "",
                    body=body,
                    signature=request.headers.get("x-pagerduty-signature"),
                )
            except SignatureError as exc:
                log.warning("rejected PagerDuty webhook: %s", exc)
                seen("pagerduty", "unauthorized")
                return JSONResponse(
                    {"status": "unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
                )

        try:
            payload = json.loads(body)
            run_request = pagerduty_request(payload)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                {"status": "bad payload", "detail": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if run_request is None:
            seen("pagerduty", "ignored")
            return JSONResponse({"status": "ignored"}, status_code=status.HTTP_202_ACCEPTED)
        submission = await request.app.state.manager.submit(run_request)
        seen("pagerduty", submission.reason or "accepted")
        return _accepted(submission)

    @router.post("/webhooks/github")
    async def github(request: Request) -> Response:
        body = await request.body()
        if settings.require_signatures:
            try:
                verify_github_signature(
                    settings.github_secret or "",
                    body=body,
                    signature=request.headers.get("x-hub-signature-256"),
                )
            except SignatureError as exc:
                log.warning("rejected GitHub webhook: %s", exc)
                seen("github", "unauthorized")
                return JSONResponse(
                    {"status": "unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
                )

        try:
            payload = json.loads(body)
            run_request = github_request(
                payload,
                event_type=request.headers.get("x-github-event"),
                delivery_id=request.headers.get("x-github-delivery"),
                watch_minutes=settings.deployment_watch_minutes,
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                {"status": "bad payload", "detail": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if run_request is None:
            seen("github", "ignored")
            return JSONResponse({"status": "ignored"}, status_code=status.HTTP_202_ACCEPTED)
        submission = await request.app.state.manager.submit(run_request)
        seen("github", submission.reason or "accepted")
        return _accepted(submission)

    @router.post("/webhooks/slack/interactions")
    async def slack_interactions(request: Request) -> Response:
        """Where an Approve click becomes a recorded decision.

        Resolved inline rather than queued: it is a single indexed UPDATE, and
        Slack replaces the message with whatever comes back within three
        seconds. Making the reviewer wait on a worker pool to learn whether
        their click landed would be its own kind of unsafe.
        """
        body = await request.body()
        if settings.require_signatures:
            try:
                verify_slack_signature(
                    settings.slack_signing_secret or "",
                    timestamp=request.headers.get("x-slack-request-timestamp", ""),
                    body=body,
                    signature=request.headers.get("x-slack-signature", ""),
                )
            except SlackSignatureError as exc:
                log.warning("rejected Slack interaction: %s", exc)
                seen("slack", "unauthorized")
                return JSONResponse(
                    {"status": "unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
                )

        try:
            payload = payload_from_form_body(body)
        except ValueError as exc:
            return JSONResponse(
                {"status": "bad payload", "detail": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        gate = request.app.state.agent.gate
        resolved, message = resolve_from_payload(
            gate.store, payload, notifier=gate.notifier
        )
        seen("slack", resolved.status.value if resolved else "unknown")
        return JSONResponse({"response_type": "ephemeral", "text": message})

    @router.get("/runs")
    async def list_runs(request: Request, limit: int = 50) -> dict[str, Any]:
        ledger: RunLedger = request.app.state.ledger
        return {"runs": [as_json(record) for record in ledger.recent(limit=limit)]}

    @router.get("/runs/{event_key:path}")
    async def get_run(request: Request, event_key: str) -> Response:
        ledger: RunLedger = request.app.state.ledger
        record = ledger.get(event_key)
        if record is None:
            return JSONResponse(
                {"status": "not found"}, status_code=status.HTTP_404_NOT_FOUND
            )
        return JSONResponse(as_json(record))

    return router


def create_app(
    settings: TriggerSettings,
    *,
    agent: DevOpsAgent | None = None,
    agent_factory: Callable[[], AsyncContextManager[DevOpsAgent]] | None = None,
    ledger: RunLedger | None = None,
    reporter: Any | None = None,
    observability: Any | None = None,
) -> FastAPI:
    """Build the trigger server.

    By default the lifespan builds the agent and tears its MCP servers down on
    shutdown. ``agent_factory`` overrides *how* it is built while keeping
    *where*: an MCP session belongs to the event loop that created it, and
    calling one from another loop deadlocks on the first ``tools/call``. Build
    it in the lifespan, which runs in the serving loop.

    ``agent`` skips construction entirely. Only safe for an agent with no
    loop-bound resources — a test double, or one whose MCP connections were
    opened in this same loop.
    """
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            live_agent = agent
            if live_agent is None:
                events = getattr(observability, "events", None)
                factory = agent_factory or (
                    lambda: build_agent(settings.agent_config, events=events)
                )
                live_agent = await stack.enter_async_context(factory())

            live_ledger = ledger or RunLedger(settings.runs_db)
            manager = RunManager(
                agent=live_agent,
                ledger=live_ledger,
                workers=settings.workers,
                queue_size=settings.queue_size,
                reporter=reporter,
            )
            await manager.start()
            if settings.resume_interrupted:
                await resubmit_interrupted(
                    live_agent, manager, max_age=settings.resume_max_age
                )
            scheduler = Scheduler(manager=manager, tasks=settings.scheduled_tasks)
            await scheduler.start()

            app.state.agent = live_agent
            app.state.ledger = live_ledger
            app.state.manager = manager
            app.state.scheduler = scheduler
            try:
                yield
            finally:
                await scheduler.stop()
                await manager.drain(timeout=settings.drain_timeout)
                if ledger is None:
                    live_ledger.close()
                if observability is not None:
                    # Flush before the process exits; the batch span
                    # processor drops whatever it is still holding.
                    observability.shutdown()

    app = FastAPI(title="agentic-devops triggers", lifespan=lifespan)
    app.include_router(build_router(settings, getattr(observability, "metrics", None)))
    return app


def slack_reporter(agent: DevOpsAgent, channel: str) -> Any:
    """Post each finished run's findings to a Slack channel."""

    async def report(record: Any, run: Any, failure: BaseException | None) -> None:
        notifier = agent.gate.notifier
        client = getattr(notifier, "_client", None)
        if client is None:
            log.info("run %s finished:\n%s", record.run_id, findings_summary(run))
            return
        header = (
            f"*Agent run `{record.run_id}`* — {record.kind} / `{record.thread_id}` "
            f"({record.status.value})"
        )
        body = f"{type(failure).__name__}: {failure}" if failure else findings_summary(run)
        client.chat_postMessage(channel=channel, text=f"{header}\n{body}")

    return report
