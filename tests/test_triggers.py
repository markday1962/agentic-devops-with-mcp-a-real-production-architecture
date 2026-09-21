from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from agentic_devops.approval import ApprovalGate, ApprovalPolicy, ApprovalRequest, RiskLevel
from agentic_devops.orchestration import AgentConfig
from agentic_devops.triggers import (
    MissingSecret,
    RunLedger,
    RunStatus,
    RunRequest,
    ScheduledTask,
    TriggerSettings,
    create_app,
)

from conftest import RecordingNotifier

GITHUB_SECRET = "gh-secret"
PAGERDUTY_SECRET = "pd-secret"
SLACK_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"


# ── doubles ──────────────────────────────────────────────────────────────


@dataclass
class FakeRun:
    thread_id: str
    goal: str
    text: str = "Root cause: memory limit."
    halted: str | None = None
    turns: int = 3
    findings: list = field(default_factory=list)
    denied_writes: list = field(default_factory=list)


@dataclass
class FakeAgent:
    gate: ApprovalGate
    runs: list[FakeRun] = field(default_factory=list)
    fail_with: Exception | None = None
    block: threading.Event | None = None
    started: threading.Event | None = None
    tool_names: list[str] = field(default_factory=lambda: ["get_pod_logs"])
    transcripts: object | None = None
    resumed: list[str] = field(default_factory=list)

    async def run(self, goal: str, *, thread_id: str) -> FakeRun:
        if self.started is not None:
            self.started.set()
        if self.block is not None:
            while not self.block.is_set():
                await asyncio.sleep(0.01)
        if self.fail_with is not None:
            raise self.fail_with
        run = FakeRun(thread_id=thread_id, goal=goal)
        self.runs.append(run)
        return run

    async def resume(self, thread_id: str) -> FakeRun:
        self.resumed.append(thread_id)
        run = FakeRun(thread_id=thread_id, goal="resumed")
        self.runs.append(run)
        return run


@pytest.fixture
def agent(store) -> FakeAgent:
    return FakeAgent(
        gate=ApprovalGate(
            store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy()
        )
    )


@pytest.fixture
def ledger(tmp_path) -> RunLedger:
    ledger = RunLedger(tmp_path / "runs.db")
    yield ledger
    ledger.close()


def make_settings(**overrides) -> TriggerSettings:
    return TriggerSettings(
        agent_config=AgentConfig(),
        github_secret=GITHUB_SECRET,
        pagerduty_secret=PAGERDUTY_SECRET,
        slack_signing_secret=SLACK_SECRET,
        workers=overrides.pop("workers", 1),
        queue_size=overrides.pop("queue_size", 8),
        drain_timeout=overrides.pop("drain_timeout", 5.0),
        **overrides,
    )


def client_for(agent, ledger, **overrides) -> TestClient:
    app = create_app(make_settings(**overrides), agent=agent, ledger=ledger)
    return TestClient(app)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ── payloads and signing ─────────────────────────────────────────────────


def pagerduty_payload(event_id="01EVENT", incident_id="PABC123", event_type="incident.triggered"):
    return {
        "event": {
            "id": event_id,
            "event_type": event_type,
            "occurred_at": "2026-09-21T14:03:00Z",
            "data": {
                "id": incident_id,
                "type": "incident",
                "title": "Elevated 5xx rate",
                "urgency": "high",
                "created_at": "2026-09-21T14:03:00Z",
                "html_url": "https://acme.pagerduty.com/incidents/PABC123",
                "service": {"summary": "payments-api"},
            },
        }
    }


def github_payload(deployment_id=99, state="success"):
    return {
        "action": "created",
        "deployment_status": {"state": state, "environment": "production"},
        "deployment": {"id": deployment_id, "ref": "v2.4.1", "environment": "production"},
        "repository": {"full_name": "acme/payments"},
    }


def post_pagerduty(client, payload, *, secret=PAGERDUTY_SECRET):
    body = json.dumps(payload).encode()
    signature = "v1=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        "/webhooks/pagerduty",
        content=body,
        headers={"x-pagerduty-signature": signature, "content-type": "application/json"},
    )


def post_github(client, payload, *, delivery="d-1", event="deployment_status", secret=GITHUB_SECRET):
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "x-hub-signature-256": signature,
            "x-github-event": event,
            "x-github-delivery": delivery,
            "content-type": "application/json",
        },
    )


def post_slack(client, approval_id, *, approve=True, secret=SLACK_SECRET, user="sre-oncall"):
    verb = "approve" if approve else "reject"
    payload = {
        "type": "block_actions",
        "user": {"id": "U1", "username": user},
        "actions": [{"action_id": f"approval_{verb}", "value": f"{verb}:{approval_id}"}],
    }
    body = urlencode({"payload": json.dumps(payload)}).encode()
    timestamp = str(int(time.time()))
    base = b"v0:" + timestamp.encode() + b":" + body
    signature = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return client.post(
        "/webhooks/slack/interactions",
        content=body,
        headers={
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": signature,
            "content-type": "application/x-www-form-urlencoded",
        },
    )


# ── PagerDuty ────────────────────────────────────────────────────────────


def test_incident_starts_a_run(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_pagerduty(client, pagerduty_payload())
        assert response.status_code == 202
        assert wait_for(lambda: agent.runs)

    run = agent.runs[0]
    assert run.thread_id == "incident-PABC123"
    assert "payments-api" in run.goal
    assert "Elevated 5xx rate" in run.goal
    assert "https://acme.pagerduty.com/incidents/PABC123" in run.goal

    record = ledger.get("pagerduty:01EVENT")
    assert record.status is RunStatus.COMPLETED
    assert record.kind == "incident"


def test_unsigned_request_is_rejected(agent, ledger):
    with client_for(agent, ledger) as client:
        response = client.post("/webhooks/pagerduty", json=pagerduty_payload())
    assert response.status_code == 401
    assert agent.runs == []


def test_forged_signature_is_rejected(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_pagerduty(client, pagerduty_payload(), secret="wrong-secret")
    assert response.status_code == 401
    assert agent.runs == []


def test_rotated_secret_accepts_either_signature(agent, ledger):
    """PagerDuty sends one signature per active secret during rotation."""
    body = json.dumps(pagerduty_payload()).encode()
    old = "v1=" + hmac.new(b"old-secret", body, hashlib.sha256).hexdigest()
    current = "v1=" + hmac.new(PAGERDUTY_SECRET.encode(), body, hashlib.sha256).hexdigest()

    with client_for(agent, ledger) as client:
        response = client.post(
            "/webhooks/pagerduty",
            content=body,
            headers={"x-pagerduty-signature": f"{old},{current}"},
        )
    assert response.status_code == 202


def test_retried_delivery_does_not_start_a_second_run(agent, ledger):
    """The failure mode `bg.add_task` has: PagerDuty retries, and two agents
    investigate the same incident."""
    with client_for(agent, ledger) as client:
        first = post_pagerduty(client, pagerduty_payload())
        second = post_pagerduty(client, pagerduty_payload())
        assert wait_for(lambda: agent.runs)

    assert first.status_code == 202
    assert second.status_code == 409
    assert len(agent.runs) == 1


def test_a_new_event_on_the_same_incident_is_new_work(agent, ledger):
    with client_for(agent, ledger) as client:
        post_pagerduty(client, pagerduty_payload(event_id="01FIRST"))
        post_pagerduty(
            client,
            pagerduty_payload(event_id="01SECOND", event_type="incident.reopened"),
        )
        assert wait_for(lambda: len(agent.runs) == 2)
    assert len(agent.runs) == 2


def test_uninteresting_events_are_acknowledged_not_run(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_pagerduty(
            client, pagerduty_payload(event_type="incident.resolved")
        )
    assert response.status_code == 202
    assert response.json()["status"] == "ignored"
    assert agent.runs == []


def test_malformed_payload_is_a_400(agent, ledger):
    with client_for(agent, ledger) as client:
        broken = {"event": {"event_type": "incident.triggered", "data": {}}}
        response = post_pagerduty(client, broken)
    assert response.status_code == 400


# ── GitHub ───────────────────────────────────────────────────────────────


def test_successful_deployment_starts_a_watch(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_github(client, github_payload())
        assert response.status_code == 202
        assert wait_for(lambda: agent.runs)

    run = agent.runs[0]
    assert run.thread_id == "deploy-99"
    assert "acme/payments production" in run.goal
    assert "v2.4.1" in run.goal
    assert ledger.get("github:d-1").kind == "deployment-watch"


def test_failed_deployment_is_not_watched(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_github(client, github_payload(state="failure"))
    assert response.json()["status"] == "ignored"
    assert agent.runs == []


def test_other_github_events_are_ignored(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_github(client, {"action": "opened"}, event="pull_request")
    assert response.json()["status"] == "ignored"
    assert agent.runs == []


def test_github_forged_signature_is_rejected(agent, ledger):
    with client_for(agent, ledger) as client:
        response = post_github(client, github_payload(), secret="wrong")
    assert response.status_code == 401
    assert agent.runs == []


# ── Slack: the endpoint layer 2 has been waiting for ─────────────────────


def pending_approval(store):
    request = ApprovalRequest.new(
        tool_name="delete_resource",
        args={"namespace": "payments", "name": "api-7f9"},
        risk=RiskLevel.HIGH,
        timeout=timedelta(minutes=10),
    )
    return store.create(request)


def test_approve_click_resolves_the_pending_approval(agent, ledger, store):
    request = pending_approval(store)
    with client_for(agent, ledger) as client:
        response = post_slack(client, request.id)

    assert response.status_code == 200
    assert "approved" in response.json()["text"]
    assert store.get(request.id).decided_by == "sre-oncall"


def test_reject_click_records_a_rejection(agent, ledger, store):
    request = pending_approval(store)
    with client_for(agent, ledger) as client:
        post_slack(client, request.id, approve=False)
    assert store.get(request.id).status.value == "rejected"


def test_forged_slack_request_cannot_approve_anything(agent, ledger, store):
    request = pending_approval(store)
    with client_for(agent, ledger) as client:
        response = post_slack(client, request.id, secret="not-the-signing-secret")

    assert response.status_code == 401
    assert store.get(request.id).status.value == "pending"


def test_second_click_gets_a_straight_answer(agent, ledger, store):
    request = pending_approval(store)
    with client_for(agent, ledger) as client:
        post_slack(client, request.id, user="first")
        response = post_slack(client, request.id, user="second")
    assert "already approved by first" in response.json()["text"]


# ── capacity, failure, lifecycle ─────────────────────────────────────────


def test_queue_full_returns_503_so_the_sender_retries(ledger, store):
    """An alert storm must not become unbounded concurrent agent runs."""
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())
    release = threading.Event()
    agent = FakeAgent(gate=gate, block=release, started=threading.Event())

    with client_for(agent, ledger, workers=1, queue_size=1) as client:
        first = post_pagerduty(client, pagerduty_payload(event_id="E1"))
        assert wait_for(agent.started.is_set)
        second = post_pagerduty(client, pagerduty_payload(event_id="E2"))
        third = post_pagerduty(client, pagerduty_payload(event_id="E3"))
        release.set()

    assert first.status_code == 202
    assert second.status_code == 202      # queued
    assert third.status_code == 503       # no room
    assert third.headers["retry-after"] == "60"
    assert ledger.get("pagerduty:E3").status is RunStatus.REJECTED


def test_a_failing_run_is_recorded_and_the_worker_survives(ledger, store):
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())
    agent = FakeAgent(gate=gate, fail_with=RuntimeError("cluster unreachable"))

    with client_for(agent, ledger) as client:
        post_pagerduty(client, pagerduty_payload(event_id="E1"))
        assert wait_for(
            lambda: (r := ledger.get("pagerduty:E1")) and r.status is RunStatus.FAILED
        )
        agent.fail_with = None
        post_pagerduty(client, pagerduty_payload(event_id="E2"))
        assert wait_for(lambda: agent.runs)

    assert "cluster unreachable" in ledger.get("pagerduty:E1").error
    assert ledger.get("pagerduty:E2").status is RunStatus.COMPLETED


def test_halted_runs_are_distinguished_from_completed(ledger, store):
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())

    class HaltingAgent(FakeAgent):
        async def run(self, goal: str, *, thread_id: str) -> FakeRun:
            return FakeRun(thread_id=thread_id, goal=goal, halted="reached the turn limit")

    agent = HaltingAgent(gate=gate)
    with client_for(agent, ledger) as client:
        post_pagerduty(client, pagerduty_payload(event_id="E1"))
        assert wait_for(
            lambda: (r := ledger.get("pagerduty:E1")) and r.status is RunStatus.HALTED
        )
    assert ledger.get("pagerduty:E1").error == "reached the turn limit"


def test_runs_interrupted_by_a_restart_are_recorded(ledger):
    ledger.claim(
        RunRequest(
            event_key="pagerduty:OLD",
            thread_id="incident-1",
            goal="investigate",
            kind="incident",
            source="pagerduty",
        )
    )
    assert ledger.get("pagerduty:OLD").status is RunStatus.QUEUED

    assert ledger.reclaim_interrupted() == 1
    record = ledger.get("pagerduty:OLD")
    assert record.status is RunStatus.INTERRUPTED
    assert "restarted" in record.error


# ── operability ──────────────────────────────────────────────────────────


def test_health_and_readiness(agent, ledger):
    with client_for(agent, ledger) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        ready = client.get("/readyz").json()
        assert ready["status"] == "ready"
        assert ready["tools"] == ["get_pod_logs"]


def test_runs_are_inspectable(agent, ledger):
    with client_for(agent, ledger) as client:
        post_pagerduty(client, pagerduty_payload())
        assert wait_for(lambda: agent.runs)
        listing = client.get("/runs").json()
        detail = client.get("/runs/pagerduty:01EVENT").json()
        missing = client.get("/runs/nope")

    assert listing["runs"][0]["thread_id"] == "incident-PABC123"
    assert detail["status"] == "completed"
    assert missing.status_code == 404


def test_refuses_to_start_without_secrets():
    settings = TriggerSettings(agent_config=AgentConfig())
    with pytest.raises(MissingSecret) as excinfo:
        create_app(settings)
    assert "GITHUB_WEBHOOK_SECRET" in str(excinfo.value)
    assert "SLACK_SIGNING_SECRET" in str(excinfo.value)


def test_signatures_can_be_disabled_for_local_development(agent, ledger):
    with client_for(agent, ledger, require_signatures=False) as client:
        response = client.post("/webhooks/pagerduty", json=pagerduty_payload())
    assert response.status_code == 202


def test_scheduled_task_submits_work(agent, ledger):
    task = ScheduledTask(
        name="cert-check",
        goal="Check for TLS certificates expiring in the next 14 days.",
        interval=timedelta(seconds=3600),
        jitter=timedelta(0),
        run_at_start=True,
    )
    with client_for(agent, ledger, scheduled_tasks=(task,)):
        assert wait_for(lambda: agent.runs)

    assert "TLS certificates" in agent.runs[0].goal
    assert agent.runs[0].thread_id.startswith("maintenance-cert-check-")


# ── picking up after a restart ───────────────────────────────────────────


def interrupted_transcript(tmp_path, *, age_seconds: float = 0.0):
    """A transcript left behind by a process that was killed mid-run."""
    from datetime import datetime, timedelta, timezone

    from agentic_devops.memory import TranscriptStore

    store = TranscriptStore(tmp_path / "transcripts.db")
    store.save(
        thread_id="incident-PABC123",
        goal="investigate payments-api",
        messages=[{"role": "user", "content": "investigate payments-api"}],
        turns=2,
        status="running",
    )
    if age_seconds:
        stale = (
            datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
        ).isoformat()
        store._conn.execute(
            "UPDATE transcripts SET updated_at = ?", (stale,)
        )
    return store


def test_a_killed_run_is_picked_up_on_startup(ledger, store, tmp_path):
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())
    transcripts = interrupted_transcript(tmp_path)
    agent = FakeAgent(gate=gate, transcripts=transcripts)

    try:
        with client_for(agent, ledger):
            assert wait_for(lambda: agent.resumed)
    finally:
        transcripts.close()

    assert agent.resumed == ["incident-PABC123"]
    assert ledger.recent()[0].kind == "resume"


def test_a_stale_run_is_left_alone(ledger, store, tmp_path):
    """Resuming last Tuesday's incident helps nobody — the cluster state it
    was reasoning about is long gone."""
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())
    transcripts = interrupted_transcript(tmp_path, age_seconds=7200)
    agent = FakeAgent(gate=gate, transcripts=transcripts)

    try:
        with client_for(agent, ledger) as client:
            post_pagerduty(client, pagerduty_payload())
            assert wait_for(lambda: agent.runs)
    finally:
        transcripts.close()

    assert agent.resumed == []


def test_resume_can_be_turned_off(ledger, store, tmp_path):
    gate = ApprovalGate(store=store, notifier=RecordingNotifier(), policy=ApprovalPolicy())
    transcripts = interrupted_transcript(tmp_path)
    agent = FakeAgent(gate=gate, transcripts=transcripts)

    try:
        with client_for(agent, ledger, resume_interrupted=False) as client:
            post_pagerduty(client, pagerduty_payload())
            assert wait_for(lambda: agent.runs)
    finally:
        transcripts.close()

    assert agent.resumed == []
