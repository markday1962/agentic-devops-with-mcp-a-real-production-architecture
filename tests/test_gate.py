from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from agentic_devops.approval import (
    ApprovalDenied,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalStatus,
    RiskLevel,
)

from conftest import AutoDecidingNotifier, RecordingNotifier


def make_gate(store, clock, notifier, **kwargs):
    return ApprovalGate(
        store=store,
        notifier=notifier,
        policy=kwargs.pop("policy", ApprovalPolicy()),
        timeout=kwargs.pop("timeout", timedelta(minutes=10)),
        poll_interval=1.0,
        agent_id="devops-agent-1",
        clock=clock.now,
        sleeper=clock.sleep,
        **kwargs,
    )


class Cluster:
    """Stand-in for an MCP tool that actually changes something."""

    def __init__(self) -> None:
        self.deleted: list[dict] = []

    def delete_resource(self, **kwargs):
        self.deleted.append(kwargs)
        return f"deleted {kwargs.get('name')}"

    def get_pod_logs(self, **kwargs):
        return "log line"


def test_request_captures_risk_and_agent(gate):
    request = gate.request("delete_resource", {"namespace": "kube-system", "name": "x"})
    assert request.risk is RiskLevel.HIGH
    assert request.requested_by == "devops-agent-1"
    assert request.status is ApprovalStatus.PENDING
    assert request.notification_ref is not None


def test_approved_action_runs(store, clock):
    cluster = Cluster()
    gate = make_gate(store, clock, AutoDecidingNotifier(store, approve=True, clock=clock))
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    result = guarded(namespace="payments", name="api-7f9")

    assert result == "deleted api-7f9"
    assert cluster.deleted == [{"namespace": "payments", "name": "api-7f9"}]


def test_rejected_action_does_not_run(store, clock):
    cluster = Cluster()
    gate = make_gate(store, clock, AutoDecidingNotifier(store, approve=False, clock=clock))
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    result = guarded(namespace="payments", name="api-7f9")

    assert cluster.deleted == []
    assert "NOT executed" in result
    assert "rejected" in result
    assert "Do not retry" in result


def test_timeout_denies_and_never_runs(store, clock):
    cluster = Cluster()
    gate = make_gate(store, clock, RecordingNotifier(), timeout=timedelta(seconds=30))
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    result = guarded(namespace="payments", name="api-7f9")

    assert cluster.deleted == []
    assert "timed out" in result
    assert store.list_pending(now=clock.now()) == []


def test_unreachable_reviewers_fail_closed(store, clock):
    """A broken Slack integration must deny immediately, not stall for the
    full timeout on every single action."""
    cluster = Cluster()
    gate = make_gate(store, clock, RecordingNotifier(fail=True))
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    started = clock.now()
    result = guarded(namespace="payments", name="api-7f9")

    assert cluster.deleted == []
    assert "NOT executed" in result
    assert clock.now() == started  # denied without waiting out the deadline


def test_read_only_tools_are_not_wrapped(store, clock):
    cluster = Cluster()
    policy = ApprovalPolicy(read_only_tools=frozenset({"get_pod_logs"}))
    gate = make_gate(store, clock, RecordingNotifier(), policy=policy)
    guarded = gate.guard("get_pod_logs", cluster.get_pod_logs)

    assert guarded == cluster.get_pod_logs  # returned untouched, not wrapped
    assert guarded(namespace="payments") == "log line"
    assert store.history() == []


def test_forbidden_tools_are_never_offered_for_approval(store, clock):
    cluster = Cluster()
    policy = ApprovalPolicy(forbidden_tools=frozenset({"delete_resource"}))
    notifier = RecordingNotifier()
    gate = make_gate(store, clock, notifier, policy=policy)
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    result = guarded(namespace="payments", name="api-7f9")

    assert cluster.deleted == []
    assert "blocked by policy" in result
    assert notifier.notified == []


def test_each_call_needs_its_own_approval(store, clock):
    """The article's 'never chain write actions' rule, enforced rather than
    asked for in the system prompt."""
    cluster = Cluster()
    gate = make_gate(store, clock, AutoDecidingNotifier(store, approve=True, clock=clock))
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    guarded(namespace="payments", name="one")
    guarded(namespace="payments", name="two")

    assert len(cluster.deleted) == 2
    assert len(store.history()) == 2
    assert all(r.status is ApprovalStatus.CONSUMED for r in store.history())


def test_raise_on_denied_mode(store, clock):
    cluster = Cluster()
    gate = make_gate(
        store,
        clock,
        AutoDecidingNotifier(store, approve=False, clock=clock),
        raise_on_denied=True,
    )
    guarded = gate.guard("delete_resource", cluster.delete_resource)

    with pytest.raises(ApprovalDenied) as excinfo:
        guarded(namespace="payments", name="api-7f9")

    assert excinfo.value.request.status is ApprovalStatus.REJECTED
    assert cluster.deleted == []


def test_summary_builder_reaches_the_reviewer(store, clock):
    notifier = RecordingNotifier()
    gate = make_gate(store, clock, notifier)
    guarded = gate.guard(
        "delete_resource",
        Cluster().delete_resource,
        summary_builder=lambda kwargs: f"Delete pod {kwargs['name']} to clear CrashLoop",
        thread_id="incident-42",
    )
    guarded(namespace="payments", name="api-7f9")

    posted = notifier.notified[0]
    assert "clear CrashLoop" in posted.describe()
    assert posted.thread_id == "incident-42"


def test_wait_returns_the_decision_made_elsewhere(store, clock):
    """The realistic shape: one process waits, another records the click."""
    gate = make_gate(store, clock, RecordingNotifier())
    request = gate.request("apply_manifest", {"namespace": "web"})

    class Reviewer:
        """Approves once the waiter has polled twice."""

        def __init__(self):
            self.calls = 0

        def __call__(self, seconds):
            clock.sleep(seconds)
            self.calls += 1
            if self.calls == 2:
                store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())

    gate.sleeper = Reviewer()
    decided = gate.wait(request.id)

    assert decided.status is ApprovalStatus.APPROVED
    assert decided.decided_by == "sre-oncall"


def test_async_guard_runs_approved_actions(store, clock):
    calls = []

    async def scale_deployment(**kwargs):
        calls.append(kwargs)
        return "scaled"

    gate = make_gate(store, clock, AutoDecidingNotifier(store, approve=True, clock=clock))
    guarded = gate.aguard("scale_deployment", scale_deployment)

    assert asyncio.run(guarded(namespace="web", replicas=3)) == "scaled"
    assert calls == [{"namespace": "web", "replicas": 3}]


def test_async_guard_blocks_rejected_actions(store, clock):
    calls = []

    async def scale_deployment(**kwargs):
        calls.append(kwargs)
        return "scaled"

    gate = make_gate(store, clock, AutoDecidingNotifier(store, approve=False, clock=clock))
    guarded = gate.aguard("scale_deployment", scale_deployment)

    result = asyncio.run(guarded(namespace="web", replicas=0))
    assert calls == []
    assert "NOT executed" in result


def test_async_waiting_honours_the_injected_clock(store, clock):
    """Regression: `await_decision` slept on the real clock while reading the
    injected one, so a deadline that never arrived spun forever."""
    cluster = Cluster()
    gate = make_gate(
        store,
        clock,
        RecordingNotifier(),
        timeout=timedelta(seconds=30),
        async_sleeper=clock.asleep,
    )

    async def scale(**kwargs):
        cluster.deleted.append(kwargs)
        return "scaled"

    guarded = gate.aguard("scale_deployment", scale)
    result = asyncio.run(guarded(namespace="web", replicas=3))

    assert cluster.deleted == []
    assert "timed out" in result
    assert clock.now() >= gate.clock()
