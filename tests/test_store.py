from __future__ import annotations

from datetime import timedelta

import pytest

from agentic_devops.approval import (
    ApprovalConflict,
    ApprovalNotFound,
    ApprovalRequest,
    ApprovalStatus,
    ArgumentMismatch,
    RiskLevel,
    SQLiteApprovalStore,
    fingerprint_args,
)


def make_request(clock, tool="delete_resource", **args):
    return ApprovalRequest.new(
        tool_name=tool,
        args=args or {"namespace": "payments", "name": "api-7f9"},
        risk=RiskLevel.HIGH,
        timeout=timedelta(minutes=10),
        now=clock.now(),
        thread_id="incident-42",
    )


def test_create_and_read_back(store, clock):
    request = make_request(clock)
    store.create(request)

    loaded = store.get(request.id, now=clock.now())
    assert loaded.status is ApprovalStatus.PENDING
    assert loaded.tool_name == "delete_resource"
    assert loaded.args == {"namespace": "payments", "name": "api-7f9"}
    assert loaded.thread_id == "incident-42"


def test_unknown_id_raises(store, clock):
    with pytest.raises(ApprovalNotFound):
        store.get("deadbeef", now=clock.now())


def test_approval_survives_a_restart(tmp_path, clock):
    path = tmp_path / "approvals.db"
    first = SQLiteApprovalStore(path)
    request = make_request(clock)
    first.create(request)
    first.close()

    # the agent pod restarts mid-incident
    second = SQLiteApprovalStore(path)
    loaded = second.get(request.id, now=clock.now())
    assert loaded.status is ApprovalStatus.PENDING

    decided = second.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())
    assert decided.status is ApprovalStatus.APPROVED
    second.close()


def test_resolve_records_the_reviewer(store, clock):
    request = store.create(make_request(clock))
    decided = store.resolve(
        request.id, approved=True, decided_by="sre-oncall", note="checked the diff", now=clock.now()
    )
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.decided_by == "sre-oncall"
    assert decided.decision_note == "checked the diff"
    assert decided.decided_at == clock.now()


def test_second_decision_loses_the_race(store, clock):
    request = store.create(make_request(clock))
    store.resolve(request.id, approved=False, decided_by="first", now=clock.now())

    with pytest.raises(ApprovalConflict) as excinfo:
        store.resolve(request.id, approved=True, decided_by="second", now=clock.now())

    assert excinfo.value.request.status is ApprovalStatus.REJECTED
    assert excinfo.value.request.decided_by == "first"


def test_pending_request_expires_on_read(store, clock):
    request = store.create(make_request(clock))
    clock.advance(seconds=601)

    loaded = store.get(request.id, now=clock.now())
    assert loaded.status is ApprovalStatus.EXPIRED
    assert loaded.decided_by == "system"


def test_approval_after_the_deadline_is_refused(store, clock):
    """The failure that matters: a reviewer clicks Approve a second too late."""
    request = store.create(make_request(clock))
    clock.advance(seconds=601)

    with pytest.raises(ApprovalConflict) as excinfo:
        store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())

    assert excinfo.value.request.status is ApprovalStatus.EXPIRED
    assert store.get(request.id, now=clock.now()).status is ApprovalStatus.EXPIRED


def test_decision_just_inside_the_deadline_holds(store, clock):
    request = store.create(make_request(clock))
    clock.advance(seconds=599)

    decided = store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())
    assert decided.status is ApprovalStatus.APPROVED


def test_consume_spends_the_approval_once(store, clock):
    request = store.create(make_request(clock))
    store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())

    spent = store.consume(
        request.id, args_fingerprint=request.args_fingerprint, now=clock.now()
    )
    assert spent.status is ApprovalStatus.CONSUMED

    with pytest.raises(ApprovalConflict):
        store.consume(request.id, args_fingerprint=request.args_fingerprint, now=clock.now())


def test_cannot_consume_an_undecided_request(store, clock):
    request = store.create(make_request(clock))
    with pytest.raises(ApprovalConflict):
        store.consume(request.id, args_fingerprint=request.args_fingerprint, now=clock.now())


def test_changed_arguments_burn_the_approval(store, clock):
    request = store.create(make_request(clock))
    store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())

    swapped = fingerprint_args({"namespace": "payments", "name": "api-EVERYTHING"})
    with pytest.raises(ArgumentMismatch):
        store.consume(request.id, args_fingerprint=swapped, now=clock.now())

    after = store.get(request.id, now=clock.now())
    assert after.status is ApprovalStatus.REJECTED
    assert after.decision_note == "arguments changed after approval"


def test_fingerprint_ignores_key_order():
    assert fingerprint_args({"a": 1, "b": 2}) == fingerprint_args({"b": 2, "a": 1})
    assert fingerprint_args({"a": 1}) != fingerprint_args({"a": 2})


def test_list_pending_excludes_expired(store, clock):
    live = store.create(make_request(clock, name="live"))
    clock.advance(seconds=601)
    fresh = store.create(make_request(clock, name="fresh"))

    pending = store.list_pending(now=clock.now())
    ids = {request.id for request in pending}
    assert fresh.id in ids
    assert live.id not in ids


def test_expire_overdue_sweeps_everything(store, clock):
    store.create(make_request(clock, name="a"))
    store.create(make_request(clock, name="b"))
    clock.advance(seconds=601)

    assert store.expire_overdue(now=clock.now()) == 2
    assert store.expire_overdue(now=clock.now()) == 0


def test_history_is_scoped_by_thread(store, clock):
    store.create(make_request(clock))
    other = ApprovalRequest.new(
        tool_name="apply_manifest",
        args={"namespace": "web"},
        risk=RiskLevel.MEDIUM,
        timeout=timedelta(minutes=5),
        now=clock.now(),
        thread_id="deploy-99",
    )
    store.create(other)

    assert [r.id for r in store.history(thread_id="deploy-99")] == [other.id]
    assert len(store.history()) == 2


def test_concurrent_reviewers_produce_one_decision(store, clock):
    """Twenty threads race to decide the same request; exactly one wins.

    This is the property the in-memory dict could not offer: the decision is a
    single conditional UPDATE, so there is no window where two reviewers both
    believe they authorized the action.
    """
    import threading

    request = store.create(make_request(clock))
    winners: list[str] = []
    barrier = threading.Barrier(20)

    def click(reviewer: str) -> None:
        barrier.wait()
        try:
            store.resolve(request.id, approved=True, decided_by=reviewer, now=clock.now())
            winners.append(reviewer)
        except ApprovalConflict:
            pass

    threads = [threading.Thread(target=click, args=(f"reviewer-{i}",)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert store.get(request.id, now=clock.now()).decided_by == winners[0]


def test_concurrent_execution_spends_the_approval_once(store, clock):
    """Two agent threads holding the same approval: only one write runs."""
    import threading

    request = store.create(make_request(clock))
    store.resolve(request.id, approved=True, decided_by="sre-oncall", now=clock.now())

    executed: list[int] = []
    barrier = threading.Barrier(10)

    def run(index: int) -> None:
        barrier.wait()
        try:
            store.consume(
                request.id, args_fingerprint=request.args_fingerprint, now=clock.now()
            )
            executed.append(index)
        except ApprovalConflict:
            pass

    threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(executed) == 1
