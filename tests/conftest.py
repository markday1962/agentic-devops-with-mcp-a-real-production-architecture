from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentic_devops.approval import ApprovalGate, ApprovalPolicy, SQLiteApprovalStore


class FakeClock:
    """Controllable time. ``sleep`` advances it instead of waiting."""

    def __init__(self, start: datetime | None = None) -> None:
        self.value = start or datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)

    async def asleep(self, seconds: float) -> None:
        """Async sleeper for the gate: advances the clock instead of waiting."""
        self.value += timedelta(seconds=seconds)


class RecordingNotifier:
    """Captures what would have been posted, and can stand in for a reviewer."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.notified: list = []
        self.updated: list = []

    def notify(self, request):
        if self.fail:
            raise RuntimeError("slack is down")
        self.notified.append(request)
        return f"C123:{request.id}"

    def update(self, request):
        self.updated.append(request)


class AutoDecidingNotifier(RecordingNotifier):
    """A reviewer who clicks the moment the message is posted."""

    def __init__(self, store, *, approve: bool, reviewer: str = "sre-oncall", clock=None):
        super().__init__()
        self.store = store
        self.approve = approve
        self.reviewer = reviewer
        self.clock = clock

    def notify(self, request):
        ref = super().notify(request)
        self.store.resolve(
            request.id,
            approved=self.approve,
            decided_by=self.reviewer,
            now=self.clock.now() if self.clock else None,
        )
        return ref


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path) -> SQLiteApprovalStore:
    store = SQLiteApprovalStore(tmp_path / "approvals.db")
    yield store
    store.close()


@pytest.fixture
def gate(store, clock) -> ApprovalGate:
    return ApprovalGate(
        store=store,
        notifier=RecordingNotifier(),
        policy=ApprovalPolicy(),
        timeout=timedelta(minutes=10),
        poll_interval=2.0,
        agent_id="devops-agent-1",
        clock=clock.now,
        sleeper=clock.sleep,
        async_sleeper=clock.asleep,
    )
