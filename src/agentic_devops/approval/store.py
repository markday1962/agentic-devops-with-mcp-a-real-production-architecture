"""Durable storage for approval requests.

The article kept pending approvals in ``self.pending``, a dict on the gate
instance. That works until the agent pod restarts mid-incident, at which point
every in-flight approval silently becomes a timeout, and it cannot support more
than one process — the webhook that receives the Slack button click has to be
the same process that is blocked waiting on it.

Everything here is written so the *decision* is a single atomic UPDATE guarded
by the current status and the deadline. Two reviewers racing, or a click that
lands a second after the timeout, resolve deterministically instead of
whichever thread happened to touch the dict last.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import ApprovalConflict, ApprovalNotFound, ArgumentMismatch
from .models import ApprovalRequest, ApprovalStatus, RiskLevel, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id                TEXT PRIMARY KEY,
    tool_name         TEXT NOT NULL,
    args              TEXT NOT NULL,
    args_fingerprint  TEXT NOT NULL,
    risk              TEXT NOT NULL,
    status            TEXT NOT NULL,
    requested_at      TEXT NOT NULL,
    expires_at        TEXT NOT NULL,
    summary           TEXT,
    thread_id         TEXT,
    requested_by      TEXT,
    notification_ref  TEXT,
    decided_at        TEXT,
    decided_by        TEXT,
    decision_note     TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals (status, expires_at);
CREATE INDEX IF NOT EXISTS idx_approvals_thread ON approvals (thread_id);
"""


@runtime_checkable
class ApprovalStore(Protocol):
    """Storage contract the gate depends on. Swap SQLite for Postgres by
    implementing these five methods with the same atomicity guarantees."""

    def create(self, request: ApprovalRequest) -> ApprovalRequest: ...

    def get(self, request_id: str, *, now: datetime | None = None) -> ApprovalRequest: ...

    def resolve(
        self,
        request_id: str,
        *,
        approved: bool,
        decided_by: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> ApprovalRequest: ...

    def consume(
        self,
        request_id: str,
        *,
        args_fingerprint: str,
        now: datetime | None = None,
    ) -> ApprovalRequest: ...

    def list_pending(self, *, now: datetime | None = None) -> list[ApprovalRequest]: ...


def _iso(value: datetime) -> str:
    return value.isoformat()


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteApprovalStore:
    """Single-file store, safe across threads and processes on one host.

    ``:memory:`` is accepted for tests; it is process-local by definition, so do
    not use it for anything that needs to survive a restart.
    """

    def __init__(self, path: str | Path = "approvals.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── reads ────────────────────────────────────────────────────────────

    def _row_to_request(self, row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            id=row["id"],
            tool_name=row["tool_name"],
            args=json.loads(row["args"]),
            args_fingerprint=row["args_fingerprint"],
            risk=RiskLevel(row["risk"]),
            status=ApprovalStatus(row["status"]),
            requested_at=_parse(row["requested_at"]),  # type: ignore[arg-type]
            expires_at=_parse(row["expires_at"]),  # type: ignore[arg-type]
            summary=row["summary"],
            thread_id=row["thread_id"],
            requested_by=row["requested_by"],
            notification_ref=row["notification_ref"],
            decided_at=_parse(row["decided_at"]),
            decided_by=row["decided_by"],
            decision_note=row["decision_note"],
        )

    def _fetch(self, request_id: str) -> ApprovalRequest:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (request_id,)
        ).fetchone()
        if row is None:
            raise ApprovalNotFound(f"no approval request with id {request_id!r}")
        return self._row_to_request(row)

    # ── writes ───────────────────────────────────────────────────────────

    def create(self, request: ApprovalRequest) -> ApprovalRequest:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO approvals (
                    id, tool_name, args, args_fingerprint, risk, status,
                    requested_at, expires_at, summary, thread_id, requested_by,
                    notification_ref, decided_at, decided_by, decision_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.id,
                    request.tool_name,
                    json.dumps(request.args, default=repr),
                    request.args_fingerprint,
                    request.risk.value,
                    request.status.value,
                    _iso(request.requested_at),
                    _iso(request.expires_at),
                    request.summary,
                    request.thread_id,
                    request.requested_by,
                    request.notification_ref,
                    _iso(request.decided_at) if request.decided_at else None,
                    request.decided_by,
                    request.decision_note,
                ),
            )
        return request

    def attach_notification(self, request_id: str, ref: str | None) -> None:
        """Record where the request was posted, so the gate can edit that
        message once the decision lands."""
        with self._lock:
            self._conn.execute(
                "UPDATE approvals SET notification_ref = ? WHERE id = ?", (ref, request_id)
            )

    def get(self, request_id: str, *, now: datetime | None = None) -> ApprovalRequest:
        """Read a request, expiring it first if its deadline has passed.

        Expiry is applied lazily on read rather than by a sweeper, so a request
        is never observed as pending past its deadline even if no sweeper runs.
        """
        moment = now or utcnow()
        with self._lock:
            self._expire_overdue_locked(moment, request_id=request_id)
            return self._fetch(request_id)

    def _expire_overdue_locked(
        self, now: datetime, *, request_id: str | None = None
    ) -> int:
        sql = (
            "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, "
            "decision_note = ? WHERE status = ? AND expires_at <= ?"
        )
        params: list[Any] = [
            ApprovalStatus.EXPIRED.value,
            _iso(now),
            "system",
            "no response before deadline; denied by default",
            ApprovalStatus.PENDING.value,
            _iso(now),
        ]
        if request_id is not None:
            sql += " AND id = ?"
            params.append(request_id)
        return self._conn.execute(sql, params).rowcount

    def expire_overdue(self, *, now: datetime | None = None) -> int:
        """Sweep every overdue request. Useful for a periodic janitor so the
        audit log shows timeouts even for requests nobody ever polls again."""
        moment = now or utcnow()
        with self._lock:
            return self._expire_overdue_locked(moment)

    def resolve(
        self,
        request_id: str,
        *,
        approved: bool,
        decided_by: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Record a human decision.

        The UPDATE requires the request to still be pending *and* inside its
        deadline, so a button click that arrives after the timeout cannot
        resurrect a denied action. Raises :class:`ApprovalConflict` when the
        decision did not take, carrying the request's real state.
        """
        moment = now or utcnow()
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        with self._lock:
            changed = self._conn.execute(
                """
                UPDATE approvals
                   SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?
                 WHERE id = ? AND status = ? AND expires_at > ?
                """,
                (
                    status.value,
                    _iso(moment),
                    decided_by,
                    note,
                    request_id,
                    ApprovalStatus.PENDING.value,
                    _iso(moment),
                ),
            ).rowcount
            if changed:
                return self._fetch(request_id)

            # Did not take: either already decided, or the deadline passed.
            self._expire_overdue_locked(moment, request_id=request_id)
            current = self._fetch(request_id)
        raise ApprovalConflict(
            f"approval {request_id} is {current.status.value}, cannot record "
            f"{status.value} from {decided_by}",
            current,
        )

    def consume(
        self,
        request_id: str,
        *,
        args_fingerprint: str,
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Spend an approval, immediately before the write executes.

        One approval authorizes exactly one call — the article's rule about
        never chaining write actions is enforced here rather than left to the
        model's goodwill. A fingerprint mismatch means the arguments changed
        between the human reading them and the tool running, so the approval is
        burned rather than reused.
        """
        moment = now or utcnow()
        with self._lock:
            current = self._fetch(request_id)
            if current.args_fingerprint != args_fingerprint:
                self._conn.execute(
                    """
                    UPDATE approvals
                       SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?
                     WHERE id = ? AND status = ?
                    """,
                    (
                        ApprovalStatus.REJECTED.value,
                        _iso(moment),
                        "system",
                        "arguments changed after approval",
                        request_id,
                        ApprovalStatus.APPROVED.value,
                    ),
                )
                raise ArgumentMismatch(
                    f"approval {request_id} was granted for different arguments",
                    self._fetch(request_id),
                )

            changed = self._conn.execute(
                "UPDATE approvals SET status = ? WHERE id = ? AND status = ?",
                (
                    ApprovalStatus.CONSUMED.value,
                    request_id,
                    ApprovalStatus.APPROVED.value,
                ),
            ).rowcount
            if changed:
                return self._fetch(request_id)
            current = self._fetch(request_id)
        raise ApprovalConflict(
            f"approval {request_id} is {current.status.value}, not an unspent approval",
            current,
        )

    def list_pending(self, *, now: datetime | None = None) -> list[ApprovalRequest]:
        moment = now or utcnow()
        with self._lock:
            self._expire_overdue_locked(moment)
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY requested_at",
                (ApprovalStatus.PENDING.value,),
            ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def history(self, *, thread_id: str | None = None, limit: int = 100) -> list[ApprovalRequest]:
        """Audit trail. Every request ever made, newest first."""
        with self._lock:
            if thread_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM approvals ORDER BY requested_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM approvals WHERE thread_id = ? "
                    "ORDER BY requested_at DESC LIMIT ?",
                    (thread_id, limit),
                ).fetchall()
        return [self._row_to_request(row) for row in rows]
