"""Data types for the approval gate."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"

    @property
    def is_terminal(self) -> bool:
        return self is not ApprovalStatus.PENDING

    @property
    def authorizes_execution(self) -> bool:
        """Only a live approval authorizes a write. Consumed approvals are spent."""
        return self is ApprovalStatus.APPROVED


def fingerprint_args(args: Mapping[str, Any]) -> str:
    """Stable hash of tool arguments.

    The gate binds an approval to the exact arguments a human saw. Anything that
    cannot be represented as JSON is coerced via ``repr``, which is enough to
    make a substitution show up as a different fingerprint.
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One human decision about one proposed write action."""

    id: str
    tool_name: str
    args: dict[str, Any]
    args_fingerprint: str
    risk: RiskLevel
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime
    summary: str | None = None
    thread_id: str | None = None
    requested_by: str | None = None
    notification_ref: str | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_note: str | None = None

    @classmethod
    def new(
        cls,
        tool_name: str,
        args: Mapping[str, Any],
        risk: RiskLevel,
        timeout: timedelta,
        *,
        now: datetime | None = None,
        summary: str | None = None,
        thread_id: str | None = None,
        requested_by: str | None = None,
    ) -> "ApprovalRequest":
        created = now or utcnow()
        snapshot = dict(args)
        return cls(
            id=uuid.uuid4().hex[:8],
            tool_name=tool_name,
            args=snapshot,
            args_fingerprint=fingerprint_args(snapshot),
            risk=risk,
            status=ApprovalStatus.PENDING,
            requested_at=created,
            expires_at=created + timeout,
            summary=summary,
            thread_id=thread_id,
            requested_by=requested_by,
        )

    def with_notification(self, ref: str | None) -> "ApprovalRequest":
        return replace(self, notification_ref=ref)

    def is_overdue(self, now: datetime) -> bool:
        return self.status is ApprovalStatus.PENDING and now >= self.expires_at

    def describe(self) -> str:
        """Human-readable rendering for the reviewer."""
        body = json.dumps(self.args, indent=2, default=repr)
        head = self.summary or f"Agent wants to call `{self.tool_name}`"
        return f"{head}\n\n```\n{self.tool_name}({body})\n```"
