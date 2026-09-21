"""Turning a Slack button click into a recorded decision.

Pure functions, no web framework: the webhook server (layer 3) owns the route
and passes the raw body and headers in. Anyone who can POST to that endpoint
can approve production writes, so signature verification is not optional.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Any, Mapping

from .errors import ApprovalConflict, ApprovalNotFound
from .models import ApprovalRequest, ApprovalStatus
from .store import ApprovalStore

log = logging.getLogger("devops-agent.approval.slack")

#: Slack's own guidance: reject anything older than five minutes to blunt replays.
MAX_SIGNATURE_AGE_SECONDS = 60 * 5


class SlackSignatureError(Exception):
    """The request did not come from Slack, or came too long ago."""


def verify_slack_signature(
    signing_secret: str,
    *,
    timestamp: str,
    body: bytes,
    signature: str,
    now: float | None = None,
    max_age: int = MAX_SIGNATURE_AGE_SECONDS,
) -> None:
    """Validate ``X-Slack-Request-Timestamp`` / ``X-Slack-Signature``.

    Raises :class:`SlackSignatureError` on any failure; returns None on success.
    """
    try:
        sent_at = int(timestamp)
    except (TypeError, ValueError) as exc:
        raise SlackSignatureError("missing or malformed request timestamp") from exc

    moment = time.time() if now is None else now
    if abs(moment - sent_at) > max_age:
        raise SlackSignatureError("request timestamp outside the accepted window")

    basestring = b"v0:" + str(sent_at).encode() + b":" + body
    digest = hmac.new(signing_secret.encode(), basestring, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(f"v0={digest}", signature or ""):
        raise SlackSignatureError("signature mismatch")


def parse_action(payload: Mapping[str, Any]) -> tuple[str, bool, str]:
    """Extract ``(approval_id, approved, reviewer)`` from an interaction payload."""
    actions = payload.get("actions") or []
    if not actions:
        raise ValueError("interaction payload carried no actions")

    value = actions[0].get("value", "")
    verb, _, approval_id = value.partition(":")
    if verb not in {"approve", "reject"} or not approval_id:
        raise ValueError(f"unrecognised action value {value!r}")

    user = payload.get("user") or {}
    reviewer = user.get("username") or user.get("name") or user.get("id") or "unknown"
    return approval_id, verb == "approve", reviewer


def resolve_from_payload(
    store: ApprovalStore,
    payload: Mapping[str, Any],
    *,
    now: datetime | None = None,
    notifier: Any | None = None,
) -> tuple[ApprovalRequest | None, str]:
    """Record the click. Returns ``(request, message_for_the_reviewer)``.

    Never raises for the ordinary unhappy paths — a reviewer clicking a button
    that already timed out should get a straight answer in Slack, not a 500.
    """
    try:
        approval_id, approved, reviewer = parse_action(payload)
    except ValueError as exc:
        return None, f"Could not read that action: {exc}"

    try:
        request = store.resolve(
            approval_id,
            approved=approved,
            decided_by=reviewer,
            note=f"decided in Slack by {reviewer}",
            now=now,
        )
    except ApprovalNotFound:
        return None, f"Approval `{approval_id}` no longer exists."
    except ApprovalConflict as exc:
        current = exc.request
        if current.status is ApprovalStatus.EXPIRED:
            message = (
                f"Too late — approval `{approval_id}` timed out at "
                f"{current.expires_at.isoformat()} and was denied. "
                "Ask the agent to propose the action again."
            )
        else:
            message = (
                f"Approval `{approval_id}` was already "
                f"{current.status.value} by {current.decided_by or 'someone else'}."
            )
        if notifier is not None:
            notifier.update(current)
        return current, message

    if notifier is not None:
        notifier.update(request)
    verb = "approved" if approved else "rejected"
    return request, f"You {verb} `{request.tool_name}` (ID `{approval_id}`)."


def payload_from_form_body(body: bytes) -> dict[str, Any]:
    """Slack posts interactions as ``payload=<url-encoded JSON>``."""
    from urllib.parse import parse_qs

    fields = parse_qs(body.decode("utf-8"))
    raw = fields.get("payload", [None])[0]
    if raw is None:
        raise ValueError("no payload field in interaction body")
    return json.loads(raw)
