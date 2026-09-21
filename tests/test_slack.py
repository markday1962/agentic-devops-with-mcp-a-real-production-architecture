from __future__ import annotations

import hashlib
import hmac
import json
from datetime import timedelta
from urllib.parse import urlencode

import pytest

from agentic_devops.approval import ApprovalRequest, ApprovalStatus, RiskLevel
from agentic_devops.approval.slack import (
    SlackSignatureError,
    parse_action,
    payload_from_form_body,
    resolve_from_payload,
    verify_slack_signature,
)

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"


def sign(body: bytes, timestamp: int, secret: str = SECRET) -> str:
    base = b"v0:" + str(timestamp).encode() + b":" + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def click_payload(approval_id: str, *, approve: bool = True, user: str = "sre-oncall") -> dict:
    verb = "approve" if approve else "reject"
    return {
        "type": "block_actions",
        "user": {"id": "U123", "username": user},
        "actions": [{"action_id": f"approval_{verb}", "value": f"{verb}:{approval_id}"}],
    }


def pending(store, clock, tool="delete_resource"):
    request = ApprovalRequest.new(
        tool_name=tool,
        args={"namespace": "payments", "name": "api-7f9"},
        risk=RiskLevel.HIGH,
        timeout=timedelta(minutes=10),
        now=clock.now(),
    )
    return store.create(request)


# ── signature verification ───────────────────────────────────────────────


def test_valid_signature_passes():
    body = b"payload=%7B%7D"
    verify_slack_signature(
        SECRET, timestamp="1750000000", body=body, signature=sign(body, 1750000000), now=1750000000
    )


def test_tampered_body_is_rejected():
    body = b"payload=%7B%7D"
    signature = sign(body, 1750000000)
    with pytest.raises(SlackSignatureError):
        verify_slack_signature(
            SECRET,
            timestamp="1750000000",
            body=b"payload=evil",
            signature=signature,
            now=1750000000,
        )


def test_wrong_secret_is_rejected():
    body = b"payload=%7B%7D"
    with pytest.raises(SlackSignatureError):
        verify_slack_signature(
            SECRET,
            timestamp="1750000000",
            body=body,
            signature=sign(body, 1750000000, secret="not-the-secret"),
            now=1750000000,
        )


def test_replayed_request_is_rejected():
    body = b"payload=%7B%7D"
    signature = sign(body, 1750000000)
    with pytest.raises(SlackSignatureError):
        verify_slack_signature(
            SECRET,
            timestamp="1750000000",
            body=body,
            signature=signature,
            now=1750000000 + 3600,
        )


def test_missing_timestamp_is_rejected():
    with pytest.raises(SlackSignatureError):
        verify_slack_signature(SECRET, timestamp="", body=b"", signature="v0=abc")


# ── payload handling ─────────────────────────────────────────────────────


def test_parse_action_reads_id_and_reviewer():
    approval_id, approved, reviewer = parse_action(click_payload("abc123"))
    assert (approval_id, approved, reviewer) == ("abc123", True, "sre-oncall")


def test_parse_action_rejects_unknown_verbs():
    payload = click_payload("abc123")
    payload["actions"][0]["value"] = "nuke:abc123"
    with pytest.raises(ValueError):
        parse_action(payload)


def test_payload_from_form_body():
    body = urlencode({"payload": json.dumps(click_payload("abc123"))}).encode()
    assert payload_from_form_body(body)["actions"][0]["value"] == "approve:abc123"


# ── resolution ───────────────────────────────────────────────────────────


def test_click_records_the_decision(store, clock):
    request = pending(store, clock)
    resolved, message = resolve_from_payload(
        store, click_payload(request.id), now=clock.now()
    )
    assert resolved.status is ApprovalStatus.APPROVED
    assert resolved.decided_by == "sre-oncall"
    assert "approved" in message


def test_late_click_is_told_it_was_too_late(store, clock):
    request = pending(store, clock)
    clock.advance(seconds=601)

    resolved, message = resolve_from_payload(
        store, click_payload(request.id), now=clock.now()
    )
    assert resolved.status is ApprovalStatus.EXPIRED
    assert "Too late" in message
    assert store.get(request.id, now=clock.now()).status is ApprovalStatus.EXPIRED


def test_second_click_reports_the_first_decision(store, clock):
    request = pending(store, clock)
    resolve_from_payload(store, click_payload(request.id, approve=False, user="first"), now=clock.now())

    _, message = resolve_from_payload(
        store, click_payload(request.id, user="second"), now=clock.now()
    )
    assert "already rejected by first" in message


def test_unknown_approval_id_is_handled(store, clock):
    resolved, message = resolve_from_payload(store, click_payload("deadbeef"), now=clock.now())
    assert resolved is None
    assert "no longer exists" in message


def test_malformed_payload_is_handled(store, clock):
    resolved, message = resolve_from_payload(store, {"actions": []}, now=clock.now())
    assert resolved is None
    assert "Could not read" in message
