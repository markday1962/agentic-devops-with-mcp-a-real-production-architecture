"""Webhook authentication.

The article's trigger server accepts any POST that parses. A webhook endpoint
that starts an agent run is a remote-code-execution surface with extra steps:
anyone who can reach it can make the agent investigate a fabricated incident,
and every fabricated incident is an opportunity to get a human to click Approve
on something. Every sender here is authenticated before its payload is read.
"""

from __future__ import annotations

import hashlib
import hmac
import time


class SignatureError(Exception):
    """The request did not come from the sender it claims to."""


def verify_github_signature(secret: str, *, body: bytes, signature: str | None) -> None:
    """Validate GitHub's ``X-Hub-Signature-256`` header."""
    if not signature:
        raise SignatureError("missing X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureError("GitHub signature mismatch")


def verify_pagerduty_signature(
    secret: str,
    *,
    body: bytes,
    signature: str | None,
) -> None:
    """Validate PagerDuty's ``X-PagerDuty-Signature`` header.

    The header carries a comma-separated list — PagerDuty sends one signature
    per active secret so keys can be rotated without downtime — and the request
    is valid if any of them matches.
    """
    if not signature:
        raise SignatureError("missing X-PagerDuty-Signature")
    expected = "v1=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    candidates = [part.strip() for part in signature.split(",") if part.strip()]
    if not any(hmac.compare_digest(expected, candidate) for candidate in candidates):
        raise SignatureError("PagerDuty signature mismatch")


def verify_shared_secret(secret: str, *, provided: str | None) -> None:
    """Constant-time comparison for simple bearer-token senders (schedulers,
    internal tooling)."""
    if not provided:
        raise SignatureError("missing authorization")
    if not hmac.compare_digest(secret, provided):
        raise SignatureError("shared secret mismatch")


def timestamp_is_fresh(sent_at: float, *, now: float | None = None, max_age: int = 300) -> bool:
    moment = time.time() if now is None else now
    return abs(moment - sent_at) <= max_age
