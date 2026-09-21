"""Translating real webhook payloads into agent work.

The article's models (``PagerDutyWebhook(event, incident)``,
``GitHubWebhook(action, pull_request, deployment)``) do not match what either
service actually sends. PagerDuty v3 nests everything under ``event.data``;
GitHub signals a finished deployment with a ``deployment_status`` event whose
state is ``success``, not with ``action == "completed"``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..orchestration.prompt import deployment_watch_goal, incident_goal
from .runs import RunRequest

log = logging.getLogger("devops-agent.triggers")

#: PagerDuty event types worth waking an agent for.
INCIDENT_EVENTS = {"incident.triggered", "incident.reopened"}


def pagerduty_request(payload: Mapping[str, Any]) -> RunRequest | None:
    """Build a run request from a PagerDuty v3 webhook, or None to ignore it."""
    event = payload.get("event") or {}
    event_type = event.get("event_type")
    if event_type not in INCIDENT_EVENTS:
        log.debug("ignoring PagerDuty event type %s", event_type)
        return None

    data = event.get("data") or {}
    incident_id = data.get("id")
    delivery_id = event.get("id")
    if not incident_id or not delivery_id:
        raise ValueError("PagerDuty payload is missing event.id or event.data.id")

    service = (data.get("service") or {}).get("summary") or "unknown service"
    goal = incident_goal(
        service=service,
        title=data.get("title") or "(no title)",
        urgency=data.get("urgency") or "unknown",
        triggered_at=data.get("created_at") or event.get("occurred_at") or "unknown",
        extra=f"PagerDuty incident: {data.get('html_url')}" if data.get("html_url") else None,
    )
    return RunRequest(
        # Keyed on the delivery, not the incident: a reopened incident is new
        # work, a retried delivery of the same event is not.
        event_key=f"pagerduty:{delivery_id}",
        thread_id=f"incident-{incident_id}",
        goal=goal,
        kind="incident",
        source="pagerduty",
    )


def github_request(
    payload: Mapping[str, Any],
    *,
    event_type: str | None,
    delivery_id: str | None,
    watch_minutes: int = 10,
) -> RunRequest | None:
    """Build a run request from a GitHub webhook, or None to ignore it."""
    if event_type != "deployment_status":
        log.debug("ignoring GitHub event type %s", event_type)
        return None
    if not delivery_id:
        raise ValueError("GitHub payload is missing X-GitHub-Delivery")

    status = payload.get("deployment_status") or {}
    if status.get("state") != "success":
        log.debug("ignoring deployment_status state %s", status.get("state"))
        return None

    deployment = payload.get("deployment") or {}
    repo = (payload.get("repository") or {}).get("full_name") or "unknown repo"
    environment = (
        deployment.get("environment") or status.get("environment") or "unknown"
    )
    ref = deployment.get("ref") or deployment.get("sha") or "unknown"

    return RunRequest(
        event_key=f"github:{delivery_id}",
        thread_id=f"deploy-{deployment.get('id') or delivery_id}",
        goal=deployment_watch_goal(
            environment=f"{repo} {environment}", ref=ref, minutes=watch_minutes
        ),
        kind="deployment-watch",
        source="github",
    )
