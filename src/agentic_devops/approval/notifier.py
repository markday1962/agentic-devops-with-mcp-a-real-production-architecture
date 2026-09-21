"""Getting a pending action in front of a human, and telling them how it ended."""

from __future__ import annotations

import logging
import sys
from typing import Protocol, TextIO, runtime_checkable

from .models import ApprovalRequest, ApprovalStatus, RiskLevel

log = logging.getLogger("devops-agent.approval")

RISK_EMOJI = {
    RiskLevel.HIGH: ":red_circle:",
    RiskLevel.MEDIUM: ":large_yellow_circle:",
    RiskLevel.LOW: ":large_green_circle:",
}

STATUS_EMOJI = {
    ApprovalStatus.APPROVED: ":white_check_mark:",
    ApprovalStatus.CONSUMED: ":white_check_mark:",
    ApprovalStatus.REJECTED: ":x:",
    ApprovalStatus.EXPIRED: ":hourglass_flowing_sand:",
    ApprovalStatus.PENDING: ":hourglass:",
}


@runtime_checkable
class Notifier(Protocol):
    def notify(self, request: ApprovalRequest) -> str | None:
        """Present the request to reviewers. Returns an opaque reference to the
        message (used later by :meth:`update`), or None if there is nothing to
        update. Raising means the humans were not reached — the gate treats
        that as a denial."""

    def update(self, request: ApprovalRequest) -> None:
        """Reflect the final decision wherever the request was posted."""


class NullNotifier:
    """Approvals nobody is told about. Only useful when another system owns the
    notification path, or in tests."""

    def notify(self, request: ApprovalRequest) -> str | None:
        return None

    def update(self, request: ApprovalRequest) -> None:
        return None


class ConsoleNotifier:
    """Prints requests to a stream. For local development, where the reviewer
    resolves requests with the CLI instead of a Slack button."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr

    def notify(self, request: ApprovalRequest) -> str | None:
        print(
            f"\n[{request.risk.value}] approval {request.id} required\n"
            f"{request.describe()}\n"
            f"expires {request.expires_at.isoformat()}\n"
            f"approve with: approvals resolve {request.id} --approve --by <you>\n",
            file=self._stream,
            flush=True,
        )
        return f"console:{request.id}"

    def update(self, request: ApprovalRequest) -> None:
        print(
            f"[{request.id}] {request.status.value}"
            f"{f' by {request.decided_by}' if request.decided_by else ''}",
            file=self._stream,
            flush=True,
        )


class SlackNotifier:
    """Block Kit message with Approve / Reject buttons.

    The buttons carry ``approve:<id>`` / ``reject:<id>`` as their value; the
    interactivity endpoint hands that to
    :func:`agentic_devops.approval.slack.resolve_from_payload`. Once decided,
    the message is edited in place so the buttons disappear — a stale message
    with live-looking buttons is how a reviewer approves the same action twice.
    """

    def __init__(
        self,
        token: str,
        channel: str,
        *,
        client: object | None = None,
        mention: str | None = None,
    ) -> None:
        if client is None:
            from slack_sdk import WebClient  # imported lazily: optional dependency

            client = WebClient(token=token)
        self._client = client
        self.channel = channel
        self.mention = mention

    def _blocks(self, request: ApprovalRequest) -> list[dict]:
        header = (
            f"{RISK_EMOJI[request.risk]} *AI Agent Action Request* "
            f"(ID: `{request.id}`, risk: *{request.risk.value}*)"
        )
        if self.mention and request.risk is RiskLevel.HIGH:
            header = f"{self.mention} {header}"
        context = [f"expires <!date^{int(request.expires_at.timestamp())}^{{time}}|soon>"]
        if request.thread_id:
            context.append(f"thread `{request.thread_id}`")
        if request.requested_by:
            context.append(f"agent `{request.requested_by}`")
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": header}},
            {"type": "section", "text": {"type": "mrkdwn", "text": request.describe()}},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " · ".join(context)}],
            },
            {
                "type": "actions",
                "block_id": f"approval:{request.id}",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": f"approve:{request.id}",
                        "action_id": "approval_approve",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "value": f"reject:{request.id}",
                        "action_id": "approval_reject",
                    },
                ],
            },
        ]

    def notify(self, request: ApprovalRequest) -> str | None:
        response = self._client.chat_postMessage(  # type: ignore[attr-defined]
            channel=self.channel,
            text=f"Approval required: {request.tool_name} ({request.risk.value})",
            blocks=self._blocks(request),
        )
        return f"{response['channel']}:{response['ts']}"

    def update(self, request: ApprovalRequest) -> None:
        if not request.notification_ref:
            return
        channel, _, ts = request.notification_ref.rpartition(":")
        emoji = STATUS_EMOJI.get(request.status, "")
        who = request.decided_by or "system"
        line = (
            f"{emoji} *{request.tool_name}* — *{request.status.value.upper()}* by {who} "
            f"(ID: `{request.id}`)"
        )
        if request.decision_note:
            line += f"\n_{request.decision_note}_"
        try:
            self._client.chat_update(  # type: ignore[attr-defined]
                channel=channel,
                ts=ts,
                text=line,
                blocks=[
                    {"type": "section", "text": {"type": "mrkdwn", "text": line}},
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": request.describe()},
                    },
                ],
            )
        except Exception:  # noqa: BLE001 - never let cosmetics fail a decision
            log.warning("could not update Slack message for approval %s", request.id, exc_info=True)
