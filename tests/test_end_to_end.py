"""All three layers, over HTTP, with a real MCP server subprocess.

A signed PagerDuty webhook starts a run; the agent proposes a destructive
tool call; the gate holds it; a signed Slack click releases it; the tool runs
on the real server. Only the model itself is a double.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from agentic_devops.approval import ApprovalStatus
from agentic_devops.mcp import MCPServerSpec
from agentic_devops.orchestration import AgentConfig
from agentic_devops.orchestration.agent import build_agent
from agentic_devops.triggers import RunStatus, TriggerSettings, create_app

from conftest import RecordingNotifier
from fake_anthropic import FakeAnthropic, calls, says
from test_triggers import (
    GITHUB_SECRET,
    PAGERDUTY_SECRET,
    SLACK_SECRET,
    pagerduty_payload,
    post_pagerduty,
    post_slack,
    wait_for,
)

SERVER = Path(__file__).parent / "mcp_test_server.py"


def test_alert_to_approval_to_execution(tmp_path, store):
    config = AgentConfig(
        servers=[
            MCPServerSpec(
                name="test-infra",
                command=sys.executable,
                args=(str(SERVER),),
                read_timeout=20.0,
            )
        ],
        approval_poll_interval=0.05,
        agent_id="devops-agent-e2e",
    )
    notifier = RecordingNotifier()
    client = FakeAnthropic(
        [
            calls(("get_pod_logs", {"namespace": "payments", "name": "api-7f9"})),
            calls(("delete_resource", {"namespace": "payments", "name": "api-7f9"})),
            says("Pod was OOMKilled and has been removed."),
        ]
    )

    settings = TriggerSettings(
        agent_config=config,
        github_secret=GITHUB_SECRET,
        pagerduty_secret=PAGERDUTY_SECRET,
        slack_signing_secret=SLACK_SECRET,
        runs_db=str(tmp_path / "runs.db"),
        workers=1,
        drain_timeout=5.0,
    )

    def scenario():
        # Built by the lifespan, inside the loop that serves requests: an MCP
        # session belongs to the loop that opened it.
        app = create_app(
            settings,
            agent_factory=lambda: build_agent(
                config, client=client, store=store, notifier=notifier
            ),
        )
        with TestClient(app) as http:
            accepted = post_pagerduty(http, pagerduty_payload())
            assert accepted.status_code == 202

            # The run reaches the gate and stops there.
            assert wait_for(lambda: notifier.notified), "no approval was requested"
            request = notifier.notified[0]
            assert request.tool_name == "delete_resource"
            assert request.risk.value == "HIGH"
            assert request.thread_id == "incident-PABC123"

            # A reviewer clicks Approve in Slack.
            clicked = post_slack(http, request.id)
            assert clicked.status_code == 200
            assert "approved" in clicked.json()["text"]

            assert wait_for(
                lambda: http.get("/runs/pagerduty:01EVENT").json()["status"]
                == RunStatus.COMPLETED.value
            ), "run never completed"

            record = http.get("/runs/pagerduty:01EVENT").json()

        return record

    record = scenario()

    assert record["turns"] == 3
    assert record["summary"] == "Pod was OOMKilled and has been removed."

    # The approval was granted, spent, and is not reusable.
    decided = store.get(notifier.notified[0].id)
    assert decided.status is ApprovalStatus.CONSUMED
    assert decided.decided_by == "sre-oncall"
