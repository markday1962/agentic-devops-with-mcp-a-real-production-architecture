"""The metrics the article listed in a comment block, actually recorded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..orchestration.loop import AgentRun
from ..orchestration.registry import ToolInvocation


def _tool_outcome(invocation: ToolInvocation) -> str:
    if invocation.result.denied:
        return "denied"
    if invocation.result.is_error:
        return "error"
    return "ok"


@dataclass
class AgentMetrics:
    """OTel instruments for the agent's own behaviour.

    The four questions these are meant to answer, which the article's comment
    list does not quite: is the agent finishing what it starts, is it spending
    tokens proportionate to what it achieves, are humans actually answering
    approval requests, and which tools fail.
    """

    meter: Any

    def __post_init__(self) -> None:
        self.runs = self.meter.create_counter(
            "agent.runs", unit="1", description="Agent runs by outcome"
        )
        self.run_duration = self.meter.create_histogram(
            "agent.run.duration", unit="s", description="Wall-clock time per run"
        )
        self.run_turns = self.meter.create_histogram(
            "agent.run.turns", unit="1", description="Model turns per run"
        )
        self.tool_calls = self.meter.create_counter(
            "agent.tool_calls", unit="1", description="Tool calls by tool and outcome"
        )
        self.tool_duration = self.meter.create_histogram(
            "agent.tool.duration", unit="ms", description="Tool call latency"
        )
        self.approvals = self.meter.create_counter(
            "agent.approvals",
            unit="1",
            description="Approval requests by outcome (approved/rejected/expired)",
        )
        self.tokens = self.meter.create_counter(
            "agent.tokens", unit="1", description="Tokens by kind"
        )
        self.findings = self.meter.create_counter(
            "agent.findings", unit="1", description="Findings recorded, by significance"
        )
        self.webhooks = self.meter.create_counter(
            "agent.webhooks", unit="1", description="Inbound webhooks by source and disposition"
        )

    def record_tool_call(self, run: AgentRun, invocation: ToolInvocation) -> None:
        attributes = {
            "tool": invocation.name,
            "server": invocation.server,
            "outcome": _tool_outcome(invocation),
            "write": invocation.was_write,
        }
        self.tool_calls.add(1, attributes)
        self.tool_duration.record(invocation.duration_ms, attributes)

        if invocation.approval_status:
            # Separating expired from rejected is the point: a rising
            # rejection rate means the agent is proposing bad actions, a
            # rising expiry rate means nobody is reading Slack. Those need
            # different people to do different things.
            self.approvals.add(
                1, {"outcome": invocation.approval_status, "tool": invocation.name}
            )

    def record_run(self, run: AgentRun, duration_seconds: float) -> None:
        outcome = "halted" if run.halted else "completed"
        attributes = {"outcome": outcome, "service": run.service or "unknown"}
        self.runs.add(1, attributes)
        self.run_duration.record(duration_seconds, attributes)
        self.run_turns.record(run.turns, attributes)

        for kind, value in (
            ("input", run.usage.input_tokens),
            ("output", run.usage.output_tokens),
            ("cache_read", run.usage.cache_read_input_tokens),
            ("cache_write", run.usage.cache_creation_input_tokens),
        ):
            if value:
                self.tokens.add(value, {"kind": kind, "model": run.model or "unknown"})

        for finding in run.findings:
            self.findings.add(1, {"significance": finding.significance})

    def record_webhook(self, source: str, disposition: str) -> None:
        self.webhooks.add(1, {"source": source, "disposition": disposition})
