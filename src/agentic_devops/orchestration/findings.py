"""``log_finding`` — the agent's running record of what it learned.

Findings outlive the run. If the agent loops, times out, or is killed halfway
through an incident, what it had established is still on disk for the human who
picks the incident up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..mcp.toolset import ToolResult, ToolSpec
from .context import active_thread_id

log = logging.getLogger("devops-agent.findings")

FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What you found, in one sentence.",
        },
        "evidence": {
            "type": "string",
            "description": (
                "The specific observation supporting it — a log line, a metric "
                "value, a deployment timestamp. Not a restatement of the summary."
            ),
        },
        "service": {"type": "string", "description": "Service the finding concerns."},
        "significance": {
            "type": "string",
            "enum": ["context", "symptom", "cause", "ruled_out"],
            "description": (
                "context: background. symptom: an observed effect. cause: "
                "something you believe is causal. ruled_out: a hypothesis the "
                "evidence eliminates — record these, they save the next person time."
            ),
        },
    },
    "required": ["summary", "evidence", "significance"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Finding:
    summary: str
    evidence: str
    significance: str
    service: str | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recorded_at"] = self.recorded_at.isoformat()
        return data


@dataclass
class FindingsRecorder:
    """A ``ToolSource`` exposing a single local tool.

    Findings are bucketed by run. One agent serves concurrent incidents, and
    incident A's evidence appearing in incident B's report is worse than no
    report at all.
    """

    name: str = "local"
    by_thread: dict[str, list[Finding]] = field(default_factory=dict)

    def for_thread(self, thread_id: str) -> list[Finding]:
        return list(self.by_thread.get(thread_id, ()))

    @property
    def findings(self) -> list[Finding]:
        """Every finding across every run, oldest first. Diagnostics only —
        use :meth:`for_thread` for anything a human will read."""
        return sorted(
            (f for bucket in self.by_thread.values() for f in bucket),
            key=lambda f: f.recorded_at,
        )

    async def list_tools(self) -> Sequence[ToolSpec]:
        return [
            ToolSpec(
                name="log_finding",
                description=(
                    "Record something you have established about this incident. "
                    "Call this as you go, before acting on what you found — not "
                    "once at the end."
                ),
                input_schema=FINDING_SCHEMA,
                server=self.name,
                read_only_hint=True,
            )
        ]

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        if name != "log_finding":
            return ToolResult.text(f"no such tool: {name}", is_error=True)
        try:
            finding = Finding(
                summary=args["summary"],
                evidence=args["evidence"],
                significance=args["significance"],
                service=args.get("service"),
            )
        except KeyError as exc:
            return ToolResult.text(f"log_finding is missing required field {exc}", is_error=True)

        thread_id = active_thread_id()
        bucket = self.by_thread.setdefault(thread_id, [])
        bucket.append(finding)
        log.info(
            "finding thread=%s [%s] %s", thread_id, finding.significance, finding.summary
        )
        return ToolResult.text(f"Recorded finding {len(bucket)}.")

    def as_json(self, thread_id: str | None = None) -> str:
        findings = self.for_thread(thread_id) if thread_id else self.findings
        return json.dumps([f.as_dict() for f in findings], indent=2)
