"""Where a tool call meets the approval gate."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..approval.gate import ApprovalGate
from ..mcp.toolset import ToolCatalog, ToolResult, ToolSpec

log = logging.getLogger("devops-agent.tools")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One executed (or refused) tool call, for the audit trail and layer 5."""

    name: str
    args: dict[str, Any]
    result: ToolResult
    duration_ms: float
    server: str
    was_write: bool
    approval_id: str | None = None
    #: approved / rejected / expired / consumed — a rejection and a timeout
    #: mean very different things about the humans, and only this
    #: distinguishes them downstream.
    approval_status: str | None = None


def summarize_call(spec: ToolSpec, args: Mapping[str, Any]) -> str:
    """The one-line headline a reviewer sees above the arguments."""
    target = args.get("name") or args.get("resource") or args.get("deployment")
    where = args.get("namespace") or args.get("environment") or args.get("cluster")
    if target and where:
        return f"Agent wants to run `{spec.name}` on `{target}` in `{where}`"
    if target:
        return f"Agent wants to run `{spec.name}` on `{target}`"
    return f"Agent wants to run `{spec.name}` (from {spec.server})"


@dataclass
class ToolRegistry:
    """The catalog plus the gate: the only path from the model to infrastructure.

    Every call routes through :meth:`call`, which decides — from policy and the
    server's own annotations — whether a human has to see it first.
    """

    catalog: ToolCatalog
    gate: ApprovalGate
    #: Recent calls across all runs, for diagnostics. Bounded: the registry
    #: outlives every run in a long-lived server, and the durable audit trail
    #: is the approval store, not this.
    invocations: deque[ToolInvocation] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    def definitions(self) -> list[dict[str, Any]]:
        return self.catalog.definitions()

    def spec(self, name: str) -> ToolSpec | None:
        return self.catalog.tools.get(name)

    def requires_approval(self, name: str) -> bool:
        spec = self.catalog.tools.get(name)
        if spec is None:
            return True
        # A server that flags a tool destructive gets to escalate, never to
        # de-escalate: destructive_hint short-circuits to "needs a human",
        # while read_only_hint is only consulted for tools policy hasn't
        # classified.
        if spec.destructive_hint is True:
            return True
        return self.gate.policy.requires_approval(name, read_only_hint=spec.read_only_hint)

    async def call(
        self, name: str, args: Mapping[str, Any], *, thread_id: str | None = None
    ) -> ToolInvocation:
        started = time.monotonic()
        spec = self.catalog.tools.get(name)
        if spec is None:
            # The model hallucinated a tool, or a server disappeared mid-run.
            return self._record(
                name,
                args,
                ToolResult.text(
                    f"No tool named `{name}` is available. Use one of: "
                    f"{', '.join(sorted(self.catalog.tools))}.",
                    is_error=True,
                ),
                started,
                server="unknown",
                was_write=False,
            )

        needs_approval = self.requires_approval(name)
        approval_id: str | None = None
        approval_status: str | None = None

        if needs_approval:
            verdict = await self.gate.aauthorize(
                name,
                args,
                summary=summarize_call(spec, args),
                thread_id=thread_id,
            )
            approval_id = verdict.request.id if verdict.request else None
            approval_status = (
                verdict.request.status.value if verdict.request else "unnotified"
            )
            if not verdict.granted:
                # A refusal is information for the model, not a tool failure:
                # is_error would invite a retry, which is the opposite of what
                # a rejected write should produce.
                return self._record(
                    name,
                    args,
                    ToolResult.text(verdict.message or "Denied.", denied=True),
                    started,
                    server=spec.server,
                    was_write=True,
                    approval_id=approval_id,
                    approval_status=approval_status,
                )

        source = self.catalog.sources[name]
        result = await source.call_tool(name, args)
        return self._record(
            name,
            args,
            result,
            started,
            server=spec.server,
            was_write=needs_approval,
            approval_id=approval_id,
            approval_status=approval_status,
        )

    def _record(
        self,
        name: str,
        args: Mapping[str, Any],
        result: ToolResult,
        started: float,
        *,
        server: str,
        was_write: bool,
        approval_id: str | None = None,
        approval_status: str | None = None,
    ) -> ToolInvocation:
        invocation = ToolInvocation(
            name=name,
            args=dict(args),
            result=result,
            duration_ms=(time.monotonic() - started) * 1000,
            server=server,
            was_write=was_write,
            approval_id=approval_id,
            approval_status=approval_status,
        )
        self.invocations.append(invocation)
        log.info(
            "tool=%s server=%s write=%s denied=%s error=%s duration_ms=%.0f",
            name,
            server,
            was_write,
            result.denied,
            result.is_error,
            invocation.duration_ms,
        )
        return invocation
