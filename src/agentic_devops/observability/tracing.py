"""OpenTelemetry spans for the agent's own reasoning.

The article's tracer opens a span and closes it in the same breath::

    def trace_agent_run(task_goal, thread_id):
        with tracer.start_as_current_span('agent.run') as span:
            span.set_attribute(...)      # and then the with-block ends

Every span is instantaneous, nothing nests, and the trace shows a flat list of
zero-duration markers. A span has to *contain* the work, which is why
:class:`AgentEvents` deals in context managers — the run span wraps the whole
loop, turn spans nest inside it, and tool spans nest inside the turn that made
them.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from ..orchestration.loop import AgentRun
from ..orchestration.registry import ToolInvocation
from .metrics import AgentMetrics, _tool_outcome
from .redaction import Redactor, allowed_attributes, argument_shape, flatten

log = logging.getLogger("devops-agent.observability")


class TracedSpan:
    """Wraps an OTel span in the small surface the loop is allowed to use."""

    def __init__(self, span: Any, events: "OTelEvents", run: AgentRun | None = None) -> None:
        self._span = span
        self._events = events
        self._run = run

    def annotate(self, **attributes: Any) -> None:
        for key, value in allowed_attributes(attributes.items()).items():
            self._span.set_attribute(key, value)

    def record(self, invocation: ToolInvocation) -> None:
        self._events.record_invocation(self._span, self._run, invocation)


@dataclass
class OTelEvents:
    """:class:`AgentEvents` backed by OpenTelemetry."""

    tracer: Any
    metrics: AgentMetrics | None = None
    redactor: Redactor = field(default_factory=Redactor)
    #: Put goals, tool arguments and results into spans. Off by default —
    #: see redaction.py for why.
    record_payloads: bool = False
    _started: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    # ── spans ────────────────────────────────────────────────────────────

    @contextmanager
    def run_span(self, run: AgentRun) -> Iterator[TracedSpan]:
        started = time.monotonic()
        self._started[run.thread_id] = started
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attributes(
                allowed_attributes(
                    {
                        "agent.thread_id": run.thread_id,
                        "agent.service": run.service,
                        "agent.goal_length": len(run.goal),
                    }.items()
                )
            )
            if self.record_payloads:
                span.set_attribute("agent.goal", self.redactor.text(run.goal))
            try:
                yield TracedSpan(span, self, run)
            except BaseException as exc:
                span.record_exception(exc)
                self._set_error(span, f"{type(exc).__name__}: {exc}")
                raise
            finally:
                span.set_attributes(
                    {
                        "agent.turns": run.turns,
                        "agent.tool_calls": len(run.invocations),
                        "agent.denied_writes": len(run.denied_writes),
                        "agent.findings": len(run.findings),
                        "agent.tokens.input": run.usage.input_tokens,
                        "agent.tokens.output": run.usage.output_tokens,
                        "agent.tokens.cache_read": run.usage.cache_read_input_tokens,
                    }
                )
                if run.halted:
                    # A halt is not an exception, but it is the thing an
                    # operator is looking for when they open the trace.
                    span.set_attribute("agent.halted", self.redactor.text(run.halted))
                    self._set_error(span, run.halted)

    @contextmanager
    def turn_span(self, run: AgentRun, turn: int) -> Iterator[TracedSpan]:
        with self.tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("agent.turn", turn)
            span.set_attribute("agent.thread_id", run.thread_id)
            yield TracedSpan(span, self, run)

    @contextmanager
    def tool_span(
        self, run: AgentRun, name: str, args: Mapping[str, Any]
    ) -> Iterator[TracedSpan]:
        with self.tracer.start_as_current_span(f"agent.tool/{name}") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute("agent.thread_id", run.thread_id)
            for key, value in argument_shape(args).items():
                span.set_attribute(f"tool.{key}", value)
            if self.record_payloads:
                for key, value in flatten("tool.arg", self.redactor.mapping(args)).items():
                    span.set_attribute(key, value)
            yield TracedSpan(span, self, run)

    # ── recording ────────────────────────────────────────────────────────

    def record_invocation(
        self, span: Any, run: AgentRun | None, invocation: ToolInvocation
    ) -> None:
        outcome = _tool_outcome(invocation)
        span.set_attributes(
            allowed_attributes(
                {
                    "tool.server": invocation.server,
                    "tool.is_write_op": invocation.was_write,
                    "tool.outcome": outcome,
                    "tool.duration_ms": invocation.duration_ms,
                    "tool.result_length": len(invocation.result.rendered()),
                    "approval.id": invocation.approval_id,
                    "approval.status": invocation.approval_status,
                }.items()
            )
        )
        if self.record_payloads:
            span.set_attribute(
                "tool.result", self.redactor.text(invocation.result.rendered())
            )
        if outcome == "error":
            self._set_error(span, "tool returned an error")
        if self.metrics is not None and run is not None:
            self.metrics.record_tool_call(run, invocation)

    def on_response(self, run: AgentRun, response: Any) -> None:
        model = getattr(response, "model", None)
        if model:
            run.model = model

    def on_finish(self, run: AgentRun) -> None:
        started = self._started.pop(run.thread_id, None)
        if self.metrics is not None:
            self.metrics.record_run(
                run, time.monotonic() - started if started else 0.0
            )

    def _set_error(self, span: Any, message: str) -> None:
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, message))
        except Exception:  # noqa: BLE001 - never let telemetry break a run
            log.debug("could not set span status", exc_info=True)
