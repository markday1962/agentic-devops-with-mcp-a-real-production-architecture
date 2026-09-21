from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from agentic_devops.approval import ApprovalGate, ApprovalPolicy
from agentic_devops.mcp import FakeToolSource
from agentic_devops.observability import (
    REDACTED,
    JSONFormatter,
    Redactor,
    TraceContextFilter,
    argument_shape,
    configure_observability,
    events_for_testing,
)
from agentic_devops.orchestration import (
    AgentLoop,
    FindingsRecorder,
    ToolRegistry,
    build_catalog_from_sources,
)

from conftest import AutoDecidingNotifier, RecordingNotifier
from fake_anthropic import FakeAnthropic, FakeMessage, TextBlock, calls, says


# ── redaction ────────────────────────────────────────────────────────────

redactor = Redactor()


def test_ordinary_values_survive():
    assert redactor.value("namespace", "payments") == "payments"
    assert redactor.value("replicas", 3) == 3


@pytest.mark.parametrize(
    "key",
    ["token", "password", "aws_secret_access_key", "dbPassword", "Authorization"],
)
def test_secret_argument_names_are_never_recorded(key):
    assert redactor.value(key, "hunter2") == REDACTED


@pytest.mark.parametrize(
    "text",
    [
        "postgres://admin:s3cret@db.internal:5432/app",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "xoxb-123456789012-abcdefghijkl",
        "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_credential_shapes_are_scrubbed_wherever_they_appear(text):
    """Secrets do not only arrive in arguments named `password` — they turn up
    inside log excerpts and stack traces the agent pastes around."""
    cleaned = redactor.text(f"context before {text} context after")
    assert REDACTED in cleaned
    assert "s3cret" not in cleaned
    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned


def test_long_values_are_truncated():
    result = redactor.value("body", "x" * 1000)
    assert len(result) < 400
    assert "+744" in result


def test_argument_shape_describes_without_disclosing():
    shape = argument_shape({"namespace": "payments-eu", "customer_id": "cus_9912"})
    flat = str(shape)
    assert "payments-eu" not in flat
    assert "cus_9912" not in flat
    assert shape["arg_names"] == "customer_id,namespace"
    assert shape["arg_count"] == 2


# ── span structure ───────────────────────────────────────────────────────


async def build_loop(store, clock, script, events, *, approve=True):
    source = FakeToolSource(name="kubernetes")
    source.add("get_pod_logs", lambda a: "OOMKilled", read_only=True)
    source.add("list_deployments", lambda a: "3/3", read_only=True)
    source.add("delete_resource", lambda a: "deleted", destructive=True)
    source.add("broken_tool", lambda a: (_ for _ in ()).throw(RuntimeError("nope")), read_only=True)
    catalog = await build_catalog_from_sources([source, FindingsRecorder()])
    gate = ApprovalGate(
        store=store,
        notifier=AutoDecidingNotifier(store, approve=approve, clock=clock)
        if approve is not None
        else RecordingNotifier(),
        policy=ApprovalPolicy(),
        # Short deadline so the timeout case expires in a handful of polls
        # rather than grinding the fake clock through ten minutes.
        timeout=timedelta(seconds=1),
        poll_interval=0.1,
        clock=clock.now,
        sleeper=clock.sleep,
        async_sleeper=clock.asleep,
    )
    return AgentLoop(
        client=FakeAnthropic(script),
        registry=ToolRegistry(catalog=catalog, gate=gate),
        recorder=FindingsRecorder(),
        events=events,
    )


def spans_by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_spans_nest_run_turn_tool(store, clock):
    """The article's tracer produced a flat list of zero-duration markers."""
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "api-7f9"})), says("Done.")],
            events,
            approve=None,
        )
        await loop.run("investigate", thread_id="incident-42")

    asyncio.run(scenario())
    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}

    tool = by_name["agent.tool/get_pod_logs"]
    turn = next(s for s in spans if s.name == "agent.turn")
    run = by_name["agent.run"]

    assert tool.parent.span_id == turn.context.span_id
    assert turn.parent.span_id == run.context.span_id
    assert run.parent is None
    # And they contain the work, rather than being instantaneous markers.
    assert run.end_time > run.start_time
    assert run.start_time <= tool.start_time and tool.end_time <= run.end_time


def test_parallel_reads_are_siblings_not_a_chain(store, clock):
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [
                calls(("get_pod_logs", {"name": "a"}), ("list_deployments", {"ns": "b"})),
                says("Both."),
            ],
            events,
            approve=None,
        )
        await loop.run("check both", thread_id="incident-42")

    asyncio.run(scenario())
    spans = exporter.get_finished_spans()
    tools = [s for s in spans if s.name.startswith("agent.tool/")]
    turn = next(s for s in spans if s.name == "agent.turn")

    assert len(tools) == 2
    assert {t.parent.span_id for t in tools} == {turn.context.span_id}


def test_run_span_carries_the_summary_an_operator_wants(store, clock):
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [calls(("delete_resource", {"name": "api-7f9"})), says("Done.")],
            events,
        )
        await loop.run("clear the pod", thread_id="incident-42", service="payments-api")

    asyncio.run(scenario())
    run = spans_by_name(exporter)["agent.run"]

    assert run.attributes["agent.thread_id"] == "incident-42"
    assert run.attributes["agent.service"] == "payments-api"
    assert run.attributes["agent.turns"] == 2
    assert run.attributes["agent.tool_calls"] == 1
    assert run.attributes["agent.tokens.input"] == 200


def test_a_halted_run_is_marked_an_error(store, clock):
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [FakeMessage(content=[TextBlock("...")], stop_reason="max_tokens")],
            events,
            approve=None,
        )
        await loop.run("investigate", thread_id="incident-42")

    asyncio.run(scenario())
    run = spans_by_name(exporter)["agent.run"]

    assert run.status.status_code.name == "ERROR"
    assert "max_tokens" in run.attributes["agent.halted"]


def test_tool_errors_are_visible_on_the_span(store, clock):
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store, clock, [calls(("broken_tool", {})), says("Oh well.")], events, approve=None
        )
        await loop.run("try it", thread_id="incident-42")

    asyncio.run(scenario())
    tool = spans_by_name(exporter)["agent.tool/broken_tool"]

    assert tool.attributes["tool.outcome"] == "error"
    assert tool.status.status_code.name == "ERROR"


def test_approval_outcome_is_on_the_tool_span(store, clock):
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [calls(("delete_resource", {"name": "x"})), says("Understood.")],
            events,
            approve=False,
        )
        await loop.run("delete it", thread_id="incident-42")

    asyncio.run(scenario())
    tool = spans_by_name(exporter)["agent.tool/delete_resource"]

    assert tool.attributes["tool.outcome"] == "denied"
    assert tool.attributes["approval.status"] == "rejected"
    assert tool.attributes["tool.is_write_op"] is True


# ── what spans must NOT contain ──────────────────────────────────────────


def test_payloads_are_not_recorded_by_default(store, clock):
    """Tool arguments and incident text leave the cluster when traces do."""
    events, exporter, _ = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [
                calls(("get_pod_logs", {"name": "api-7f9", "customer": "cus_9912"})),
                says("Done."),
            ],
            events,
            approve=None,
        )
        await loop.run(
            "customer ACME reports 5xx on payments-api", thread_id="incident-42"
        )

    asyncio.run(scenario())
    everything = str([s.attributes for s in exporter.get_finished_spans()])

    assert "cus_9912" not in everything
    assert "ACME" not in everything
    assert "api-7f9" not in everything
    # The shape is still there, which is what makes a malformed call debuggable.
    assert "customer,name" in everything


def test_payloads_can_be_opted_into_and_are_still_redacted(store, clock):
    events, exporter, _ = events_for_testing(record_payloads=True)

    async def scenario():
        loop = await build_loop(
            store,
            clock,
            [
                calls(("get_pod_logs", {"name": "api-7f9", "token": "ghp_secret000000000000"})),
                says("Done."),
            ],
            events,
            approve=None,
        )
        await loop.run("investigate payments-api", thread_id="incident-42")

    asyncio.run(scenario())
    everything = str([s.attributes for s in exporter.get_finished_spans()])

    assert "api-7f9" in everything          # opted in
    assert "ghp_secret000000000000" not in everything  # still never
    assert REDACTED in everything


# ── metrics ──────────────────────────────────────────────────────────────


def collect(reader) -> dict[str, list]:
    data = reader.get_metrics_data()
    points: dict[str, list] = {}
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                points[metric.name] = list(metric.data.data_points)
    return points


def test_run_and_token_metrics_are_recorded(store, clock):
    events, _, reader = events_for_testing()

    async def scenario():
        loop = await build_loop(
            store, clock, [calls(("get_pod_logs", {"name": "x"})), says("Done.")], events,
            approve=None,
        )
        await loop.run("investigate", thread_id="incident-42", service="payments-api")

    asyncio.run(scenario())
    points = collect(reader)

    runs = points["agent.runs"][0]
    assert runs.value == 1
    assert runs.attributes["outcome"] == "completed"
    assert runs.attributes["service"] == "payments-api"

    tokens = {p.attributes["kind"]: p.value for p in points["agent.tokens"]}
    assert tokens["input"] == 200
    assert tokens["output"] == 100

    assert points["agent.run.turns"][0].sum == 2
    assert points["agent.tool_calls"][0].attributes["outcome"] == "ok"


def test_timeouts_and_rejections_are_counted_separately(store, clock):
    """A rising rejection rate means the agent is proposing bad actions; a
    rising expiry rate means nobody is reading Slack. Different problems."""
    events, _, reader = events_for_testing()

    async def rejected():
        loop = await build_loop(
            store, clock, [calls(("delete_resource", {"name": "a"})), says("ok")], events,
            approve=False,
        )
        await loop.run("delete a", thread_id="incident-1")

    async def timed_out():
        loop = await build_loop(
            store, clock, [calls(("delete_resource", {"name": "b"})), says("ok")], events,
            approve=None,
        )
        await loop.run("delete b", thread_id="incident-2")

    asyncio.run(rejected())
    asyncio.run(timed_out())

    outcomes = {
        p.attributes["outcome"]: p.value for p in collect(reader)["agent.approvals"]
    }
    assert outcomes["rejected"] == 1
    assert outcomes["expired"] == 1


# ── logs ─────────────────────────────────────────────────────────────────


def test_log_records_carry_the_active_trace(store, clock):
    """Traces say an approval took nine minutes; logs say who it waited on.
    The join only works if records carry the trace id."""
    events, _, _ = events_for_testing()
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = Capture()
    handler.addFilter(TraceContextFilter())
    logger = logging.getLogger("devops-agent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    async def scenario():
        loop = await build_loop(store, clock, [says("Done.")], events, approve=None)
        await loop.run("investigate", thread_id="incident-42")

    try:
        logger.info("before the run")
        asyncio.run(scenario())
    finally:
        logger.removeHandler(handler)

    outside = [r for r in captured if r.getMessage() == "before the run"]
    inside = [r for r in captured if r.getMessage() != "before the run"]

    assert outside and outside[0].trace_id == "0"      # no span, no id, no crash
    assert inside, "the run logged nothing"
    assert any(r.trace_id != "0" for r in inside)
    assert all(hasattr(r, "span_id") for r in captured)


def test_json_formatter_emits_trace_ids():
    record = logging.LogRecord(
        "devops-agent", logging.INFO, __file__, 1, "tool call", (), None
    )
    TraceContextFilter().filter(record)
    line = JSONFormatter().format(record)

    assert '"message": "tool call"' in line
    assert '"trace_id"' in line


# ── setup ────────────────────────────────────────────────────────────────


def test_no_endpoint_means_no_telemetry_not_a_crash(monkeypatch):
    """An agent that will not start because a collector is unreachable is
    worse than one running blind."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    observability = configure_observability(configure_logs=False)

    assert observability.enabled is False
    assert observability.metrics is None
    observability.shutdown()  # must be safe


def test_null_events_are_a_working_no_op(store, clock):
    from agentic_devops.orchestration.loop import NullEvents

    async def scenario():
        loop = await build_loop(
            store, clock, [calls(("get_pod_logs", {"name": "x"})), says("Done.")],
            NullEvents(), approve=None,
        )
        return await loop.run("investigate", thread_id="incident-42")

    result = asyncio.run(scenario())
    assert result.completed
