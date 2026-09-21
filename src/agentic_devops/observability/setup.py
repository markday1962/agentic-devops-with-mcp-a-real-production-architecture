"""Turning observability on, and degrading cleanly when it is off."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ..orchestration.loop import AgentEvents, NullEvents
from .logs import configure_logging
from .metrics import AgentMetrics
from .redaction import Redactor
from .tracing import OTelEvents

log = logging.getLogger("devops-agent.observability")

DEFAULT_SERVICE_NAME = "devops-agent"


@dataclass
class Observability:
    """Handle for whatever got wired up. ``events`` is always usable."""

    events: AgentEvents
    metrics: AgentMetrics | None = None
    enabled: bool = False
    _shutdown: list[Any] = None  # type: ignore[assignment]

    def shutdown(self) -> None:
        """Flush pending spans and metrics. Call before the process exits —
        the batch processor drops whatever it is still holding otherwise."""
        for provider in self._shutdown or ():
            try:
                provider.shutdown()
            except Exception:  # noqa: BLE001
                log.warning("error shutting down telemetry provider", exc_info=True)


def configure_observability(
    *,
    service_name: str | None = None,
    endpoint: str | None = None,
    record_payloads: bool | None = None,
    console: bool = False,
    configure_logs: bool = True,
    log_level: str | None = None,
    redactor: Redactor | None = None,
) -> Observability:
    """Wire up tracing and metrics, or return a no-op if that is not possible.

    Observability is an optional extra, and an agent that will not start
    because a collector is unreachable is worse than one running blind. A
    missing SDK or a bad endpoint degrades to :class:`NullEvents` with a
    warning.
    """
    if configure_logs:
        configure_logging(
            log_level or os.environ.get("LOG_LEVEL", "INFO"),
            json_output=os.environ.get("LOG_FORMAT", "json") == "json",
        )

    if record_payloads is None:
        record_payloads = os.environ.get("OTEL_RECORD_PAYLOADS", "").lower() in {
            "1",
            "true",
            "yes",
        }
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint and not console:
        log.info("no OTLP endpoint configured; running without telemetry")
        return Observability(events=NullEvents(), enabled=False, _shutdown=[])

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "opentelemetry is not installed (pip install '.[otel]'); "
            "running without telemetry"
        )
        return Observability(events=NullEvents(), enabled=False, _shutdown=[])

    resource = Resource.create(
        {
            "service.name": service_name
            or os.environ.get("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME),
            "service.version": os.environ.get("AGENT_VERSION", "dev"),
            "deployment.environment": os.environ.get("DEPLOY_ENV", "unknown"),
        }
    )

    span_exporters: list[Any] = []
    metric_readers: list[Any] = []
    try:
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            span_exporters.append(OTLPSpanExporter(endpoint=endpoint))
            metric_readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
            )
        if console:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            span_exporters.append(ConsoleSpanExporter())
            metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    except Exception:  # noqa: BLE001
        log.warning("could not build OTLP exporters; running without telemetry", exc_info=True)
        return Observability(events=NullEvents(), enabled=False, _shutdown=[])

    tracer_provider = TracerProvider(resource=resource)
    for exporter in span_exporters:
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    otel_metrics.set_meter_provider(meter_provider)

    metrics = AgentMetrics(meter=meter_provider.get_meter("devops-agent"))
    events = OTelEvents(
        tracer=tracer_provider.get_tracer("devops-agent"),
        metrics=metrics,
        record_payloads=record_payloads,
        redactor=redactor or Redactor(),
    )
    log.info(
        "telemetry enabled endpoint=%s record_payloads=%s", endpoint, record_payloads
    )
    return Observability(
        events=events,
        metrics=metrics,
        enabled=True,
        _shutdown=[tracer_provider, meter_provider],
    )


def events_for_testing(
    *, record_payloads: bool = False
) -> tuple[OTelEvents, Any, Any]:
    """An in-memory tracer and meter, for asserting on what got recorded."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])

    events = OTelEvents(
        tracer=tracer_provider.get_tracer("test"),
        metrics=AgentMetrics(meter=meter_provider.get_meter("test")),
        record_payloads=record_payloads,
    )
    return events, exporter, reader
