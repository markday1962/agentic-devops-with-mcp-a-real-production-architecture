"""Layer 5: observability for the agent itself."""

from .logs import JSONFormatter, TraceContextFilter, configure_logging
from .metrics import AgentMetrics
from .redaction import REDACTED, Redactor, argument_shape
from .setup import Observability, configure_observability, events_for_testing
from .tracing import OTelEvents, TracedSpan

__all__ = [
    "AgentMetrics",
    "JSONFormatter",
    "OTelEvents",
    "Observability",
    "REDACTED",
    "Redactor",
    "TraceContextFilter",
    "TracedSpan",
    "argument_shape",
    "configure_logging",
    "configure_observability",
    "events_for_testing",
]
