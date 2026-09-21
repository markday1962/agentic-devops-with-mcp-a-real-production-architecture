"""Logs that can be joined to traces.

A trace tells you an approval took nine minutes; the log line tells you which
reviewer it was waiting on. That join only works if every log record carries
the trace and span id of the work that emitted it, which is what the filter
below does.
"""

from __future__ import annotations

import json
import logging
from typing import Any

RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class TraceContextFilter(logging.Filter):
    """Stamps the active trace and span id onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = span_id = "0"
        try:
            from opentelemetry import trace

            context = trace.get_current_span().get_span_context()
            if context.is_valid:
                trace_id = format(context.trace_id, "032x")
                span_id = format(context.span_id, "016x")
        except Exception:  # noqa: BLE001 - logging must never raise
            pass
        record.trace_id = trace_id
        record.span_id = span_id
        return True


class JSONFormatter(logging.Formatter):
    """One JSON object per line, for a log pipeline rather than a human."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "0"),
            "span_id": getattr(record, "span_id", "0"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in RESERVED and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(TraceContextFilter())
    handler.setFormatter(
        JSONFormatter()
        if json_output
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
