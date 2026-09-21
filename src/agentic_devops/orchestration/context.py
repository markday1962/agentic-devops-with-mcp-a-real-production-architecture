"""Which run is currently executing, for code too deep to be passed a run.

One agent serves many concurrent incidents — layer 3 hands the same
:class:`DevOpsAgent` to every webhook. Anything that accumulates per-run state
(findings today, transcripts in layer 4, spans in layer 5) has to know which
run it is accumulating for, and a tool executor is several frames below the
loop that knows.

A ContextVar is the right shape here: ``asyncio`` copies the current context
into each task, so a tool call fanned out with ``gather`` sees the run that
spawned it, and two runs in the same event loop never see each other's.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

#: The thread_id of the run on this task, or None outside a run.
current_thread_id: ContextVar[str | None] = ContextVar("current_thread_id", default=None)

UNSCOPED = "unscoped"


@contextmanager
def run_scope(thread_id: str) -> Iterator[None]:
    token = current_thread_id.set(thread_id)
    try:
        yield
    finally:
        current_thread_id.reset(token)


def active_thread_id() -> str:
    return current_thread_id.get() or UNSCOPED
