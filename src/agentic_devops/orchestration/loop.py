"""The ReAct loop: reason, act, observe, repeat — until done or stopped."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence

from ..memory.knowledge import KnowledgeIndex, render_hits
from ..memory.transcripts import TranscriptStore, prepare_resume
from ..mcp.toolset import ToolResult
from .context import run_scope
from .findings import Finding, FindingsRecorder
from .registry import ToolInvocation, ToolRegistry
from .prompt import SYSTEM_PROMPT

log = logging.getLogger("devops-agent.loop")

MODEL = "claude-opus-5"

#: A run that has not concluded in this many turns is not about to. The article
#: had no bound; a reasoning loop on a HIGH-risk tool is a loop that asks a
#: human the same question forever.
DEFAULT_MAX_TURNS = 25


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        for attribute in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = getattr(usage, attribute, None)
            if value:
                setattr(self, attribute, getattr(self, attribute) + value)


@dataclass
class AgentRun:
    """Everything that happened, whether or not it went well."""

    thread_id: str
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    usage: UsageTotals = field(default_factory=UsageTotals)
    turns: int = 0
    service: str | None = None
    #: Which model actually served the run — not necessarily the one asked
    #: for, since a refusal can hand the turn to a fallback.
    model: str | None = None
    stop_reason: str | None = None
    #: Set when the run ended for a reason other than the agent finishing.
    halted: str | None = None
    text: str = ""

    @property
    def completed(self) -> bool:
        return self.halted is None

    @property
    def denied_writes(self) -> list[ToolInvocation]:
        return [i for i in self.invocations if i.result.denied]


class SpanHandle(Protocol):
    """What the loop can say about work that is still in progress."""

    def annotate(self, **attributes: Any) -> None: ...

    def record(self, invocation: ToolInvocation) -> None: ...


class AgentEvents(Protocol):
    """Layer 5's seam.

    Spans rather than point events: a span needs to *contain* the work to have
    a duration or a parent, which is the flaw in instrumenting an agent with
    ``on_turn(run, turn)`` callbacks.
    """

    def run_span(self, run: AgentRun) -> AbstractContextManager[SpanHandle]: ...

    def turn_span(self, run: AgentRun, turn: int) -> AbstractContextManager[SpanHandle]: ...

    def tool_span(
        self, run: AgentRun, name: str, args: Mapping[str, Any]
    ) -> AbstractContextManager[SpanHandle]: ...

    def on_response(self, run: AgentRun, response: Any) -> None: ...

    def on_finish(self, run: AgentRun) -> None: ...


class NullSpan:
    def annotate(self, **attributes: Any) -> None: ...

    def record(self, invocation: ToolInvocation) -> None: ...


@contextmanager
def _null_span() -> Iterator[NullSpan]:
    yield NullSpan()


class NullEvents:
    """The default: instrumentation costs nothing when nobody is collecting."""

    def run_span(self, run: AgentRun) -> AbstractContextManager[NullSpan]:
        return _null_span()

    def turn_span(self, run: AgentRun, turn: int) -> AbstractContextManager[NullSpan]:
        return _null_span()

    def tool_span(
        self, run: AgentRun, name: str, args: Mapping[str, Any]
    ) -> AbstractContextManager[NullSpan]:
        return _null_span()

    def on_response(self, run: AgentRun, response: Any) -> None: ...

    def on_finish(self, run: AgentRun) -> None: ...


@dataclass
class AgentLoop:
    """Drives one agent run to completion.

    Stateless between runs: conversation state lives in the returned
    :class:`AgentRun`, which layer 4 will persist and replay.
    """

    client: Any
    registry: ToolRegistry
    system_prompt: str = SYSTEM_PROMPT
    model: str = MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = 32_000
    effort: str = "high"
    #: Route a policy refusal to a fallback model instead of failing the run.
    #: Infrastructure work trips security classifiers more than most domains.
    refusal_fallbacks: bool = True
    recorder: FindingsRecorder | None = None
    events: AgentEvents = field(default_factory=NullEvents)
    #: Episodic memory. Checkpointed every turn, so a crash costs one turn.
    transcripts: TranscriptStore | None = None
    #: Semantic memory, consulted once before the first turn. The agent can
    #: also query it mid-run through `recall_knowledge`.
    knowledge: KnowledgeIndex | None = None
    recall_limit: int = 4
    #: Drop stale tool results server-side as the window fills. A DevOps agent
    #: accumulates log dumps faster than almost any other kind.
    context_editing: bool = True

    async def run(
        self,
        goal: str,
        *,
        thread_id: str,
        history: Sequence[dict[str, Any]] | None = None,
        service: str | None = None,
    ) -> AgentRun:
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": await self._opening(goal, service)})
        run = AgentRun(
            thread_id=thread_id, goal=goal, messages=messages, service=service
        )
        with run_scope(thread_id):
            return await self._drive(run, messages)

    async def resume(self, thread_id: str) -> AgentRun:
        """Pick a run back up from its last checkpoint."""
        if self.transcripts is None:
            raise RuntimeError("resume needs a TranscriptStore")
        checkpoint = self.transcripts.load(thread_id)
        if checkpoint is None:
            raise KeyError(f"no transcript for thread {thread_id!r}")

        messages, note = prepare_resume(checkpoint.messages)
        if not messages:
            raise ValueError(f"transcript for {thread_id!r} has nothing resumable")
        if note or messages[-1].get("role") == "assistant":
            messages.append(
                {
                    "role": "user",
                    "content": (note + "\n\n" if note else "")
                    + "Continue from where you left off.",
                }
            )
        run = AgentRun(
            thread_id=thread_id,
            goal=checkpoint.goal,
            messages=messages,
            service=checkpoint.service,
        )
        log.info("resuming %s from turn %d", thread_id, checkpoint.turns)
        with run_scope(thread_id):
            return await self._drive(run, messages)

    async def _opening(self, goal: str, service: str | None) -> str:
        """The first user message: what we already know, then the task.

        Recalled context goes in the messages, never the system prompt — the
        system block carries the cache breakpoint, and rewriting it per
        incident would throw away the cached prefix on every run.
        """
        if self.knowledge is None:
            return goal
        hits = await asyncio.to_thread(
            self.knowledge.search, goal, limit=self.recall_limit, service=service
        )
        if not hits:
            return goal
        log.info("recalled %d prior document(s) for %s", len(hits), service or "any service")
        return (
            "Before you begin — prior knowledge that may be relevant. It may "
            "also be irrelevant or wrong; check it against what you observe.\n\n"
            f"{render_hits(hits)}\n\n---\n\nYour task:\n\n{goal}"
        )

    async def _checkpoint(self, run: AgentRun, status: str) -> None:
        if self.transcripts is None:
            return
        try:
            await asyncio.to_thread(
                self.transcripts.save,
                thread_id=run.thread_id,
                goal=run.goal,
                messages=run.messages,
                turns=run.turns,
                status=status,
                service=run.service,
                summary=run.text or None,
            )
        except Exception:  # noqa: BLE001 - losing a checkpoint must not lose the run
            log.warning("could not checkpoint %s", run.thread_id, exc_info=True)

    async def _drive(self, run: AgentRun, messages: list[dict[str, Any]]) -> AgentRun:
        with self.events.run_span(run):
            return await self._turns(run, messages)

    async def _turns(self, run: AgentRun, messages: list[dict[str, Any]]) -> AgentRun:
        for turn in range(1, self.max_turns + 1):
            run.turns = turn
            with self.events.turn_span(run, turn) as span:
                finished = await self._turn(run, messages, span)
            if finished:
                break
        else:
            run.halted = f"reached the {self.max_turns}-turn limit without concluding"

        if self.recorder is not None:
            run.findings = self.recorder.for_thread(run.thread_id)
        await self._checkpoint(run, "halted" if run.halted else "completed")
        self.events.on_finish(run)
        log.info(
            "run finished thread=%s turns=%d tools=%d denied=%d halted=%s",
            run.thread_id,
            run.turns,
            len(run.invocations),
            len(run.denied_writes),
            run.halted,
        )
        return run

    async def _turn(
        self, run: AgentRun, messages: list[dict[str, Any]], span: Any
    ) -> bool:
        """One reason-act-observe cycle. Returns True when the run is over."""
        response = await self._respond(messages)
        run.usage.add(response.usage)
        self.events.on_response(run, response)
        run.stop_reason = response.stop_reason
        # Append the content verbatim — thinking blocks must go back
        # unedited, and compaction state (layer 4) rides along in here too.
        messages.append({"role": "assistant", "content": response.content})
        run.text = _text_of(response.content) or run.text
        await self._checkpoint(run, "running")

        span.annotate(**{"agent.stop_reason": response.stop_reason or "unknown"})

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            run.halted = (
                "the model declined this request"
                f" (category: {getattr(details, 'category', None)})"
            )
            return True

        if response.stop_reason == "max_tokens":
            run.halted = "response hit max_tokens; the turn was truncated"
            return True

        if response.stop_reason == "pause_turn":
            # A server tool paused mid-turn; resending continues it.
            return False

        if response.stop_reason == "tool_use":
            results = await self._run_tools(run, response.content)
            messages.append({"role": "user", "content": results})
            return False

        return True

    async def _respond(self, messages: Sequence[dict[str, Any]]) -> Any:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Tools render before system in the cache prefix, and the catalog
            # is sorted, so this breakpoint covers a stable prefix across runs.
            "system": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": list(messages),
            "tools": self.registry.definitions(),
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
        }

        betas: list[str] = []
        extra: dict[str, Any] = {}
        if self.refusal_fallbacks:
            betas.append("server-side-fallback-2026-07-01")
            extra["fallbacks"] = "default"
        if self.context_editing:
            betas.append("context-management-2025-06-27")
            params["context_management"] = {
                "edits": [{"type": "clear_tool_uses_20250919"}]
            }

        if betas:
            stream = self.client.beta.messages.stream(**params, betas=betas, **extra)
        else:
            stream = self.client.messages.stream(**params)

        async with stream as active:
            return await active.get_final_message()

    async def _run_tools(
        self, run: AgentRun, content: Sequence[Any]
    ) -> list[dict[str, Any]]:
        """Execute every tool call in one assistant turn.

        Reads go concurrently; writes go one at a time. Firing three approval
        requests into Slack simultaneously and executing whichever comes back
        first is not something a reviewer can reason about — and each write
        wants to see the state the previous one left behind.
        """
        calls = [block for block in content if getattr(block, "type", None) == "tool_use"]
        results: list[dict[str, Any] | None] = [None] * len(calls)

        reads = [i for i, c in enumerate(calls) if not self.registry.requires_approval(c.name)]
        writes = [i for i, c in enumerate(calls) if i not in set(reads)]

        if reads:
            gathered = await asyncio.gather(*(self._call(run, calls[i]) for i in reads))
            for index, block in zip(reads, gathered):
                results[index] = block

        for index in writes:
            results[index] = await self._call(run, calls[index])

        return [block for block in results if block is not None]

    async def _call(self, run: AgentRun, call: Any) -> dict[str, Any]:
        """Execute one tool call inside its own span.

        ``gather`` wraps each coroutine in a task, and a task copies the
        current context — so parallel reads become sibling spans under this
        turn rather than a flat list or, worse, children of each other.
        """
        with self.events.tool_span(run, call.name, call.input) as span:
            invocation = await self.registry.call(
                call.name, call.input, thread_id=run.thread_id
            )
            span.record(invocation)
        run.invocations.append(invocation)
        return tool_result_block(call.id, invocation.result)


def tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result.blocks,
    }
    if result.is_error:
        block["is_error"] = True
    return block


def _text_of(content: Sequence[Any]) -> str:
    return "\n".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    ).strip()
