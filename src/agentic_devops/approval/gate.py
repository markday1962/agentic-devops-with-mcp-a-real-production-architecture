"""The approval gate: intercept a write, ask a human, act only on a live yes."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Coroutine, Mapping, TypeVar

from .errors import ApprovalConflict, ApprovalDenied, ArgumentMismatch
from .models import ApprovalRequest, ApprovalStatus, fingerprint_args, utcnow
from .notifier import Notifier, NullNotifier
from .policy import ApprovalPolicy
from .store import ApprovalStore

log = logging.getLogger("devops-agent.approval")

T = TypeVar("T")

DEFAULT_TIMEOUT = timedelta(minutes=10)


def _forbidden_message(tool_name: str) -> str:
    return (
        f"Action `{tool_name}` is blocked by policy and cannot be approved by "
        "anyone. Do not retry; escalate to a human operator."
    )


def denial_message(request: ApprovalRequest) -> str:
    """What the agent is told when it does not get its approval.

    Phrased for a model to act on: say what happened and what not to do next,
    so it reports back to the human instead of retrying the same call.
    """
    if request.status is ApprovalStatus.EXPIRED:
        return (
            f"Action `{request.tool_name}` was NOT executed: approval request "
            f"{request.id} timed out with no human response, which counts as a "
            "refusal. Do not retry. Report that the action is still pending "
            "human sign-off."
        )
    reason = f" ({request.decision_note})" if request.decision_note else ""
    by = f" by {request.decided_by}" if request.decided_by else ""
    return (
        f"Action `{request.tool_name}` was NOT executed: approval request "
        f"{request.id} was rejected{by}{reason}. Do not retry this action. "
        "Consider an alternative, or explain to the human why you believe it is needed."
    )


@dataclass(frozen=True, slots=True)
class Authorization:
    """The gate's verdict on one proposed call.

    ``granted`` means the approval was given *and* spent — the caller may run
    the action exactly once. ``message`` is agent-readable prose explaining a
    refusal, suitable for handing straight back as a tool result.
    """

    granted: bool
    request: ApprovalRequest | None = None
    message: str | None = None


@dataclass(slots=True)
class ApprovalGate:
    """Wraps write tools so each call needs a fresh, live human approval.

    ``timeout`` is a deadline stored with the request, not a sleep budget: the
    request expires at that wall-clock moment whether or not this process is
    still waiting on it. A restart loses the waiter, not the decision.
    """

    store: ApprovalStore
    notifier: Notifier = NullNotifier()
    policy: ApprovalPolicy = ApprovalPolicy()
    timeout: timedelta = DEFAULT_TIMEOUT
    poll_interval: float = 2.0
    agent_id: str | None = None
    #: Raise ApprovalDenied instead of returning an explanatory string. Agents
    #: usually want the string; library callers usually want the exception.
    raise_on_denied: bool = False
    clock: Callable[[], datetime] = utcnow
    sleeper: Callable[[float], None] = time.sleep
    #: The async path needs its own sleeper. Reading an injected clock while
    #: sleeping on the real one makes the two disagree, and a waiter whose
    #: deadline never arrives spins until the process is killed.
    async_sleeper: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep

    # ── request lifecycle ────────────────────────────────────────────────

    def request(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        summary: str | None = None,
        thread_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> ApprovalRequest:
        """Create a pending request and put it in front of a human.

        If the notifier fails the request is rejected immediately: an approval
        nobody can see would otherwise sit there until it times out, turning a
        broken Slack integration into a silent ten-minute stall per action.
        """
        request = ApprovalRequest.new(
            tool_name=tool_name,
            args=args,
            risk=self.policy.assess(tool_name, args),
            timeout=timeout or self.timeout,
            now=self.clock(),
            summary=summary,
            thread_id=thread_id,
            requested_by=self.agent_id,
        )
        self.store.create(request)
        log.info(
            "approval requested id=%s tool=%s risk=%s thread=%s",
            request.id,
            request.tool_name,
            request.risk.value,
            request.thread_id,
        )

        try:
            ref = self.notifier.notify(request)
        except Exception:  # noqa: BLE001 - unreachable reviewers must fail closed
            log.exception("notifier failed for approval %s; denying", request.id)
            return self._fail_closed(request, "reviewers could not be notified")

        if ref is not None:
            request = request.with_notification(ref)
            attach = getattr(self.store, "attach_notification", None)
            if attach is not None:
                attach(request.id, ref)
        return request

    def _fail_closed(self, request: ApprovalRequest, note: str) -> ApprovalRequest:
        try:
            return self.store.resolve(
                request.id,
                approved=False,
                decided_by="system",
                note=note,
                now=self.clock(),
            )
        except ApprovalConflict as exc:
            return exc.request

    def wait(self, request_id: str, *, deadline: datetime | None = None) -> ApprovalRequest:
        """Block until the request leaves pending, or its deadline passes."""
        while True:
            current = self.store.get(request_id, now=self.clock())
            if current.status.is_terminal:
                self._announce(current)
                return current
            limit = deadline or current.expires_at
            remaining = (limit - self.clock()).total_seconds()
            if remaining <= 0:
                # Next get() applies expiry; loop once more to read the real state.
                continue
            self.sleeper(min(self.poll_interval, remaining))

    async def await_decision(
        self, request_id: str, *, deadline: datetime | None = None
    ) -> ApprovalRequest:
        """Async twin of :meth:`wait`, for the event-driven trigger layer."""
        while True:
            current = self.store.get(request_id, now=self.clock())
            if current.status.is_terminal:
                self._announce(current)
                return current
            limit = deadline or current.expires_at
            remaining = (limit - self.clock()).total_seconds()
            if remaining <= 0:
                continue
            await self.async_sleeper(min(self.poll_interval, remaining))

    def _announce(self, request: ApprovalRequest) -> None:
        """Tell the notifier how it ended — unless a human already told it."""
        if request.decided_by in (None, "system"):
            try:
                self.notifier.update(request)
            except Exception:  # noqa: BLE001
                log.warning("notifier update failed for %s", request.id, exc_info=True)

    def require(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        summary: str | None = None,
        thread_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> ApprovalRequest:
        """Ask, then wait. Returns the request in its final state."""
        request = self.request(
            tool_name, args, summary=summary, thread_id=thread_id, timeout=timeout
        )
        if request.status.is_terminal:
            return request
        return self.wait(request.id)

    async def arequire(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        summary: str | None = None,
        thread_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> ApprovalRequest:
        request = self.request(
            tool_name, args, summary=summary, thread_id=thread_id, timeout=timeout
        )
        if request.status.is_terminal:
            return request
        return await self.await_decision(request.id)

    # ── execution ────────────────────────────────────────────────────────

    def execute_approved(
        self,
        request: ApprovalRequest,
        args: Mapping[str, Any],
        fn: Callable[..., T],
    ) -> T:
        """Spend the approval, then run the action.

        The approval is consumed *before* execution, so a crash mid-write
        cannot leave a reusable approval behind, and the same grant can never
        authorize a second call.
        """
        spent = self.store.consume(
            request.id, args_fingerprint=fingerprint_args(args), now=self.clock()
        )
        log.info(
            "executing approved action id=%s tool=%s approved_by=%s",
            spent.id,
            spent.tool_name,
            spent.decided_by,
        )
        return fn(**args)

    def authorize(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        summary: str | None = None,
        thread_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> Authorization:
        """Ask, wait, and spend the approval — the whole gate in one call.

        Returns before the action runs, having already consumed the approval,
        so the caller is free to execute however it likes (a local function, an
        MCP round trip) without the gate needing to know.
        """
        if self.policy.is_forbidden(tool_name):
            log.warning("blocked forbidden tool %s", tool_name)
            return Authorization(granted=False, message=_forbidden_message(tool_name))

        request = self.require(
            tool_name, args, summary=summary, thread_id=thread_id, timeout=timeout
        )
        if not request.status.authorizes_execution:
            return Authorization(
                granted=False, request=request, message=denial_message(request)
            )
        return self._spend(request, args)

    async def aauthorize(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        summary: str | None = None,
        thread_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> Authorization:
        """Async twin of :meth:`authorize`."""
        if self.policy.is_forbidden(tool_name):
            log.warning("blocked forbidden tool %s", tool_name)
            return Authorization(granted=False, message=_forbidden_message(tool_name))

        request = await self.arequire(
            tool_name, args, summary=summary, thread_id=thread_id, timeout=timeout
        )
        if not request.status.authorizes_execution:
            return Authorization(
                granted=False, request=request, message=denial_message(request)
            )
        return self._spend(request, args)

    def _spend(self, request: ApprovalRequest, args: Mapping[str, Any]) -> Authorization:
        try:
            spent = self.store.consume(
                request.id, args_fingerprint=fingerprint_args(args), now=self.clock()
            )
        except (ApprovalConflict, ArgumentMismatch) as exc:
            log.warning("refusing to execute %s: %s", request.tool_name, exc)
            return Authorization(
                granted=False,
                request=exc.request,
                message=(
                    f"Action `{request.tool_name}` was NOT executed: {exc}. "
                    "Request a fresh approval before trying again."
                ),
            )
        log.info(
            "authorized id=%s tool=%s approved_by=%s",
            spent.id,
            spent.tool_name,
            spent.decided_by,
        )
        return Authorization(granted=True, request=spent)

    def guard(
        self,
        tool_name: str,
        fn: Callable[..., T],
        *,
        thread_id: str | None = None,
        summary_builder: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> Callable[..., T | str]:
        """Wrap a callable so it cannot run without a live approval.

        Read-only tools are returned untouched. Keyword arguments only: the
        human approves a named set of arguments, and positional args cannot be
        fingerprinted against what they saw.
        """
        if self.policy.is_forbidden(tool_name):
            @functools.wraps(fn)
            def forbidden(**kwargs: Any) -> str:
                log.warning("blocked forbidden tool %s", tool_name)
                return _forbidden_message(tool_name)

            return forbidden

        if not self.policy.requires_approval(tool_name):
            return fn

        @functools.wraps(fn)
        def guarded(**kwargs: Any) -> T | str:
            summary = summary_builder(kwargs) if summary_builder else None
            verdict = self.authorize(
                tool_name, kwargs, summary=summary, thread_id=thread_id
            )
            if not verdict.granted:
                if self.raise_on_denied:
                    raise ApprovalDenied(verdict.message or "denied", verdict.request)
                return verdict.message or "denied"
            return fn(**kwargs)

        return guarded

    def aguard(
        self,
        tool_name: str,
        fn: Callable[..., Awaitable[T]],
        *,
        thread_id: str | None = None,
        summary_builder: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> Callable[..., Awaitable[T | str]]:
        """Async twin of :meth:`guard`."""
        if self.policy.is_forbidden(tool_name):
            @functools.wraps(fn)
            async def forbidden(**kwargs: Any) -> str:
                log.warning("blocked forbidden tool %s", tool_name)
                return _forbidden_message(tool_name)

            return forbidden

        if not self.policy.requires_approval(tool_name):
            return fn

        @functools.wraps(fn)
        async def guarded(**kwargs: Any) -> T | str:
            summary = summary_builder(kwargs) if summary_builder else None
            verdict = await self.aauthorize(
                tool_name, kwargs, summary=summary, thread_id=thread_id
            )
            if not verdict.granted:
                if self.raise_on_denied:
                    raise ApprovalDenied(verdict.message or "denied", verdict.request)
                return verdict.message or "denied"
            return await fn(**kwargs)

        return guarded
