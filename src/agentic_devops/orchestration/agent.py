"""Wiring: MCP servers + approval gate + loop, assembled into one agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from ..approval.gate import DEFAULT_TIMEOUT, ApprovalGate
from ..approval.notifier import ConsoleNotifier, Notifier, SlackNotifier
from ..approval.policy import ApprovalPolicy
from ..approval.store import ApprovalStore, SQLiteApprovalStore
from ..mcp.servers import MCPServerSpec, MCPToolLayer
from ..memory.knowledge import KnowledgeIndex
from ..memory.learn import record_run
from ..memory.recall import KnowledgeRecall
from ..memory.transcripts import TranscriptStore
from ..mcp.toolset import ToolCatalog, ToolSource
from .findings import FindingsRecorder
from .loop import DEFAULT_MAX_TURNS, MODEL, AgentEvents, AgentLoop, AgentRun, NullEvents
from .prompt import SYSTEM_PROMPT
from .registry import ToolRegistry

log = logging.getLogger("devops-agent")


@dataclass
class AgentConfig:
    servers: Sequence[MCPServerSpec] = ()
    #: Extra in-process tool sources (the findings recorder is added for you).
    local_sources: Sequence[ToolSource] = ()
    approvals_db: str = "approvals.db"
    slack_token: str | None = None
    slack_channel: str | None = None
    slack_mention: str | None = None
    approval_timeout: timedelta = DEFAULT_TIMEOUT
    #: How often a waiting run re-reads the approval store.
    approval_poll_interval: float = 2.0
    policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    agent_id: str = "devops-agent-1"
    model: str = MODEL
    effort: str = "high"
    max_turns: int = DEFAULT_MAX_TURNS
    system_prompt: str = SYSTEM_PROMPT
    refusal_fallbacks: bool = True
    context_editing: bool = True
    #: Semantic memory. None disables recall entirely.
    knowledge: KnowledgeIndex | None = None
    recall_limit: int = 4
    #: Episodic memory. None means a crashed run cannot be resumed.
    transcripts: TranscriptStore | None = None
    #: File each concluded run back into the knowledge index, unverified.
    learn_from_runs: bool = True

    @classmethod
    def from_env(cls, servers: Sequence[MCPServerSpec], **overrides: Any) -> "AgentConfig":
        return cls(
            servers=servers,
            approvals_db=os.environ.get("APPROVALS_DB", "approvals.db"),
            slack_token=os.environ.get("SLACK_BOT_TOKEN"),
            slack_channel=os.environ.get("SLACK_APPROVAL_CHANNEL"),
            slack_mention=os.environ.get("SLACK_APPROVAL_MENTION"),
            agent_id=os.environ.get("AGENT_ID", "devops-agent-1"),
            **overrides,
        )


def load_servers(path: str | Path) -> list[MCPServerSpec]:
    """Read MCP server definitions from JSON.

    ``{"servers": [{"name": "kubernetes", "command": "kubernetes-mcp",
    "args": ["--readonly"], "prefix": "k8s"}]}``
    """
    data = json.loads(Path(path).read_text())
    return [
        MCPServerSpec(
            name=entry["name"],
            command=entry.get("command"),
            args=tuple(entry.get("args", ())),
            env=entry.get("env"),
            cwd=entry.get("cwd"),
            url=entry.get("url"),
            prefix=entry.get("prefix"),
            read_timeout=entry.get("read_timeout", 60.0),
        )
        for entry in data["servers"]
    ]


def build_notifier(config: AgentConfig) -> Notifier:
    if config.slack_token and config.slack_channel:
        return SlackNotifier(
            token=config.slack_token,
            channel=config.slack_channel,
            mention=config.slack_mention,
        )
    log.warning(
        "no Slack credentials configured; approvals will be requested on the "
        "console and must be resolved with the `approvals` CLI"
    )
    return ConsoleNotifier()


@dataclass
class DevOpsAgent:
    """One configured agent. Reusable across runs; not safe to share across
    event loops."""

    loop: AgentLoop
    registry: ToolRegistry
    gate: ApprovalGate
    recorder: FindingsRecorder
    knowledge: KnowledgeIndex | None = None
    transcripts: TranscriptStore | None = None
    learn_from_runs: bool = True

    async def run(
        self, goal: str, *, thread_id: str, service: str | None = None
    ) -> AgentRun:
        log.info("starting run thread=%s", thread_id)
        run = await self.loop.run(goal, thread_id=thread_id, service=service)
        await self._learn(run)
        return run

    async def resume(self, thread_id: str) -> AgentRun:
        """Continue a run that a restart interrupted."""
        run = await self.loop.resume(thread_id)
        await self._learn(run)
        return run

    def resumable(self, limit: int = 20) -> list[str]:
        if self.transcripts is None:
            return []
        return [c.thread_id for c in self.transcripts.resumable(limit=limit)]

    async def _learn(self, run: AgentRun) -> None:
        if not (self.learn_from_runs and self.knowledge is not None):
            return
        try:
            await asyncio.to_thread(
                record_run, self.knowledge, run, service=run.service
            )
        except Exception:  # noqa: BLE001 - never fail a finished run over memory
            log.warning("could not record run %s", run.thread_id, exc_info=True)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self.registry.catalog.tools)


@asynccontextmanager
async def build_agent(
    config: AgentConfig,
    *,
    client: Any | None = None,
    store: ApprovalStore | None = None,
    notifier: Notifier | None = None,
    events: AgentEvents | None = None,
) -> AsyncIterator[DevOpsAgent]:
    """Connect everything, yield an agent, and tear the servers down after.

    The MCP servers are subprocesses; leaving this context kills them, so a
    failed run does not leave a `kubernetes-mcp` holding a cluster connection.
    """
    if client is None:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()

    gate = ApprovalGate(
        store=store or SQLiteApprovalStore(config.approvals_db),
        notifier=notifier or build_notifier(config),
        policy=config.policy,
        timeout=config.approval_timeout,
        poll_interval=config.approval_poll_interval,
        agent_id=config.agent_id,
    )
    recorder = FindingsRecorder()
    memory_sources: list[ToolSource] = []
    if config.knowledge is not None:
        memory_sources.append(
            KnowledgeRecall(index=config.knowledge, limit=config.recall_limit)
        )

    async with MCPToolLayer(config.servers) as layer:
        catalog = layer.catalog
        for source in (recorder, *memory_sources, *config.local_sources):
            for spec in await source.list_tools():
                catalog.add(source, spec)

        registry = ToolRegistry(catalog=catalog, gate=gate)
        loop = AgentLoop(
            client=client,
            registry=registry,
            system_prompt=config.system_prompt,
            model=config.model,
            max_turns=config.max_turns,
            effort=config.effort,
            refusal_fallbacks=config.refusal_fallbacks,
            context_editing=config.context_editing,
            recorder=recorder,
            events=events or NullEvents(),
            transcripts=config.transcripts,
            knowledge=config.knowledge,
            recall_limit=config.recall_limit,
        )
        yield DevOpsAgent(
            loop=loop,
            registry=registry,
            gate=gate,
            recorder=recorder,
            knowledge=config.knowledge,
            transcripts=config.transcripts,
            learn_from_runs=config.learn_from_runs,
        )


async def build_catalog_from_sources(sources: Sequence[ToolSource]) -> ToolCatalog:
    """Build a catalog from already-connected sources. For tests, and for
    callers that manage their own MCP connections."""
    catalog = ToolCatalog()
    for source in sources:
        for spec in await source.list_tools():
            catalog.add(source, spec)
    return catalog
