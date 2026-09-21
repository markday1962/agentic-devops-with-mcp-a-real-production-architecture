"""Client-side MCP connections.

The agent runs the MCP servers itself and executes every tool call locally.
That is not an incidental choice: the approval gate can only intercept a write
that passes through this process. Handing the server list to Anthropic's hosted
MCP connector would move execution server-side, where nothing can stop it.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .toolset import (
    ToolCatalog,
    ToolResult,
    ToolSpec,
    blocks_from_mcp_content,
    sanitize_tool_name,
    truncate_blocks,
)

log = logging.getLogger("devops-agent.mcp")

DEFAULT_READ_TIMEOUT = 60.0


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """How to reach one MCP server.

    Either ``command`` (stdio: the server is a subprocess) or ``url``
    (streamable HTTP), never both.
    """

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    #: Prepended to every tool name from this server. Use when two servers
    #: expose the same tool name.
    prefix: str | None = None
    read_timeout: float = DEFAULT_READ_TIMEOUT

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError(
                f"server {self.name!r}: set exactly one of command (stdio) or url (http)"
            )


class MCPConnection:
    """A live session with one MCP server, exposed as a ``ToolSource``."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self._session: Any | None = None
        #: model-facing name -> the name the server actually knows
        self._remote_names: dict[str, str] = {}

    async def connect(self, stack: AsyncExitStack) -> "MCPConnection":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if self.spec.command:
            params = StdioServerParameters(
                command=self.spec.command,
                args=list(self.spec.args),
                env=dict(self.spec.env) if self.spec.env else None,
                cwd=self.spec.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.streamable_http import streamable_http_client

            streams = await stack.enter_async_context(streamable_http_client(self.spec.url))
            read, write = streams[0], streams[1]

        session = await stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=self.spec.read_timeout)
        )
        await session.initialize()
        self._session = session
        log.info("connected to MCP server %s", self.spec.name)
        return self

    async def list_tools(self) -> Sequence[ToolSpec]:
        if self._session is None:
            raise RuntimeError(f"MCP server {self.spec.name!r} is not connected")

        listed = await self._session.list_tools()
        specs: list[ToolSpec] = []
        for tool in listed.tools:
            exposed = sanitize_tool_name(tool.name, prefix=self.spec.prefix)
            self._remote_names[exposed] = tool.name
            annotations = getattr(tool, "annotations", None)
            specs.append(
                ToolSpec(
                    name=exposed,
                    description=tool.description or f"{tool.name} (no description provided)",
                    input_schema=tool.input_schema or {"type": "object", "properties": {}},
                    server=self.spec.name,
                    read_only_hint=getattr(annotations, "read_only_hint", None),
                    destructive_hint=getattr(annotations, "destructive_hint", None),
                )
            )
        return specs

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        if self._session is None:
            raise RuntimeError(f"MCP server {self.spec.name!r} is not connected")

        remote = self._remote_names.get(name, name)
        try:
            result = await self._session.call_tool(remote, dict(args))
        except Exception as exc:  # noqa: BLE001 - a dead server is a tool error,
            # not an agent crash; the model can read this and try something else.
            log.warning("MCP call %s on %s failed", remote, self.spec.name, exc_info=True)
            return ToolResult.text(
                f"{type(exc).__name__} calling {name} on server "
                f"{self.spec.name}: {exc}",
                is_error=True,
            )

        content = getattr(result, "content", None) or []
        return ToolResult(
            blocks=truncate_blocks(blocks_from_mcp_content(content)),
            is_error=bool(getattr(result, "is_error", False)),
        )


@dataclass
class MCPToolLayer:
    """Layer 1's connection to the world: every configured server, connected,
    with their tools merged into one catalog.

    Used as an async context manager so a crashed agent run tears its server
    subprocesses down with it.
    """

    specs: Sequence[MCPServerSpec]
    catalog: ToolCatalog = field(default_factory=ToolCatalog)
    _stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPToolLayer":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for spec in self.specs:
                connection = await MCPConnection(spec).connect(self._stack)
                for tool in await connection.list_tools():
                    self.catalog.add(connection, tool)
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise
        log.info(
            "tool layer ready: %d tools from %d servers",
            len(self.catalog),
            len(self.specs),
        )
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
