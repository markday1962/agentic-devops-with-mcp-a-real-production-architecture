"""An in-process tool source for tests and local dry runs.

Real MCP servers need a cluster, a GitHub token, and an AWS account. This gives
the orchestration layer something to call that behaves like a server —
including failing, returning oversized output, and lying about its own
annotations — without any of that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .toolset import ToolResult, ToolSpec, truncate_blocks

#: A fake tool body: takes the call arguments, returns text or a ToolResult.
FakeHandler = Callable[[Mapping[str, Any]], Any]


@dataclass
class FakeTool:
    name: str
    handler: FakeHandler
    description: str = "fake tool"
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None


@dataclass
class FakeToolSource:
    """A ``ToolSource`` backed by Python callables."""

    name: str = "fake-mcp"
    tools: list[FakeTool] = field(default_factory=list)
    #: Every (tool_name, args) pair this source was asked to run.
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def add(
        self,
        name: str,
        handler: FakeHandler,
        *,
        read_only: bool | None = None,
        destructive: bool | None = None,
        description: str = "fake tool",
        input_schema: dict[str, Any] | None = None,
    ) -> "FakeToolSource":
        self.tools.append(
            FakeTool(
                name=name,
                handler=handler,
                description=description,
                input_schema=input_schema or {"type": "object", "properties": {}},
                read_only_hint=read_only,
                destructive_hint=destructive,
            )
        )
        return self

    async def list_tools(self) -> Sequence[ToolSpec]:
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                server=self.name,
                read_only_hint=tool.read_only_hint,
                destructive_hint=tool.destructive_hint,
            )
            for tool in self.tools
        ]

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, dict(args)))
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            return ToolResult.text(f"no such tool: {name}", is_error=True)
        try:
            outcome = tool.handler(args)
        except Exception as exc:  # noqa: BLE001 - mirrors MCPConnection
            return ToolResult.text(f"{type(exc).__name__}: {exc}", is_error=True)
        if isinstance(outcome, ToolResult):
            return outcome
        return ToolResult(blocks=truncate_blocks([{"type": "text", "text": str(outcome)}]))
