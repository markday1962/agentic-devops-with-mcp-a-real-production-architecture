"""The tool surface the agent sees, assembled from MCP servers.

A ``ToolSource`` is anything that can list tools and call them: a real MCP
connection, or an in-process fake. The agent loop only ever talks to this
module, so swapping transports does not reach the orchestration layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

log = logging.getLogger("devops-agent.mcp")

#: Anthropic caps tool names at 128 chars from [a-zA-Z0-9_-].
MAX_TOOL_NAME = 128

#: Tool results go straight into the context window. An unbounded `get_logs`
#: can bury the conversation, so results are clipped with a visible marker.
DEFAULT_MAX_RESULT_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A normalized tool outcome, independent of transport."""

    blocks: list[dict[str, Any]]
    is_error: bool = False
    #: Set when the approval gate refused; not a failure, but not an execution.
    denied: bool = False

    @classmethod
    def text(cls, body: str, *, is_error: bool = False, denied: bool = False) -> "ToolResult":
        return cls(blocks=[{"type": "text", "text": body}], is_error=is_error, denied=denied)

    def rendered(self) -> str:
        return "\n".join(b.get("text", "") for b in self.blocks if b.get("type") == "text")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One callable tool, as advertised by a server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server: str
    #: From MCP annotations. ``None`` means the server said nothing.
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None

    def definition(self) -> dict[str, Any]:
        """The Anthropic tool definition for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@runtime_checkable
class ToolSource(Protocol):
    """What the registry needs from a server, real or fake."""

    name: str

    async def list_tools(self) -> Sequence[ToolSpec]: ...

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> ToolResult: ...


class DuplicateToolName(Exception):
    """Two servers advertise the same tool name.

    Not resolved silently: which `delete_resource` ran is exactly the kind of
    ambiguity an audit log must never contain. Give one server a `prefix`.
    """


def sanitize_tool_name(name: str, *, prefix: str | None = None) -> str:
    full = f"{prefix}_{name}" if prefix else name
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in full)
    return cleaned[:MAX_TOOL_NAME]


def truncate_blocks(
    blocks: Sequence[Mapping[str, Any]], *, limit: int = DEFAULT_MAX_RESULT_CHARS
) -> list[dict[str, Any]]:
    """Clip oversized text blocks, saying so where the model can read it."""
    out: list[dict[str, Any]] = []
    budget = limit
    for block in blocks:
        if block.get("type") != "text":
            out.append(dict(block))
            continue
        body = block.get("text", "")
        if len(body) <= budget:
            out.append(dict(block))
            budget -= len(body)
            continue
        dropped = len(body) - budget
        out.append(
            {
                "type": "text",
                "text": body[:budget]
                + f"\n\n[truncated: {dropped} more characters. Narrow the query "
                "— filter by time range, label, or line count — rather than "
                "re-running this call unchanged.]",
            }
        )
        budget = 0
    return out


def blocks_from_mcp_content(content: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert MCP content blocks to Anthropic tool_result content blocks."""
    blocks: list[dict[str, Any]] = []
    for item in content:
        kind = getattr(item, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": item.text})
        elif kind == "image":
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": getattr(item, "mime_type", "image/png"),
                        "data": item.data,
                    },
                }
            )
        else:
            # Resource links, embedded resources, anything newer: hand the
            # model the JSON rather than dropping it.
            dumped = getattr(item, "model_dump_json", None)
            blocks.append(
                {"type": "text", "text": dumped(exclude_none=True) if dumped else json.dumps(str(item))}
            )
    return blocks or [{"type": "text", "text": "(tool returned no content)"}]


@dataclass
class ToolCatalog:
    """Every tool the agent can reach, keyed by the name the model will use."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)
    sources: dict[str, ToolSource] = field(default_factory=dict)

    def add(self, source: ToolSource, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            existing = self.tools[spec.name]
            raise DuplicateToolName(
                f"both {existing.server!r} and {spec.server!r} advertise a tool "
                f"named {spec.name!r}; set a prefix on one of the servers"
            )
        self.tools[spec.name] = spec
        self.sources[spec.name] = source

    def definitions(self) -> list[dict[str, Any]]:
        """Anthropic tool definitions, in a stable order.

        Sorted because tools render before the system prompt in the cache
        prefix: a set-ordered tool list would silently invalidate the cache on
        every process restart.
        """
        return [self.tools[name].definition() for name in sorted(self.tools)]

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self.tools
