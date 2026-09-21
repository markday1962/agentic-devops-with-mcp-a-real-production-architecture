"""MCP tool layer: connect to servers, expose their tools to the agent."""

from .fake import FakeTool, FakeToolSource
from .servers import DEFAULT_READ_TIMEOUT, MCPConnection, MCPServerSpec, MCPToolLayer
from .toolset import (
    DEFAULT_MAX_RESULT_CHARS,
    DuplicateToolName,
    ToolCatalog,
    ToolResult,
    ToolSource,
    ToolSpec,
    sanitize_tool_name,
    truncate_blocks,
)

__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_READ_TIMEOUT",
    "DuplicateToolName",
    "FakeTool",
    "FakeToolSource",
    "MCPConnection",
    "MCPServerSpec",
    "MCPToolLayer",
    "ToolCatalog",
    "ToolResult",
    "ToolSource",
    "ToolSpec",
    "sanitize_tool_name",
    "truncate_blocks",
]
