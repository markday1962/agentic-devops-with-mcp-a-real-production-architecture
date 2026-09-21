"""A real MCP server, run as a subprocess by the stdio integration test."""

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

server = MCPServer("test-infra")


@server.tool(
    description="Read pod logs.",
    annotations=ToolAnnotations(read_only_hint=True),
)
def get_pod_logs(namespace: str, name: str) -> str:
    return f"logs for {name} in {namespace}: OOMKilled at 14:02"


@server.tool(
    description="Delete a resource.",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
def delete_resource(namespace: str, name: str) -> str:
    return f"deleted {name} from {namespace}"


if __name__ == "__main__":
    server.run("stdio")
