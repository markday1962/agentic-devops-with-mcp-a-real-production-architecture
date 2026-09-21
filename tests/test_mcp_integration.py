"""Exercises the real MCP stdio transport against a live server subprocess.

The fake tool source covers loop behaviour; this covers the thing the fake
cannot: that discovery, annotations, and calls actually work over the wire.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from agentic_devops.approval import ApprovalGate, ApprovalPolicy
from agentic_devops.mcp import MCPServerSpec, MCPToolLayer
from agentic_devops.orchestration import ToolRegistry

from conftest import AutoDecidingNotifier, RecordingNotifier

SERVER = Path(__file__).parent / "mcp_test_server.py"


def server_spec(**overrides) -> MCPServerSpec:
    return MCPServerSpec(
        name="test-infra",
        command=sys.executable,
        args=(str(SERVER),),
        read_timeout=20.0,
        **overrides,
    )


def test_tools_and_annotations_come_over_the_wire():
    async def scenario():
        async with MCPToolLayer([server_spec()]) as layer:
            return dict(layer.catalog.tools)

    tools = asyncio.run(scenario())

    assert set(tools) == {"get_pod_logs", "delete_resource"}
    assert tools["get_pod_logs"].read_only_hint is True
    assert tools["delete_resource"].destructive_hint is True
    assert tools["get_pod_logs"].server == "test-infra"
    assert "namespace" in tools["get_pod_logs"].input_schema["properties"]


def test_prefix_disambiguates_two_instances_of_one_server():
    async def scenario():
        specs = [
            server_spec(prefix="prod"),
            MCPServerSpec(
                name="test-infra-staging",
                command=sys.executable,
                args=(str(SERVER),),
                prefix="staging",
                read_timeout=20.0,
            ),
        ]
        async with MCPToolLayer(specs) as layer:
            return sorted(layer.catalog.tools)

    assert asyncio.run(scenario()) == [
        "prod_delete_resource",
        "prod_get_pod_logs",
        "staging_delete_resource",
        "staging_get_pod_logs",
    ]


def test_read_call_executes_against_the_server(store, clock):
    async def scenario():
        async with MCPToolLayer([server_spec()]) as layer:
            gate = ApprovalGate(
                store=store,
                notifier=RecordingNotifier(),
                policy=ApprovalPolicy(),
                clock=clock.now,
                sleeper=clock.sleep,
                async_sleeper=clock.asleep,
            )
            registry = ToolRegistry(catalog=layer.catalog, gate=gate)
            return await registry.call(
                "get_pod_logs", {"namespace": "payments", "name": "api-7f9"}
            )

    invocation = asyncio.run(scenario())

    assert "OOMKilled at 14:02" in invocation.result.rendered()
    assert not invocation.was_write
    assert store.history() == []


def test_destructive_call_is_gated_then_executed(store, clock):
    async def scenario():
        async with MCPToolLayer([server_spec()]) as layer:
            gate = ApprovalGate(
                store=store,
                notifier=AutoDecidingNotifier(store, approve=True, clock=clock),
                policy=ApprovalPolicy(),
                timeout=timedelta(minutes=10),
                poll_interval=0.01,
                clock=clock.now,
                sleeper=clock.sleep,
                async_sleeper=clock.asleep,
            )
            registry = ToolRegistry(catalog=layer.catalog, gate=gate)
            return await registry.call(
                "delete_resource",
                {"namespace": "payments", "name": "api-7f9"},
                thread_id="incident-42",
            )

    invocation = asyncio.run(scenario())

    assert "deleted api-7f9" in invocation.result.rendered()
    assert invocation.was_write
    assert invocation.approval_id is not None
    assert len(store.history()) == 1


def test_denied_call_never_reaches_the_server(store, clock):
    async def scenario():
        async with MCPToolLayer([server_spec()]) as layer:
            gate = ApprovalGate(
                store=store,
                notifier=AutoDecidingNotifier(store, approve=False, clock=clock),
                policy=ApprovalPolicy(),
                poll_interval=0.01,
                clock=clock.now,
                sleeper=clock.sleep,
                async_sleeper=clock.asleep,
            )
            registry = ToolRegistry(catalog=layer.catalog, gate=gate)
            return await registry.call(
                "delete_resource", {"namespace": "payments", "name": "api-7f9"}
            )

    invocation = asyncio.run(scenario())

    assert invocation.result.denied
    assert "NOT executed" in invocation.result.rendered()
    assert "deleted" not in invocation.result.rendered()


def test_server_error_becomes_a_tool_error_not_a_crash(store, clock):
    """A missing required argument is the server's problem to report."""

    async def scenario():
        async with MCPToolLayer([server_spec()]) as layer:
            gate = ApprovalGate(
                store=store,
                notifier=RecordingNotifier(),
                policy=ApprovalPolicy(),
                clock=clock.now,
                sleeper=clock.sleep,
                async_sleeper=clock.asleep,
            )
            registry = ToolRegistry(catalog=layer.catalog, gate=gate)
            return await registry.call("get_pod_logs", {"namespace": "payments"})

    invocation = asyncio.run(scenario())
    assert invocation.result.is_error


def test_bad_server_spec_is_rejected_up_front():
    with pytest.raises(ValueError, match="exactly one of command"):
        MCPServerSpec(name="broken")
    with pytest.raises(ValueError, match="exactly one of command"):
        MCPServerSpec(name="broken", command="x", url="http://example.com/mcp")
