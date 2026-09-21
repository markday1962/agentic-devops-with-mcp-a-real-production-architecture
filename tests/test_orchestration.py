from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from agentic_devops.approval import ApprovalGate, ApprovalPolicy
from agentic_devops.mcp import DuplicateToolName, FakeToolSource
from agentic_devops.orchestration import (
    AgentLoop,
    FindingsRecorder,
    SYSTEM_PROMPT,
    ToolRegistry,
    build_catalog_from_sources,
)

from conftest import AutoDecidingNotifier, RecordingNotifier
from fake_anthropic import FakeAnthropic, calls, says


def infra_source() -> FakeToolSource:
    source = FakeToolSource(name="kubernetes")
    source.add(
        "get_pod_logs",
        lambda args: f"logs for {args.get('name')}: OOMKilled",
        read_only=True,
        description="Read pod logs.",
    )
    source.add(
        "list_deployments",
        lambda args: "payments-api  3/3",
        read_only=True,
        description="List deployments.",
    )
    source.add(
        "delete_resource",
        lambda args: f"deleted {args.get('name')}",
        read_only=False,
        destructive=True,
        description="Delete a resource.",
    )
    source.add(
        "rotate_credentials",
        lambda args: "rotated",
        description="Rotate credentials. Deliberately unannotated and not in "
        "the default write list.",
    )
    return source


async def build(store, clock, script, *, notifier=None, policy=None, **loop_kwargs):
    source = infra_source()
    recorder = FindingsRecorder()
    catalog = await build_catalog_from_sources([source, recorder])
    gate = ApprovalGate(
        store=store,
        notifier=notifier or AutoDecidingNotifier(store, approve=True, clock=clock),
        policy=policy or ApprovalPolicy(),
        timeout=timedelta(minutes=10),
        poll_interval=0.01,
        agent_id="devops-agent-1",
        clock=clock.now,
        sleeper=clock.sleep,
        async_sleeper=clock.asleep,
    )
    registry = ToolRegistry(catalog=catalog, gate=gate)
    client = FakeAnthropic(script)
    loop = AgentLoop(
        client=client, registry=registry, recorder=recorder, **loop_kwargs
    )
    return loop, client, source, recorder


def run(coro):
    return asyncio.run(coro)


# ── the loop ─────────────────────────────────────────────────────────────


def test_read_only_call_runs_without_approval(store, clock):
    async def scenario():
        loop, client, source, _ = await build(
            store,
            clock,
            [calls(("get_pod_logs", {"namespace": "payments", "name": "api-7f9"})), says("OOMKilled.")],
            notifier=RecordingNotifier(),
        )
        result = await loop.run("why is payments-api down?", thread_id="incident-42")
        return result, client, source

    result, client, source = run(scenario())

    assert source.calls == [("get_pod_logs", {"namespace": "payments", "name": "api-7f9"})]
    assert store.history() == []  # nobody was asked
    assert result.text == "OOMKilled."
    assert result.completed


def test_write_call_waits_for_approval_then_runs(store, clock):
    async def scenario():
        loop, _, source, _ = await build(
            store,
            clock,
            [calls(("delete_resource", {"namespace": "payments", "name": "api-7f9"})), says("Deleted.")],
        )
        return await loop.run("clear the crashlooping pod", thread_id="incident-42"), source

    result, source = run(scenario())

    assert source.calls == [("delete_resource", {"namespace": "payments", "name": "api-7f9"})]
    assert len(store.history()) == 1
    assert store.history()[0].thread_id == "incident-42"
    assert result.denied_writes == []


def test_rejected_write_is_reported_to_the_model_not_executed(store, clock):
    async def scenario():
        loop, client, source, _ = await build(
            store,
            clock,
            [
                calls(("delete_resource", {"namespace": "payments", "name": "api-7f9"})),
                says("Understood, escalating instead."),
            ],
            notifier=AutoDecidingNotifier(store, approve=False, clock=clock),
        )
        return await loop.run("clear the pod", thread_id="incident-42"), client, source

    result, client, source = run(scenario())

    assert source.calls == []  # never reached the cluster
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert "NOT executed" in tool_result["content"][0]["text"]
    # A refusal is not an error: is_error would read as "retry me".
    assert "is_error" not in tool_result
    assert len(result.denied_writes) == 1


def test_destructive_hint_gates_a_tool_policy_never_heard_of(store, clock):
    """`rotate_credentials` is not in DEFAULT_WRITE_TOOLS. It still stops."""

    async def scenario():
        loop, _, source, _ = await build(
            store,
            clock,
            [calls(("rotate_credentials", {"service": "payments"})), says("Rotated.")],
            notifier=AutoDecidingNotifier(store, approve=False, clock=clock),
        )
        return await loop.run("rotate the creds", thread_id="incident-42"), source

    _, source = run(scenario())
    assert source.calls == []
    assert len(store.history()) == 1


def test_unknown_tool_is_reported_with_the_real_tool_list(store, clock):
    async def scenario():
        loop, client, _, _ = await build(
            store,
            clock,
            [calls(("kubectl_nuke", {})), says("Sorry.")],
            notifier=RecordingNotifier(),
        )
        return await loop.run("nuke it", thread_id="incident-42"), client

    _, client = run(scenario())
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "No tool named `kubectl_nuke`" in tool_result["content"][0]["text"]
    assert "get_pod_logs" in tool_result["content"][0]["text"]


def test_parallel_reads_return_in_one_user_message(store, clock):
    """Splitting tool results across messages trains the model out of
    parallel calls."""

    async def scenario():
        loop, client, _, _ = await build(
            store,
            clock,
            [
                calls(
                    ("get_pod_logs", {"name": "api-7f9"}),
                    ("list_deployments", {"namespace": "payments"}),
                ),
                says("Both checked."),
            ],
            notifier=RecordingNotifier(),
        )
        return await loop.run("check both", thread_id="incident-42"), client

    _, client = run(scenario())
    final_message = client.requests[1]["messages"][-1]
    assert final_message["role"] == "user"
    assert len(final_message["content"]) == 2
    assert [b["tool_use_id"] for b in final_message["content"]] == ["toolu_0", "toolu_1"]


def test_two_writes_in_one_turn_need_two_approvals(store, clock):
    async def scenario():
        loop, _, source, _ = await build(
            store,
            clock,
            [
                calls(
                    ("delete_resource", {"name": "one"}),
                    ("delete_resource", {"name": "two"}),
                ),
                says("Both gone."),
            ],
        )
        return await loop.run("clear both", thread_id="incident-42"), source

    _, source = run(scenario())
    assert len(store.history()) == 2
    assert [name for name, _ in source.calls] == ["delete_resource", "delete_resource"]


def test_findings_survive_the_run(store, clock):
    async def scenario():
        loop, _, _, recorder = await build(
            store,
            clock,
            [
                calls(
                    (
                        "log_finding",
                        {
                            "summary": "payments-api OOMKilled after the 14:00 deploy",
                            "evidence": "3 OOMKilled events, memory limit 512Mi unchanged",
                            "significance": "cause",
                            "service": "payments-api",
                        },
                    )
                ),
                says("Root cause: memory limit."),
            ],
            notifier=RecordingNotifier(),
        )
        return await loop.run("investigate", thread_id="incident-42"), recorder

    result, recorder = run(scenario())
    assert len(result.findings) == 1
    assert result.findings[0].significance == "cause"
    assert recorder.for_thread("incident-42") == result.findings


# ── stopping conditions ──────────────────────────────────────────────────


def test_turn_limit_halts_the_run(store, clock):
    async def scenario():
        loop, _, _, _ = await build(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "x"})) for _ in range(5)],
            notifier=RecordingNotifier(),
            max_turns=3,
        )
        return await loop.run("loop forever", thread_id="incident-42")

    result = run(scenario())
    assert result.turns == 3
    assert "3-turn limit" in result.halted
    assert not result.completed


def test_refusal_halts_the_run(store, clock):
    from fake_anthropic import FakeMessage, StopDetails, TextBlock

    async def scenario():
        loop, _, _, _ = await build(
            store,
            clock,
            [
                FakeMessage(
                    content=[TextBlock("I can't help with that.")],
                    stop_reason="refusal",
                    stop_details=StopDetails(category="cyber"),
                )
            ],
            notifier=RecordingNotifier(),
        )
        return await loop.run("do something alarming", thread_id="incident-42")

    result = run(scenario())
    assert "declined" in result.halted
    assert "cyber" in result.halted


def test_truncated_response_halts_rather_than_looping(store, clock):
    from fake_anthropic import FakeMessage, TextBlock

    async def scenario():
        loop, _, _, _ = await build(
            store,
            clock,
            [FakeMessage(content=[TextBlock("a very long")], stop_reason="max_tokens")],
            notifier=RecordingNotifier(),
        )
        return await loop.run("write an essay", thread_id="incident-42")

    result = run(scenario())
    assert "max_tokens" in result.halted


def test_pause_turn_resumes(store, clock):
    from fake_anthropic import FakeMessage, TextBlock

    async def scenario():
        loop, client, _, _ = await build(
            store,
            clock,
            [
                FakeMessage(content=[TextBlock("thinking...")], stop_reason="pause_turn"),
                says("Done."),
            ],
            notifier=RecordingNotifier(),
        )
        return await loop.run("long task", thread_id="incident-42"), client

    result, client = run(scenario())
    assert len(client.requests) == 2
    assert result.completed
    assert result.text == "Done."


# ── request shape ────────────────────────────────────────────────────────


def test_request_uses_current_api_surface(store, clock):
    async def scenario():
        loop, client, _, _ = await build(
            store, clock, [says("hello")], notifier=RecordingNotifier()
        )
        await loop.run("hi", thread_id="incident-42")
        return client

    client = run(scenario())
    request = client.requests[0]

    assert request["model"] == "claude-opus-5"
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "high"}
    assert request["fallbacks"] == "default"
    assert "budget_tokens" not in str(request["thinking"])


def test_system_prompt_is_cached_and_tools_are_ordered(store, clock):
    """Tools render before system in the cache prefix, so an unstable tool
    order would silently cost a cache hit on every run."""

    async def scenario():
        loop, client, _, _ = await build(
            store, clock, [says("hello")], notifier=RecordingNotifier()
        )
        await loop.run("hi", thread_id="incident-42")
        return client

    client = run(scenario())
    request = client.requests[0]

    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["text"] == SYSTEM_PROMPT
    names = [t["name"] for t in request["tools"]]
    assert names == sorted(names)


def test_thinking_blocks_are_returned_unedited(store, clock):
    async def scenario():
        loop, client, _, _ = await build(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "x"})), says("done")],
            notifier=RecordingNotifier(),
        )
        await loop.run("check", thread_id="incident-42")
        return client

    client = run(scenario())
    assistant_turn = client.requests[1]["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"][0].type == "thinking"
    assert assistant_turn["content"][0].signature == "sig-abc"


def test_usage_is_accumulated(store, clock):
    async def scenario():
        loop, _, _, _ = await build(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "x"})), says("done")],
            notifier=RecordingNotifier(),
        )
        return await loop.run("check", thread_id="incident-42")

    result = run(scenario())
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100


# ── catalog ──────────────────────────────────────────────────────────────


def test_duplicate_tool_names_are_refused():
    async def scenario():
        one = FakeToolSource(name="k8s-prod").add("delete_resource", lambda a: "x")
        two = FakeToolSource(name="k8s-staging").add("delete_resource", lambda a: "x")
        await build_catalog_from_sources([one, two])

    with pytest.raises(DuplicateToolName) as excinfo:
        run(scenario())
    assert "k8s-prod" in str(excinfo.value) and "k8s-staging" in str(excinfo.value)


def test_oversized_tool_output_is_clipped_with_a_hint():
    from agentic_devops.mcp import truncate_blocks

    blocks = truncate_blocks([{"type": "text", "text": "x" * 100}], limit=20)
    assert len(blocks) == 1
    assert blocks[0]["text"].startswith("x" * 20)
    assert "truncated: 80 more characters" in blocks[0]["text"]
    assert "Narrow the query" in blocks[0]["text"]


def test_concurrent_runs_do_not_share_findings(store, clock):
    """One agent serves many incidents. Incident A's evidence appearing in
    incident B's report is worse than no report at all."""

    async def scenario():
        source = infra_source()
        recorder = FindingsRecorder()
        catalog = await build_catalog_from_sources([source, recorder])
        gate = ApprovalGate(
            store=store,
            notifier=RecordingNotifier(),
            policy=ApprovalPolicy(),
            clock=clock.now,
            sleeper=clock.sleep,
        )
        registry = ToolRegistry(catalog=catalog, gate=gate)

        def loop_for(label):
            return AgentLoop(
                client=FakeAnthropic(
                    [
                        calls(
                            (
                                "log_finding",
                                {
                                    "summary": f"{label} summary",
                                    "evidence": f"{label} evidence",
                                    "significance": "cause",
                                },
                            )
                        ),
                        says(f"{label} done"),
                    ]
                ),
                registry=registry,
                recorder=recorder,
            )

        return await asyncio.gather(
            loop_for("alpha").run("investigate alpha", thread_id="incident-alpha"),
            loop_for("beta").run("investigate beta", thread_id="incident-beta"),
        )

    alpha, beta = run(scenario())

    assert [f.summary for f in alpha.findings] == ["alpha summary"]
    assert [f.summary for f in beta.findings] == ["beta summary"]
