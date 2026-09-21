from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from agentic_devops.approval import ApprovalGate, ApprovalPolicy
from agentic_devops.mcp import FakeToolSource
from agentic_devops.memory import (
    Document,
    FTSKnowledgeIndex,
    HybridIndex,
    Kind,
    KnowledgeRecall,
    SearchHit,
    TranscriptStore,
    add_postmortem,
    add_runbook,
    prepare_resume,
    record_run,
    run_to_document,
    serialize_messages,
)
from agentic_devops.orchestration import (
    AgentLoop,
    FindingsRecorder,
    ToolRegistry,
    build_catalog_from_sources,
)
from agentic_devops.orchestration.findings import Finding

from conftest import RecordingNotifier
from fake_anthropic import FakeAnthropic, calls, says


@pytest.fixture
def index(tmp_path) -> FTSKnowledgeIndex:
    ix = FTSKnowledgeIndex(tmp_path / "knowledge.db")
    yield ix
    ix.close()


@pytest.fixture
def transcripts(tmp_path) -> TranscriptStore:
    store = TranscriptStore(tmp_path / "transcripts.db")
    yield store
    store.close()


def seed(index) -> None:
    add_postmortem(
        index,
        service="payments-api",
        title="payments-api 5xx spike after v2.4 deploy",
        root_cause="Memory limit of 512Mi was too low for the new cache; pods OOMKilled.",
        resolution="Raised the limit to 1Gi and added a memory alert.",
        incident_id="PINC1",
    )
    add_runbook(
        index,
        service="billing-worker",
        title="Restarting the billing worker safely",
        steps=["Drain the queue", "Scale to zero", "Scale back to three"],
    )
    index.add(
        Document.new(
            kind=Kind.PATTERN,
            title="CrashLoopBackOff with exit code 137",
            body="Exit code 137 is SIGKILL, nearly always the OOM killer.",
            verified=True,
        )
    )


# ── lexical index ────────────────────────────────────────────────────────


def test_retrieves_the_relevant_postmortem(index):
    seed(index)
    hits = index.search("payments-api high error rate after deployment")
    assert hits
    assert "payments-api 5xx spike" in hits[0].document.title


def test_a_query_full_of_fts_syntax_does_not_explode(index):
    """Real incident titles contain colons, hyphens, parens and the word NOT,
    all of which are FTS5 grammar."""
    seed(index)
    hits = index.search('payments-api: 5xx (NOT resolved) * "quoted"')
    assert hits[0].document.service == "payments-api"


def test_empty_query_returns_nothing_rather_than_everything(index):
    seed(index)
    assert index.search("!!! ???") == []


def test_filters_by_kind_and_service(index):
    seed(index)
    assert index.search("restart", kinds=[Kind.RUNBOOK])[0].document.kind is Kind.RUNBOOK
    assert index.search("deploy", kinds=[Kind.RUNBOOK]) == []
    assert index.search("memory", service="payments-api")
    assert index.search("memory", service="billing-worker") == []


def test_documents_are_upserted_not_duplicated(index):
    first = add_postmortem(
        index, service="a", title="t", root_cause="rc", resolution="r"
    )
    second = add_postmortem(
        index, service="a", title="t", root_cause="rc", resolution="r"
    )
    assert first.id == second.id
    assert len(index.all()) == 1


def test_delete_and_verify(index):
    seed(index)
    document = index.search("exit code 137")[0].document
    assert index.set_verified(document.id, False)
    assert index.get(document.id).verified is False
    assert index.delete(document.id)
    assert index.get(document.id) is None


# ── provenance ───────────────────────────────────────────────────────────


def test_unverified_documents_announce_themselves():
    """An agent's own guess must not be recalled as established fact."""
    guess = Document.new(kind=Kind.RUN, title="t", body="b", verified=False)
    trusted = Document.new(kind=Kind.POSTMORTEM, title="t", body="b", verified=True)

    assert "UNVERIFIED" in guess.render()
    assert "treat as a lead, not a fact" in guess.render()
    assert "verified by a human" in trusted.render()
    assert "UNVERIFIED" not in trusted.render()


# ── hybrid fusion ────────────────────────────────────────────────────────


class StubIndex:
    def __init__(self, name, ordering):
        self.name = name
        self.ordering = ordering

    def add(self, document):
        return document

    def search(self, query, *, limit=5, kinds=None, service=None):
        return [
            SearchHit(document=doc, score=1.0 / rank, index=self.name)
            for rank, doc in enumerate(self.ordering, start=1)
        ][:limit]

    def get(self, document_id):
        return next((d for d in self.ordering if d.id == document_id), None)

    def delete(self, document_id):
        return False

    def all(self, *, limit=100):
        return list(self.ordering)


def test_fusion_prefers_what_both_indexes_like():
    a = Document.new(kind=Kind.RUN, title="a", body="a")
    b = Document.new(kind=Kind.RUN, title="b", body="b")
    c = Document.new(kind=Kind.RUN, title="c", body="c")

    # b is second in both; a and c each top one list and are absent from the other.
    lexical = StubIndex("lex", [a, b])
    dense = StubIndex("vec", [c, b])
    hybrid = HybridIndex(indexes=[lexical, dense])

    ranked = [hit.document.title for hit in hybrid.search("anything", limit=3)]
    assert ranked[0] == "b"


# ── episodic memory ──────────────────────────────────────────────────────


class Block:
    """Stands in for an SDK content block with model_dump."""

    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self, exclude_none=True, mode="json"):
        return dict(self.fields)


def test_thinking_signatures_survive_serialization():
    """A resumed turn is rejected if a thinking block loses its signature."""
    messages = [
        {"role": "user", "content": "investigate"},
        {
            "role": "assistant",
            "content": [Block(type="thinking", thinking="", signature="sig-abc")],
        },
    ]
    serialized = serialize_messages(messages)
    assert serialized[1]["content"][0]["signature"] == "sig-abc"


def test_transcript_round_trip(transcripts):
    transcripts.save(
        thread_id="incident-42",
        goal="investigate",
        messages=[{"role": "user", "content": "investigate"}],
        turns=1,
        status="running",
        service="payments-api",
    )
    loaded = transcripts.load("incident-42")
    assert loaded.turns == 1
    assert loaded.service == "payments-api"
    assert loaded.messages == [{"role": "user", "content": "investigate"}]
    assert [c.thread_id for c in transcripts.resumable()] == ["incident-42"]


def test_checkpoints_overwrite_in_place(transcripts):
    for turn in (1, 2, 3):
        transcripts.save(
            thread_id="incident-42",
            goal="investigate",
            messages=[{"role": "user", "content": "x"}] * turn,
            turns=turn,
            status="running",
        )
    assert transcripts.load("incident-42").turns == 3
    assert len(transcripts.recent()) == 1


def test_resume_drops_an_unanswered_tool_call(transcripts):
    """The state a killed run actually leaves behind: a tool_use with no
    result, which the API rejects."""
    messages = [
        {"role": "user", "content": "investigate"},
        {"role": "assistant", "content": [{"type": "text", "text": "checking logs"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "delete_resource", "input": {}}
            ],
        },
    ]
    kept, note = prepare_resume(messages)

    assert len(kept) == 3
    assert kept[-1]["role"] == "user"
    assert "delete_resource" in note
    assert "may or may not have taken effect" in note
    assert "re-request approval" in note


def test_resume_of_a_clean_transcript_needs_no_warning():
    messages = [
        {"role": "user", "content": "investigate"},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    kept, note = prepare_resume(messages)
    assert kept == messages
    assert note is None


def test_purge_drops_old_transcripts(transcripts):
    transcripts.save(
        thread_id="old", goal="g", messages=[], turns=1, status="completed"
    )
    assert transcripts.purge(older_than=timedelta(days=30)) == 0
    assert transcripts.purge(older_than=timedelta(seconds=-1)) == 1


# ── writing back ─────────────────────────────────────────────────────────


class FakeRun:
    def __init__(self, findings=(), halted=None, text="Conclusion.", service=None):
        self.thread_id = "incident-42"
        self.goal = "Investigate payments-api\nmore detail"
        self.findings = list(findings)
        self.halted = halted
        self.text = text
        self.turns = 4
        self.denied_writes = []
        self.service = service


def cause(summary="Memory limit too low", service="payments-api"):
    return Finding(
        summary=summary,
        evidence="3 OOMKilled events",
        significance="cause",
        service=service,
    )


def test_a_halted_run_teaches_nothing():
    run = FakeRun(findings=[cause()], halted="reached the turn limit")
    assert run_to_document(run) is None


def test_a_run_with_no_conclusion_is_not_recorded():
    inconclusive = Finding(
        summary="error rate is elevated", evidence="4% 5xx", significance="symptom"
    )
    assert run_to_document(FakeRun(findings=[inconclusive])) is None


def test_a_concluded_run_is_recorded_but_not_trusted(index):
    run = FakeRun(findings=[cause()])
    document = record_run(index, run)

    assert document.kind is Kind.RUN
    assert document.verified is False
    assert document.service == "payments-api"
    assert "Memory limit too low" in document.body
    assert index.get(document.id) is not None
    assert "UNVERIFIED" in index.search("memory limit")[0].document.render()


def test_ruled_out_findings_are_kept(index):
    run = FakeRun(
        findings=[
            cause(),
            Finding(
                summary="not a network partition",
                evidence="peer latency flat throughout",
                significance="ruled_out",
            ),
        ]
    )
    document = record_run(index, run)
    assert "Ruled out" in document.body
    assert "not a network partition" in document.body


# ── the loop ─────────────────────────────────────────────────────────────


async def build_loop(store, clock, script, *, index=None, transcripts=None):
    source = FakeToolSource(name="kubernetes")
    source.add("get_pod_logs", lambda a: "OOMKilled", read_only=True)
    recorder = FindingsRecorder()
    sources = [source, recorder]
    if index is not None:
        sources.append(KnowledgeRecall(index=index))
    catalog = await build_catalog_from_sources(sources)
    gate = ApprovalGate(
        store=store,
        notifier=RecordingNotifier(),
        policy=ApprovalPolicy(),
        clock=clock.now,
        sleeper=clock.sleep,
        async_sleeper=clock.asleep,
    )
    client = FakeAnthropic(script)
    loop = AgentLoop(
        client=client,
        registry=ToolRegistry(catalog=catalog, gate=gate),
        recorder=recorder,
        knowledge=index,
        transcripts=transcripts,
    )
    return loop, client


def test_prior_knowledge_is_injected_before_the_task(store, clock, index):
    seed(index)

    async def scenario():
        loop, client = await build_loop(store, clock, [says("ok")], index=index)
        await loop.run(
            "payments-api is returning 5xx after a deploy",
            thread_id="incident-42",
            service="payments-api",
        )
        return client

    client = run_async(scenario())
    opening = client.requests[0]["messages"][0]["content"]

    assert "prior knowledge" in opening
    assert "Memory limit of 512Mi" in opening
    assert opening.endswith("payments-api is returning 5xx after a deploy")
    # Never in the system block: that carries the cache breakpoint.
    assert "512Mi" not in client.requests[0]["system"][0]["text"]


def test_no_recall_no_preamble(store, clock, index):
    async def scenario():
        loop, client = await build_loop(store, clock, [says("ok")], index=index)
        await loop.run("something nobody has seen", thread_id="incident-42")
        return client

    client = run_async(scenario())
    assert client.requests[0]["messages"][0]["content"] == "something nobody has seen"


def test_the_agent_can_query_memory_mid_run(store, clock, index):
    seed(index)

    async def scenario():
        loop, client = await build_loop(
            store,
            clock,
            [
                calls(("recall_knowledge", {"query": "exit code 137"})),
                says("It's the OOM killer."),
            ],
            index=index,
        )
        await loop.run("pods keep restarting", thread_id="incident-42")
        return client

    client = run_async(scenario())
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert "SIGKILL" in tool_result["content"][0]["text"]


def test_every_turn_is_checkpointed(store, clock, transcripts):
    async def scenario():
        loop, _ = await build_loop(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "x"})), says("Done.")],
            transcripts=transcripts,
        )
        return await loop.run("investigate", thread_id="incident-42", service="payments-api")

    result = run_async(scenario())
    checkpoint = transcripts.load("incident-42")

    assert result.turns == 2
    assert checkpoint.turns == 2
    assert checkpoint.status == "completed"
    assert checkpoint.service == "payments-api"
    assert checkpoint.summary == "Done."


def test_a_halted_run_is_checkpointed_as_halted(store, clock, transcripts):
    async def scenario():
        loop, _ = await build_loop(
            store,
            clock,
            [calls(("get_pod_logs", {"name": "x"}))] * 3,
            transcripts=transcripts,
        )
        loop.max_turns = 2
        return await loop.run("investigate", thread_id="incident-42")

    run_async(scenario())
    assert transcripts.load("incident-42").status == "halted"


def test_a_killed_run_resumes_from_its_checkpoint(store, clock, transcripts):
    """The whole point of episodic memory: the pod died mid-incident."""
    transcripts.save(
        thread_id="incident-42",
        goal="investigate payments-api",
        messages=[
            {"role": "user", "content": "investigate payments-api"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "t1", "name": "get_pod_logs", "input": {}},
                ],
            },
        ],
        turns=1,
        status="running",
    )

    async def scenario():
        loop, client = await build_loop(
            store, clock, [says("Resumed and finished.")], transcripts=transcripts
        )
        return await loop.resume("incident-42"), client

    result, client = run_async(scenario())
    messages = client.requests[0]["messages"]

    assert len(messages) == 2                      # dangling tool_use dropped
    assert messages[-1]["role"] == "user"
    assert "interrupted" in messages[-1]["content"]
    assert "get_pod_logs" in messages[-1]["content"]
    assert result.text == "Resumed and finished."
    assert transcripts.load("incident-42").status == "completed"


def test_context_editing_is_requested(store, clock):
    async def scenario():
        loop, client = await build_loop(store, clock, [says("ok")])
        await loop.run("investigate", thread_id="incident-42")
        return client

    client = run_async(scenario())
    request = client.requests[0]
    assert request["context_management"] == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }
    assert "context-management-2025-06-27" in request["betas"]


def run_async(coro):
    return asyncio.run(coro)
