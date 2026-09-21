## Ollama commands for Mark Day
ollama run mistral-small
ollama launch claude --model mistral-small

# agentic-devops-with-mcp

A production agentic DevOps architecture, built out layer by layer, following
*Agentic DevOps With MCP: A Real Production Architecture* (Neel Shah, Jun 2026).

| Layer | Status |
| --- | --- |
| 1. **Orchestration core (ReAct loop + MCP tools)** | **done** |
| 2. **Approval gate** | **done** |
| 3. **Event-driven triggers (PagerDuty / GitHub webhooks)** | **done** |
| 4. **Memory (episodic + semantic)** | **done** |
| 5. **Agent observability (OTel)** | **done** |

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,slack]"
.venv/bin/python -m pytest
```

## Layer 5 — observability

```python
from agentic_devops.observability import configure_observability

observability = configure_observability()        # reads OTEL_EXPORTER_OTLP_ENDPOINT
app = create_app(settings, observability=observability)
```

No endpoint configured, or the SDK not installed? You get `NullEvents` and a
log line. An agent that won't start because a collector is unreachable is worse
than one running blind.

### Spans that contain the work

The article's tracer opens a span and closes it in the same breath:

```python
def trace_agent_run(task_goal, thread_id):
    with tracer.start_as_current_span('agent.run') as span:
        span.set_attribute(...)      # and then the with-block ends
```

Every span is instantaneous, nothing nests, and the trace is a flat list of
zero-duration markers. A span has to *contain* the work — so `AgentEvents` deals
in context managers:

```
agent.run                                  9.4s   thread_id, service, turns, tokens
├─ agent.turn                              1.2s   stop_reason=tool_use
│  ├─ agent.tool/get_pod_logs              0.4s   ┐ parallel reads are
│  └─ agent.tool/list_deployments          0.3s   ┘ siblings, not a chain
└─ agent.turn                              7.9s   stop_reason=end_turn
   └─ agent.tool/delete_resource           7.6s   approval.status=approved
```

`asyncio.gather` copies the current context into each task, which is what makes
parallel reads come out as siblings under their turn. A halted run sets the run
span to ERROR — a turn limit isn't an exception, but it's exactly what someone
opens the trace to find.

### Metrics, actually recorded

The article lists five metrics in a comment block. These are instruments:
`agent.runs`, `agent.run.duration`, `agent.run.turns`, `agent.tool_calls`,
`agent.tool.duration`, `agent.approvals`, `agent.tokens`, `agent.findings`,
`agent.webhooks`.

`agent.approvals` is broken out by outcome, and the split that matters is
**rejected vs expired**: a rising rejection rate means the agent is proposing
bad actions, a rising expiry rate means nobody is reading Slack. Different
problems, different people, and the article's single approval counter can't
tell them apart.

### Traces leave the cluster

`agent.task_goal[:500]` and `json.dumps(args)[:500]` — the article's two span
attributes — are incident data. Tool arguments carry hostnames, customer
identifiers and connection strings; incident titles carry customer names. The
tracing backend has its own access control, retention and vendor.

So the default records the *shape* of a call, not its contents:

```
tool.name        = "get_pod_logs"
tool.arg_count   = 2
tool.arg_names   = "customer,name"
tool.arg_types   = "customer:str,name:str"
tool.duration_ms = 412
tool.outcome     = "ok"
```

Enough to debug a malformed call; nothing that identifies a customer.
`record_payloads=True` (or `OTEL_RECORD_PAYLOADS=true`) turns values back on for
debugging, and everything then passes through `Redactor` — secret-ish argument
names by substring, plus value patterns for AWS keys, GitHub/Slack tokens, JWTs,
bearer headers, PEM private keys and credentials embedded in URLs. It's a
backstop, not a guarantee.

### Logs join to traces

`TraceContextFilter` stamps `trace_id`/`span_id` on every record and
`JSONFormatter` emits one object per line. The trace says an approval took nine
minutes; the log says which reviewer it was waiting on.

## Layer 4 — memory

```python
from agentic_devops.memory import FTSKnowledgeIndex, TranscriptStore

config = AgentConfig(
    servers=[...],
    knowledge=FTSKnowledgeIndex("/data/knowledge.db"),   # semantic
    transcripts=TranscriptStore("/data/transcripts.db"), # episodic
)
```

Curate it with the `knowledge` CLI: `add-postmortem`, `add-runbook`, `search`,
`verify`, `rm`, `transcripts --resumable`.

### Semantic — what the organisation knows

Retrieval is a protocol with three implementations:

| Index | Backing | Needs |
| --- | --- | --- |
| `FTSKnowledgeIndex` (default) | SQLite FTS5 + BM25 | nothing |
| `VectorKnowledgeIndex` | Voyage embeddings | `pip install '.[voyage]'`, `VOYAGE_API_KEY` |
| `HybridIndex` | reciprocal rank fusion of both | both |

Lexical is the default deliberately. Infrastructure recall is unusually
keyword-shaped — `payments-api`, `CrashLoopBackOff`, `OOMKilled`, `5xx` are
exact tokens, and BM25 matches them precisely where an embedding blurs them
into neighbours. Dense retrieval earns its keep on the other half ("pods keep
restarting" finding a CrashLoopBackOff postmortem), which is what the hybrid is
for. Fusion is by *rank*, not score: BM25 scores and cosine similarities aren't
on a comparable scale, so averaging them just lets the larger-magnitude index
win every time.

Knowledge reaches the agent two ways — retrieved once from the incident title
before the first turn, and available throughout as a `recall_knowledge` tool.
The tool matters more: at turn one the agent doesn't yet know what the incident
is about, and a query written after it has seen the logs is a much better query.

**Retrieved context goes in the messages, never the system prompt.** The system
block carries the cache breakpoint; rewriting it per incident would throw away
the cached prefix on every run.

### Trust is tracked

The article's `add_postmortem` files a root cause and a resolution as
knowledge. Done automatically from agent output, that's a machine for turning
one wrong diagnosis into the house opinion: the next incident retrieves it,
agrees, and writes a second document agreeing with the first.

So documents carry a `verified` flag and render their provenance:

```
[run] Memory limit too low for the new cache  (service: payments-api)
recorded 2026-09-21 · UNVERIFIED — an agent's own account, treat as a lead, not a fact
```

Agent-written documents start unverified. Only runs that actually *concluded*
— no halt, and at least one finding the agent labelled a `cause` — get recorded
at all. A human promotes one with `knowledge verify <id>`.

### Episodic — resuming a killed run

Every turn is checkpointed, so a crash costs one turn rather than the whole
investigation. The interesting part isn't the storage, it's what a half-finished
conversation looks like on reload: an interrupted run's last message is usually
an assistant turn with `tool_use` blocks that were never answered, and sending
that back is a 400. `prepare_resume` trims to the last complete exchange and
tells the agent what it dropped:

> This run was interrupted. You had called `delete_resource` but the process
> stopped before the result came back, so those calls may or may not have taken
> effect — check current state before assuming either way, and re-request
> approval for anything that writes.

That last clause matters: a resumed run must not assume its pre-crash approvals
still stand. They don't — layer 2 consumed them.

Layer 3 resubmits interrupted runs at startup, but only recent ones
(`resume_max_age`, default 1 hour): resuming last Tuesday's incident helps
nobody, and the cluster state it was reasoning about is long gone.

### Context editing

Also memory, in the sense that matters at runtime: `clear_tool_uses_20250919`
drops stale tool results server-side as the window fills. A DevOps agent
accumulates log dumps faster than almost any other kind, and the article had no
answer for a run that outgrows its context.

`TranscriptStore.purge(older_than=...)` exists because transcripts hold log
excerpts, hostnames, and occasionally a secret that leaked into a stack trace.

## Layer 3 — event-driven triggers

```
uvicorn agentic_devops.main:app --host 0.0.0.0 --port 8080
```

| Route | Purpose |
| --- | --- |
| `POST /webhooks/pagerduty` | `incident.triggered` / `incident.reopened` → investigation run |
| `POST /webhooks/github` | `deployment_status` state `success` → deployment watch |
| `POST /webhooks/slack/interactions` | **Approve / Reject clicks — this is what layer 2 has been waiting on** |
| `GET /healthz`, `GET /readyz` | liveness; readiness reports queue depth, active runs, connected tools |
| `GET /runs`, `GET /runs/{event_key}` | what the agent has been asked to do, and how it went |

Handlers do four things and return — authenticate, translate, claim, enqueue.
Nothing slow happens in a request: PagerDuty and GitHub give you seconds before
they call it a failure and retry.

### Every sender is authenticated

The article's trigger server accepts any POST that parses. An endpoint that
starts an agent run is a remote-code-execution surface with extra steps —
anyone who can reach it can fabricate an incident, and every fabricated
incident is a chance to get a human to click Approve on something.

- GitHub: `X-Hub-Signature-256` (HMAC-SHA256 over the raw body)
- PagerDuty: `X-PagerDuty-Signature`, accepting any of the comma-separated
  signatures so secret rotation doesn't need downtime
- Slack: signing secret + a 5-minute timestamp window against replays

`create_app` **refuses to start** if a secret is missing. `require_signatures=False`
exists for laptops and logs a loud warning.

### Deliveries are claimed exactly once

`bg.add_task(run_agent, ...)` — the article's mechanism — has three properties
that only appear in production:

| Problem | What happens | Here |
| --- | --- | --- |
| Retried deliveries | PagerDuty retries anything it doesn't get a timely 2xx for; a second agent investigates the same incident, racing the first and raising its own approvals | `event_key` is the ledger's primary key. The INSERT *is* the claim, so concurrent deliveries can't both win. Duplicates get a 409 |
| No ceiling | An alert storm is exactly when a hundred webhooks arrive and exactly when you least want a hundred concurrent runs posting a hundred approval requests | Bounded queue + fixed worker pool. Full → 503 + `Retry-After`, so the sender retries once the storm passes |
| Amnesia | A pod restart drops in-flight work silently; nobody can tell afterwards whether the agent looked | Durable run ledger; startup marks orphaned runs `interrupted` rather than leaving them looking live |

Shutdown drains: stop accepting, let in-flight runs finish (an agent killed
mid-run may have already spent an approval), then cancel. Keep `drain_timeout`
under the pod's `terminationGracePeriodSeconds`.

### Payload shapes

The article's `PagerDutyWebhook(event, incident)` and
`GitHubWebhook(action, pull_request, deployment)` don't match what either
service sends. PagerDuty v3 nests everything under `event.data`; GitHub signals
a finished deployment with a `deployment_status` event whose `state` is
`success`, not `action == "completed"`. Unrecognised events get a 202
`{"status": "ignored"}` — they're not errors.

### Scheduled runs

The third trigger. Interval-based rather than cron, deliberately: a cron
expression needs a dependency and a timezone policy. Anything that must fire at
a wall-clock time belongs in a CronJob posting to the webhook endpoint.

```python
ScheduledTask(
    name="cert-check",
    goal="Check for TLS certificates expiring in the next 14 days.",
    interval=timedelta(hours=6),
)
```

A scheduled task that can't get a worker is skipped, not queued — incidents
outrank housekeeping.

## Layer 1 — the orchestration core

A ReAct loop over the Anthropic Messages API on `claude-opus-5`, with tools
supplied by MCP servers we run ourselves.

```python
from agentic_devops.mcp import MCPServerSpec
from agentic_devops.orchestration import AgentConfig, build_agent, incident_goal

config = AgentConfig.from_env(servers=[
    MCPServerSpec(name="kubernetes", command="kubernetes-mcp", args=("--readonly",)),
    MCPServerSpec(name="github", command="github-mcp"),
    MCPServerSpec(name="datadog", url="https://mcp.internal/datadog"),
])

async with build_agent(config) as agent:
    run = await agent.run(
        incident_goal(
            service="payments-api",
            title="Elevated 5xx rate",
            urgency="high",
            triggered_at="2026-09-21T14:03:00Z",
        ),
        thread_id=incident_id,
    )

print(run.text)          # the agent's conclusion
print(run.findings)      # what it established, recorded as it went
print(run.denied_writes) # what a human said no to
print(run.usage)         # tokens, including cache hits
```

`build_agent` is an async context manager: leaving it kills the MCP server
subprocesses, so a failed run doesn't leave a `kubernetes-mcp` holding a
cluster connection.

### Why client-side MCP

Anthropic's hosted MCP connector (`mcp_servers` + `mcp_toolset`) is far less
code, and we can't use it: it executes tools server-side, where layer 2's
approval gate cannot intercept a write. We run the servers and execute every
call locally. Both stdio (subprocess) and streamable HTTP transports work; the
integration tests drive a real server over stdio.

### Which calls stop for a human

Three signals, in this order — a server can escalate a tool's risk but never
clear one the operator has classified as a write:

1. `destructive_hint: true` in the tool's MCP annotations → always gated.
2. The operator's policy: `write_tools` → gated, `read_only_tools` → not.
3. `read_only_hint: true` → runs unattended.
4. Nothing said → gated, because `unknown_tools_require_approval` defaults to
   true. An agent picks up tools from servers it didn't ship with.

### Loop behaviour worth knowing

- **Reads run concurrently, writes run one at a time.** Firing three approval
  requests into Slack at once and executing whichever returns first isn't
  something a reviewer can reason about.
- **All tool results go back in a single user message.** Splitting them trains
  the model out of parallel calls.
- **A refused write is not a tool error.** It comes back as an ordinary result
  saying the action did not run and not to retry; `is_error` would read as
  "try again".
- **Every stop reason is handled**: `refusal` (with its category), `max_tokens`,
  `pause_turn` (resumes), and a turn limit, because the article's loop had no
  bound and a reasoning loop on a HIGH-risk tool asks a human the same question
  forever.
- **Oversized tool output is clipped** with a note telling the model to narrow
  the query rather than re-run it.
- **Thinking blocks go back unedited** — required to continue a turn on the
  same model.
- `log_finding` is a local tool, so what the agent established survives a run
  that halts or times out.

### How it differs from the article

The article's layer 1 doesn't run as written: `langchain_mcp` isn't a package,
`AnthropicEmbeddings` doesn't exist, and `ChatAnthropic(...)` is missing a
closing paren. Beyond that:

| Article | Here |
| --- | --- |
| `claude-3-7-sonnet-20250219` | `claude-opus-5`, adaptive thinking, `effort: high` |
| LangGraph `create_react_agent` | Our own loop — the gate wraps execution directly, and layers 4/5 hook in where we choose |
| Prompt rule: "call `request_approval()` before any write" | The gate intercepts regardless; the prompt describes it as a fact of the environment, because a rule the model can forget is not a control |
| "Only propose remediation if confidence > 85%" | Ask for the evidence behind a hypothesis — a generated percentage isn't a measurement |
| Write tools identified by a hardcoded name set | MCP annotations first, policy second, fail closed on unknowns |
| No turn limit, no truncation, no stop-reason handling | All four |

Prompt caching is set up deliberately: tools render before `system` in the cache
prefix, so `ToolCatalog.definitions()` sorts by name — an unstable tool order
would silently cost a cache hit on every process restart. Check
`run.usage.cache_read_input_tokens` to confirm it's working.

Refusal fallbacks (`fallbacks: "default"`) are on by default — infrastructure
work trips security classifiers more than most domains, and a refused incident
investigation is worse than a slower one. Set `refusal_fallbacks=False` on
`AgentConfig` if your account doesn't have the beta.

## Layer 2 — the approval gate

Every write an agent proposes stops here until a named human says yes: inside a
deadline, for those exact arguments, once.

```python
from agentic_devops.approval import ApprovalGate, SQLiteApprovalStore, SlackNotifier

gate = ApprovalGate(
    store=SQLiteApprovalStore("/data/approvals.db"),
    notifier=SlackNotifier(token=SLACK_BOT_TOKEN, channel="#infra-approvals"),
    agent_id="devops-agent-1",
)

delete_resource = gate.guard("delete_resource", k8s.delete_resource, thread_id=incident_id)
delete_resource(namespace="payments", name="api-7f9")
# → blocks on Slack; returns the tool result, or a sentence telling the agent
#   it was denied and not to retry
```

Read-only tools pass through `guard()` untouched — an agent that needs sign-off
to read logs is an agent nobody uses.

### What it guarantees

- **Denied by default.** A request that reaches its deadline is `expired`, and
  expiry is applied on read, so it is never observed as pending past the
  deadline even if no sweeper is running.
- **Late clicks do not authorize.** The decision is a single conditional
  `UPDATE ... WHERE status='pending' AND expires_at > now`. A reviewer who
  clicks Approve a second after the timeout gets told they were too late.
- **One approval, one call.** Approvals are `consume()`d immediately before the
  write. The article's "never chain write actions" rule is enforced here rather
  than asked for in the system prompt.
- **Approved arguments are the executed arguments.** Each request stores a hash
  of its arguments; a mismatch at execution time burns the approval instead of
  reusing it.
- **Unreachable reviewers fail closed.** If the notifier raises, the request is
  rejected immediately — a broken Slack integration should not turn into a
  silent ten-minute stall on every action.
- **Everything is auditable.** Every request, decision, reviewer, and timeout
  is a row: `approvals history --thread incident-42`.

### How it differs from the article

The article's `ApprovalGate` kept pending approvals in `self.pending`, a dict on
the instance, and monkey-patched `tool._run`. Three consequences:

1. An agent restart mid-incident silently converts every in-flight approval
   into a timeout.
2. It cannot span processes — the webhook receiving the Slack click has to be
   the same process blocked in `_wait_for_response`.
3. Nothing stops one approval from authorizing a second call, so rule 2 of the
   system prompt ("never take multiple write actions in sequence without
   approval for each") is enforced only by the model's goodwill.

Risk assessment also took the call arguments and then ignored them. Here the
tool name sets a floor and escalation rules raise it: protected namespaces,
production environments, scale-to-zero, and bulk `--all` operations all push a
call to HIGH.

### Layout

```
src/agentic_devops/mcp/
  toolset.py    ToolSpec/ToolResult, the catalog, truncation, MCP→Anthropic blocks
  servers.py    stdio + streamable-HTTP connections, MCPToolLayer
  fake.py       in-process tool source for tests and dry runs

src/agentic_devops/orchestration/
  prompt.py     system prompt + goal builders for layer 3
  registry.py   where a tool call meets the approval gate
  loop.py       the ReAct loop, stop reasons, parallelism, usage
  findings.py   log_finding
  agent.py      wiring: config, notifier selection, build_agent

src/agentic_devops/observability/
  redaction.py  what is safe to put in a span
  tracing.py    OTelEvents: run → turn → tool spans
  metrics.py    the instruments
  logs.py       trace-correlated JSON logging
  setup.py      configure_observability(), degrades to no-op

src/agentic_devops/memory/
  knowledge.py  Document, KnowledgeIndex protocol, hybrid RRF fusion
  fts.py        SQLite FTS5 lexical index (the default)
  vectors.py    Voyage-backed dense index (optional)
  transcripts.py episodic store, checkpointing, resume trimming
  recall.py     the recall_knowledge tool
  learn.py      writing runs back, unverified
  cli.py        `knowledge search | add-postmortem | verify | transcripts`

src/agentic_devops/triggers/
  signatures.py GitHub + PagerDuty HMAC verification
  events.py     webhook payloads → RunRequests
  runs.py       durable run ledger, bounded queue, worker pool
  schedule.py   interval-based maintenance runs
  app.py        FastAPI routes + lifespan

src/agentic_devops/approval/
  models.py     ApprovalRequest, statuses, argument fingerprinting
  policy.py     which tools need approval, and how risky each call is
  store.py      ApprovalStore protocol + SQLite implementation
  notifier.py   Slack Block Kit, console, and null notifiers
  slack.py      signature verification + turning a click into a decision
  gate.py       request → wait → consume → execute
  cli.py        `approvals pending | show | resolve | history`
```

### Wiring the Slack interactivity endpoint

`slack.py` is framework-free so layer 3 owns the route:

```python
@app.post("/webhooks/slack/interactions")
async def slack_interaction(request: Request):
    body = await request.body()
    verify_slack_signature(
        SIGNING_SECRET,
        timestamp=request.headers["X-Slack-Request-Timestamp"],
        body=body,
        signature=request.headers["X-Slack-Signature"],
    )
    payload = payload_from_form_body(body)
    _, message = resolve_from_payload(store, payload, notifier=notifier)
    return {"text": message}
```

Signature verification is not optional: anyone who can POST to that endpoint
can approve production writes.

### Operational notes

- **The agent must be built in the loop that serves requests.** An MCP session
  belongs to the event loop that opened it; calling one from another loop
  deadlocks on the first `tools/call`. `create_app`'s lifespan does this
  correctly — use `agent_factory` to change *how* the agent is built while
  keeping *where*. (There's an end-to-end test covering this path; it caught
  the deadlock.)
- `.venv/bin/ruff check --select F src tests` before committing.

- `SQLiteApprovalStore` is safe across threads and processes **on one host**
  (WAL + conditional updates). The moment the agent and the webhook server run
  on different nodes, implement the same five-method `ApprovalStore` protocol
  against Postgres — the atomicity requirements are documented in `store.py`.
- Run `store.expire_overdue()` periodically so timeouts land in the audit log
  even for requests nobody polls again.
- The gate deliberately holds a thread while waiting. Use `aguard`/`arequire`
  from async trigger handlers so a ten-minute approval does not pin the event
  loop.
