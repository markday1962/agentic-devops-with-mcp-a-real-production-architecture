"""``recall_knowledge`` — letting the agent ask memory a question mid-run.

Pre-flight retrieval on the incident title is useful but blunt: the agent does
not yet know what the incident is about when it fires. Once it has seen the
logs and has a real symptom in hand, a second, better-informed query is worth
far more — so recall is also a tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..mcp.toolset import ToolResult, ToolSpec
from .knowledge import Kind, KnowledgeIndex, render_hits

log = logging.getLogger("devops-agent.memory")

RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "What you want to know, in the words you would use to describe "
                "the symptom — e.g. 'payments-api 5xx spike after deploy' or "
                "'pods restarting with exit code 137'."
            ),
        },
        "service": {
            "type": "string",
            "description": "Restrict to one service. Omit to search everything.",
        },
        "kinds": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [kind.value for kind in Kind],
            },
            "description": (
                "Restrict to document types. 'postmortem' and 'runbook' are "
                "human-written; 'run' is a previous agent's own account and is "
                "not authoritative."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass
class KnowledgeRecall:
    """A ``ToolSource`` over the knowledge index."""

    index: KnowledgeIndex
    name: str = "memory"
    limit: int = 4

    async def list_tools(self) -> Sequence[ToolSpec]:
        return [
            ToolSpec(
                name="recall_knowledge",
                description=(
                    "Search past postmortems, runbooks, service documentation "
                    "and previous agent runs. Use it when you have a concrete "
                    "symptom — this has probably happened before."
                ),
                input_schema=RECALL_SCHEMA,
                server=self.name,
                read_only_hint=True,
            )
        ]

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        if name != "recall_knowledge":
            return ToolResult.text(f"no such tool: {name}", is_error=True)

        query = args.get("query")
        if not query:
            return ToolResult.text("recall_knowledge needs a query", is_error=True)

        kinds = None
        raw_kinds = args.get("kinds")
        if raw_kinds:
            try:
                kinds = [Kind(value) for value in raw_kinds]
            except ValueError as exc:
                return ToolResult.text(f"unknown document kind: {exc}", is_error=True)

        hits = self.index.search(
            query, limit=self.limit, kinds=kinds, service=args.get("service")
        )
        log.info("recall query=%r hits=%d", query, len(hits))
        return ToolResult.text(render_hits(hits))
