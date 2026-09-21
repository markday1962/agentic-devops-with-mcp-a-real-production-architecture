"""Writing back what a run learned — carefully.

The article's ``add_postmortem`` takes a root cause and a resolution and files
them as knowledge. Done automatically from agent output, that is a machine for
turning one wrong diagnosis into the house opinion: the next incident retrieves
it, agrees with it, and writes a second document that agrees with the first.

Two guards here. Agent-written documents are stored ``verified=False`` and
render with a line saying so, and only runs that actually concluded — with a
finding the agent labelled a cause — are recorded at all. A human promotes a
document to verified with ``knowledge verify <id>``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .knowledge import Document, Kind, KnowledgeIndex

log = logging.getLogger("devops-agent.memory")


def _findings_by(run: Any, significance: str) -> list[Any]:
    return [f for f in getattr(run, "findings", ()) if f.significance == significance]


def run_to_document(run: Any, *, service: str | None = None) -> Document | None:
    """Render a finished run as a knowledge document, or None if it taught
    us nothing worth keeping."""
    if getattr(run, "halted", None):
        # A run that hit the turn limit or was refused did not reach a
        # conclusion; filing its partial reasoning as knowledge is worse than
        # filing nothing.
        return None

    causes = _findings_by(run, "cause")
    if not causes:
        return None

    service = service or next(
        (f.service for f in getattr(run, "findings", ()) if f.service), None
    )
    sections = [f"Goal: {run.goal.strip().splitlines()[0]}"]
    for label, significance in (
        ("Cause", "cause"),
        ("Ruled out", "ruled_out"),
        ("Symptoms", "symptom"),
    ):
        items = _findings_by(run, significance)
        if items:
            sections.append(
                f"{label}:\n"
                + "\n".join(f"  - {f.summary} ({f.evidence})" for f in items)
            )

    denied = getattr(run, "denied_writes", ())
    if denied:
        names = ", ".join(sorted({i.name for i in denied}))
        sections.append(f"Proposed but not approved: {names}")
    if getattr(run, "text", None):
        sections.append(f"Conclusion:\n  {run.text.strip()}")

    title = causes[0].summary[:120]
    return Document.new(
        kind=Kind.RUN,
        title=title,
        body="\n\n".join(sections),
        service=service,
        thread_id=getattr(run, "thread_id", None),
        verified=False,
        metadata={"turns": getattr(run, "turns", 0)},
    )


def record_run(
    index: KnowledgeIndex, run: Any, *, service: str | None = None
) -> Document | None:
    document = run_to_document(run, service=service)
    if document is None:
        log.info(
            "not recording run %s: no concluded cause",
            getattr(run, "thread_id", "?"),
        )
        return None
    index.add(document)
    log.info("recorded run %s as document %s", run.thread_id, document.id)
    return document


def add_postmortem(
    index: KnowledgeIndex,
    *,
    service: str,
    title: str,
    root_cause: str,
    resolution: str,
    incident_id: str | None = None,
    verified: bool = True,
) -> Document:
    """The human-curated path: a real postmortem, trusted by default."""
    document = Document.new(
        kind=Kind.POSTMORTEM,
        title=title,
        body=f"Root cause:\n  {root_cause}\n\nResolution:\n  {resolution}",
        service=service,
        thread_id=incident_id,
        verified=verified,
        metadata={"incident_id": incident_id} if incident_id else {},
    )
    index.add(document)
    return document


def add_runbook(
    index: KnowledgeIndex, *, service: str, title: str, steps: Sequence[str]
) -> Document:
    body = "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, start=1))
    document = Document.new(
        kind=Kind.RUNBOOK, title=title, body=body, service=service, verified=True
    )
    index.add(document)
    return document
