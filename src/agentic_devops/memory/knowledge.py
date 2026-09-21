"""Semantic memory: what the organisation knows about its own infrastructure.

The article reached straight for Chroma and ``AnthropicEmbeddings`` — a class
that does not exist, because Anthropic does not ship an embeddings API. Rather
than swap in one vendor, retrieval is a protocol here with three
implementations: lexical (SQLite FTS5, no dependencies, no keys, works
offline), dense (Voyage), and a fusion of both.

Lexical is the default on purpose. Infrastructure recall is unusually
keyword-shaped — `payments-api`, `CrashLoopBackOff`, `OOMKilled`, `5xx` are
exact tokens, and BM25 matches them precisely where an embedding blurs them
into neighbours. Dense retrieval earns its keep on the other half of the
problem ("pods keep restarting" finding a CrashLoopBackOff postmortem), which
is what :class:`HybridIndex` is for.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Kind(str, Enum):
    """What a document is, which decides how much weight it deserves."""

    POSTMORTEM = "postmortem"
    RUNBOOK = "runbook"
    SERVICE = "service"
    PATTERN = "pattern"
    #: An agent's own account of a past run. Written automatically, so never
    #: trusted the way a human-written postmortem is.
    RUN = "run"


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    kind: Kind
    title: str
    body: str
    service: str | None = None
    thread_id: str | None = None
    #: True only when a human has confirmed it. Agent-written documents start
    #: false and say so at retrieval time — an unverified diagnosis recalled
    #: as fact is how one wrong conclusion becomes the house opinion.
    verified: bool = False
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(kind: Kind, title: str, body: str) -> str:
        digest = hashlib.sha256(f"{kind.value}:{title}:{body}".encode()).hexdigest()
        return digest[:16]

    @classmethod
    def new(
        cls,
        *,
        kind: Kind,
        title: str,
        body: str,
        service: str | None = None,
        thread_id: str | None = None,
        verified: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "Document":
        return cls(
            id=cls.make_id(kind, title, body),
            kind=kind,
            title=title,
            body=body,
            service=service,
            thread_id=thread_id,
            verified=verified,
            metadata=metadata or {},
        )

    def render(self) -> str:
        """How the document appears to the model.

        The provenance line is not decoration: the agent behaves differently
        when it knows it is reading its own unconfirmed guess from last Tuesday
        rather than a postmortem a human signed off.
        """
        trust = (
            "verified by a human"
            if self.verified
            else "UNVERIFIED — an agent's own account, treat as a lead, not a fact"
        )
        header = f"[{self.kind.value}] {self.title}"
        if self.service:
            header += f"  (service: {self.service})"
        return (
            f"{header}\n"
            f"recorded {self.created_at.date().isoformat()} · {trust}\n"
            f"{self.body}"
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: Document
    score: float
    index: str


@runtime_checkable
class KnowledgeIndex(Protocol):
    """Storage and retrieval for infrastructure knowledge."""

    name: str

    def add(self, document: Document) -> Document: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: Sequence[Kind] | None = None,
        service: str | None = None,
    ) -> list[SearchHit]: ...

    def get(self, document_id: str) -> Document | None: ...

    def delete(self, document_id: str) -> bool: ...

    def all(self, *, limit: int = 100) -> list[Document]: ...


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")


def query_terms(query: str, *, max_terms: int = 24) -> list[str]:
    """Extract searchable terms from free text.

    FTS5's MATCH grammar treats quotes, colons, hyphens, ``NOT`` and ``*`` as
    syntax, so a raw incident title like ``payments-api: 5xx (NOT resolved)``
    is a syntax error rather than a query. Terms are extracted and re-quoted
    individually.
    """
    seen: list[str] = []
    for match in _WORD.finditer(query):
        term = match.group(0).strip(".-").lower()
        if len(term) < 2 or term in seen:
            continue
        seen.append(term)
        if len(seen) >= max_terms:
            break
    return seen


@dataclass
class HybridIndex:
    """Reciprocal rank fusion over several indexes.

    RRF rather than score averaging because BM25 scores and cosine
    similarities are not on a comparable scale — mixing them numerically lets
    whichever index happens to produce larger magnitudes decide every result.
    Ranks are comparable; scores are not.
    """

    indexes: Sequence[KnowledgeIndex]
    name: str = "hybrid"
    #: The usual RRF constant: damps the influence of top ranks enough that a
    #: document both indexes like beats one that only one index loves.
    k: int = 60

    def add(self, document: Document) -> Document:
        for index in self.indexes:
            index.add(document)
        return document

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: Sequence[Kind] | None = None,
        service: str | None = None,
    ) -> list[SearchHit]:
        fused: dict[str, float] = {}
        documents: dict[str, Document] = {}
        for index in self.indexes:
            hits = index.search(query, limit=limit * 3, kinds=kinds, service=service)
            for rank, hit in enumerate(hits, start=1):
                fused[hit.document.id] = fused.get(hit.document.id, 0.0) + 1.0 / (
                    self.k + rank
                )
                documents[hit.document.id] = hit.document

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchHit(document=documents[doc_id], score=score, index=self.name)
            for doc_id, score in ordered[:limit]
        ]

    def get(self, document_id: str) -> Document | None:
        for index in self.indexes:
            found = index.get(document_id)
            if found is not None:
                return found
        return None

    def delete(self, document_id: str) -> bool:
        return any([index.delete(document_id) for index in self.indexes])

    def all(self, *, limit: int = 100) -> list[Document]:
        return self.indexes[0].all(limit=limit) if self.indexes else []


def render_hits(hits: Iterable[SearchHit]) -> str:
    """Retrieved context, as the model sees it."""
    rendered = [hit.document.render() for hit in hits]
    if not rendered:
        return "No relevant prior incidents or documentation found."
    return "\n\n---\n\n".join(rendered)
