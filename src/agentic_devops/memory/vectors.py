"""Dense knowledge index backed by Voyage embeddings.

Optional: ``pip install '.[voyage]'`` and set ``VOYAGE_API_KEY``. Anthropic has
no embeddings endpoint, so the article's ``AnthropicEmbeddings`` had nothing
behind it; Voyage is the partner Anthropic points at.

Vectors live in the same SQLite file as everything else and similarity is
computed in Python. That is the right trade at this scale — an organisation's
postmortems number in the thousands, not the millions, and a brute-force scan
over a few thousand vectors costs less than the network round trip that
produced the query embedding. Swap in a vector database when the corpus stops
fitting in memory, not before.
"""

from __future__ import annotations

import array
import json
import logging
import math
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .knowledge import Document, Kind, SearchHit

log = logging.getLogger("devops-agent.memory")

DEFAULT_MODEL = "voyage-3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    service     TEXT,
    thread_id   TEXT,
    verified    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB NOT NULL
);
"""


def _pack(values: Sequence[float]) -> bytes:
    return array.array("f", values).tobytes()


def _unpack(blob: bytes) -> array.array:
    vector = array.array("f")
    vector.frombytes(blob)
    return vector


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class VoyageEmbedder:
    """Thin wrapper so tests can substitute a deterministic embedder."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        import voyageai  # imported lazily: optional dependency

        self.model = model
        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        response = self._client.embed(
            list(texts), model=self.model, input_type="query" if is_query else "document"
        )
        return response.embeddings


class VectorKnowledgeIndex:
    name = "vector"

    def __init__(self, path: str | Path = "knowledge.db", *, embedder: Any = None) -> None:
        self.path = str(path)
        self.embedder = embedder or VoyageEmbedder()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, document: Document) -> Document:
        text = f"{document.title}\n{document.body}"
        embedding = self.embedder.embed([text])[0]
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO vectors (
                    id, kind, title, body, service, thread_id, verified,
                    created_at, metadata, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title, body = excluded.body,
                    service = excluded.service, verified = excluded.verified,
                    metadata = excluded.metadata, embedding = excluded.embedding
                """,
                (
                    document.id,
                    document.kind.value,
                    document.title,
                    document.body,
                    document.service,
                    document.thread_id,
                    int(document.verified),
                    document.created_at.isoformat(),
                    json.dumps(document.metadata),
                    _pack(embedding),
                ),
            )
        return document

    def _row(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            kind=Kind(row["kind"]),
            title=row["title"],
            body=row["body"],
            service=row["service"],
            thread_id=row["thread_id"],
            verified=bool(row["verified"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: Sequence[Kind] | None = None,
        service: str | None = None,
    ) -> list[SearchHit]:
        sql = "SELECT * FROM vectors WHERE 1=1"
        params: list[Any] = []
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kind.value for kind in kinds)
        if service:
            sql += " AND service = ?"
            params.append(service)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        target = self.embedder.embed([query], is_query=True)[0]
        scored = [
            (cosine(target, _unpack(row["embedding"])), row) for row in rows
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchHit(document=self._row(row), score=score, index=self.name)
            for score, row in scored[:limit]
        ]

    def get(self, document_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM vectors WHERE id = ?", (document_id,)
            ).fetchone()
        return self._row(row) if row else None

    def delete(self, document_id: str) -> bool:
        with self._lock:
            return bool(
                self._conn.execute(
                    "DELETE FROM vectors WHERE id = ?", (document_id,)
                ).rowcount
            )

    def all(self, *, limit: int = 100) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM vectors ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]
