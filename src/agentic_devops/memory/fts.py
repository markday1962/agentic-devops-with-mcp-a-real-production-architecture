"""Lexical knowledge index on SQLite FTS5.

No service to run, no key to hold, no embedding cost per query, and it works
on a laptop with the network off. For infrastructure text it is also simply
good: BM25 over `CrashLoopBackOff` and `payments-api` beats a dense model that
has never seen either token.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .knowledge import Document, Kind, SearchHit, query_terms

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    service     TEXT,
    thread_id   TEXT,
    verified    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents (kind);
CREATE INDEX IF NOT EXISTS idx_documents_service ON documents (service);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, body, service,
    content='documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts (rowid, title, body, service)
    VALUES (new.rowid, new.title, new.body, new.service);
END;
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts (documents_fts, rowid, title, body, service)
    VALUES ('delete', old.rowid, old.title, old.body, old.service);
END;
CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts (documents_fts, rowid, title, body, service)
    VALUES ('delete', old.rowid, old.title, old.body, old.service);
    INSERT INTO documents_fts (rowid, title, body, service)
    VALUES (new.rowid, new.title, new.body, new.service);
END;
"""


class FTSKnowledgeIndex:
    name = "fts"

    def __init__(self, path: str | Path = "knowledge.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, document: Document) -> Document:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents (
                    id, kind, title, body, service, thread_id, verified, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    service = excluded.service,
                    verified = excluded.verified,
                    metadata = excluded.metadata
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
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)

        sql = [
            "SELECT d.*, bm25(documents_fts, 2.0, 1.0, 1.5) AS score",
            "  FROM documents_fts",
            "  JOIN documents d ON d.rowid = documents_fts.rowid",
            " WHERE documents_fts MATCH ?",
        ]
        params: list[Any] = [match]
        if kinds:
            sql.append(f" AND d.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kind.value for kind in kinds)
        if service:
            sql.append(" AND d.service = ?")
            params.append(service)
        # bm25() returns negative numbers, better matches more negative.
        sql.append(" ORDER BY score LIMIT ?")
        params.append(limit)

        with self._lock:
            rows = self._conn.execute("\n".join(sql), params).fetchall()
        return [
            SearchHit(document=self._row(row), score=-row["score"], index=self.name)
            for row in rows
        ]

    def get(self, document_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._row(row) if row else None

    def delete(self, document_id: str) -> bool:
        with self._lock:
            return bool(
                self._conn.execute(
                    "DELETE FROM documents WHERE id = ?", (document_id,)
                ).rowcount
            )

    def set_verified(self, document_id: str, verified: bool = True) -> bool:
        with self._lock:
            return bool(
                self._conn.execute(
                    "UPDATE documents SET verified = ? WHERE id = ?",
                    (int(verified), document_id),
                ).rowcount
            )

    def all(self, *, limit: int = 100) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]
