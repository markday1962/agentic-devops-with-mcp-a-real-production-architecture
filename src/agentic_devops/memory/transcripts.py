"""Episodic memory: what happened, turn by turn, and how to pick it back up.

The article used LangGraph's ``PostgresSaver`` for this. The job is the same —
a run that dies mid-incident should not start over from nothing — but the
interesting part is not the storage, it is what a half-finished conversation
looks like when you reload it.

An interrupted run's last message is usually an assistant turn containing
``tool_use`` blocks whose results were never appended. Sending that back is a
400: every ``tool_use`` must be answered. :func:`prepare_resume` trims back to
the last complete exchange and says what it dropped.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("devops-agent.memory")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    thread_id   TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    service     TEXT,
    kind        TEXT,
    messages    TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL,
    summary     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_service ON transcripts (service, updated_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_status ON transcripts (status, updated_at);
"""


def serialize_block(block: Any) -> Any:
    """Turn one content block into something JSON can hold and the API accepts.

    ``model_dump(mode="json")`` is the important branch: it round-trips SDK
    blocks — including a thinking block's ``signature``, without which a
    resumed turn is rejected.
    """
    if isinstance(block, (str, int, float, bool)) or block is None:
        return block
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True, mode="json")
    if is_dataclass(block):
        return asdict(block)
    return str(block)


def serialize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [serialize_block(block) for block in content]
        else:
            content = serialize_block(content)
        out.append({"role": message["role"], "content": content})
    return out


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def prepare_resume(
    messages: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Trim a checkpoint back to a state the API will accept.

    Returns the usable prefix and a note describing what was dropped, so the
    agent can be told it was interrupted rather than silently losing a step it
    believes it took.
    """
    trimmed = list(messages)
    dropped: list[str] = []

    while trimmed:
        last = trimmed[-1]
        if last.get("role") != "assistant":
            break
        pending = [
            block.get("name", "a tool")
            for block in _blocks(last)
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not pending:
            break
        dropped.extend(pending)
        trimmed.pop()

    # A conversation cannot end on a user turn we are about to answer twice,
    # but it also cannot be empty or start with an assistant turn.
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)

    if not dropped:
        return trimmed, None
    return trimmed, (
        "This run was interrupted. You had called "
        + ", ".join(f"`{name}`" for name in dropped)
        + " but the process stopped before the result came back, so those calls "
        "may or may not have taken effect — check current state before "
        "assuming either way, and re-request approval for anything that writes."
    )


@dataclass
class Checkpoint:
    thread_id: str
    goal: str
    messages: list[dict[str, Any]]
    turns: int
    status: str
    service: str | None = None
    kind: str | None = None
    summary: str | None = None
    created_at: datetime = None  # type: ignore[assignment]
    updated_at: datetime = None  # type: ignore[assignment]


class TranscriptStore:
    """Durable conversation state, one row per run."""

    def __init__(self, path: str | Path = "transcripts.db") -> None:
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

    def save(
        self,
        *,
        thread_id: str,
        goal: str,
        messages: Sequence[dict[str, Any]],
        turns: int,
        status: str,
        service: str | None = None,
        kind: str | None = None,
        summary: str | None = None,
    ) -> None:
        now = utcnow().isoformat()
        payload = json.dumps(serialize_messages(messages))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO transcripts (
                    thread_id, goal, service, kind, messages, turns, status,
                    summary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages = excluded.messages,
                    turns = excluded.turns,
                    status = excluded.status,
                    summary = excluded.summary,
                    service = COALESCE(excluded.service, transcripts.service),
                    updated_at = excluded.updated_at
                """,
                (thread_id, goal, service, kind, payload, turns, status, summary, now, now),
            )

    def _row(self, row: sqlite3.Row) -> Checkpoint:
        return Checkpoint(
            thread_id=row["thread_id"],
            goal=row["goal"],
            messages=json.loads(row["messages"]),
            turns=row["turns"],
            status=row["status"],
            service=row["service"],
            kind=row["kind"],
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def load(self, thread_id: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM transcripts WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return self._row(row) if row else None

    def recent(
        self, *, service: str | None = None, status: str | None = None, limit: int = 20
    ) -> list[Checkpoint]:
        sql = "SELECT * FROM transcripts WHERE 1=1"
        params: list[Any] = []
        if service:
            sql += " AND service = ?"
            params.append(service)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row(row) for row in rows]

    def resumable(self, limit: int = 20) -> list[Checkpoint]:
        """Runs that were in flight when a process died."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM transcripts WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                ("running", limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            return bool(
                self._conn.execute(
                    "DELETE FROM transcripts WHERE thread_id = ?", (thread_id,)
                ).rowcount
            )

    def purge(self, older_than: timedelta) -> int:
        """Transcripts hold log excerpts, hostnames and occasionally secrets
        that leaked into a stack trace. They are not kept forever."""
        cutoff = (utcnow() - older_than).isoformat()
        with self._lock:
            return self._conn.execute(
                "DELETE FROM transcripts WHERE updated_at < ?", (cutoff,)
            ).rowcount
