"""Audit trail: every query's full lifecycle is persisted for accountability.

A record captures the query, the retrieved sources, the model draft, the human
decision, the final answer, the cost/latency, and timestamps. This is the
auditability requirement for a human-in-the-loop production system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AuditRecord:
    thread_id: str
    query: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    top_score: float = 0.0
    draft: str | None = None
    decision: str | None = None  # approve|edit|reject|abstain
    final_answer: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    created_at: str = field(default_factory=_now)
    decided_at: str | None = None


class AuditRepository(Protocol):
    def save(self, record: AuditRecord) -> None: ...
    def get(self, thread_id: str) -> AuditRecord | None: ...


class InMemoryAuditRepository:
    """Used by tests and local runs."""

    def __init__(self) -> None:
        self._store: dict[str, AuditRecord] = {}

    def save(self, record: AuditRecord) -> None:
        self._store[record.thread_id] = record

    def get(self, thread_id: str) -> AuditRecord | None:
        return self._store.get(thread_id)


class PostgresAuditRepository:
    """Persists audit records to Postgres (JSONB), keyed by thread_id."""

    def __init__(self, dsn: str, table: str = "prag_audit") -> None:
        self._dsn = dsn
        self._table = table
        self._ready = False

    def _conn(self):  # type: ignore[no-untyped-def]
        import psycopg

        return psycopg.connect(self._dsn)

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._conn() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"thread_id TEXT PRIMARY KEY, record JSONB NOT NULL, "
                f"updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.commit()
        self._ready = True

    def save(self, record: AuditRecord) -> None:
        self._ensure()
        payload = json.dumps(asdict(record))
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO {self._table} (thread_id, record) VALUES (%s, %s) "
                f"ON CONFLICT (thread_id) DO UPDATE SET record = EXCLUDED.record, "
                f"updated_at = now()",
                (record.thread_id, payload),
            )
            conn.commit()

    def get(self, thread_id: str) -> AuditRecord | None:
        self._ensure()
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT record FROM {self._table} WHERE thread_id = %s", (thread_id,)
            ).fetchone()
        if not row:
            return None
        data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return AuditRecord(**data)
