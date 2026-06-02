"""Engine: the single wiring point shared by the API and the CLI.

It owns the dependency graph (LLM, store, audit, checkpointer, compiled graph) and
exposes three operations: ingest, query (may pause for approval), and resume.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from langgraph.types import Command

from prag.agent.graph import build_graph, make_checkpointer
from prag.agent.nodes import Nodes
from prag.audit.repository import (
    AuditRecord,
    AuditRepository,
    InMemoryAuditRepository,
    PostgresAuditRepository,
)
from prag.config import Settings, get_settings
from prag.ingestion.pipeline import ingest_path
from prag.llm.factory import build_embedder, build_llm
from prag.observability.logging import get_logger
from prag.observability.metrics import QUERY_LATENCY
from prag.retrieval.store import InMemoryVectorStore, PgVectorStore, VectorStore

log = get_logger("engine")


@dataclass
class QueryResult:
    thread_id: str
    status: str  # answered | abstained | pending_approval
    answer: str | None = None
    draft: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class Engine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._stack = ExitStack()
        self._started = False
        self.store: VectorStore
        self.audit: AuditRepository
        self.graph: Any

    def start(self) -> Engine:
        if self._started:
            return self
        s = self.settings
        embedder = build_embedder(s)
        llm = build_llm(s)

        if s.vector_backend == "pgvector":
            self.store = PgVectorStore(embedder=embedder, dsn=s.database_url)
        else:
            self.store = InMemoryVectorStore(embedder=embedder)

        self.audit = (
            PostgresAuditRepository(s.database_url)
            if s.checkpointer == "postgres"
            else InMemoryAuditRepository()
        )

        nodes = Nodes(llm=llm, store=self.store, settings=s)
        checkpointer = self._stack.enter_context(make_checkpointer(s))
        self.graph = build_graph(nodes, checkpointer)
        self._started = True
        log.info("engine_started", vector_backend=s.vector_backend, checkpointer=s.checkpointer)
        return self

    def close(self) -> None:
        self._stack.close()
        self._started = False

    # --- operations ---
    def ingest(self, path: str) -> int:
        return ingest_path(self.store, self.settings, path)

    def query(self, query: str, thread_id: str) -> QueryResult:
        config = {"configurable": {"thread_id": thread_id}}
        start = time.perf_counter()
        result = self.graph.invoke({"query": query, "thread_id": thread_id}, config)
        elapsed = time.perf_counter() - start

        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            self._persist(thread_id, query, result, decision=None, latency=elapsed)
            return QueryResult(
                thread_id=thread_id,
                status="pending_approval",
                draft=payload.get("draft"),
                sources=payload.get("sources", []),
                confidence=payload.get("confidence", 0.0),
            )

        QUERY_LATENCY.observe(elapsed)
        self._persist(thread_id, query, result, decision=result.get("decision"), latency=elapsed)
        return self._terminal_result(thread_id, result)

    def resume(self, thread_id: str, action: str, text: str | None = None) -> QueryResult:
        config = {"configurable": {"thread_id": thread_id}}
        start = time.perf_counter()
        resume_value: dict[str, Any] = {"action": action}
        if text is not None:
            resume_value["text"] = text
        result = self.graph.invoke(Command(resume=resume_value), config)
        elapsed = time.perf_counter() - start
        QUERY_LATENCY.observe(elapsed)

        prior = self.audit.get(thread_id)
        query = prior.query if prior else ""
        self._persist(thread_id, query, result, decision=result.get("decision"), latency=elapsed)
        return self._terminal_result(thread_id, result)

    # --- helpers ---
    def _terminal_result(self, thread_id: str, state: dict[str, Any]) -> QueryResult:
        outcome = state.get("outcome", "answered")
        status = {"answered": "answered", "abstained": "abstained", "rejected": "rejected"}.get(
            outcome, outcome
        )
        return QueryResult(
            thread_id=thread_id,
            status=status,
            answer=state.get("final_answer"),
            sources=[
                {"source": r["source"], "ordinal": r["ordinal"], "score": r["score"]}
                for r in state.get("retrieved", [])
            ],
            confidence=state.get("top_score", 0.0),
        )

    def _persist(
        self,
        thread_id: str,
        query: str,
        state: dict[str, Any],
        *,
        decision: str | None,
        latency: float,
    ) -> None:
        existing = self.audit.get(thread_id)
        record = existing or AuditRecord(thread_id=thread_id, query=query)
        record.query = query or record.query
        record.retrieved_doc_ids = [r["doc_id"] for r in state.get("retrieved", [])]
        record.top_score = state.get("top_score", record.top_score)
        record.draft = state.get("draft", record.draft)
        record.input_tokens = state.get("input_tokens", record.input_tokens)
        record.output_tokens = state.get("output_tokens", record.output_tokens)
        record.cost_usd = state.get("cost_usd", record.cost_usd)
        record.latency_seconds = round(record.latency_seconds + latency, 4)
        if decision is not None:
            record.decision = decision
            record.final_answer = state.get("final_answer")
            from datetime import datetime

            record.decided_at = datetime.now(UTC).isoformat()
        self.audit.save(record)
