"""HTTP routes. The engine is resolved from app state."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from prag.api.schemas import (
    ApprovalDecision,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from prag.engine import Engine

router = APIRouter()


def _engine(request: Request) -> Engine:
    return request.app.state.engine


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    thread_id = req.thread_id or uuid.uuid4().hex
    result = _engine(request).query(req.query, thread_id)
    return QueryResponse(
        thread_id=result.thread_id,
        status=result.status,
        answer=result.answer,
        draft=result.draft,
        sources=result.sources,
        confidence=result.confidence,
    )


@router.get("/approvals/{thread_id}", response_model=QueryResponse)
def get_approval(thread_id: str, request: Request) -> QueryResponse:
    """Return the pending draft for a paused thread (for the reviewer UI)."""
    engine = _engine(request)
    record = engine.audit.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown thread_id")
    return QueryResponse(
        thread_id=thread_id,
        status="pending_approval" if record.decision is None else "answered",
        answer=record.final_answer,
        draft=record.draft,
        confidence=record.top_score,
    )


@router.post("/approvals/{thread_id}", response_model=QueryResponse)
def submit_approval(thread_id: str, decision: ApprovalDecision, request: Request) -> QueryResponse:
    if decision.action == "edit" and not decision.text:
        raise HTTPException(status_code=422, detail="text is required when action=edit")
    result = _engine(request).resume(thread_id, decision.action, decision.text)
    return QueryResponse(
        thread_id=result.thread_id,
        status=result.status,
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request, path: str = "data/sample") -> IngestResponse:
    count = _engine(request).ingest(path)
    return IngestResponse(ingested_chunks=count)
