"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str | None = Field(
        default=None, description="Reuse to correlate; auto-generated if omitted."
    )


class SourceRef(BaseModel):
    source: str
    ordinal: int
    score: float


class QueryResponse(BaseModel):
    thread_id: str
    status: Literal["answered", "abstained", "rejected", "pending_approval"]
    answer: str | None = None
    draft: str | None = None
    sources: list[SourceRef] = []
    confidence: float = 0.0


class ApprovalDecision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    text: str | None = Field(default=None, description="Required edited answer when action=edit.")


class IngestResponse(BaseModel):
    ingested_chunks: int
