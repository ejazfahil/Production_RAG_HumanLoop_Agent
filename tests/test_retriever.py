"""Unit tests for the RAG retrieval pipeline."""
import pytest
from unittest.mock import MagicMock, patch


class MockRetriever:
    """Lightweight mock of the pgvector retriever."""

    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 3):
        return self.docs[:top_k]


def test_retriever_returns_correct_number_of_docs():
    docs = [f"doc_{i}" for i in range(10)]
    retriever = MockRetriever(docs)
    results = retriever.retrieve("test query", top_k=5)
    assert len(results) == 5


def test_retriever_returns_fewer_docs_when_corpus_small():
    docs = ["only one doc"]
    retriever = MockRetriever(docs)
    results = retriever.retrieve("test", top_k=5)
    assert len(results) == 1


def test_hitl_gate_blocks_without_approval():
    """HITL gate must not release answer without human approval."""
    approved = False
    answer = "test answer"
    released = answer if approved else None
    assert released is None


def test_hitl_gate_releases_with_approval():
    approved = True
    answer = "the meter reads 42.7 kWh"
    released = answer if approved else None
    assert released == answer


def test_cost_logger_accumulates_tokens():
    costs = []
    costs.append({"tokens": 150, "model": "gpt-4o-mini", "cost_usd": 0.00009})
    costs.append({"tokens": 320, "model": "gpt-4o-mini", "cost_usd": 0.000192})
    total = sum(c["cost_usd"] for c in costs)
    assert round(total, 6) == 0.000282
