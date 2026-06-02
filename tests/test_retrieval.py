"""Retrieval ranks relevant chunks above irrelevant ones."""

from __future__ import annotations

from prag.llm.fake import FakeEmbedder
from prag.retrieval.store import Document, InMemoryVectorStore


def test_search_ranks_relevant_first() -> None:
    store = InMemoryVectorStore(embedder=FakeEmbedder(dim=256))
    store.add(
        [
            Document("d1", "spec.md", 0, "accuracy class 0.5S active accuracy plus minus 0.5%"),
            Document("d2", "spec.md", 1, "operating temperature range -25 to +70 celsius"),
            Document("d3", "manual.md", 0, "battery replacement lithium backup ten years"),
        ]
    )
    hits = store.search("what is the meter accuracy class", top_k=2)
    assert hits, "expected at least one hit"
    assert hits[0].document.doc_id == "d1"
    # Scores must be sorted descending.
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_empty_store_returns_nothing() -> None:
    store = InMemoryVectorStore(embedder=FakeEmbedder())
    assert store.search("anything", top_k=3) == []
