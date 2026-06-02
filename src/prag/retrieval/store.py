"""Vector store: an in-memory backend for dev/CI and a pgvector backend for prod.

Both expose the same `add`/`search` contract so the agent and ingestion code are
storage-agnostic. Scores are cosine similarity in [-1, 1]; for normalized vectors
they land in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from prag.llm.base import Embedder


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class Retrieved:
    document: Document
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore(Protocol):
    def add(self, docs: list[Document]) -> int: ...
    def search(self, query: str, *, top_k: int) -> list[Retrieved]: ...


@dataclass
class InMemoryVectorStore:
    """Brute-force cosine search. Fine for dev, tests and small corpora."""

    embedder: Embedder
    _docs: list[Document] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vectors = self.embedder.embed([d.text for d in docs])
        self._docs.extend(docs)
        self._vectors.extend(vectors)
        return len(docs)

    def search(self, query: str, *, top_k: int) -> list[Retrieved]:
        if not self._docs:
            return []
        qvec = self.embedder.embed([query])[0]
        scored = [
            Retrieved(document=doc, score=_cosine(qvec, vec))
            for doc, vec in zip(self._docs, self._vectors, strict=True)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


@dataclass
class PgVectorStore:
    """Postgres + pgvector backend. Lazily creates the table/extension on first use."""

    embedder: Embedder
    dsn: str
    table: str = "prag_documents"
    _ready: bool = False

    def _conn(self):  # type: ignore[no-untyped-def]
        import psycopg

        return psycopg.connect(self.dsn)

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        from pgvector.psycopg import register_vector

        with self._conn() as conn:
            register_vector(conn)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ordinal INT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({self.embedder.dim})
                )
                """
            )
            conn.commit()
        self._ready = True

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        self._ensure_schema()
        from pgvector.psycopg import register_vector

        vectors = self.embedder.embed([d.text for d in docs])
        with self._conn() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} (doc_id, source, ordinal, text, embedding) "
                    f"VALUES (%s, %s, %s, %s, %s)",
                    [
                        (d.doc_id, d.source, d.ordinal, d.text, vec)
                        for d, vec in zip(docs, vectors, strict=True)
                    ],
                )
            conn.commit()
        return len(docs)

    def search(self, query: str, *, top_k: int) -> list[Retrieved]:
        self._ensure_schema()
        from pgvector.psycopg import register_vector

        qvec = self.embedder.embed([query])[0]
        with self._conn() as conn:
            register_vector(conn)
            rows = conn.execute(
                f"SELECT doc_id, source, ordinal, text, "
                f"1 - (embedding <=> %s) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                (qvec, qvec, top_k),
            ).fetchall()
        return [
            Retrieved(
                document=Document(doc_id=r[0], source=r[1], ordinal=r[2], text=r[3]),
                score=float(r[4]),
            )
            for r in rows
        ]
