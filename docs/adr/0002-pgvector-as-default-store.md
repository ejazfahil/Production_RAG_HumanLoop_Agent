# ADR-0002: pgvector as the Production Vector Store Backend

**Date**: June 2026  
**Status**: Accepted

## Context

We need vector similarity search for retrieval. The choices are:

1. **Dedicated vector DBs** (Weaviate, Pinecone, Qdrant): Purpose-built, expensive, operational overhead.
2. **In-memory (numpy)**: Fast locally, doesn't scale, no persistence.
3. **pgvector** (Postgres extension): Postgres table + cosine/L2 distance, embedded within the database.

We already use Postgres for audit tables and checkpoints, so the operational footprint is unchanged.

## Decision

**Use pgvector** (`postgres pgvector` extension) as the production retrieval backend.

### Rationale

1. **No new infrastructure**: Postgres is already running for audit + checkpointing.
2. **Cost-effective**: No per-query fees; you pay only for the database server.
3. **Semantic search**: Supports cosine similarity + L2 distance at scale.
4. **Transactions**: Audit + documents + vectors in the same ACID context.
5. **Simplicity**: No API gateway or external queue; Postgres is your entire data layer.

### Dev/Test

Use `InMemoryVectorStore` (in `src/prag/retrieval/store.py`) for local dev and CI. No database needed.

## Consequences

### Positive

- **Consolidated data layer**: Audit, state, and vectors all in Postgres.
- **No vendor lock-in**: pgvector is open-source; migration to Weaviate/Qdrant is straightforward if needed.
- **Cost visibility**: Compute cost is database server cost; easy to forecast.
- **Exact recall**: No approximation like HNSW; for small-to-medium corpora (millions of vectors), exact search is fast enough.

### Negative

- **Latency for huge corpora**: pgvector exact search is O(n) after index. For >100M vectors, HNSW approximation (Qdrant/Weaviate) would be faster.
- **Postgres scaling**: pgvector performance is bound by Postgres instance size; you scale vertically, not horizontally.
- **DBA maintenance**: Vacuum, reindex, and WAL configuration become responsibility (vs a managed vector DB).

## Alternatives Considered

### 1. Pinecone (managed)
- **Why rejected**: Per-query cost adds up; vendor lock-in; overkill for industrial docs (typically <1M vectors).

### 2. Weaviate
- **Why rejected**: Extra Kubernetes service to operate; adds operational complexity for a single-node system.

### 3. Embedding cache (Redis)
- **Why rejected**: Adds latency; cache invalidation hard; pgvector is simpler and more durable.

## Implementation Notes

- Use `pgvector.psycopg` for embedding type safety in Python.
- Create a GIN or HNSW index on the embedding column for large corpora:
  ```sql
  CREATE INDEX ON prag_documents USING ivfflat (embedding vector_cosine_ops);
  ```
- Monitor table bloat with `VACUUM ANALYZE prag_documents` in maintenance windows.
- Document the embedding dimension in config (`embedding_dim`); Postgres schema must match.

## Migration Path

If you hit pgvector scaling limits:
1. Implement a `WeaviateVectorStore` class matching the `VectorStore` protocol.
2. Swap in config: `vector_backend=weaviate`.
3. No changes to agent or API code.
