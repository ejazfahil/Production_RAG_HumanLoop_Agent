# Architecture Notes — 2026-06-09

## Components
1. **Ingestion**: PDF/DOCX → chunks → embeddings → pgvector
2. **Retrieval**: cosine similarity, HNSW index, top-k=5
3. **Generation**: gpt-4o-mini, temperature=0
4. **HITL Gate**: human approval required before answer release
5. **Observability**: Prometheus metrics → Grafana dashboard

## Key Design Decisions
- LangGraph for stateful, resumable agent
- pgvector over Pinecone (self-hosted, no vendor lock-in)
- HITL timeout: 300s → auto-reject

# ts:2026-06-09T17:45:00
