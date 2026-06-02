# Production RAG + Human-in-the-Loop Agent

A production-grade retrieval-augmented generation (RAG) system for industrial document intelligence, with stateful human-in-the-loop approval gates, full auditability, and observability. Built with **LangGraph** for orchestration, **pgvector** for semantic search, and **Prometheus/Grafana** for metrics.

## Why This Matters

RAG systems fail in production when:
1. **No audit trail**: You can't prove why the system said what it said.
2. **No human oversight**: Hallucinations ship unchecked.
3. **No visibility**: You don't know cost or latency until it's too late.

This repo addresses all three. Every query is persisted, pauseable for human review, and instrumented with cost/latency metrics.

## Architecture

```mermaid
graph LR
    User["User Query"] --> API["FastAPI /query"]
    API --> Retrieve["Retrieve Node"]
    Retrieve --> Score{Score ≥<br/>min_score?}
    Score -->|No| Abstain["Abstain<br/>(refuse to guess)"]
    Score -->|Yes| Draft["Draft Answer<br/>(LLM)"]
    Draft --> Ground["Grounding Check<br/>(word overlap)"]
    Ground -->|Grounded| HITL["HITL Approval<br/>(interrupt)"]
    Ground -->|Not Grounded| Abstain
    HITL -->|API /approvals POST| Review["Human Reviewer"]
    Review -->|approve|edit|reject| Finalize["Finalize Response"]
    Finalize --> Respond["Return Answer"]
    Abstain --> Respond
    Respond --> Metrics["Prometheus Metrics<br/>(cost, latency, tokens)"]
    Metrics --> Grafana["Grafana Dashboards"]
    Respond --> Audit["Postgres Audit Log<br/>(full record)"]
```

## Key Features

- **Stateful HITL**: Graph pauses at approval, survives restarts via LangGraph checkpointer.
- **Grounding**: Drafts are checked to ensure they're grounded in retrieved context before review.
- **Abstention**: Low-confidence queries refuse to guess; high-confidence threshold is configurable.
- **Audit trail**: Every decision (approve/edit/reject) + cost/tokens persisted to Postgres.
- **Cost accounting**: Per-node cost estimates on all LLM calls; aggregate to dashboard.
- **Provider-agnostic**: Swap OpenAI ↔ Mistral ↔ self-hosted vLLM via config alone.
- **EU-friendly**: Built with Haystack + Mistral options; production-ready for EU deployments.

## Quick Start

### 1. Install & Setup

```bash
git clone https://github.com/ejazfahil/Production_RAG_HumanLoop_Agent.git
cd Production_RAG_HumanLoop_Agent
make setup
```

### 2. Ingest Sample Documents

The repo includes sample meter specs and maintenance manuals.

```bash
make ingest
```

### 3. Run Tests

```bash
make test
```

### 4. Start the API

```bash
make run
```

The API runs on `http://localhost:8000`.

### 5. Query Locally

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the accuracy class of the X200?", "thread_id": "user-1"}'
```

Response (query pauses for approval):

```json
{
  "thread_id": "user-1",
  "status": "pending_approval",
  "draft": "[meter_spec_acme_x200.md#0] Class 0.5S (active) ±0.5%",
  "sources": [
    {"source": "meter_spec_acme_x200.md", "ordinal": 0, "score": 0.58}
  ],
  "confidence": 0.58
}
```

### 6. Approve or Edit

```bash
curl -X POST http://localhost:8000/approvals/user-1 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

Final response:

```json
{
  "thread_id": "user-1",
  "status": "answered",
  "answer": "[meter_spec_acme_x200.md#0] Class 0.5S (active) ±0.5%",
  "sources": [
    {"source": "meter_spec_acme_x200.md", "ordinal": 0, "score": 0.58}
  ]
}
```

### 7. View Metrics

```bash
curl http://localhost:8000/metrics | grep prag_
```

Exposed metrics:

- `prag_queries_total{outcome="answered|abstained|rejected"}`
- `prag_cost_usd_total{model="..."}`
- `prag_tokens_total{model="...", direction="input|output"}`
- `prag_node_latency_seconds{node="retrieve|draft|grounding_check|hitl_approval|finalize"}`
- `prag_query_latency_seconds` (end-to-end, excluding HITL pause time)
- `prag_approvals_total{decision="approve|edit|reject"}`

## Configuration

All settings are environment variables or `.env` file. See `.env.example`:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Key Settings

| Var | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `fake` | `fake`, `openai`, `anthropic`, `mistral` |
| `LLM_MODEL` | `fake-small` | E.g., `gpt-4o-mini`, `claude-3-5-sonnet` |
| `LLM_API_KEY` | (none) | Required if not using `fake` provider |
| `EMBEDDING_PROVIDER` | `fake` | `fake`, `openai`, `mistral` |
| `VECTOR_BACKEND` | `memory` | `memory` (dev), `pgvector` (prod) |
| `CHECKPOINTER` | `memory` | `memory`, `sqlite`, `postgres` (for durable HITL) |
| `MIN_RETRIEVAL_SCORE` | `0.25` | Abstain if top hit scores below this |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `AUTO_APPROVE` | `false` | Auto-approve (CI/smoke only) |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `DATABASE_URL` | postgres://... | Postgres for pgvector + audit |

## Docker & Postgres

For production, use `docker-compose` to spin up Postgres + the API:

```bash
docker-compose up -d
```

This starts:
- **postgres:16-alpine** with pgvector extension
- **prag:latest** FastAPI service on port 8000
- **pgadmin** (optional) on port 5050

The environment in `docker-compose.yml` defaults to pgvector + postgres checkpointer (durable).

### Build the Docker Image

```bash
make docker-build
```

## Production Notes

### Cost Tracking

Cost is computed from token counts and the per-model pricing table in `src/prag/llm/base.py`. Update the `PRICING` dict as vendor rates change. In production, export `prag_cost_usd_total` to your billing system.

### Grounding Heuristic

The grounding check is a word-overlap proxy. For more rigorous entailment checking, replace `_groundedness()` in `src/prag/agent/nodes.py` with an LLM-as-judge call.

### Retrieval Quality

With the `fake` embedder (offline), retrieval is weak bag-of-words. Real deployments use `embedding_provider=openai` or `mistral` — quality jumps significantly. The architecture is agnostic; no code changes needed.

### Multi-Tenant HITL at Scale

For thousands of concurrent paused threads, use the `postgres` checkpointer (not `memory`) and ensure the database is scaled. Each paused graph thread is one Postgres row; handle cardinality accordingly.

### EU AI Act Compliance

An example compliance report template is generated by `eval/` (if enabled). Document:
- Metric definitions (faithfulness, answer relevancy, etc.)
- Test dataset characteristics
- Known limitations (e.g., grounding is word-overlap, not entailment)
- Human oversight process (HITL workflow)
- Cost and latency SLOs

See `docs/eval-report-template.md` for the structure.

## Testing

```bash
# Unit + integration + smoke tests
make test

# Coverage report
make test-cov

# Lint + format
make lint
make format

# Type-check
make type
```

## Architecture Decision Records

See `docs/adr/` for design rationale:

- **ADR-0001**: Why LangGraph for stateful HITL (durable execution, native interrupts)
- **ADR-0002**: Why pgvector as the default retrieval backend
- **ADR-0003**: Provider-agnostic LLM layer rationale

## Extending

### Add a Real LLM Provider

1. Implement a class matching the `LLM` protocol in `src/prag/llm/base.py`.
2. Register it in the factory (`src/prag/llm/factory.py`).
3. Add pricing to `PRICING` dict.
4. Update config validation in `src/prag/config.py`.

### Swap the Vector Store

1. Implement a class matching the `VectorStore` protocol in `src/prag/retrieval/store.py`.
2. Wire it in the engine (`src/prag/engine.py`).
3. No changes to agent or API needed.

### Custom Grounding Logic

Replace `_groundedness()` in `src/prag/agent/nodes.py` with your own. Examples:

- **LLM-as-judge**: Call the LLM with a strict rubric.
- **NLI model**: Use a small entailment model (e.g., Roberta-NLI).
- **Embedding distance**: Ensure draft embedding is close to context.

### Add Metrics

Use `prometheus_client` in any node. New metrics automatically expose on `/metrics`.

## License

MIT

## Support & Contributing

This is a portfolio project demonstrating production-grade ML systems. For questions, open an issue or reach out to [@ejazfahil](https://github.com/ejazfahil).

---

**Built with ❤️ for EU industrial AI.**
