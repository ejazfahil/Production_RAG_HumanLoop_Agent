# Production RAG + Human-in-the-Loop Agent

A production-grade Retrieval-Augmented Generation (RAG) system for industrial document intelligence, featuring stateful human-in-the-loop approval gates, complete auditability, and extensive observability. Built with **LangGraph** for orchestration, **pgvector** for semantic search, and **Prometheus/Grafana** for metrics and telemetry.

---

## 🎯 Aim

The primary objective of this project is to bridge the gap between experimental RAG setups and reliable production environments. In high-stakes industrial, legal, or medical domains, fully autonomous AI generation is a significant liability due to the risk of hallucinations. This system aims to enforce a strict validation framework where AI drafts are thoroughly grounded in trusted references and verified by human subject-matter experts before publication.

---

## 📄 Context

Standard RAG architectures often suffer from critical operational issues:
1. **Hallucinations**: Generative LLMs can produce inaccurate or fabricated information that is difficult to detect programmatically.
2. **Missing Audit Trails**: Organizations cannot trace why an answer was generated, which sources were consulted, or who approved the output.
3. **No Operational Visibility**: Teams lack clear insights into operational latencies, token consumption, and actual API usage costs until invoices arrive.

This project addresses these challenges by introducing a stateful, auditable, and highly observable agent. Every step of the query lifecycle is tracked, persisted to a relational database, checked for source groundedness, and subjected to a human approval step.

---

## 🏆 Project Achievements

- **Stateful Human-in-the-Loop (HITL) Gates**: Uses LangGraph's native `interrupt()` and `Command` mechanisms to pause execution graphs mid-run. Paused runs survive system restarts by persisting state to SQLite or production-scale PostgreSQL checkpointers.
- **Strict Grounding Heuristics**: Implements a transparent, configurable grounding verification check on the drafted response against the retrieved source texts before passing the draft to human reviewers.
- **Storage-Agnostic Retrieval Engine**: Features an abstract `VectorStore` protocol supporting a fast in-memory store for development/CI and a scalable **pgvector** database backend for production deployment.
- **Vendor-Agnostic LLM Layer**: Standardized on Python `Protocol` definitions, enabling seamless switching between OpenAI, Anthropic, Mistral, or self-hosted models solely through configuration changes.
- **Production-Ready Observability**: Configured with Prometheus to export detailed real-time metrics, paired with a pre-configured Grafana dashboard for visualizing operational KPIs.
- **Detailed Audit Trail**: Records a granular history of every query, including exact source chunks, token counts, calculated costs in USD, processing latency, and final human review decisions (approve, edit, or reject).
- **Robust Quality Control**: Achieved complete type safety (100% clean mypy checks), formatting compliance (ruff clean), and automated test coverage (passing all 15 tests, 73% coverage).

---

## 📐 Architecture

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

---

## 🚀 Quick Start

### 1. Installation & Environment Setup

Clone the repository and install the development dependencies using the `make` target:

```bash
git clone https://github.com/ejazfahil/Production_RAG_HumanLoop_Agent.git
cd Production_RAG_HumanLoop_Agent
make setup
```

### 2. Ingest Reference Documents

The project includes sample maintenance manuals and technical specifications. To parse and embed these documents:

```bash
make ingest
```

### 3. Run Verification Suite

Ensure type checking, linting, formatting, and unit tests are passing:

```bash
make type    # mypy type checks
make lint    # ruff analysis
make test    # pytest execution
```

### 4. Launch the FastAPI API

Start the local development server:

```bash
make run
```

The API will run on `http://localhost:8000`.

---

## 📡 API Usage Guide

### A. Submitting a Query

To initiate a query through the agent:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the accuracy class of the X200?", "thread_id": "user-1"}'
```

If the response is grounded and has high confidence, the graph pauses at the approval stage and returns the state:

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

### B. Fetching Paused Approvals

To inspect the pending draft for a specific user thread:

```bash
curl http://localhost:8000/approvals/user-1
```

### C. Reviewer Decision (Approve / Edit / Reject)

To approve the draft and finalize the response:

```bash
curl -X POST http://localhost:8000/approvals/user-1 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

To edit the draft and submit a modified version:

```bash
curl -X POST http://localhost:8000/approvals/user-1 \
  -H "Content-Type: application/json" \
  -d '{"action": "edit", "text": "[Revised] Accuracy is ±0.5% active class."}'
```

---

## 📊 Observability Metrics

The application exposes Prometheus-compatible metrics at `http://localhost:8000/metrics`.

Key exported metrics:
- `prag_queries_total{outcome="answered|abstained|rejected"}`: Total query counter.
- `prag_cost_usd_total{model="..."}`: Cumulative API costs.
- `prag_tokens_total{model="...", direction="input|output"}`: Input and output token counts.
- `prag_node_latency_seconds{node="..."}`: Processing duration per graph stage.
- `prag_approvals_total{decision="approve|edit|reject"}`: Record of human reviewer decisions.

---

## ⚙️ Configuration

Environment variables can be configured in a `.env` file (see `.env.example` as a template):

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `fake` | Model provider (`fake`, `openai`, `anthropic`, `mistral`) |
| `LLM_MODEL` | `fake-small` | Model name (e.g., `gpt-4o-mini`, `claude-3-5-sonnet`) |
| `LLM_API_KEY` | (none) | API key for the chosen LLM provider |
| `EMBEDDING_PROVIDER` | `fake` | Embedding model provider (`fake`, `openai`, `mistral`) |
| `VECTOR_BACKEND` | `memory` | Backend engine (`memory` for dev, `pgvector` for prod) |
| `CHECKPOINTER` | `memory` | LangGraph checkpointer (`memory`, `sqlite`, `postgres`) |
| `MIN_RETRIEVAL_SCORE` | `0.25` | Minimum similarity score before abstaining |
| `TOP_K` | `4` | Number of document chunks to retrieve |
| `AUTO_APPROVE` | `false` | Enable automatic approval for integration smoke tests |
| `DATABASE_URL` | (none) | PostgreSQL connection string for pgvector + audit |

---

## 🐳 Docker Deployment

To spin up a fully integrated production stack containing the FastAPI app, Postgres (with pgvector), and pgAdmin:

```bash
# Build the application image
make docker-build

# Start up the environment
make docker-up
```

---

## ⚖️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
