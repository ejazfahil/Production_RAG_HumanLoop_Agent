# 🤖 Production RAG + Human-in-the-Loop Agent

[![CI](https://github.com/ejazfahil/Production_RAG_HumanLoop_Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ejazfahil/Production_RAG_HumanLoop_Agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-stateful-green)](https://langchain-ai.github.io/langgraph/)

A **production-grade stateful RAG agent** for industrial/technical document intelligence (utility-meter spec sheets, datasheets, maintenance manuals). Ingests documents, retrieves over them with pgvector, and answers — with a **human approval gate** before any answer is committed, plus full cost + latency logging and Prometheus/Grafana observability.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph State Machine            │
│                                                     │
│  Document ──► Ingest ──► pgvector ──► Retriever     │
│                                           │         │
│                                        LLM Node     │
│                                           │         │
│                                    ┌──── ▼ ──────┐  │
│                                    │  HITL Gate  │  │  ◄── Human approves
│                                    └──── ▼ ──────┘  │
│                                     Answer + Audit  │
└─────────────────────────────────────────────────────┘
           │
    Prometheus ──► Grafana Dashboard
```

## ✨ Features

- 📄 **Document ingestion** — PDF, DOCX, TXT with metadata tagging
- 🔍 **pgvector retrieval** — cosine similarity with HNSW indexing
- 🤖 **Stateful LangGraph agent** — resumable, auditable state
- 🛑 **Human-in-the-Loop gate** — every answer requires human approval before release
- 💰 **Cost + latency logging** — per-query USD cost, token counts, p50/p99 latency
- 📊 **Prometheus + Grafana** — real-time observability out of the box
- 🗂️ **Full audit trail** — every query, retrieval, and approval logged to PostgreSQL

## 🚀 Quickstart

```bash
git clone https://github.com/ejazfahil/Production_RAG_HumanLoop_Agent.git
cd Production_RAG_HumanLoop_Agent
cp .env.example .env  # add your OPENAI_API_KEY and DATABASE_URL
docker-compose up -d  # starts postgres + grafana
pip install -r requirements.txt
python ingest.py --docs ./docs/sample_manual.pdf
python agent.py --query "What is the rated voltage of meter model X42?"
```

### 🔌 Run fully offline with a local LLM (Ollama)

The generation provider is OpenAI-compatible and swappable via a single env var, so
the agent runs with **no external API, no API key, and zero per-token cost** — every
request stays on the machine (useful for GDPR / on-prem document data). Point it at a
host [Ollama](https://ollama.com) server:

```yaml
# docker-compose.yml (app service)
LLM_PROVIDER: openai
LLM_MODEL: qwen3:8b                              # or llama3.2:3b for lower latency
LLM_API_KEY: ollama                              # placeholder; Ollama ignores it
LLM_BASE_URL: http://host.docker.internal:11434/v1
```

Verified end-to-end against `qwen3:8b` from inside the container. Switching back to a
hosted provider is just `LLM_PROVIDER`/`LLM_MODEL`/`LLM_BASE_URL` — no code changes.

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `DATABASE_URL` | PostgreSQL + pgvector URL | required |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | Generation model | `gpt-4o-mini` |
| `HITL_TIMEOUT_SECONDS` | Seconds before auto-reject | `300` |
| `TOP_K_RETRIEVAL` | Documents to retrieve | `5` |

## 💰 Cost Profile

| Operation | Model | Avg tokens | Cost/query |
|-----------|-------|-----------|------------|
| Embedding | text-embedding-3-small | 150 | $0.000015 |
| Generation | gpt-4o-mini | 800 | $0.00048 |
| **Total** | | | **~$0.00050** |

## 📁 Project Structure

```
Production_RAG_HumanLoop_Agent/
├── agent.py              # Main LangGraph agent
├── ingest.py             # Document ingestion pipeline
├── retriever.py          # pgvector retrieval
├── hitl_gate.py          # Human approval gate
├── cost_logger.py        # Token cost + latency logging
├── tests/
│   └── test_retriever.py
├── .github/workflows/ci.yml
├── docker-compose.yml    # Postgres + Grafana
├── prometheus.yml        # Metrics config
└── README.md
```

## 🔭 Roadmap
- [ ] Slack integration for HITL approval notifications
- [ ] Multi-document cross-referencing
- [ ] Streaming responses with partial HITL
- [ ] Fine-tuned embedding model for industrial vocab

## 📄 License
MIT — see [LICENSE](LICENSE)
