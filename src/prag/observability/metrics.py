"""Prometheus metrics. Exposed on /metrics by the API.

Cost/latency/token counters are the production-discipline signal: every node run
records latency, tokens and estimated USD cost so they can be graphed in Grafana.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

NODE_LATENCY = Histogram(
    "prag_node_latency_seconds",
    "Per-node execution latency.",
    labelnames=("node",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

QUERY_LATENCY = Histogram(
    "prag_query_latency_seconds",
    "End-to-end query latency (excludes time blocked on human approval).",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

TOKENS = Counter(
    "prag_tokens_total",
    "LLM tokens consumed.",
    labelnames=("model", "direction"),  # direction: input|output
)

COST_USD = Counter(
    "prag_cost_usd_total",
    "Estimated LLM spend in USD.",
    labelnames=("model",),
)

QUERIES = Counter(
    "prag_queries_total",
    "Queries by terminal outcome.",
    labelnames=("outcome",),  # answered|abstained|rejected
)

APPROVALS = Counter(
    "prag_approvals_total",
    "Human review decisions.",
    labelnames=("decision",),  # approve|edit|reject
)

ERRORS = Counter(
    "prag_errors_total",
    "Unhandled errors by component.",
    labelnames=("component",),
)

INGESTED_CHUNKS = Counter(
    "prag_ingested_chunks_total",
    "Document chunks indexed.",
)
