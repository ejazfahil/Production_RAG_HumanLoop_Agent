# ADR-0001: Use LangGraph for Stateful, Human-in-the-Loop Orchestration

**Date**: June 2026  
**Status**: Accepted

## Context

We need to build a RAG agent that can pause mid-execution to let a human reviewer approve, edit, or reject the draft answer before it's returned. This is a **human-in-the-loop (HITL)** pattern.

The pause must be **durable**: if the system crashes while waiting for human input, the graph state should persist so the workflow can resume exactly where it left off without re-running the LLM.

Traditional LLM frameworks (LangChain Sequential Chains, simple if/else orchestration) lack native support for:
- Persistent execution state across restarts
- Checkpointing to Postgres/SQLite
- Graph-based conditional logic with cycles
- Native interrupts (pause points)

## Decision

**Use LangGraph** as the core orchestration framework for the agent graph.

LangGraph is a low-level, graph-based orchestration library built for exactly this use case:
- **Durable execution**: Checkpointer backends (memory, sqlite, postgres) persist state after every node.
- **Native interrupts**: `interrupt()` and `Command(resume=...)` implement HITL pause/resume without custom code.
- **Rich conditionals**: `conditional_edges()` for routing (e.g., abstain if confidence is low).
- **Multi-tenant ready**: Thread IDs + checkpointer isolate concurrent sessions.

## Consequences

### Positive

- **Durability for free**: Graph state survives crashes; human approval workflows are resilient.
- **Clean separation**: Agent logic (nodes) is decoupled from persistence.
- **Auditability**: Every state transition can be logged; full trace exists in checkpointer.
- **No external queue needed**: Kafka/RabbitMQ unnecessary for paused workflows; the checkpointer is the queue.

### Negative

- **New abstraction to learn**: LangGraph is lower-level than LangChain; developers must understand graph terminology.
- **Checkpointer scaling**: In-memory checkpointer is fine for dev; postgres checkpointer must scale for thousands of concurrent paused threads.
- **Migration path**: If you outgrow LangGraph (unlikely), rebuilding state logic would be non-trivial.

## Alternatives Considered

### 1. LangChain with a custom queue (Redis/Postgres)
- **Why rejected**: Requires hand-wiring persist/restore logic; much boilerplate; error-prone. LangGraph is purpose-built for this.

### 2. Temporal.io (workflow orchestration)
- **Why rejected**: Overkill for a single agent; adds operational complexity (Temporal cluster to run).

### 3. Simple async/await with asyncio.Event
- **Why rejected**: No durable state; single-machine only; no built-in audit trail.

## Implementation Notes

- Use `langgraph.checkpoint.postgres.PostgresSaver` for prod; `MemorySaver` for dev.
- Always pass `thread_id` in the checkpoint config so HITL threads are isolated.
- Log graph state transitions to structured logs for debugging.
- Expose checkpointer metrics (e.g., checkpoint latency) so scaling bottlenecks are visible.
