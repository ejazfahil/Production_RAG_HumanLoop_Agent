"""Assemble the LangGraph StateGraph and pick a checkpointer.

The checkpointer is what makes human-in-the-loop durable: when the graph hits the
approval interrupt, state is persisted and the run can resume later (even after a
restart) from exactly where it paused.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph

from prag.agent.nodes import Nodes
from prag.agent.state import AgentState
from prag.config import Settings


def build_graph(nodes: Nodes, checkpointer: Any) -> Any:
    """Wire nodes and edges into a compiled graph with the given checkpointer."""
    g: StateGraph = StateGraph(AgentState)

    g.add_node("retrieve", nodes.retrieve)
    g.add_node("draft_answer", nodes.draft_answer)
    g.add_node("grounding_check", nodes.grounding_check)
    g.add_node("hitl_approval", nodes.hitl_approval)
    g.add_node("finalize", nodes.finalize)
    g.add_node("abstain", nodes.abstain)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", nodes.route_after_retrieval, ["draft_answer", "abstain"])
    g.add_edge("draft_answer", "grounding_check")
    g.add_conditional_edges(
        "grounding_check", nodes.route_after_grounding, ["hitl_approval", "abstain"]
    )
    g.add_edge("hitl_approval", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)

    return g.compile(checkpointer=checkpointer)


@contextmanager
def make_checkpointer(settings: Settings) -> Iterator[object]:
    """Yield a checkpointer appropriate for the configured backend."""
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
    elif settings.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(settings.sqlite_path) as cp:
            yield cp
    elif settings.checkpointer == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(settings.database_url) as cp:
            cp.setup()
            yield cp
    else:  # pragma: no cover - guarded by config typing
        raise ValueError(f"unknown checkpointer: {settings.checkpointer}")
