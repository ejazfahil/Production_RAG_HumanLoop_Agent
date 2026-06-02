"""
Compiled graph for LangGraph Cloud deployment.

This module is referenced in langgraph.json for the platform to discover and load the graph.
"""

from typing import Any

from prag.agent.graph import build_graph, make_checkpointer
from prag.agent.nodes import Nodes
from prag.config import get_settings
from prag.llm.factory import build_embedder, build_llm
from prag.retrieval.store import InMemoryVectorStore


def _build_default_graph() -> Any:
    """Build the default graph with in-memory backends (for local/Cloud)."""
    settings = get_settings()
    embedder = build_embedder(settings)
    llm = build_llm(settings)
    store = InMemoryVectorStore(embedder=embedder)
    nodes = Nodes(llm=llm, store=store, settings=settings)
    with make_checkpointer(settings) as checkpointer:
        return build_graph(nodes, checkpointer)


# Module-level graph for LangGraph Cloud / CLI
graph = _build_default_graph()
