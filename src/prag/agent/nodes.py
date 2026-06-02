"""Graph nodes. Each node is small, typed, instrumented, and side-effect-aware.

Flow:  retrieve -> (abstain | draft) -> grounding_check -> (abstain | hitl_approval)
       -> finalize.

The HITL approval node uses LangGraph's `interrupt()` to pause the run and hand
control to a human reviewer; the run resumes via `Command(resume=...)`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langgraph.types import interrupt

from prag.agent.state import AgentState, RetrievedRef
from prag.config import Settings
from prag.llm.base import LLM
from prag.observability.logging import get_logger
from prag.observability.metrics import (
    APPROVALS,
    COST_USD,
    NODE_LATENCY,
    QUERIES,
    TOKENS,
)
from prag.retrieval.store import VectorStore

log = get_logger("agent")

ABSTENTION = "Not supported by the provided documents."

SYSTEM_PROMPT = (
    "You are an industrial document-intelligence assistant. Answer strictly from the "
    "provided CONTEXT. If the context does not support an answer, say so. Cite sources."
)

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _groundedness(draft: str, context: str) -> float:
    """Fraction of the draft's content words supported by the context.

    A deliberately transparent proxy for an LLM-as-judge groundedness scorer; swap in
    a stricter judge by replacing this function.
    """
    draft_words = _words(draft) - _STOPWORDS
    if not draft_words:
        return 0.0
    ctx_words = _words(context)
    supported = sum(1 for w in draft_words if w in ctx_words)
    return supported / len(draft_words)


_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "by",
    "be",
    "this",
    "that",
    "it",
    "as",
    "at",
    "from",
    "based",
}


@dataclass
class Nodes:
    """Holds dependencies and exposes the graph node callables."""

    llm: LLM
    store: VectorStore
    settings: Settings

    # --- retrieve ---
    def retrieve(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        hits = self.store.search(state["query"], top_k=self.settings.top_k)
        refs: list[RetrievedRef] = [
            {
                "doc_id": h.document.doc_id,
                "source": h.document.source,
                "ordinal": h.document.ordinal,
                "text": h.document.text,
                "score": h.score,
            }
            for h in hits
        ]
        top = refs[0]["score"] if refs else 0.0
        NODE_LATENCY.labels("retrieve").observe(time.perf_counter() - start)
        log.info("retrieve", query=state["query"], hits=len(refs), top_score=round(top, 3))
        return {"retrieved": refs, "top_score": top}

    # --- draft ---
    def draft_answer(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        context = "\n\n".join(
            f"[{r['source']}#{r['ordinal']}] {r['text']}" for r in state["retrieved"]
        )
        completion = self.llm.complete(
            SYSTEM_PROMPT, f"QUESTION: {state['query']}\n\nCONTEXT:\n{context}"
        )
        u = completion.usage
        TOKENS.labels(u.model, "input").inc(u.input_tokens)
        TOKENS.labels(u.model, "output").inc(u.output_tokens)
        COST_USD.labels(u.model).inc(u.cost_usd)
        NODE_LATENCY.labels("draft").observe(time.perf_counter() - start)
        log.info("draft", tokens_in=u.input_tokens, tokens_out=u.output_tokens, cost=u.cost_usd)
        return {
            "draft": completion.text,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cost_usd": u.cost_usd,
        }

    # --- grounding check ---
    def grounding_check(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        context = " ".join(r["text"] for r in state["retrieved"])
        score = _groundedness(state["draft"], context)
        grounded = score >= 0.6
        NODE_LATENCY.labels("grounding_check").observe(time.perf_counter() - start)
        log.info("grounding_check", score=round(score, 3), grounded=grounded)
        return {"grounded": grounded}

    # --- human-in-the-loop approval (interrupt) ---
    def hitl_approval(self, state: AgentState) -> AgentState:
        if self.settings.auto_approve:
            APPROVALS.labels("approve").inc()
            return {"decision": "approve", "final_answer": state["draft"]}

        decision = interrupt(
            {
                "query": state["query"],
                "draft": state["draft"],
                "sources": [
                    {"source": r["source"], "ordinal": r["ordinal"], "score": r["score"]}
                    for r in state["retrieved"]
                ],
                "confidence": state["top_score"],
                "instructions": "Respond with {action: approve|edit|reject, text?: str}.",
            }
        )
        action = decision.get("action", "reject")
        APPROVALS.labels(action).inc()
        final = decision.get("text", state["draft"]) if action != "reject" else ""
        log.info("hitl_decision", action=action)
        return {"decision": action, "final_answer": final}

    # --- finalize ---
    def finalize(self, state: AgentState) -> AgentState:
        decision = state.get("decision", "reject")
        if decision == "reject":
            QUERIES.labels("rejected").inc()
            return {"final_answer": "", "outcome": "rejected"}
        QUERIES.labels("answered").inc()
        return {"outcome": "answered"}

    # --- abstain ---
    def abstain(self, state: AgentState) -> AgentState:
        QUERIES.labels("abstained").inc()
        log.info("abstain", top_score=state.get("top_score", 0.0))
        return {"decision": "abstain", "final_answer": ABSTENTION, "outcome": "abstained"}

    # --- conditional edges ---
    def route_after_retrieval(self, state: AgentState) -> str:
        return (
            "draft_answer" if state["top_score"] >= self.settings.min_retrieval_score else "abstain"
        )

    def route_after_grounding(self, state: AgentState) -> str:
        return "hitl_approval" if state.get("grounded") else "abstain"
