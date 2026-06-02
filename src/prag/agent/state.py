"""The agent's typed state, threaded through every node in the graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RetrievedRef(TypedDict):
    doc_id: str
    source: str
    ordinal: int
    text: str
    score: float


class AgentState(TypedDict, total=False):
    # Inputs
    query: str
    thread_id: str

    # Retrieval
    retrieved: list[RetrievedRef]
    top_score: float

    # Generation / review
    draft: str
    grounded: bool
    decision: str  # approve|edit|reject|abstain
    final_answer: str

    # Accounting (summed across nodes)
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]

    # Terminal outcome label for metrics
    outcome: str  # answered|abstained|rejected
