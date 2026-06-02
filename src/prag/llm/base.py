"""Provider-agnostic LLM + embedding interfaces and cost accounting.

Keeping a thin abstraction here means the agent code never imports a vendor SDK
directly, so swapping OpenAI -> Mistral (EU) -> a self-hosted vLLM endpoint is a
config change, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# USD per 1K tokens (input, output). Extend as providers/models are added.
# Numbers are illustrative defaults; confirm against current provider pricing.
PRICING: dict[str, tuple[float, float]] = {
    "fake-small": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-5-sonnet": (0.003, 0.015),
    "mistral-small-latest": (0.0002, 0.0006),
    "mistral-large-latest": (0.002, 0.006),
}


@dataclass(frozen=True)
class Usage:
    """Token usage and derived cost for a single LLM call."""

    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        in_rate, out_rate = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens / 1000) * in_rate + (self.output_tokens / 1000) * out_rate


@dataclass(frozen=True)
class Completion:
    """An LLM completion plus its usage accounting."""

    text: str
    usage: Usage


@runtime_checkable
class LLM(Protocol):
    """Minimal chat-completion interface the agent depends on."""

    model: str

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion: ...


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface the retrieval layer depends on."""

    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
