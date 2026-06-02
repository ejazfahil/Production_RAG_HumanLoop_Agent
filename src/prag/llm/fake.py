"""Deterministic, offline LLM/embedder used for local dev and CI.

The fake LLM produces an extractive answer from the provided context so that the
grounding check and the full graph can be exercised end-to-end without any API key.
"""

from __future__ import annotations

import hashlib
import math
import re

from prag.llm.base import Completion, Embedder, Usage


def _count_tokens(text: str) -> int:
    # Cheap word-based proxy; the real providers report exact usage.
    return max(1, len(text.split()))


class FakeLLM:
    """Returns an extractive 'answer' grounded in the supplied context."""

    def __init__(self, model: str = "fake-small") -> None:
        self.model = model

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion:
        # Pull the first 1-2 sentences out of the CONTEXT block, if present.
        context = ""
        if "CONTEXT:" in prompt:
            context = prompt.split("CONTEXT:", 1)[1]
        sentences = re.split(r"(?<=[.!?])\s+", context.strip())
        answer = " ".join(s for s in sentences[:2] if s).strip()
        if not answer:
            answer = "Based on the provided documents, no grounded answer is available."
        usage = Usage(
            model=self.model,
            input_tokens=_count_tokens(system) + _count_tokens(prompt),
            output_tokens=_count_tokens(answer),
        )
        return Completion(text=answer, usage=usage)


class FakeEmbedder:
    """Hashing-based bag-of-words embedding: deterministic and dependency-free."""

    def __init__(self, model: str = "fake-embed", dim: int = 384) -> None:
        self.model = model
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# Static type sanity: these satisfy the LLM / Embedder protocols.
_llm_check: type = FakeLLM
_emb_check: type[Embedder] = FakeEmbedder
