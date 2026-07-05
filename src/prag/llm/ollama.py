"""Ollama-backed LLM + Embedder — local, free, no API key.

Satisfies the same `LLM` / `Embedder` protocols as the fake and OpenAI-compatible
providers (``prag.llm.base``), so switching to a real local embedder/model is a
config change (`embedding_provider=ollama`, `llm_provider=ollama`), not a rewrite.
Uses Ollama's native REST API over stdlib urllib (no extra dependency).
"""
from __future__ import annotations

import json
import urllib.request

from prag.llm.base import Completion, Usage

_DEFAULT_HOST = "http://localhost:11434"


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OllamaLLM:
    """Chat completion via Ollama ``/api/generate`` (deterministic, temperature 0)."""

    def __init__(self, model: str = "llama3.2:3b", host: str = _DEFAULT_HOST, timeout: float = 120.0):
        self.model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion:
        data = _post(
            f"{self._host}/api/generate",
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": max_tokens},
            },
            self._timeout,
        )
        return Completion(
            text=data.get("response", ""),
            usage=Usage(
                model=self.model,
                input_tokens=int(data.get("prompt_eval_count", 0)),
                output_tokens=int(data.get("eval_count", 0)),
            ),
        )


class OllamaEmbedder:
    """Embeddings via Ollama ``/api/embeddings`` (default ``nomic-embed-text``, dim 768)."""

    def __init__(self, model: str = "nomic-embed-text", host: str = _DEFAULT_HOST,
                 dim: int = 768, timeout: float = 120.0):
        self.model = model
        self._host = host.rstrip("/")
        self.dim = dim
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            data = _post(
                f"{self._host}/api/embeddings",
                {"model": self.model, "prompt": text},
                self._timeout,
            )
            vec = [float(x) for x in (data.get("embedding") or [])]
            if vec:
                self.dim = len(vec)  # trust the model's true dimension
            vectors.append(vec)
        return vectors
