"""Factory that builds the configured LLM and Embedder.

`fake` is fully offline. The OpenAI-compatible client also covers Mistral and any
self-hosted vLLM endpoint by setting `llm_base_url`. Anthropic uses its own path.
"""

from __future__ import annotations

from prag.config import Settings
from prag.llm.base import Completion, Embedder, Usage
from prag.llm.fake import FakeEmbedder, FakeLLM


class OpenAICompatLLM:
    """Works with OpenAI, Mistral, Azure OpenAI, or a self-hosted vLLM server."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        usage = Usage(
            model=self.model,
            input_tokens=int(u.get("prompt_tokens", 0)),
            output_tokens=int(u.get("completion_tokens", 0)),
        )
        return Completion(text=text, usage=usage)


def build_llm(settings: Settings) -> FakeLLM | OpenAICompatLLM:
    """Return the configured chat LLM."""
    if settings.llm_provider == "fake":
        return FakeLLM(model=settings.llm_model)
    if settings.llm_provider == "ollama":
        from prag.llm.ollama import OllamaLLM

        return OllamaLLM(model=settings.llm_model, host=settings.ollama_host)
    if settings.llm_provider in ("openai", "mistral"):
        if not settings.llm_api_key:
            raise ValueError(f"llm_api_key required for provider '{settings.llm_provider}'")
        return OpenAICompatLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    # Anthropic / others would be wired here behind the same `complete` contract.
    raise NotImplementedError(f"LLM provider '{settings.llm_provider}' not wired yet")


def build_embedder(settings: Settings) -> Embedder:
    """Return the configured embedder. Fake is the offline default; Ollama is a real,
    free, local embedder (e.g. nomic-embed-text)."""
    if settings.embedding_provider == "fake":
        return FakeEmbedder(model=settings.embedding_model, dim=settings.embedding_dim)
    if settings.embedding_provider == "ollama":
        from prag.llm.ollama import OllamaEmbedder

        return OllamaEmbedder(model=settings.embedding_model, host=settings.ollama_host,
                              dim=settings.embedding_dim)
    # Other real embedding providers slot in here; they must satisfy the Embedder protocol.
    raise NotImplementedError(f"Embedding provider '{settings.embedding_provider}' not wired yet")
