"""Unit tests for the Ollama LLM + Embedder adapters.

HTTP is mocked, so these run offline in CI (no Ollama server needed) while still
exercising request shaping, response parsing, usage accounting, and protocol
conformance.
"""
from __future__ import annotations

import prag.llm.ollama as ollama_mod
from prag.llm.base import Embedder, Usage


def test_ollama_llm_complete(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"response": "hello world", "prompt_eval_count": 7, "eval_count": 3}

    monkeypatch.setattr(ollama_mod, "_post", fake_post)
    llm = ollama_mod.OllamaLLM(model="llama3.2:3b")
    out = llm.complete("system prompt", "user question", max_tokens=32)

    assert captured["url"].endswith("/api/generate")
    assert captured["payload"]["system"] == "system prompt"
    assert captured["payload"]["options"]["num_predict"] == 32
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert out.text == "hello world"
    assert isinstance(out.usage, Usage)
    assert out.usage.input_tokens == 7 and out.usage.output_tokens == 3


def test_ollama_embedder_parses_and_sets_dim(monkeypatch):
    def fake_post(url, payload, timeout):
        assert url.endswith("/api/embeddings")
        # one call per text
        return {"embedding": [0.1, 0.2, 0.3, 0.4]}

    monkeypatch.setattr(ollama_mod, "_post", fake_post)
    emb = ollama_mod.OllamaEmbedder(model="nomic-embed-text", dim=768)
    vectors = emb.embed(["first text", "second text"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3, 0.4]
    assert emb.dim == 4  # trusts the model's true dimension
    assert isinstance(emb, Embedder)


def test_ollama_embedder_handles_empty(monkeypatch):
    monkeypatch.setattr(ollama_mod, "_post", lambda url, payload, timeout: {})
    emb = ollama_mod.OllamaEmbedder()
    assert emb.embed(["x"]) == [[]]
