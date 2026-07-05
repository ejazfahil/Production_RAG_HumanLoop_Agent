"""Application configuration, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, typed configuration. Never put secrets in code; load them here."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "production-rag-humanloop-agent"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # --- LLM provider (provider-agnostic; "fake" needs no API key, "ollama" is free/local) ---
    llm_provider: Literal["fake", "ollama", "openai", "anthropic", "mistral"] = "fake"
    llm_model: str = "fake-small"
    llm_api_key: str | None = None
    llm_base_url: str | None = None  # e.g. Mistral / Azure / self-hosted vLLM
    ollama_host: str = "http://localhost:11434"  # for llm/embedding_provider == "ollama"
    embedding_provider: Literal["fake", "ollama", "openai", "mistral"] = "fake"
    embedding_model: str = "fake-embed"
    embedding_dim: int = 384

    # --- Retrieval ---
    vector_backend: Literal["memory", "pgvector"] = "memory"
    top_k: int = 4
    # If the best retrieved score is below this, the agent abstains instead of guessing.
    min_retrieval_score: float = 0.25
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Persistence ---
    database_url: str = "postgresql://prag:prag@localhost:5432/prag"
    # Checkpointer backend for LangGraph state (durable HITL across restarts).
    checkpointer: Literal["memory", "sqlite", "postgres"] = "memory"
    sqlite_path: str = "./prag_checkpoints.sqlite"

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "production-rag-humanloop-agent"
    otel_exporter_otlp_endpoint: str | None = None

    # --- HITL ---
    # Auto-approve is for CI/smoke runs only; real deployments keep this False.
    auto_approve: bool = False

    seed: int = 1234


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
