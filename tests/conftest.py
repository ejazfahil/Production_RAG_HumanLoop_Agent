"""Shared fixtures. Everything runs offline with the fake provider + in-memory store."""

from __future__ import annotations

import pytest

from prag.config import Settings
from prag.engine import Engine


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="memory",
        checkpointer="memory",
        min_retrieval_score=0.1,
        auto_approve=False,
    )


@pytest.fixture
def engine(settings: Settings) -> Engine:
    eng = Engine(settings).start()
    eng.ingest("data/sample")
    yield eng
    eng.close()
