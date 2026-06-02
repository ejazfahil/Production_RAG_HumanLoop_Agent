"""FastAPI application factory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from prag.api.routes import router
from prag.config import get_settings
from prag.engine import Engine
from prag.observability.logging import configure_logging, get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    app.state.engine = Engine(settings).start()
    log.info("api_ready")
    try:
        yield
    finally:
        app.state.engine.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Production RAG + Human-in-the-Loop Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if settings.otel_exporter_otlp_endpoint:
        try:  # pragma: no cover - optional dependency path
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:
            log.warning("otel_instrumentation_failed", error=str(exc))

    return app


app = create_app()
