"""FastAPI application factory and runtime lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from datalens.api.dependencies import AppServices
from datalens.api.logging import request_logging_middleware
from datalens.api.routes import feedback, health, models, runs, scoring
from datalens.application.orchestration import ScoringCoordinator
from datalens.application.persistence import Repository
from datalens.application.retraining import RetrainingService
from datalens.application.scoring import ScoringService
from datalens.configuration.loader import RuntimeConfig, load_runtime_config
from datalens.modeling.reranker import RerankerStore


def create_app(runtime_config: RuntimeConfig | None = None) -> FastAPI:
    """Create the versioned DataLens API with explicit application services."""
    config = runtime_config or load_runtime_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        repository = Repository(config.settings.database_url)
        repository.initialize()
        scoring_service = ScoringService(config)
        reranker_store = RerankerStore(config.settings.artifact_dir / "feedback-reranker")
        app.state.services = AppServices(
            config=config,
            scoring=scoring_service,
            scoring_coordinator=ScoringCoordinator(
                config,
                scoring_service,
                repository,
                reranker_store,
            ),
            repository=repository,
            retraining=RetrainingService(config, repository, reranker_store),
        )
        yield

    app = FastAPI(
        title="DataLens API",
        version="1.0.0",
        description=(
            "Human-in-the-loop procurement data quality scoring. DataLens does not detect fraud."
        ),
        lifespan=lifespan,
    )
    app.middleware("http")(request_logging_middleware)
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(scoring.router, prefix=prefix)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(feedback.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    return app


app = create_app()
