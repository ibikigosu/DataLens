"""Liveness and readiness routes."""

from fastapi import APIRouter, HTTPException

from datalens.api.contracts import HealthResponse
from datalens.api.dependencies import Services

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def liveness(services: Services) -> HealthResponse:
    return HealthResponse(status="ok", environment=services.config.settings.environment)


@router.get("/health/ready", response_model=HealthResponse)
def readiness(services: Services) -> HealthResponse:
    if not services.repository.ready():
        raise HTTPException(status_code=503, detail="Database is not ready")
    return HealthResponse(status="ready", environment=services.config.settings.environment)
