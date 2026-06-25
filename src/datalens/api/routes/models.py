"""Manual candidate retraining route."""

from fastapi import APIRouter

from datalens.api.contracts import ActiveModelResetResponse, RetrainingResponse
from datalens.api.dependencies import Services
from datalens.application.retraining import retraining_result

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/retrain", response_model=RetrainingResponse)
def retrain(services: Services) -> RetrainingResponse:
    return RetrainingResponse.model_validate(retraining_result(services.retraining.run()))


@router.post("/active-reranker/deactivate", response_model=ActiveModelResetResponse)
def deactivate_active_reranker(services: Services) -> ActiveModelResetResponse:
    return ActiveModelResetResponse.model_validate(services.retraining.deactivate_active_reranker())
