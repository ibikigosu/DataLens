"""FastAPI dependency boundary for application services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from datalens.application.orchestration import ScoringCoordinator
from datalens.application.persistence import Repository
from datalens.application.retraining import RetrainingService
from datalens.application.scoring import ScoringService
from datalens.configuration.loader import RuntimeConfig


@dataclass(frozen=True)
class AppServices:
    config: RuntimeConfig
    scoring: ScoringService
    scoring_coordinator: ScoringCoordinator
    repository: Repository
    retraining: RetrainingService


def get_services(request: Request) -> AppServices:
    return request.app.state.services


Services = Annotated[AppServices, Depends(get_services)]
