"""Application services owned by FastAPI and reusable by trusted clients."""

from datalens.application.scoring import ScoringResult, ScoringService

__all__ = ["ScoringResult", "ScoringService"]
