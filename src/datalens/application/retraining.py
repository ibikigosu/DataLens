"""Manual candidate training orchestration owned by the application layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from datalens.application.persistence import Repository, RetrainingRecord
from datalens.configuration.loader import RuntimeConfig
from datalens.modeling.reranker import RerankerStore, train_feedback_reranker


class RetrainingService:
    """Run the configured comparison workflow and retain its promotion decision."""

    def __init__(
        self,
        config: RuntimeConfig,
        repository: Repository,
        reranker_store: RerankerStore,
    ) -> None:
        self._config = config
        self._repository = repository
        self._reranker_store = reranker_store

    def run(self) -> RetrainingRecord:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate_version = f"feedback-reranker-{timestamp}"
        training = train_feedback_reranker(
            pd.DataFrame(self._repository.feedback_examples()),
            config=self._config.model.feedback_reranker,
            top_k=self._config.model.top_k,
            seed=self._config.model.seed,
        )
        result = training.result
        if training.model is not None:
            self._reranker_store.save_candidate(
                training.model,
                version=candidate_version,
                result=result,
            )
        return self._repository.save_retraining_run(
            active_model_version=(
                self._reranker_store.active_version() or self._config.model.active_model_version
            ),
            candidate_model_version=candidate_version,
            result=result,
        )

    def deactivate_active_reranker(self) -> dict[str, str | bool]:
        was_active = self._reranker_store.deactivate()
        return {
            "active_model_version": self._config.model.active_model_version,
            "deactivated": was_active,
        }


def retraining_result(record: RetrainingRecord) -> dict[str, Any]:
    """Return the stored candidate comparison as a response-safe dictionary."""
    result = json.loads(record.result_json)
    return {
        "retraining_run_id": record.id,
        "active_model_version": record.active_model_version,
        "candidate_model_version": record.candidate_model_version,
        "promoted": record.promoted,
        "promotion": result["promotion"],
        "active": result["active"],
        "candidate": result["candidate"],
        "training_examples": result["training_examples"],
        "validation_examples": result.get("validation_examples", 0),
        "evaluation_top_k": result.get("evaluation_top_k", 0),
    }
