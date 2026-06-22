"""MLflow tracking with an explicit PostgreSQL production boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow import MlflowClient

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def require_postgresql_tracking_uri(explicit_uri: str | None = None) -> str:
    """Resolve the production tracking URI and reject non-PostgreSQL backends."""
    tracking_uri = explicit_uri or os.environ.get(TRACKING_URI_ENV)
    if not tracking_uri:
        raise RuntimeError(
            f"{TRACKING_URI_ENV} is required and must use postgresql:// or postgresql+psycopg://"
        )
    scheme = urlparse(tracking_uri).scheme
    if scheme not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError("MLflow production tracking must use PostgreSQL")
    return tracking_uri


class ExperimentTracker:
    """Small MLflow adapter so experiment logic does not own persistence details."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        experiment_name: str,
        artifact_root: Path,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = artifact_root

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, str | int | float | bool],
        metrics: Mapping[str, float],
        artifacts: tuple[Path, ...],
    ) -> str:
        """Log one complete run and return its immutable MLflow run identifier."""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            experiment_id = client.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_root.resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ) as run:
            mlflow.log_params(dict(params))
            mlflow.log_metrics(dict(metrics))
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
            return run.info.run_id
