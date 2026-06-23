"""MLflow experiment tracking and selected-candidate registration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from datalens.features.pipeline import FeatureTable
from datalens.modeling.models import TableModelBundle, skops_trusted_types_for_model
from datalens.modeling.registry import (
    CandidateRegistration,
    build_registry_manifest,
    register_candidate,
    write_registry_manifest,
)


class PeriodTrackingContext(Protocol):
    fiscal_year: str
    role: str
    dataset_identity: dict[str, Any]


def log_experiments(
    *,
    tracking_uri: str,
    experiment_name: str,
    development: PeriodTrackingContext,
    temporal_holdout: PeriodTrackingContext,
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    table_winners: dict[str, str],
    promotion: dict[str, Any],
    bundles: dict[str, dict[FeatureTable, TableModelBundle]],
    training_frames: dict[FeatureTable, pd.DataFrame],
    artifact_paths: dict[str, Path],
    summary: dict[str, Any],
    active_model_version: str,
    registry_name_prefix: str,
) -> dict[str, Any]:
    """Log experiments, register selected candidates, and retain activation evidence."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_experiment(
        experiment_id=_experiment_id(
            client,
            experiment_name=experiment_name,
            artifact_root=artifact_paths["summary"].parent / "mlflow-artifacts",
        )
    )
    registrations = _log_candidate_runs(
        client=client,
        development=development,
        temporal_holdout=temporal_holdout,
        development_metrics=development_metrics,
        holdout_metrics=holdout_metrics,
        table_winners=table_winners,
        promotion=promotion,
        bundles=bundles,
        training_frames=training_frames,
        artifact_paths=artifact_paths,
        registry_name_prefix=registry_name_prefix,
    )
    registry_manifest = build_registry_manifest(
        active_model_version=active_model_version,
        candidates=tuple(registrations),
        promotion=promotion,
    )
    _persist_registry_evidence(
        summary=summary,
        registry_manifest=registry_manifest,
        artifact_paths=artifact_paths,
    )
    _log_selection_run(
        development=development,
        table_winners=table_winners,
        promotion=promotion,
        registry_manifest=registry_manifest,
        artifact_paths=artifact_paths,
    )
    return registry_manifest


def _experiment_id(
    client: MlflowClient,
    *,
    experiment_name: str,
    artifact_root: Path,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(
        experiment_name,
        artifact_location=artifact_root.resolve().as_uri(),
    )


def _log_candidate_runs(
    *,
    client: MlflowClient,
    development: PeriodTrackingContext,
    temporal_holdout: PeriodTrackingContext,
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    table_winners: dict[str, str],
    promotion: dict[str, Any],
    bundles: dict[str, dict[FeatureTable, TableModelBundle]],
    training_frames: dict[FeatureTable, pd.DataFrame],
    artifact_paths: dict[str, Path],
    registry_name_prefix: str,
) -> list[CandidateRegistration]:
    registrations: list[CandidateRegistration] = []
    for model_name, table_bundles in bundles.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_family": model_name,
                    "development_fiscal_year": development.fiscal_year,
                    "temporal_holdout_fiscal_year": temporal_holdout.fiscal_year,
                    "holdout_used_for_selection": False,
                }
            )
            mlflow.log_metrics(
                _flatten_numeric_metrics(
                    development_metrics[model_name],
                    prefix="development",
                )
            )
            mlflow.log_metrics(
                _flatten_numeric_metrics(
                    holdout_metrics[model_name],
                    prefix="fy2025",
                )
            )
            mlflow.log_dict(
                {
                    "development": development.dataset_identity,
                    "temporal_holdout": temporal_holdout.dataset_identity,
                },
                "dataset-identity.json",
            )
            mlflow.log_dict(
                {
                    "development": development_metrics[model_name],
                    "temporal_holdout": holdout_metrics[model_name],
                },
                "metrics.json",
            )
            registrations.extend(
                _log_table_runs(
                    client=client,
                    model_name=model_name,
                    table_bundles=table_bundles,
                    table_winners=table_winners,
                    promotion=promotion,
                    training_frames=training_frames,
                    artifact_paths=artifact_paths,
                    registry_name_prefix=registry_name_prefix,
                )
            )
    return registrations


def _log_table_runs(
    *,
    client: MlflowClient,
    model_name: str,
    table_bundles: Mapping[FeatureTable, TableModelBundle],
    table_winners: Mapping[str, str],
    promotion: dict[str, Any],
    training_frames: Mapping[FeatureTable, pd.DataFrame],
    artifact_paths: Mapping[str, Path],
    registry_name_prefix: str,
) -> list[CandidateRegistration]:
    registrations = []
    for table, bundle in table_bundles.items():
        with mlflow.start_run(run_name=table.value, nested=True):
            mlflow.log_params(
                {
                    "table": table.value,
                    "schema_version": bundle.model.schema_version,
                    "review_fraction": bundle.model.spec.review_fraction,
                    "seed": bundle.model.spec.seed,
                    **bundle.model.spec.parameters,
                }
            )
            mlflow.log_dict(bundle.pipeline.metadata(), "feature-pipeline.json")
            input_example = bundle.pipeline.transform(
                training_frames[table].head(5)
            ).values.to_numpy(dtype=float)
            selected = table_winners[table.value] == model_name
            registered_model_name = (
                f"{registry_name_prefix}-{table.value}-anomaly" if selected else None
            )
            model_info = mlflow.sklearn.log_model(
                bundle.model.estimator,
                name=f"{table.value}-model",
                serialization_format="skops",
                input_example=input_example,
                skops_trusted_types=skops_trusted_types_for_model(bundle.model),
                registered_model_name=registered_model_name,
            )
            if selected:
                version = model_info.registered_model_version
                if version is None or registered_model_name is None:
                    raise RuntimeError("MLflow did not return a registered model version")
                registrations.append(
                    register_candidate(
                        client,
                        table=table.value,
                        model_family=model_name,
                        registered_model_name=registered_model_name,
                        version=str(version),
                        promotion=promotion,
                    )
                )
            mlflow.log_artifacts(
                str(artifact_paths[f"bundle_{model_name}_{table.value}"]),
                artifact_path="model-bundle",
            )
    return registrations


def _persist_registry_evidence(
    *,
    summary: dict[str, Any],
    registry_manifest: dict[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    summary["registry"] = registry_manifest
    artifact_paths["summary"].write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_registry_manifest(
        artifact_paths["summary"].parent / "model-registry.json",
        registry_manifest,
    )


def _log_selection_run(
    *,
    development: PeriodTrackingContext,
    table_winners: Mapping[str, str],
    promotion: dict[str, Any],
    registry_manifest: dict[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    with mlflow.start_run(run_name="selection-and-promotion"):
        mlflow.log_dict(dict(table_winners), "table-winners.json")
        mlflow.log_dict(promotion, "promotion-decision.json")
        mlflow.log_dict(registry_manifest, "model-registry.json")
        mlflow.log_params(
            {
                "vendor_winner": table_winners["vendor"],
                "transaction_winner": table_winners["transaction"],
                "promoted": promotion["promoted"],
                "selection_period": development.fiscal_year,
                "selection_period_role": development.role,
            }
        )
        for name in (
            "summary",
            "development_scores",
            "holdout_scores",
            "guarded_queue",
            "evidence",
        ):
            mlflow.log_artifact(str(artifact_paths[name]))


def _flatten_numeric_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}_{key}"
        if isinstance(value, bool | int | float):
            flattened[metric_name] = float(value)
        elif isinstance(value, Mapping) and key != "rank_calibration":
            flattened.update(_flatten_numeric_metrics(value, prefix=metric_name))
    return flattened
