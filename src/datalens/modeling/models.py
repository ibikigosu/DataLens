"""Training, scoring, and persistence for table-specific Isolation Forests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
from sklearn.ensemble import IsolationForest

from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import (
    FeatureMatrix,
    FeaturePipeline,
    FeaturePipelineState,
    FeatureSchema,
    FeatureTable,
)
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.modeling.evidence import build_bounded_evidence

MODEL_FILENAME = "estimator.skops"
PIPELINE_FILENAME = "feature-pipeline.json"


@dataclass(frozen=True)
class IsolationForestSpec:
    """Versioned controls for reproducible Isolation Forest fitting."""

    n_estimators: int = 200
    max_samples: int | str = "auto"
    review_fraction: float = 0.01
    seed: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.review_fraction < 1:
            raise ValueError("review_fraction must be between zero and one")


@dataclass
class TrainedTableModel:
    """One fitted preprocessing and Isolation Forest scoring boundary."""

    fit_fiscal_year: str
    pipeline: FeaturePipeline
    estimator: IsolationForest
    threshold: float
    sorted_training_scores: np.ndarray
    reference_medians: np.ndarray
    reference_scales: np.ndarray
    spec: IsolationForestSpec

    @property
    def table(self) -> FeatureTable:
        return self.pipeline.schema.table

    def score(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Transform and score records without changing fitted state."""
        matrix = self.pipeline.transform(feature_frame)
        values = matrix.values.to_numpy(dtype=float)
        raw_scores = -self.estimator.decision_function(values)
        percentiles = np.searchsorted(
            self.sorted_training_scores,
            raw_scores,
            side="right",
        ) / len(self.sorted_training_scores)
        anomaly_scores = percentiles * 100
        evidence_json = [
            build_bounded_evidence(
                row,
                feature_names=matrix.feature_names,
                reference_medians=self.reference_medians,
                reference_scales=self.reference_scales,
                anomaly_score=score,
                anomaly_percentile=percentile,
            ).to_json()
            for row, score, percentile in zip(
                values,
                anomaly_scores,
                percentiles,
                strict=True,
            )
        ]
        return pd.DataFrame(
            {
                "target_table": self.table.value,
                "record_id": matrix.record_ids.astype("string"),
                "model_name": "isolation_forest",
                "anomaly_score": anomaly_scores,
                "rank_percentile": percentiles,
                "predicted": raw_scores >= self.threshold,
                "priority_score": anomaly_scores,
                "evidence_json": evidence_json,
            }
        ).sort_values(
            ["priority_score", "record_id"],
            ascending=[False, True],
            ignore_index=True,
        )


@dataclass(frozen=True)
class TrainedModels:
    """Separate vendor and transaction models fitted from one development dataset."""

    vendor: TrainedTableModel
    transaction: TrainedTableModel

    def for_table(self, table: FeatureTable) -> TrainedTableModel:
        return self.vendor if table is FeatureTable.VENDOR else self.transaction


def train_isolation_forests(
    dataset: DevelopmentFeatureDataset,
    *,
    vendor_spec: IsolationForestSpec | None = None,
    transaction_spec: IsolationForestSpec | None = None,
) -> TrainedModels:
    """Fit separate models from a development-only typed dataset."""
    return TrainedModels(
        vendor=_train_table(
            dataset.vendors,
            schema=VENDOR_FEATURE_SCHEMA,
            fiscal_year=dataset.fiscal_year,
            spec=vendor_spec or IsolationForestSpec(review_fraction=0.025),
        ),
        transaction=_train_table(
            dataset.transactions,
            schema=TRANSACTION_FEATURE_SCHEMA,
            fiscal_year=dataset.fiscal_year,
            spec=transaction_spec or IsolationForestSpec(review_fraction=0.005),
        ),
    )


def save_trained_model(model: TrainedTableModel, directory: Path) -> None:
    """Persist one candidate package using reviewed, non-pickle formats."""
    directory.mkdir(parents=True, exist_ok=True)
    estimator_path = directory / MODEL_FILENAME
    skops.io.dump(model.estimator, estimator_path)
    untrusted_types = skops.io.get_untrusted_types(file=estimator_path)
    if untrusted_types:
        estimator_path.unlink(missing_ok=True)
        raise ValueError(f"Estimator contains untrusted types: {untrusted_types}")
    payload = {
        "fit_fiscal_year": model.fit_fiscal_year,
        "threshold": model.threshold,
        "sorted_training_scores": model.sorted_training_scores.tolist(),
        "reference_medians": model.reference_medians.tolist(),
        "reference_scales": model.reference_scales.tolist(),
        "spec": {
            "n_estimators": model.spec.n_estimators,
            "max_samples": model.spec.max_samples,
            "review_fraction": model.spec.review_fraction,
            "seed": model.spec.seed,
            "n_jobs": model.spec.n_jobs,
        },
        "pipeline": model.pipeline.export_state(),
    }
    (directory / PIPELINE_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_trained_model(directory: Path, *, schema: FeatureSchema) -> TrainedTableModel:
    """Load one candidate package after validating its explicit contracts."""
    payload = json.loads((directory / PIPELINE_FILENAME).read_text(encoding="utf-8"))
    pipeline_state: FeaturePipelineState = payload["pipeline"]
    pipeline = FeaturePipeline.from_state(schema, pipeline_state)
    estimator = skops.io.load(directory / MODEL_FILENAME, trusted=[])
    if not isinstance(estimator, IsolationForest):
        raise TypeError("Candidate package does not contain an IsolationForest")
    return TrainedTableModel(
        fit_fiscal_year=str(payload["fit_fiscal_year"]),
        pipeline=pipeline,
        estimator=estimator,
        threshold=float(payload["threshold"]),
        sorted_training_scores=np.asarray(
            payload["sorted_training_scores"],
            dtype=float,
        ),
        reference_medians=np.asarray(payload["reference_medians"], dtype=float),
        reference_scales=np.asarray(payload["reference_scales"], dtype=float),
        spec=IsolationForestSpec(**payload["spec"]),
    )


def _train_table(
    feature_frame: pd.DataFrame,
    *,
    schema: FeatureSchema,
    fiscal_year: str,
    spec: IsolationForestSpec,
) -> TrainedTableModel:
    pipeline = FeaturePipeline(schema)
    matrix: FeatureMatrix = pipeline.fit_transform(feature_frame)
    values = matrix.values.to_numpy(dtype=float)
    estimator = IsolationForest(
        n_estimators=spec.n_estimators,
        max_samples=spec.max_samples,
        contamination="auto",
        random_state=spec.seed,
        n_jobs=spec.n_jobs,
    ).fit(values)
    training_scores = -estimator.decision_function(values)
    reference_medians = np.median(values, axis=0)
    reference_scales = np.quantile(values, 0.75, axis=0) - np.quantile(
        values,
        0.25,
        axis=0,
    )
    standard_deviations = np.std(values, axis=0)
    reference_scales = np.where(
        reference_scales > 0,
        reference_scales,
        standard_deviations,
    )
    reference_scales = np.where(reference_scales > 0, reference_scales, 1.0)
    return TrainedTableModel(
        fit_fiscal_year=fiscal_year,
        pipeline=pipeline,
        estimator=estimator,
        threshold=float(
            np.quantile(
                training_scores,
                1 - spec.review_fraction,
                method="higher",
            )
        ),
        sorted_training_scores=np.sort(training_scores),
        reference_medians=reference_medians,
        reference_scales=reference_scales,
        spec=spec,
    )
