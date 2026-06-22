"""Stable training and scoring interfaces for table-specific anomaly models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from datalens.features.pipeline import FeatureMatrix, FeaturePipeline, FeatureSchema, FeatureTable
from datalens.modeling.evidence import MAX_EVIDENCE_FEATURES, build_bounded_evidence


class ModelFamily(StrEnum):
    """Supported statistical anomaly-model families."""

    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"


@dataclass(frozen=True)
class ModelSpec:
    """Reproducible model and operational review-threshold configuration."""

    family: ModelFamily
    review_fraction: float
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.review_fraction < 1:
            raise ValueError("review_fraction must be between zero and one")

    @property
    def name(self) -> str:
        return self.family.value


@dataclass
class FittedAnomalyModel:
    """A fitted estimator plus the immutable scoring reference learned in training."""

    table: FeatureTable
    spec: ModelSpec
    schema_version: int
    feature_names: tuple[str, ...]
    estimator: Any
    threshold: float
    sorted_training_scores: np.ndarray
    reference_medians: np.ndarray
    reference_scales: np.ndarray

    def score(
        self,
        matrix: FeatureMatrix,
        *,
        evidence_feature_limit: int = MAX_EVIDENCE_FEATURES,
    ) -> pd.DataFrame:
        """Score one compatible feature matrix without fitting or changing state."""
        if matrix.table is not self.table:
            raise ValueError(
                f"Model for {self.table.value} cannot score {matrix.table.value} features"
            )
        if matrix.feature_names != self.feature_names:
            raise ValueError("Scoring feature names do not match the fitted model")

        values = matrix.values.to_numpy(dtype=float)
        raw_anomaly_scores = _anomaly_scores(self.estimator, values)
        percentiles = np.searchsorted(
            self.sorted_training_scores,
            raw_anomaly_scores,
            side="right",
        ) / len(self.sorted_training_scores)
        anomaly_scores = percentiles * 100
        predicted = raw_anomaly_scores >= self.threshold
        evidence = [
            build_bounded_evidence(
                row,
                feature_names=self.feature_names,
                reference_medians=self.reference_medians,
                reference_scales=self.reference_scales,
                anomaly_score=score,
                anomaly_percentile=percentile,
                feature_limit=evidence_feature_limit,
            )
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
                "model_name": self.spec.name,
                "anomaly_score": anomaly_scores,
                "rank_percentile": percentiles,
                "predicted": predicted,
                "priority_score": percentiles * 100,
                "evidence": evidence,
            }
        ).sort_values(
            ["priority_score", "record_id"],
            ascending=[False, True],
            ignore_index=True,
        )


@dataclass
class TableModelBundle:
    """Serializable preprocessing and anomaly-scoring boundary for one table."""

    fit_fiscal_year: str
    pipeline: FeaturePipeline
    model: FittedAnomalyModel

    @property
    def table(self) -> FeatureTable:
        return self.model.table

    def score(
        self,
        feature_frame: pd.DataFrame,
        *,
        evidence_feature_limit: int = MAX_EVIDENCE_FEATURES,
    ) -> pd.DataFrame:
        """Transform and score a table feature frame through the fitted workflow."""
        matrix = self.pipeline.transform(feature_frame)
        return self.model.score(
            matrix,
            evidence_feature_limit=evidence_feature_limit,
        )


def train_table_model(
    feature_frame: pd.DataFrame,
    *,
    schema: FeatureSchema,
    spec: ModelSpec,
    fit_fiscal_year: int | str,
    period_role: str,
) -> TableModelBundle:
    """Fit preprocessing and one anomaly model on development-period records only."""
    if period_role != "development":
        raise ValueError(
            "Statistical models may only be fitted on a development period, "
            f"but FY{fit_fiscal_year} is {period_role}"
        )

    pipeline = FeaturePipeline(schema)
    matrix = pipeline.fit_transform(feature_frame)
    values = matrix.values.to_numpy(dtype=float)
    estimator = _build_estimator(spec, sample_count=len(values))
    estimator.fit(values)
    training_scores = _anomaly_scores(estimator, values)
    threshold = float(
        np.quantile(
            training_scores,
            1 - spec.review_fraction,
            method="higher",
        )
    )
    reference_medians = np.median(values, axis=0)
    reference_scales = np.quantile(values, 0.75, axis=0) - np.quantile(
        values,
        0.25,
        axis=0,
    )
    fallback_scales = np.std(values, axis=0)
    reference_scales = np.where(reference_scales > 0, reference_scales, fallback_scales)
    reference_scales = np.where(reference_scales > 0, reference_scales, 1.0)
    model = FittedAnomalyModel(
        table=schema.table,
        spec=spec,
        schema_version=schema.schema_version,
        feature_names=matrix.feature_names,
        estimator=estimator,
        threshold=threshold,
        sorted_training_scores=np.sort(training_scores),
        reference_medians=reference_medians,
        reference_scales=reference_scales,
    )
    return TableModelBundle(
        fit_fiscal_year=str(fit_fiscal_year),
        pipeline=pipeline,
        model=model,
    )


def save_model_bundle(bundle: TableModelBundle, path: Path) -> None:
    """Persist a fitted table workflow for later API integration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> TableModelBundle:
    """Load a previously persisted table workflow."""
    bundle = joblib.load(path)
    if not isinstance(bundle, TableModelBundle):
        raise TypeError("The model artifact does not contain a TableModelBundle")
    return bundle


def _build_estimator(spec: ModelSpec, *, sample_count: int) -> Any:
    if sample_count < 3:
        raise ValueError("Anomaly models require at least three training records")
    parameters = dict(spec.parameters)
    if spec.family is ModelFamily.ISOLATION_FOREST:
        parameters.setdefault("n_estimators", 200)
        parameters.setdefault("max_samples", "auto")
        parameters.setdefault("n_jobs", -1)
        return IsolationForest(
            contamination="auto",
            random_state=spec.seed,
            **parameters,
        )
    if spec.family is ModelFamily.LOCAL_OUTLIER_FACTOR:
        requested_neighbors = int(parameters.pop("n_neighbors", 35))
        parameters.setdefault("n_jobs", -1)
        return LocalOutlierFactor(
            contamination="auto",
            n_neighbors=min(requested_neighbors, sample_count - 1),
            novelty=True,
            **parameters,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def _anomaly_scores(estimator: Any, values: np.ndarray) -> np.ndarray:
    return -np.asarray(estimator.decision_function(values), dtype=float)
