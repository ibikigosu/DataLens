"""Stable training and scoring interfaces for table-specific anomaly models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skops.io
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from datalens.features.pipeline import FeatureMatrix, FeaturePipeline, FeatureSchema, FeatureTable
from datalens.modeling.evidence import MAX_EVIDENCE_FEATURES, build_bounded_evidence


class ModelFamily(StrEnum):
    """Supported statistical anomaly-model families."""

    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"


MODEL_BUNDLE_FORMAT_VERSION = 1
_BUNDLE_MANIFEST_NAME = "bundle.json"
_BUNDLE_ESTIMATOR_NAME = "estimator.skops"
_BUNDLE_ARRAYS_NAME = "arrays.npz"
_SKOPS_TRUSTED_TYPES = {
    ModelFamily.ISOLATION_FOREST: frozenset(),
    ModelFamily.LOCAL_OUTLIER_FACTOR: frozenset(
        {
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
        }
    ),
}


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
    """Persist a fitted table workflow as data-only state plus a skops estimator."""
    if path.exists() and not path.is_dir():
        raise ValueError("Model bundle path must be a directory")
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / _BUNDLE_MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)

    estimator_data = skops.io.dumps(bundle.model.estimator)
    _validated_skops_trusted_types(
        skops.io.get_untrusted_types(data=estimator_data),
        bundle.model.spec.family,
    )
    _write_bytes_atomically(path / _BUNDLE_ESTIMATOR_NAME, estimator_data)
    _write_arrays_atomically(
        path / _BUNDLE_ARRAYS_NAME,
        sorted_training_scores=bundle.model.sorted_training_scores,
        reference_medians=bundle.model.reference_medians,
        reference_scales=bundle.model.reference_scales,
    )
    manifest = {
        "format_version": MODEL_BUNDLE_FORMAT_VERSION,
        "fit_fiscal_year": bundle.fit_fiscal_year,
        "pipeline": bundle.pipeline.export_state(),
        "model": {
            "table": bundle.model.table.value,
            "family": bundle.model.spec.family.value,
            "review_fraction": bundle.model.spec.review_fraction,
            "seed": bundle.model.spec.seed,
            "parameters": bundle.model.spec.parameters,
            "schema_version": bundle.model.schema_version,
            "feature_names": list(bundle.model.feature_names),
            "threshold": bundle.model.threshold,
        },
    }
    _write_text_atomically(
        manifest_path,
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
    )


def load_model_bundle(path: Path) -> TableModelBundle:
    """Load a validated data-only table workflow without executing pickle payloads."""
    if not path.is_dir():
        raise ValueError("Model bundle path must be a directory")
    manifest = _load_manifest(path / _BUNDLE_MANIFEST_NAME)
    if manifest.get("format_version") != MODEL_BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported model bundle format version")

    fit_fiscal_year = manifest.get("fit_fiscal_year")
    if not isinstance(fit_fiscal_year, str) or not fit_fiscal_year:
        raise ValueError("fit_fiscal_year must be a non-empty string")
    pipeline_state = _mapping_value(manifest, "pipeline")
    pipeline = FeaturePipeline.from_state(pipeline_state)
    model_state = _mapping_value(manifest, "model")
    table = FeatureTable(_string_value(model_state, "table"))
    family = ModelFamily(_string_value(model_state, "family"))
    parameters = dict(_mapping_value(model_state, "parameters"))
    spec = ModelSpec(
        family=family,
        review_fraction=_bounded_fraction(model_state, "review_fraction"),
        seed=_integer_value(model_state, "seed"),
        parameters=parameters,
    )
    schema_version = _integer_value(model_state, "schema_version")
    feature_names = _string_tuple(model_state, "feature_names")
    threshold = _finite_float(model_state, "threshold")

    if pipeline.schema.table is not table:
        raise ValueError("Pipeline and model tables do not match")
    if pipeline.schema.schema_version != schema_version:
        raise ValueError("Pipeline and model schema versions do not match")
    if pipeline.schema.output_feature_names != feature_names:
        raise ValueError("Pipeline and model feature names do not match")

    arrays = _load_model_arrays(path / _BUNDLE_ARRAYS_NAME, feature_count=len(feature_names))
    estimator_path = path / _BUNDLE_ESTIMATOR_NAME
    trusted_types = _validated_skops_trusted_types(
        skops.io.get_untrusted_types(file=estimator_path),
        family,
    )
    estimator = skops.io.load(estimator_path, trusted=trusted_types)
    _validate_estimator_family(estimator, family)
    if getattr(estimator, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError("Estimator feature count does not match the model bundle")

    return TableModelBundle(
        fit_fiscal_year=fit_fiscal_year,
        pipeline=pipeline,
        model=FittedAnomalyModel(
            table=table,
            spec=spec,
            schema_version=schema_version,
            feature_names=feature_names,
            estimator=estimator,
            threshold=threshold,
            sorted_training_scores=arrays["sorted_training_scores"],
            reference_medians=arrays["reference_medians"],
            reference_scales=arrays["reference_scales"],
        ),
    )


def skops_trusted_types_for_model(model: FittedAnomalyModel) -> list[str]:
    """Return only fixed, reviewed skops types required by a supported model."""
    serialized = skops.io.dumps(model.estimator)
    return _validated_skops_trusted_types(
        skops.io.get_untrusted_types(data=serialized),
        model.spec.family,
    )


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


def _validated_skops_trusted_types(
    untrusted_types: list[str],
    family: ModelFamily,
) -> list[str]:
    reported = set(untrusted_types)
    unexpected = sorted(reported - _SKOPS_TRUSTED_TYPES[family])
    if unexpected:
        raise ValueError(f"Unexpected skops types for {family.value}: {unexpected}")
    return sorted(reported)


def _validate_estimator_family(estimator: Any, family: ModelFamily) -> None:
    expected_type = {
        ModelFamily.ISOLATION_FOREST: IsolationForest,
        ModelFamily.LOCAL_OUTLIER_FACTOR: LocalOutlierFactor,
    }[family]
    if not isinstance(estimator, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} estimator for model family {family.value}"
        )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Model bundle manifest is missing or invalid") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Model bundle manifest must be an object")
    return manifest


def _load_model_arrays(path: Path, *, feature_count: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_names = {
                "sorted_training_scores",
                "reference_medians",
                "reference_scales",
            }
            if set(archive.files) != expected_names:
                raise ValueError("Model bundle arrays have unexpected entries")
            arrays = {name: np.asarray(archive[name], dtype=float) for name in expected_names}
    except (OSError, ValueError) as error:
        raise ValueError("Model bundle arrays are missing or invalid") from error

    scores = arrays["sorted_training_scores"]
    medians = arrays["reference_medians"]
    scales = arrays["reference_scales"]
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Training scores must be a non-empty finite vector")
    if np.any(np.diff(scores) < 0):
        raise ValueError("Training scores must be sorted")
    if medians.shape != (feature_count,) or not np.isfinite(medians).all():
        raise ValueError("Reference medians do not match the feature schema")
    if scales.shape != (feature_count,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Reference scales do not match the feature schema")
    return arrays


def _write_arrays_atomically(path: Path, **arrays: np.ndarray) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary_path.replace(path)


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(data)
    temporary_path.replace(path)


def _write_text_atomically(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _string_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer_value(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _finite_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _bounded_fraction(mapping: Mapping[str, Any], key: str) -> float:
    value = _finite_float(mapping, key)
    if not 0 < value < 1:
        raise ValueError(f"{key} must be between zero and one")
    return value


def _string_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{key} must be a non-empty list of strings")
    return tuple(value)
