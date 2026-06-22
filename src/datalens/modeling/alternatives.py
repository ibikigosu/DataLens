"""One-Class SVM comparison models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
from sklearn.svm import OneClassSVM

from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import (
    FeaturePipeline,
    FeaturePipelineState,
    FeatureSchema,
    FeatureTable,
)
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.modeling.evidence import build_bounded_evidence


@dataclass(frozen=True)
class OneClassSvmSpec:
    nu: float = 0.01
    review_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not 0 < self.nu <= 1:
            raise ValueError("nu must be between zero and one")
        if not 0 < self.review_fraction < 1:
            raise ValueError("review_fraction must be between zero and one")


@dataclass
class OneClassSvmModel:
    fit_fiscal_year: str
    pipeline: FeaturePipeline
    estimator: OneClassSVM
    threshold: float
    sorted_training_scores: np.ndarray
    reference_medians: np.ndarray
    reference_scales: np.ndarray
    spec: OneClassSvmSpec

    @property
    def table(self) -> FeatureTable:
        return self.pipeline.schema.table

    def score(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
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
                "model_name": "one_class_svm",
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
class OneClassSvmModels:
    vendor: OneClassSvmModel
    transaction: OneClassSvmModel

    def for_table(self, table: FeatureTable) -> OneClassSvmModel:
        return self.vendor if table is FeatureTable.VENDOR else self.transaction


def train_one_class_svms(
    dataset: DevelopmentFeatureDataset,
    *,
    vendor_spec: OneClassSvmSpec | None = None,
    transaction_spec: OneClassSvmSpec | None = None,
) -> OneClassSvmModels:
    return OneClassSvmModels(
        vendor=_train_table(
            dataset.vendors,
            schema=VENDOR_FEATURE_SCHEMA,
            fiscal_year=dataset.fiscal_year,
            spec=vendor_spec or OneClassSvmSpec(nu=0.025, review_fraction=0.025),
        ),
        transaction=_train_table(
            dataset.transactions,
            schema=TRANSACTION_FEATURE_SCHEMA,
            fiscal_year=dataset.fiscal_year,
            spec=transaction_spec or OneClassSvmSpec(nu=0.005, review_fraction=0.005),
        ),
    )


def save_one_class_svm(model: OneClassSvmModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    estimator_path = directory / "estimator.skops"
    skops.io.dump(model.estimator, estimator_path)
    if skops.io.get_untrusted_types(file=estimator_path):
        estimator_path.unlink(missing_ok=True)
        raise ValueError("One-Class SVM contains untrusted serialized types")
    payload = {
        "fit_fiscal_year": model.fit_fiscal_year,
        "threshold": model.threshold,
        "sorted_training_scores": model.sorted_training_scores.tolist(),
        "reference_medians": model.reference_medians.tolist(),
        "reference_scales": model.reference_scales.tolist(),
        "spec": {
            "nu": model.spec.nu,
            "review_fraction": model.spec.review_fraction,
        },
        "pipeline": model.pipeline.export_state(),
    }
    (directory / "feature-pipeline.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_one_class_svm(
    directory: Path,
    *,
    schema: FeatureSchema,
) -> OneClassSvmModel:
    payload = json.loads((directory / "feature-pipeline.json").read_text(encoding="utf-8"))
    state: FeaturePipelineState = payload["pipeline"]
    estimator = skops.io.load(directory / "estimator.skops", trusted=[])
    if not isinstance(estimator, OneClassSVM):
        raise TypeError("Candidate package does not contain a OneClassSVM")
    return OneClassSvmModel(
        fit_fiscal_year=str(payload["fit_fiscal_year"]),
        pipeline=FeaturePipeline.from_state(schema, state),
        estimator=estimator,
        threshold=float(payload["threshold"]),
        sorted_training_scores=np.asarray(payload["sorted_training_scores"]),
        reference_medians=np.asarray(payload["reference_medians"]),
        reference_scales=np.asarray(payload["reference_scales"]),
        spec=OneClassSvmSpec(**payload["spec"]),
    )


def _train_table(
    feature_frame: pd.DataFrame,
    *,
    schema: FeatureSchema,
    fiscal_year: str,
    spec: OneClassSvmSpec,
) -> OneClassSvmModel:
    pipeline = FeaturePipeline(schema)
    matrix = pipeline.fit_transform(feature_frame)
    values = matrix.values.to_numpy(dtype=float)
    estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=spec.nu).fit(values)
    training_scores = -estimator.decision_function(values)
    medians = np.median(values, axis=0)
    scales = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    scales = np.where(scales > 0, scales, np.std(values, axis=0))
    scales = np.where(scales > 0, scales, 1.0)
    return OneClassSvmModel(
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
        reference_medians=medians,
        reference_scales=scales,
        spec=spec,
    )
