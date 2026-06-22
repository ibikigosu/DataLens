"""Bounded, non-causal evidence for statistical anomaly scores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

MAX_EVIDENCE_FEATURES = 3
MAX_EVIDENCE_CHARACTERS = 1_000


@dataclass(frozen=True)
class FeatureDeviation:
    feature: str
    value: float
    reference_median: float
    scaled_deviation: float


@dataclass(frozen=True)
class AnomalyEvidence:
    interpretation: str
    anomaly_score: float
    anomaly_percentile: float
    top_feature_deviations: tuple[FeatureDeviation, ...]

    def to_json(self) -> str:
        """Serialize evidence and enforce the public size boundary."""
        serialized = json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)
        if len(serialized) > MAX_EVIDENCE_CHARACTERS:
            raise ValueError("Bounded anomaly evidence exceeded its size limit")
        return serialized


def build_bounded_evidence(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    reference_medians: np.ndarray,
    reference_scales: np.ndarray,
    anomaly_score: float,
    anomaly_percentile: float,
    feature_limit: int = MAX_EVIDENCE_FEATURES,
) -> AnomalyEvidence:
    """Describe the strongest deviations without asserting an issue or severity."""
    if not 1 <= feature_limit <= MAX_EVIDENCE_FEATURES:
        raise ValueError(f"feature_limit must be between 1 and {MAX_EVIDENCE_FEATURES}")
    deviations = (values - reference_medians) / reference_scales
    ordered_indices = np.argsort(np.abs(deviations))[::-1][:feature_limit]
    return AnomalyEvidence(
        interpretation=(
            "Statistical review evidence only. Anomaly score is not business "
            "severity and does not identify a specific data-quality issue."
        ),
        anomaly_score=round(float(anomaly_score), 6),
        anomaly_percentile=round(float(anomaly_percentile), 6),
        top_feature_deviations=tuple(
            FeatureDeviation(
                feature=feature_names[index],
                value=round(float(values[index]), 6),
                reference_median=round(float(reference_medians[index]), 6),
                scaled_deviation=round(float(deviations[index]), 6),
            )
            for index in ordered_indices
        ),
    )
