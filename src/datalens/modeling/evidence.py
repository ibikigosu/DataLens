"""Bounded, non-causal evidence for statistical anomaly rankings."""

from __future__ import annotations

import json

import numpy as np

MAX_EVIDENCE_FEATURES = 3
MAX_EVIDENCE_CHARACTERS = 1_000


def build_bounded_evidence(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    reference_medians: np.ndarray,
    reference_scales: np.ndarray,
    anomaly_score: float,
    anomaly_percentile: float,
    feature_limit: int = MAX_EVIDENCE_FEATURES,
) -> str:
    """Describe the strongest feature deviations without claiming issue causality."""
    if feature_limit < 1 or feature_limit > MAX_EVIDENCE_FEATURES:
        raise ValueError(f"feature_limit must be between 1 and {MAX_EVIDENCE_FEATURES}")

    deviations = (values - reference_medians) / reference_scales
    ordered_indices = np.argsort(np.abs(deviations))[::-1][:feature_limit]
    evidence = {
        "interpretation": (
            "Statistical review evidence only. Anomaly score is not business severity "
            "and does not identify a specific data-quality issue."
        ),
        "anomaly_score": round(float(anomaly_score), 6),
        "anomaly_percentile": round(float(anomaly_percentile), 6),
        "top_feature_deviations": [
            {
                "feature": feature_names[index],
                "value": round(float(values[index]), 6),
                "reference_median": round(float(reference_medians[index]), 6),
                "scaled_deviation": round(float(deviations[index]), 6),
            }
            for index in ordered_indices
        ],
    }
    serialized = json.dumps(evidence, separators=(",", ":"), sort_keys=True)
    if len(serialized) > MAX_EVIDENCE_CHARACTERS:
        raise ValueError("Bounded anomaly evidence exceeded its size limit")
    return serialized
