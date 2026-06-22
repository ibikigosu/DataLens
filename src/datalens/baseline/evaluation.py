"""Evaluation for issue-level and record-level baseline behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd

from datalens.baseline.rules import finding_keys, record_keys


def _classification_metrics(truth: set[Any], predicted: set[Any]) -> dict[str, float | int]:
    true_positives = len(truth & predicted)
    false_positives = len(predicted - truth)
    false_negatives = len(truth - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_baseline(
    labels: pd.DataFrame,
    findings: pd.DataFrame,
    *,
    evaluated_records: int,
    top_k: int = 50,
) -> dict[str, Any]:
    """Evaluate issue detection, record ranking, and operational review metrics."""
    truth_issue_keys = finding_keys(labels)
    predicted_issue_keys = finding_keys(findings)
    issue_metrics = _classification_metrics(truth_issue_keys, predicted_issue_keys)
    record_metrics = _classification_metrics(record_keys(labels), record_keys(findings))

    issue_type_metrics = {}
    for issue_type in sorted(labels["issue_type"].unique()):
        truth = finding_keys(labels.loc[labels["issue_type"].eq(issue_type)])
        predicted = finding_keys(findings.loc[findings["issue_type"].eq(issue_type)])
        issue_type_metrics[issue_type] = _classification_metrics(truth, predicted)

    high_critical_labels = labels.loc[labels["severity"].isin(["high", "critical"])]
    high_critical_truth = finding_keys(high_critical_labels)
    high_critical_recall = (
        len(high_critical_truth & predicted_issue_keys) / len(high_critical_truth)
        if high_critical_truth
        else 0.0
    )

    top_findings = findings.head(top_k)
    top_keys = list(
        top_findings[["target_table", "record_id", "issue_type"]]
        .astype("string")
        .itertuples(index=False, name=None)
    )
    top_k_precision = (
        sum(key in truth_issue_keys for key in top_keys) / len(top_keys) if top_keys else 0.0
    )
    false_alarms_per_1000 = (
        issue_metrics["false_positives"] / evaluated_records * 1000 if evaluated_records else 0.0
    )
    macro_issue_f1 = sum(metrics["f1"] for metrics in issue_type_metrics.values()) / len(
        issue_type_metrics
    )

    return {
        "evaluated_records": evaluated_records,
        "controlled_defects": len(labels),
        "findings": len(findings),
        "issue_level": issue_metrics,
        "record_level": record_metrics,
        "issue_type_metrics": issue_type_metrics,
        "macro_issue_f1": macro_issue_f1,
        "high_critical_issue_recall": high_critical_recall,
        f"top_{top_k}_precision": top_k_precision,
        "false_alarms_per_1000_records": false_alarms_per_1000,
    }
