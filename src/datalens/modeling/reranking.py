"""Feedback-trained false-alarm reranking for deterministic findings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import skops.io
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = (
    "risk_score",
    "severity_rank",
    "record_issue_count",
    "model_rank_percentile",
)


@dataclass
class FalseAlarmReranker:
    """One supervised model that reorders findings without detecting issue types."""

    estimator: LogisticRegression

    def rerank(self, finding_features: pd.DataFrame) -> pd.DataFrame:
        reviewed = finding_features.copy()
        reviewed["review_confidence"] = self.estimator.predict_proba(
            reviewed.loc[:, FEATURE_COLUMNS]
        )[:, 1]
        reviewed["protected_critical"] = reviewed["severity"].eq("critical")
        reviewed["priority_score"] = (
            reviewed["protected_critical"].astype(float) * 10_000
            + reviewed["risk_score"] * 10
            + reviewed["review_confidence"]
        )
        return reviewed.sort_values(
            ["priority_score", "target_table", "record_id", "issue_type"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )


def save_false_alarm_reranker(
    reranker: FalseAlarmReranker,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skops.io.dump(reranker.estimator, path)
    if skops.io.get_untrusted_types(file=path):
        path.unlink(missing_ok=True)
        raise ValueError("False-alarm reranker contains untrusted serialized types")


def load_false_alarm_reranker(path: Path) -> FalseAlarmReranker:
    estimator = skops.io.load(path, trusted=[])
    if not isinstance(estimator, LogisticRegression):
        raise TypeError("Reranker package does not contain LogisticRegression")
    return FalseAlarmReranker(estimator=estimator)


def train_false_alarm_reranker(
    finding_features: pd.DataFrame,
    feedback: pd.DataFrame,
    *,
    seed: int = 42,
) -> FalseAlarmReranker:
    """Fit one global false-alarm model from issue-level review feedback."""
    training = finding_features.merge(
        feedback,
        on=["target_table", "record_id", "issue_type"],
        how="inner",
        validate="one_to_one",
    )
    if set(training["is_correct_flag"].unique()) != {False, True}:
        raise ValueError("Reranker training requires correct and false-alarm feedback")
    estimator = LogisticRegression(
        random_state=seed,
        class_weight="balanced",
        max_iter=1_000,
    ).fit(
        training.loc[:, FEATURE_COLUMNS],
        training["is_correct_flag"].astype(int),
    )
    return FalseAlarmReranker(estimator=estimator)


def build_finding_features(
    deterministic_findings: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Build issue-level reranker inputs while retaining deterministic identity."""
    anomaly = anomaly_scores[["target_table", "record_id", "rank_percentile"]].rename(
        columns={"rank_percentile": "model_rank_percentile"}
    )
    issue_counts = (
        deterministic_findings.groupby(["target_table", "record_id"])
        .size()
        .rename("record_issue_count")
        .reset_index()
    )
    features = deterministic_findings.merge(
        anomaly,
        on=["target_table", "record_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        issue_counts,
        on=["target_table", "record_id"],
        how="left",
        validate="many_to_one",
    )
    features["model_rank_percentile"] = features["model_rank_percentile"].fillna(0.0)
    return features


def simulated_feedback(
    deterministic_findings: pd.DataFrame,
    controlled_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Create reproducible demonstration feedback without calling it ground truth."""
    truth = set(
        controlled_labels[["target_table", "record_id", "issue_type"]]
        .astype("string")
        .itertuples(index=False, name=None)
    )
    feedback = deterministic_findings[["target_table", "record_id", "issue_type"]].copy()
    feedback["is_correct_flag"] = [
        key in truth for key in feedback.astype("string").itertuples(index=False, name=None)
    ]
    return feedback


def build_guarded_review_queue(
    reranked_findings: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Union deterministic findings and anomalies while protecting critical records."""
    deterministic_records = (
        reranked_findings.groupby(["target_table", "record_id"], as_index=False)
        .agg(
            deterministic_priority=("priority_score", "max"),
            deterministic_critical=("protected_critical", "max"),
        )
        .assign(has_deterministic_finding=True)
    )
    model_records = anomaly_scores.loc[
        anomaly_scores["predicted"],
        ["target_table", "record_id", "rank_percentile"],
    ].assign(has_model_finding=True)
    queue = deterministic_records.merge(
        model_records,
        on=["target_table", "record_id"],
        how="outer",
        validate="one_to_one",
    )
    queue["deterministic_critical"] = queue["deterministic_critical"].eq(True)
    queue["has_deterministic_finding"] = queue["has_deterministic_finding"].eq(True)
    queue["has_model_finding"] = queue["has_model_finding"].eq(True)
    queue["predicted"] = True
    tier = (
        queue["deterministic_critical"].astype(int) * 3
        + (queue["has_deterministic_finding"] & ~queue["deterministic_critical"]).astype(int) * 2
        + (queue["has_model_finding"] & ~queue["has_deterministic_finding"]).astype(int)
    )
    within_tier = queue["deterministic_priority"].fillna(queue["rank_percentile"].fillna(0.0) * 100)
    queue["priority_score"] = tier * 100_000 + within_tier
    return queue.sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )
