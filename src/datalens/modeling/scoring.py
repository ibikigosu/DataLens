"""Review-queue construction with deterministic critical-finding protection."""

from __future__ import annotations

import json

import pandas as pd


def baseline_record_ranking(findings: pd.DataFrame) -> pd.DataFrame:
    """Collapse issue findings into one deterministic record-level ranking."""
    if findings.empty:
        return pd.DataFrame(
            columns=[
                "target_table",
                "record_id",
                "predicted",
                "priority_score",
                "deterministic_critical",
                "deterministic_issue_types",
            ]
        )

    prepared = findings.copy()
    prepared["deterministic_critical"] = prepared["severity"].eq("critical")
    ranking = (
        prepared.groupby(["target_table", "record_id"], as_index=False)
        .agg(
            priority_score=("risk_score", "max"),
            deterministic_critical=("deterministic_critical", "max"),
            deterministic_issue_types=(
                "issue_type",
                lambda values: json.dumps(sorted(set(values))),
            ),
        )
        .assign(predicted=True)
    )
    return ranking.sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def build_guarded_review_queue(
    model_scores: pd.DataFrame,
    deterministic_findings: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model anomalies without allowing deterministic findings to disappear."""
    baseline = baseline_record_ranking(deterministic_findings).rename(
        columns={"priority_score": "deterministic_risk_score"}
    )
    model = model_scores.loc[
        model_scores["predicted"],
        [
            "target_table",
            "record_id",
            "model_name",
            "anomaly_score",
            "rank_percentile",
            "evidence",
        ],
    ].rename(columns={"evidence": "model_evidence"})

    queue = baseline.merge(
        model,
        on=["target_table", "record_id"],
        how="outer",
        validate="one_to_one",
    )
    queue["deterministic_critical"] = queue["deterministic_critical"].eq(True)
    queue["has_deterministic_finding"] = queue["deterministic_risk_score"].notna()
    queue["has_model_finding"] = queue["model_name"].notna()
    queue["predicted"] = True
    priority_tier = (
        queue["deterministic_critical"].astype(int) * 3
        + (queue["has_deterministic_finding"] & ~queue["deterministic_critical"]).astype(int) * 2
        + (queue["has_model_finding"] & ~queue["has_deterministic_finding"]).astype(int)
    )
    within_tier = queue["deterministic_risk_score"].fillna(queue["rank_percentile"].fillna(0) * 100)
    queue["priority_score"] = priority_tier * 1_000 + within_tier
    queue = queue.sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )

    critical_keys = set(
        deterministic_findings.loc[
            deterministic_findings["severity"].eq("critical"),
            ["target_table", "record_id"],
        ]
        .astype("string")
        .itertuples(index=False, name=None)
    )
    queue_keys = set(
        queue[["target_table", "record_id"]].astype("string").itertuples(index=False, name=None)
    )
    if not critical_keys.issubset(queue_keys):
        raise RuntimeError("A deterministic critical finding was suppressed")
    return queue
