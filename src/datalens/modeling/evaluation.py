"""Record-level anomaly evaluation and development-only model promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def evaluate_record_ranking(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    evaluated_records: int,
    top_k: int = 50,
) -> dict[str, Any]:
    """Evaluate record detection and operational ranking at a shared grain."""
    truth = _record_keys(labels)
    predicted = _record_keys(ranking.loc[ranking["predicted"]])
    overall = _classification_metrics(truth, predicted)
    deduplicated_ranking = ranking.sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
    ).drop_duplicates(["target_table", "record_id"])
    top_records = list(
        deduplicated_ranking.head(top_k)[["target_table", "record_id"]]
        .astype("string")
        .itertuples(index=False, name=None)
    )
    top_k_precision = (
        sum(record in truth for record in top_records) / len(top_records) if top_records else 0.0
    )

    tables = sorted(
        set(labels["target_table"].astype(str)) | set(ranking["target_table"].astype(str))
    )
    per_table = {}
    for table in tables:
        table_truth = {key for key in truth if key[0] == table}
        table_predicted = {key for key in predicted if key[0] == table}
        table_metrics = _classification_metrics(table_truth, table_predicted)
        table_ranking = deduplicated_ranking.loc[deduplicated_ranking["target_table"].eq(table)]
        table_top_records = list(
            table_ranking.head(top_k)[["target_table", "record_id"]]
            .astype("string")
            .itertuples(index=False, name=None)
        )
        table_metrics[f"top_{top_k}_precision"] = (
            sum(record in table_truth for record in table_top_records) / len(table_top_records)
            if table_top_records
            else 0.0
        )
        table_metrics["false_alarms_per_1000_records"] = (
            table_metrics["false_positives"] / evaluated_records * 1_000
            if evaluated_records
            else 0.0
        )
        per_table[table] = table_metrics

    high_critical_truth = _record_keys(labels.loc[labels["severity"].isin(["high", "critical"])])
    high_critical_recall = (
        len(high_critical_truth & predicted) / len(high_critical_truth)
        if high_critical_truth
        else 0.0
    )
    macro_f1 = (
        sum(metrics["f1"] for metrics in per_table.values()) / len(per_table) if per_table else 0.0
    )
    return {
        "evaluated_records": evaluated_records,
        "controlled_defect_records": len(truth),
        "predicted_records": len(predicted),
        **overall,
        "macro_f1": macro_f1,
        "high_critical_recall": high_critical_recall,
        f"top_{top_k}_precision": top_k_precision,
        "false_alarms_per_1000_records": (
            overall["false_positives"] / evaluated_records * 1_000 if evaluated_records else 0.0
        ),
        "per_table": per_table,
        "rank_calibration": _rank_calibration(labels, deduplicated_ranking),
    }


def combine_selected_scores(
    scores_by_model: dict[str, pd.DataFrame],
    table_winners: dict[str, str],
) -> pd.DataFrame:
    """Combine table-specific development winners into one scoring workflow."""
    selected = [
        scores_by_model[model_name].loc[scores_by_model[model_name]["target_table"].eq(table)]
        for table, model_name in table_winners.items()
    ]
    if not selected:
        raise ValueError("At least one table winner is required")
    return pd.concat(selected, ignore_index=True).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def select_table_winners(
    development_metrics: dict[str, dict[str, Any]],
    *,
    top_k: int = 50,
) -> dict[str, str]:
    """Select each table's model using development metrics only."""
    if not development_metrics:
        raise ValueError("At least one development result is required")
    tables = sorted(
        {table for metrics in development_metrics.values() for table in metrics["per_table"]}
    )
    winners = {}
    for table in tables:
        ranked_candidates = sorted(development_metrics)
        winners[table] = max(
            ranked_candidates,
            key=lambda model_name: _selection_key(
                development_metrics[model_name]["per_table"][table],
                model_name,
                top_k=top_k,
            ),
        )
    return winners


@dataclass(frozen=True)
class PromotionCriteria:
    """Non-inferiority gates for replacing deterministic primary ranking."""

    minimum_top_k_ratio: float = 1.0
    minimum_macro_f1_ratio: float = 1.0
    maximum_false_alarm_increase_per_1000: float = 0.0
    required_guarded_high_critical_recall: float = 1.0

    def assess(
        self,
        candidate_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        guarded_metrics: dict[str, Any],
        *,
        selection_period_role: str,
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Return auditable promotion gates without consulting holdout selection data."""
        top_key = f"top_{top_k}_precision"
        gates = {
            "development_only_selection": selection_period_role == "development",
            "top_k_precision_non_inferior": candidate_metrics[top_key]
            >= baseline_metrics[top_key] * self.minimum_top_k_ratio,
            "macro_f1_non_inferior": candidate_metrics["macro_f1"]
            >= baseline_metrics["macro_f1"] * self.minimum_macro_f1_ratio,
            "false_alarm_rate_non_inferior": candidate_metrics["false_alarms_per_1000_records"]
            <= (
                baseline_metrics["false_alarms_per_1000_records"]
                + self.maximum_false_alarm_increase_per_1000
            ),
            "deterministic_critical_findings_preserved": guarded_metrics["high_critical_recall"]
            >= self.required_guarded_high_critical_recall,
        }
        return {
            "promoted": all(gates.values()),
            "gates": gates,
            "criteria": {
                "minimum_top_k_ratio": self.minimum_top_k_ratio,
                "minimum_macro_f1_ratio": self.minimum_macro_f1_ratio,
                "maximum_false_alarm_increase_per_1000": (
                    self.maximum_false_alarm_increase_per_1000
                ),
                "required_guarded_high_critical_recall": (
                    self.required_guarded_high_critical_recall
                ),
            },
        }


def _classification_metrics(
    truth: set[tuple[str, str]],
    predicted: set[tuple[str, str]],
) -> dict[str, float | int]:
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


def _record_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        frame[["target_table", "record_id"]].astype("string").itertuples(index=False, name=None)
    )


def _selection_key(
    metrics: dict[str, Any],
    model_name: str,
    *,
    top_k: int,
) -> tuple[Any, ...]:
    return (
        metrics[f"top_{top_k}_precision"],
        metrics["f1"],
        metrics["precision"],
        -metrics["false_positives"],
        model_name,
    )


def _rank_calibration(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
) -> list[dict[str, float | int]]:
    if "rank_percentile" not in ranking or ranking["rank_percentile"].isna().all():
        return []
    truth = _record_keys(labels)
    calibrated = ranking.copy()
    calibrated["is_controlled_defect"] = [
        key in truth
        for key in calibrated[["target_table", "record_id"]]
        .astype("string")
        .itertuples(index=False, name=None)
    ]
    calibrated["percentile_bin"] = pd.cut(
        calibrated["rank_percentile"],
        bins=[index / 10 for index in range(11)],
        include_lowest=True,
        labels=False,
    )
    return [
        {
            "percentile_bin": int(bin_number),
            "records": int(len(group)),
            "controlled_defect_rate": float(group["is_controlled_defect"].mean()),
        }
        for bin_number, group in calibrated.groupby("percentile_bin", observed=True)
    ]
