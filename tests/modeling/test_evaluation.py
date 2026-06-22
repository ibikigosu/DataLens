import pandas as pd

from datalens.modeling.evaluation import (
    PromotionCriteria,
    evaluate_record_ranking,
    select_table_winners,
)
from datalens.modeling.scoring import build_guarded_review_queue


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_table": ["vendor", "vendor", "transaction"],
            "record_id": ["V1", "V2", "T1"],
            "severity": ["critical", "high", "critical"],
        }
    )


def test_record_evaluation_reports_requested_operational_metrics() -> None:
    ranking = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction", "vendor", "transaction"],
            "record_id": ["V1", "T2", "V3", "T1"],
            "predicted": [True, True, False, False],
            "priority_score": [100.0, 90.0, 80.0, 70.0],
            "rank_percentile": [1.0, 0.9, 0.8, 0.7],
        }
    )

    metrics = evaluate_record_ranking(
        _labels(),
        ranking,
        evaluated_records_by_table={"vendor": 4, "transaction": 6},
        top_k=2,
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1 / 3
    assert metrics["top_2_precision"] == 0.5
    assert metrics["false_alarms_per_1000_records"] == 100.0
    assert "macro_f1" in metrics
    assert metrics["rank_calibration"]


def test_per_table_false_alarm_rates_use_table_specific_denominators() -> None:
    ranking = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": ["V1", "T2"],
            "predicted": [True, True],
            "priority_score": [100.0, 90.0],
        }
    )

    metrics = evaluate_record_ranking(
        _labels(),
        ranking,
        evaluated_records_by_table={"vendor": 8, "transaction": 2},
    )

    assert metrics["false_alarms_per_1000_records"] == 100.0
    assert metrics["per_table"]["vendor"]["false_alarms_per_1000_records"] == 0.0
    assert metrics["per_table"]["transaction"]["false_alarms_per_1000_records"] == 500.0


def test_table_winners_are_selected_only_from_development_metrics() -> None:
    development_metrics = {
        "isolation_forest": {
            "per_table": {
                "vendor": {
                    "top_50_precision": 0.4,
                    "f1": 0.3,
                    "precision": 0.25,
                    "false_positives": 5,
                },
                "transaction": {
                    "top_50_precision": 0.1,
                    "f1": 0.1,
                    "precision": 0.1,
                    "false_positives": 10,
                },
            }
        },
        "local_outlier_factor": {
            "per_table": {
                "vendor": {
                    "top_50_precision": 0.2,
                    "f1": 0.4,
                    "precision": 0.4,
                    "false_positives": 2,
                },
                "transaction": {
                    "top_50_precision": 0.2,
                    "f1": 0.2,
                    "precision": 0.2,
                    "false_positives": 8,
                },
            }
        },
    }

    winners = select_table_winners(development_metrics)

    assert winners == {
        "transaction": "local_outlier_factor",
        "vendor": "isolation_forest",
    }


def test_guarded_queue_cannot_suppress_deterministic_critical_findings() -> None:
    model_scores = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": ["V9", "T9"],
            "model_name": ["isolation_forest", "isolation_forest"],
            "anomaly_score": [0.9, 0.8],
            "rank_percentile": [0.99, 0.98],
            "predicted": [True, True],
            "priority_score": [99.0, 98.0],
            "evidence": ["{}", "{}"],
        }
    )
    deterministic_findings = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": ["V1", "T1"],
            "issue_type": ["invalid_vendor_uei", "orphan_vendor_reference"],
            "severity": ["critical", "critical"],
            "risk_score": [100, 100],
        }
    )

    queue = build_guarded_review_queue(model_scores, deterministic_findings)

    queue_keys = set(queue[["target_table", "record_id"]].itertuples(index=False, name=None))
    assert {("vendor", "V1"), ("transaction", "T1")}.issubset(queue_keys)
    assert queue.head(2)["deterministic_critical"].all()


def test_promotion_requires_baseline_non_inferiority_and_critical_preservation() -> None:
    criteria = PromotionCriteria()
    candidate = {
        "top_50_precision": 0.2,
        "macro_f1": 0.2,
        "false_alarms_per_1000_records": 5.0,
    }
    baseline = {
        "top_50_precision": 0.8,
        "macro_f1": 0.8,
        "false_alarms_per_1000_records": 3.0,
    }
    guarded = {"high_critical_recall": 1.0}

    decision = criteria.assess(
        candidate,
        baseline,
        guarded,
        selection_period_role="development",
    )

    assert not decision["promoted"]
    assert decision["gates"]["development_only_selection"]
    assert decision["gates"]["deterministic_critical_findings_preserved"]
    assert not decision["gates"]["top_k_precision_non_inferior"]
