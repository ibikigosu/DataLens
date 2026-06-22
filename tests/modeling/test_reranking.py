import pandas as pd

from datalens.modeling.reranking import (
    build_finding_features,
    build_guarded_review_queue,
    simulated_feedback,
    train_false_alarm_reranker,
)


def test_reranker_changes_order_without_suppressing_critical_findings() -> None:
    findings = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor", "transaction"],
            "record_id": ["V1", "V2", "T1"],
            "issue_type": ["critical_true", "critical_false", "medium_true"],
            "severity": ["critical", "critical", "medium"],
            "severity_rank": [4, 4, 2],
            "risk_score": [100, 100, 50],
        }
    )
    labels = findings.iloc[[0, 2]][["target_table", "record_id", "issue_type", "severity"]]
    anomaly_scores = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor", "transaction"],
            "record_id": ["V1", "V2", "T1"],
            "rank_percentile": [0.9, 0.1, 0.8],
            "predicted": [True, True, True],
        }
    )
    features = build_finding_features(findings, anomaly_scores)
    feedback = simulated_feedback(findings, labels)

    reranker = train_false_alarm_reranker(features, feedback)
    reranked = reranker.rerank(features)
    queue = build_guarded_review_queue(reranked, anomaly_scores)

    assert set(reranked["issue_type"]) == set(findings["issue_type"])
    assert queue.loc[queue["deterministic_critical"], "record_id"].tolist() == [
        "V1",
        "V2",
    ]
    assert queue["predicted"].all()
