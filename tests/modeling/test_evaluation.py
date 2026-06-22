import pandas as pd

from datalens.modeling.evaluation import evaluate_record_ranking


def test_per_table_false_alarm_rates_use_table_specific_denominators() -> None:
    labels = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": ["V1", "T1"],
            "severity": ["critical", "critical"],
        }
    )
    ranking = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor", "transaction"],
            "record_id": ["V1", "V2", "T2"],
            "predicted": [True, True, True],
            "priority_score": [100.0, 90.0, 80.0],
        }
    )

    metrics = evaluate_record_ranking(
        labels,
        ranking,
        evaluated_records_by_table={"vendor": 100, "transaction": 1_000},
        top_k=1,
    )

    assert metrics.per_table["vendor"].false_alarms_per_1000_records == 10.0
    assert metrics.per_table["transaction"].false_alarms_per_1000_records == 1.0
    assert metrics.false_alarms_per_1000_records == 2 / 1_100 * 1_000
