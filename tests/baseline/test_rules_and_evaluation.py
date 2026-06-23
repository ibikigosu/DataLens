import pandas as pd

from datalens.baseline.evaluation import evaluate_baseline
from datalens.baseline.issues import ISSUES
from datalens.baseline.rules import run_rules
from datalens.configuration.loader import load_runtime_config


def test_rules_detect_expected_vendor_and_transaction_issues() -> None:
    vendors = pd.DataFrame(
        {
            "_record_id": ["V1", "V2"],
            "vendor_id": ["UEI:ONE", "UEI:ONE"],
            "recipient_name": [pd.NA, "Vendor"],
            "recipient_uei": ["BAD!", "ABCDEFGHIJKL"],
            "recipient_country_code": ["USA", "USA"],
            "recipient_state_code": ["ZZ", "VA"],
        }
    )
    transactions = pd.DataFrame(
        {
            "_record_id": ["T1", "T2"],
            "contract_transaction_unique_key": ["TX", "TX"],
            "vendor_id": ["UEI:MISSING", "UEI:ONE"],
            "period_of_performance_start_date": pd.to_datetime(
                ["2024-02-01", "2024-02-01"],
                utc=True,
            ),
            "period_of_performance_current_end_date": pd.to_datetime(
                ["2024-01-01", "2024-03-01"],
                utc=True,
            ),
            "number_of_offers_received": [-1, 2],
            "action_date": pd.to_datetime(["2020-01-01", "2024-03-01"], utc=True),
        }
    )

    findings = run_rules(
        vendors,
        transactions,
        fiscal_year=2024,
        scoring_weights=load_runtime_config().schema.scoring_weights,
    )

    assert {
        "missing_vendor_name",
        "invalid_vendor_uei",
        "invalid_domestic_state",
        "duplicate_vendor_id",
        "orphan_vendor_reference",
        "duplicate_transaction_key",
        "invalid_performance_date_order",
        "negative_offer_count",
        "action_date_outside_fiscal_year",
    }.issubset(set(findings["issue_type"]))
    assert findings["risk_score"].between(0, 100).all()


def test_evaluation_distinguishes_issue_and_record_metrics() -> None:
    labels = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor"],
            "record_id": ["V1", "V1"],
            "issue_type": ["missing_vendor_name", "invalid_vendor_uei"],
            "severity": ["high", "critical"],
        }
    )
    findings = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor"],
            "record_id": ["V1", "V2"],
            "issue_type": ["missing_vendor_name", "invalid_vendor_uei"],
            "severity": ["high", "critical"],
            "risk_score": [75, 100],
        }
    )

    metrics = evaluate_baseline(labels, findings, evaluated_records=10, top_k=2)

    assert metrics["issue_level"]["true_positives"] == 1
    assert metrics["issue_level"]["false_positives"] == 1
    assert metrics["issue_level"]["false_negatives"] == 1
    assert metrics["record_level"]["true_positives"] == 1
    assert metrics["record_level"]["false_positives"] == 1
    assert metrics["record_level"]["false_negatives"] == 0
    assert metrics["top_2_precision"] == 0.5
    assert metrics["false_alarms_per_1000_records"] == 100.0


def test_schema_scoring_weights_control_finding_priority() -> None:
    vendors = pd.DataFrame(
        {
            "_record_id": ["V1"],
            "vendor_id": ["V1"],
            "recipient_name": [pd.NA],
            "recipient_uei": ["ABCDEFGHIJKL"],
            "recipient_country_code": ["USA"],
            "recipient_state_code": ["VA"],
        }
    )
    transactions = pd.DataFrame(
        {
            "_record_id": ["T1"],
            "contract_transaction_unique_key": ["T1"],
            "vendor_id": ["V1"],
            "period_of_performance_start_date": pd.to_datetime(["2024-01-01"], utc=True),
            "period_of_performance_current_end_date": pd.to_datetime(
                ["2024-12-31"],
                utc=True,
            ),
            "number_of_offers_received": [1],
            "action_date": pd.to_datetime(["2024-01-15"], utc=True),
        }
    )
    weights = dict.fromkeys(ISSUES, 10)

    findings = run_rules(
        vendors,
        transactions,
        fiscal_year=2024,
        scoring_weights=weights,
    )

    assert (
        findings.loc[
            findings["issue_type"].eq("missing_vendor_name"),
            "risk_score",
        ].item()
        == 10
    )
