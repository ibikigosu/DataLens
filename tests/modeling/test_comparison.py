import pandas as pd

from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.modeling.comparison import compare_development_models
from datalens.modeling.workflow import EvaluationPeriod


def _dataset() -> DevelopmentFeatureDataset:
    return DevelopmentFeatureDataset(
        fiscal_year="2024",
        vendors=pd.DataFrame(
            {
                "_record_id": [f"vendor:{index}" for index in range(40)],
                "vendor_id": [f"V{index}" for index in range(40)],
                "source_transaction_count": range(1, 41),
                "address_variant_count": [1 + index % 3 for index in range(40)],
                "recipient_country_code": ["USA"] * 40,
                "recipient_state_code": ["VA", "MD"] * 20,
                "contracting_officers_determination_of_business_size_code": [
                    "S",
                    "O",
                ]
                * 20,
            }
        ),
        transactions=pd.DataFrame(
            {
                "_record_id": [f"transaction:{index}" for index in range(80)],
                "contract_transaction_unique_key": [f"T{index}" for index in range(80)],
                "federal_action_obligation": [float(index - 10) for index in range(80)],
                "total_dollars_obligated": [float(index * 10) for index in range(80)],
                "number_of_offers_received": [index % 5 for index in range(80)],
                "award_type_code": ["A", "B"] * 40,
                "type_of_contract_pricing_code": ["J", "K"] * 40,
                "action_type_code": ["A", "B"] * 40,
                "product_or_service_code": ["X", "Y"] * 40,
                "naics_code": ["1", "2"] * 40,
                "extent_competed_code": ["A", "B"] * 40,
                "solicitation_procedures_code": ["A", "B"] * 40,
                "type_of_set_aside_code": ["SBA", None] * 40,
            }
        ),
    )


def _period() -> EvaluationPeriod:
    dataset = _dataset()
    labels = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": ["vendor:0", "transaction:0"],
            "issue_type": ["invalid_vendor_uei", "negative_offer_count"],
            "severity": ["critical", "medium"],
        }
    )
    findings = pd.DataFrame(
        {
            "target_table": ["vendor", "vendor", "transaction"],
            "record_id": ["vendor:0", "vendor:1", "transaction:0"],
            "issue_type": [
                "invalid_vendor_uei",
                "duplicate_vendor_id",
                "negative_offer_count",
            ],
            "severity": ["critical", "critical", "medium"],
            "severity_rank": [4, 4, 2],
            "risk_score": [100, 100, 50],
        }
    )
    return EvaluationPeriod(
        fiscal_year="2024",
        role="development",
        vendors=dataset.vendors,
        transactions=dataset.transactions,
        labels=labels,
        deterministic_findings=findings,
        dataset_identity={"fixture": True},
    )


def test_comparison_selects_table_winners_and_protects_critical_findings() -> None:
    result = compare_development_models(_dataset(), _period())

    assert set(result.model_metrics) == {"isolation_forest", "one_class_svm"}
    assert set(result.table_winners) == {"vendor", "transaction"}
    assert result.guarded_metrics.high_critical_recall == 1.0
    assert result.guarded_queue.iloc[0]["deterministic_critical"]
    assert "review_confidence" in result.reranked_findings
    assert result.promotion.development_only_selection
