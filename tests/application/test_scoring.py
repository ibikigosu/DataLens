import pandas as pd
import pytest

from datalens.application.scoring import ScoringService
from datalens.configuration.loader import load_runtime_config


def test_single_record_scoring_skips_cross_table_rules() -> None:
    service = ScoringService(load_runtime_config())
    transaction = pd.DataFrame(
        [
            {
                "contract_transaction_unique_key": "T1",
                "vendor_id": "UNKNOWN",
                "federal_action_obligation": 1,
                "total_dollars_obligated": 1,
                "number_of_offers_received": -1,
                "action_date": "2024-01-15",
                "period_of_performance_start_date": "2024-01-01",
                "period_of_performance_current_end_date": "2024-12-31",
                "award_type_code": "A",
                "type_of_contract_pricing_code": "J",
                "action_type_code": "A",
                "product_or_service_code": "X",
                "naics_code": "1",
                "extent_competed_code": "A",
                "solicitation_procedures_code": "A",
                "type_of_set_aside_code": "SBA",
            }
        ]
    )

    result = service.score_record(transaction, table="transaction", fiscal_year=2024)

    assert set(result.findings["issue_type"]) == {"negative_offer_count"}


def test_scoring_rejects_unknown_single_record_table() -> None:
    service = ScoringService(load_runtime_config())

    with pytest.raises(ValueError, match="Unknown table"):
        service.score_record(pd.DataFrame([{"id": "1"}]), table="invoice", fiscal_year=2024)


def test_batch_scoring_detects_duplicate_business_keys_after_schema_validation() -> None:
    service = ScoringService(load_runtime_config())
    vendors = pd.DataFrame(
        [
            {
                "vendor_id": "V1",
                "recipient_name": "Vendor",
                "recipient_uei": "ABCDEFGHIJKL",
                "recipient_country_code": "USA",
                "recipient_state_code": "VA",
                "source_transaction_count": 1,
                "address_variant_count": 1,
                "contracting_officers_determination_of_business_size_code": "S",
            },
            {
                "vendor_id": "V1",
                "recipient_name": "Vendor duplicate",
                "recipient_uei": "ABCDEFGHIJKL",
                "recipient_country_code": "USA",
                "recipient_state_code": "VA",
                "source_transaction_count": 1,
                "address_variant_count": 1,
                "contracting_officers_determination_of_business_size_code": "S",
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "contract_transaction_unique_key": "T1",
                "vendor_id": "V1",
                "federal_action_obligation": 1,
                "total_dollars_obligated": 1,
                "number_of_offers_received": 1,
                "action_date": "2024-01-15",
                "period_of_performance_start_date": "2024-01-01",
                "period_of_performance_current_end_date": "2024-12-31",
                "award_type_code": "A",
                "type_of_contract_pricing_code": "J",
                "action_type_code": "A",
                "product_or_service_code": "X",
                "naics_code": "1",
                "extent_competed_code": "A",
                "solicitation_procedures_code": "A",
                "type_of_set_aside_code": "SBA",
            }
        ]
    )

    result = service.score_batch(vendors, transactions, fiscal_year=2024)

    assert result.findings["issue_type"].eq("duplicate_vendor_id").sum() == 2
