"""Reproducible user-demo procurement dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SIMULATED_CORRECT_ISSUE_TYPES = frozenset(
    {
        "invalid_vendor_uei",
        "duplicate_vendor_id",
        "duplicate_transaction_key",
    }
)


def simulated_feedback_verdict(issue_type: str) -> str:
    """Return the demo feedback label for a deterministic finding type."""
    return "correct_flag" if issue_type in SIMULATED_CORRECT_ISSUE_TYPES else "false_alarm"


def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a paired dataset with balanced simulated review examples."""
    vendors = []
    for index in range(40):
        vendors.append(
            {
                "vendor_id": f"V{index:03d}",
                "recipient_name": f"Demo vendor {index:03d}",
                "recipient_uei": "BAD!",
                "recipient_country_code": "USA",
                "recipient_state_code": "VA",
                "source_transaction_count": 1,
                "address_variant_count": 1,
                "contracting_officers_determination_of_business_size_code": "S",
            }
        )
    for index in range(20):
        for copy in range(2):
            vendors.append(
                {
                    "vendor_id": f"D{index:03d}",
                    "recipient_name": f"Legacy duplicate {index:03d}-{copy}",
                    "recipient_uei": f"A{index:011d}",
                    "recipient_country_code": "USA",
                    "recipient_state_code": "VA",
                    "source_transaction_count": 1,
                    "address_variant_count": 1,
                    "contracting_officers_determination_of_business_size_code": "S",
                }
            )
    transactions = []
    for index in range(10):
        for copy in range(2):
            transactions.append(
                _transaction_record(
                    key=f"DEMO-DUP-{index:03d}",
                    vendor_id=f"V{index:03d}",
                    obligation=2_000 + copy,
                )
            )
    for index in range(10):
        transactions.append(
            _transaction_record(
                key=f"DEMO-DATE-{index:03d}",
                vendor_id=f"V{index:03d}",
                start_date="2024-09-30",
                end_date="2024-01-01",
            )
        )
    for index in range(10):
        transactions.append(
            _transaction_record(
                key=f"DEMO-OFFERS-{index:03d}",
                vendor_id=f"V{index:03d}",
                offers=-1,
            )
        )
    for index in range(10):
        transactions.append(
            _transaction_record(
                key=f"DEMO-FY-{index:03d}",
                vendor_id=f"V{index:03d}",
                action_date="2023-01-15",
            )
        )
    return pd.DataFrame(vendors), pd.DataFrame(transactions)


def _transaction_record(
    *,
    key: str,
    vendor_id: str,
    obligation: int = 1_000,
    offers: int = 2,
    action_date: str = "2024-01-15",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> dict[str, object]:
    return {
        "contract_transaction_unique_key": key,
        "vendor_id": vendor_id,
        "federal_action_obligation": obligation,
        "total_dollars_obligated": obligation,
        "number_of_offers_received": offers,
        "action_date": action_date,
        "period_of_performance_start_date": start_date,
        "period_of_performance_current_end_date": end_date,
        "award_type_code": "A",
        "type_of_contract_pricing_code": "J",
        "action_type_code": "A",
        "product_or_service_code": "X",
        "naics_code": "1",
        "extent_competed_code": "A",
        "solicitation_procedures_code": "A",
        "type_of_set_aside_code": "SBA",
    }


def write_demo_files(output_dir: Path) -> None:
    vendors, transactions = demo_frames()
    output_dir.mkdir(parents=True, exist_ok=True)
    vendors.to_csv(output_dir / "vendors.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
