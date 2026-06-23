"""Reproducible mentor-demo procurement dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a paired dataset with balanced simulated review examples."""
    vendors = []
    for index in range(200):
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
    for index in range(100):
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
    transactions = pd.DataFrame(
        [
            {
                "contract_transaction_unique_key": "DEMO-T001",
                "vendor_id": "V000",
                "federal_action_obligation": 1000,
                "total_dollars_obligated": 1000,
                "number_of_offers_received": 2,
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
    return pd.DataFrame(vendors), transactions


def write_demo_files(output_dir: Path) -> None:
    vendors, transactions = demo_frames()
    output_dir.mkdir(parents=True, exist_ok=True)
    vendors.to_csv(output_dir / "vendors.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
