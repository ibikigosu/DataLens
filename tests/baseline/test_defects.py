import pandas as pd

from datalens.baseline.defects import inject_controlled_defects


def _vendors(count: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": [f"UEI:V{index:011d}" for index in range(count)],
            "recipient_uei": [f"V{index:011d}" for index in range(count)],
            "recipient_name": [f"Vendor {index}" for index in range(count)],
            "recipient_country_code": ["USA"] * count,
            "recipient_state_code": ["VA"] * count,
        }
    )


def _transactions(count: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_transaction_unique_key": [f"T{index}" for index in range(count)],
            "vendor_id": [f"UEI:V{index % 50:011d}" for index in range(count)],
            "period_of_performance_start_date": pd.to_datetime(
                ["2024-01-01"] * count,
                utc=True,
            ),
            "period_of_performance_current_end_date": pd.to_datetime(
                ["2024-12-31"] * count,
                utc=True,
            ),
            "number_of_offers_received": [2] * count,
            "action_date": pd.to_datetime(["2024-03-01"] * count, utc=True),
        }
    )


def test_controlled_defects_are_reproducible_and_issue_labeled() -> None:
    first = inject_controlled_defects(
        _vendors(),
        _transactions(),
        fiscal_year=2024,
        seed=42,
        defects_per_type=2,
    )
    second = inject_controlled_defects(
        _vendors(),
        _transactions(),
        fiscal_year=2024,
        seed=42,
        defects_per_type=2,
    )

    first_vendors, first_transactions, first_labels = first
    second_vendors, second_transactions, second_labels = second

    pd.testing.assert_frame_equal(first_vendors, second_vendors)
    pd.testing.assert_frame_equal(first_transactions, second_transactions)
    pd.testing.assert_frame_equal(first_labels, second_labels)
    assert len(first_labels) == 18
    assert first_labels["issue_type"].nunique() == 9
    assert first_labels["original_value"].notna().all()
    assert first_vendors["_record_id"].is_unique
    assert first_transactions["_record_id"].is_unique
