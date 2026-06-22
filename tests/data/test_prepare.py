import json
from pathlib import Path

import pandas as pd
import pytest

from datalens.data import prepare
from datalens.data.prepare import build_vendor_id, build_vendors, prepare_transactions


def test_build_vendor_id_prefers_uei_and_falls_back_to_duns() -> None:
    frame = pd.DataFrame(
        {
            "recipient_uei": [" UEI123 ", None, None],
            "recipient_duns": ["DUNS1", " 123456789 ", None],
        }
    )

    result = build_vendor_id(frame)

    assert result.iloc[0] == "UEI:UEI123"
    assert result.iloc[1] == "DUNS:123456789"
    assert pd.isna(result.iloc[2])


def test_prepare_transactions_preserves_identifiers_and_converts_types() -> None:
    raw = pd.DataFrame(
        {
            "contract_transaction_unique_key": ["T1"],
            "recipient_uei": ["ABC"],
            "recipient_duns": ["001234567"],
            "action_date": ["2024-03-01"],
            "federal_action_obligation": ["125.50"],
        }
    )

    result = prepare_transactions(raw, "2024")

    assert result.loc[0, "recipient_duns"] == "001234567"
    assert result.loc[0, "vendor_id"] == "UEI:ABC"
    assert result.loc[0, "fiscal_year"] == 2024
    assert result.loc[0, "federal_action_obligation"] == 125.5
    assert str(result["action_date"].dtype) == "datetime64[ns, UTC]"


def test_build_vendors_uses_latest_record_and_reports_variants() -> None:
    transactions = pd.DataFrame(
        {
            "vendor_id": ["UEI:A", "UEI:A"],
            "contract_transaction_unique_key": ["T1", "T2"],
            "recipient_uei": ["A", "A"],
            "recipient_duns": [pd.NA, pd.NA],
            "recipient_name": ["Old Name", "Current Name"],
            "recipient_name_raw": ["Old Name", "Current Name"],
            "recipient_doing_business_as_name": [pd.NA, pd.NA],
            "cage_code": ["CAGE", "CAGE"],
            "recipient_parent_uei": [pd.NA, pd.NA],
            "recipient_parent_duns": [pd.NA, pd.NA],
            "recipient_parent_name": [pd.NA, pd.NA],
            "recipient_country_code": ["USA", "USA"],
            "recipient_address_line_1": ["1 Old St", "2 New St"],
            "recipient_address_line_2": [pd.NA, pd.NA],
            "recipient_city_name": ["Old", "New"],
            "recipient_county_name": ["County", "County"],
            "recipient_state_code": ["VA", "VA"],
            "recipient_zip_4_code": ["00000", "11111"],
            "contracting_officers_determination_of_business_size_code": ["S", "S"],
            "contracting_officers_determination_of_business_size": ["SMALL", "SMALL"],
            "last_modified_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01"],
                utc=True,
            ),
            "action_date": pd.to_datetime(["2024-01-01", "2024-02-01"], utc=True),
        }
    )

    vendors = build_vendors(transactions)

    assert len(vendors) == 1
    assert vendors.loc[0, "recipient_name"] == "Current Name"
    assert vendors.loc[0, "source_transaction_count"] == 2
    assert vendors.loc[0, "recipient_name_variant_count"] == 2
    assert vendors.loc[0, "address_variant_count"] == 2


def test_load_raw_transactions_requires_acquisition(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prepare, "RAW_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Run acquisition first"):
        prepare.load_raw_transactions("2024")


def test_prepare_period_writes_tables_and_manifest(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"
    period_dir = raw_dir / "fy2024"
    period_dir.mkdir(parents=True)
    source = pd.DataFrame(
        {
            "contract_transaction_unique_key": ["T1", "T2"],
            "recipient_uei": ["A", "A"],
            "recipient_duns": ["001", "001"],
            "recipient_name": ["Vendor A", "Vendor A"],
            "recipient_name_raw": ["Vendor A", "Vendor A"],
            "recipient_doing_business_as_name": [None, None],
            "cage_code": ["CAGE", "CAGE"],
            "recipient_parent_uei": [None, None],
            "recipient_parent_duns": [None, None],
            "recipient_parent_name": [None, None],
            "recipient_country_code": ["USA", "USA"],
            "recipient_address_line_1": ["1 Main St", "1 Main St"],
            "recipient_address_line_2": [None, None],
            "recipient_city_name": ["City", "City"],
            "recipient_county_name": ["County", "County"],
            "recipient_state_code": ["VA", "VA"],
            "recipient_zip_4_code": ["00123", "00123"],
            "contracting_officers_determination_of_business_size_code": ["S", "S"],
            "contracting_officers_determination_of_business_size": ["SMALL", "SMALL"],
            "action_date": ["2024-01-01", "2024-02-01"],
            "last_modified_date": ["2024-01-02", "2024-02-02"],
            "federal_action_obligation": ["10", "20"],
        }
    )
    source.to_csv(period_dir / "Contracts_PrimeTransactions_part.csv", index=False)
    monkeypatch.setattr(prepare, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(prepare, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(prepare, "MANIFEST_DIR", manifest_dir)

    manifest_path = prepare.prepare_period("2024")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["transaction_rows"] == 2
    assert manifest["vendor_rows"] == 1
    assert (processed_dir / "fy2024" / "transactions.parquet").exists()
    assert (processed_dir / "fy2024" / "vendors.parquet").exists()
