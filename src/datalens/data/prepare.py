"""Prepare vendor and transaction tables from raw USAspending extracts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from datalens.data.plan import load_dataset_plan
from datalens.data.usaspending import sha256_file
from datalens.paths import MANIFEST_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

DATE_COLUMNS = (
    "action_date",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "period_of_performance_potential_end_date",
    "solicitation_date",
    "initial_report_date",
    "last_modified_date",
)
NUMERIC_COLUMNS = (
    "federal_action_obligation",
    "total_dollars_obligated",
    "base_and_exercised_options_value",
    "current_total_value_of_award",
    "base_and_all_options_value",
    "potential_total_value_of_award",
    "number_of_offers_received",
)
VENDOR_COLUMNS = (
    "vendor_id",
    "recipient_uei",
    "recipient_duns",
    "recipient_name",
    "recipient_name_raw",
    "recipient_doing_business_as_name",
    "cage_code",
    "recipient_parent_uei",
    "recipient_parent_duns",
    "recipient_parent_name",
    "recipient_country_code",
    "recipient_address_line_1",
    "recipient_address_line_2",
    "recipient_city_name",
    "recipient_county_name",
    "recipient_state_code",
    "recipient_zip_4_code",
    "contracting_officers_determination_of_business_size_code",
    "contracting_officers_determination_of_business_size",
)


def clean_string_series(series: pd.Series) -> pd.Series:
    """Strip strings and normalize empty values without damaging identifier formatting."""
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def build_vendor_id(frame: pd.DataFrame) -> pd.Series:
    """Build a stable vendor join key, preferring UEI and falling back to DUNS."""
    uei = clean_string_series(frame["recipient_uei"])
    duns = clean_string_series(frame["recipient_duns"])
    vendor_id = pd.Series(pd.NA, index=frame.index, dtype="string")
    vendor_id.loc[uei.notna()] = "UEI:" + uei.loc[uei.notna()]
    fallback = uei.isna() & duns.notna()
    vendor_id.loc[fallback] = "DUNS:" + duns.loc[fallback]
    return vendor_id


def load_raw_transactions(fiscal_year: str) -> pd.DataFrame:
    """Load all transaction CSV parts for one fiscal year."""
    files = sorted((RAW_DATA_DIR / f"fy{fiscal_year}").glob("*PrimeTransactions*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No raw transaction files found for FY{fiscal_year}. Run acquisition first."
        )
    frames = [pd.read_csv(path, dtype="string", low_memory=False) for path in files]
    return pd.concat(frames, ignore_index=True)


def prepare_transactions(raw: pd.DataFrame, fiscal_year: str) -> pd.DataFrame:
    """Normalize source types and add the vendor relationship key."""
    transactions = raw.copy()
    for column in transactions.select_dtypes(include=["object", "string"]).columns:
        transactions[column] = clean_string_series(transactions[column])
    transactions["vendor_id"] = build_vendor_id(transactions)
    transactions["fiscal_year"] = int(fiscal_year)
    for column in DATE_COLUMNS:
        if column in transactions:
            transactions[column] = pd.to_datetime(transactions[column], errors="coerce", utc=True)
    for column in NUMERIC_COLUMNS:
        if column in transactions:
            transactions[column] = pd.to_numeric(transactions[column], errors="coerce")
    return transactions


def _variant_count(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def build_vendors(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive one latest-known vendor record per stable vendor identifier."""
    identified = transactions.loc[transactions["vendor_id"].notna()].copy()
    if identified.empty:
        return pd.DataFrame(columns=[*VENDOR_COLUMNS, "source_transaction_count"])

    summary = (
        identified.groupby("vendor_id", sort=True)
        .agg(
            source_transaction_count=("contract_transaction_unique_key", "size"),
            recipient_name_variant_count=("recipient_name", _variant_count),
            address_variant_count=("recipient_address_line_1", _variant_count),
        )
        .reset_index()
    )
    latest = (
        identified.sort_values(
            ["last_modified_date", "action_date"],
            ascending=[False, False],
            na_position="last",
        )
        .drop_duplicates("vendor_id", keep="first")
        .loc[:, VENDOR_COLUMNS]
    )
    return latest.merge(summary, on="vendor_id", how="left", validate="one_to_one")


def _file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROCESSED_DATA_DIR.parent)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_period(fiscal_year: str) -> Path:
    """Prepare and persist vendor and transaction tables for one fiscal year."""
    raw = load_raw_transactions(fiscal_year)
    transactions = prepare_transactions(raw, fiscal_year)
    vendors = build_vendors(transactions)

    destination = PROCESSED_DATA_DIR / f"fy{fiscal_year}"
    destination.mkdir(parents=True, exist_ok=True)
    transaction_path = destination / "transactions.parquet"
    vendor_path = destination / "vendors.parquet"
    transactions.to_parquet(transaction_path, index=False)
    vendors.to_parquet(vendor_path, index=False)

    manifest = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "fiscal_year": int(fiscal_year),
        "transaction_rows": len(transactions),
        "vendor_rows": len(vendors),
        "transaction_columns": list(transactions.columns),
        "vendor_columns": list(vendors.columns),
        "duplicate_transaction_keys": int(
            transactions["contract_transaction_unique_key"].duplicated(keep=False).sum()
        ),
        "missing_vendor_keys": int(transactions["vendor_id"].isna().sum()),
        "unmatched_vendor_keys": int(
            (
                (~transactions["vendor_id"].isin(vendors["vendor_id"]))
                & transactions["vendor_id"].notna()
            ).sum()
        ),
        "files": [_file_metadata(transaction_path), _file_metadata(vendor_path)],
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"prepared_pbs_fy{fiscal_year}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiscal-year")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_dataset_plan()
    fiscal_years = [args.fiscal_year] if args.fiscal_year else plan.fiscal_years
    for fiscal_year in fiscal_years:
        manifest = prepare_period(fiscal_year)
        print(f"Prepared FY{fiscal_year}: {manifest}")


if __name__ == "__main__":
    main()
