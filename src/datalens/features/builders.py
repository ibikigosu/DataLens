"""Table-specific construction of vendor and transaction feature frames."""

from __future__ import annotations

import pandas as pd

from datalens.data.records import add_record_id
from datalens.features.pipeline import FeatureSchema
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA

VENDOR_BUSINESS_KEY = "vendor_id"
TRANSACTION_BUSINESS_KEY = "contract_transaction_unique_key"


def build_vendor_features(vendors: pd.DataFrame) -> pd.DataFrame:
    """Build the vendor feature frame while preserving the vendor business key."""
    return _build_table_features(
        vendors,
        table_name="vendor",
        business_key=VENDOR_BUSINESS_KEY,
        schema=VENDOR_FEATURE_SCHEMA,
    )


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build the transaction feature frame while preserving its business key."""
    return _build_table_features(
        transactions,
        table_name="transaction",
        business_key=TRANSACTION_BUSINESS_KEY,
        schema=TRANSACTION_FEATURE_SCHEMA,
    )


def _build_table_features(
    frame: pd.DataFrame,
    *,
    table_name: str,
    business_key: str,
    schema: FeatureSchema,
) -> pd.DataFrame:
    required_columns = (business_key, *schema.source_columns)
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{table_name.title()} records are missing columns: {missing_columns}")

    identified = add_record_id(frame, table_name=table_name)
    output_columns = (schema.record_id_column, *required_columns)
    return identified.loc[:, output_columns].copy()
