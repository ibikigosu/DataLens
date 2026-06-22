import json
from pathlib import Path

import pytest

from datalens.features.builders import (
    TRANSACTION_BUSINESS_KEY,
    VENDOR_BUSINESS_KEY,
)
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.paths import MANIFEST_DIR


@pytest.mark.parametrize(
    ("fiscal_year", "manifest_column_key", "business_key", "schema"),
    [
        (2024, "vendor_columns", VENDOR_BUSINESS_KEY, VENDOR_FEATURE_SCHEMA),
        (2024, "transaction_columns", TRANSACTION_BUSINESS_KEY, TRANSACTION_FEATURE_SCHEMA),
        (2025, "vendor_columns", VENDOR_BUSINESS_KEY, VENDOR_FEATURE_SCHEMA),
        (2025, "transaction_columns", TRANSACTION_BUSINESS_KEY, TRANSACTION_FEATURE_SCHEMA),
    ],
)
def test_feature_schema_sources_exist_in_prepared_manifest(
    fiscal_year: int,
    manifest_column_key: str,
    business_key: str,
    schema,
) -> None:
    manifest_path = MANIFEST_DIR / f"prepared_pbs_fy{fiscal_year}.json"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prepared_columns = set(manifest[manifest_column_key])

    assert business_key in prepared_columns
    assert set(schema.source_columns) <= prepared_columns
