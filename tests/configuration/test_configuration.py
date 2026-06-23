import json
from pathlib import Path

import pandas as pd
import pytest

from datalens.configuration.loader import load_runtime_config
from datalens.configuration.models import ProcurementSchema
from datalens.configuration.schema import (
    validate_procurement_frames,
    validate_table_frame,
)


def _vendor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": ["V1"],
            "recipient_name": ["Vendor One"],
            "recipient_uei": ["ABCDEFGHIJKL"],
            "recipient_country_code": ["USA"],
            "recipient_state_code": ["VA"],
            "source_transaction_count": [1],
            "address_variant_count": [1],
            "contracting_officers_determination_of_business_size_code": ["S"],
        }
    )


def _transaction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_transaction_unique_key": ["T1"],
            "vendor_id": ["V1"],
            "federal_action_obligation": [100.0],
            "total_dollars_obligated": [100.0],
            "number_of_offers_received": [2],
            "action_date": ["2024-01-15"],
            "period_of_performance_start_date": ["2024-01-01"],
            "period_of_performance_current_end_date": ["2024-12-31"],
            "award_type_code": ["A"],
            "type_of_contract_pricing_code": ["J"],
            "action_type_code": ["A"],
            "product_or_service_code": ["X"],
            "naics_code": ["1"],
            "extent_competed_code": ["A"],
            "solicitation_procedures_code": ["A"],
            "type_of_set_aside_code": ["SBA"],
        }
    )


def test_runtime_configuration_records_reproducibility_versions() -> None:
    config = load_runtime_config()

    assert config.versions == {
        "schema": "1.0.0",
        "feature": "procurement-features-v1",
        "model": "anomaly-v1",
    }
    assert config.schema.scoring_weights["invalid_vendor_uei"] == 100
    assert config.model.families["isolation_forest"].vendor.parameters["n_estimators"] == 200


def test_environment_values_override_versioned_application_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATALENS_API_PORT", "9000")

    config = load_runtime_config()

    assert config.settings.api_port == 9000


def test_schema_validation_normalizes_a_valid_procurement_dataset() -> None:
    config = load_runtime_config()

    vendors, transactions = validate_procurement_frames(
        _vendor(),
        _transaction(),
        schema=config.schema,
    )

    assert str(vendors["vendor_id"].dtype) == "string"
    assert str(transactions["number_of_offers_received"].dtype) == "Int64"
    assert str(transactions["action_date"].dtype) == "datetime64[ns, UTC]"


def test_schema_validation_fails_before_processing_invalid_input() -> None:
    config = load_runtime_config()
    invalid = _transaction().drop(columns=["action_date"])

    with pytest.raises(ValueError, match="missing required columns.*action_date"):
        validate_table_frame(
            invalid,
            schema=config.schema,
            table_name="transaction",
        )


def test_invalid_configuration_has_a_clear_file_specific_error(tmp_path: Path) -> None:
    application = json.loads(Path("config/application/default.json").read_text(encoding="utf-8"))
    invalid_schema = tmp_path / "schema.json"
    invalid_schema.write_text('{"schema_id": "broken"}\n', encoding="utf-8")
    application["schema_config_path"] = str(invalid_schema)
    application_path = tmp_path / "application.json"
    application_path.write_text(json.dumps(application), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid configuration at .*schema\.json"):
        load_runtime_config(application_path)


def test_schema_rejects_missing_scoring_weights() -> None:
    payload = json.loads(Path("config/schema/gsa-pbs-v1.json").read_text(encoding="utf-8"))
    payload["scoring_weights"].pop("missing_vendor_name")

    with pytest.raises(ValueError, match="Scoring weights must match canonical issues"):
        ProcurementSchema.model_validate(payload)


def test_schema_rejects_relationships_to_undeclared_columns() -> None:
    payload = json.loads(Path("config/schema/gsa-pbs-v1.json").read_text(encoding="utf-8"))
    payload["relationships"][0]["from_column"] = "missing_column"

    with pytest.raises(ValueError, match="references undeclared columns"):
        ProcurementSchema.model_validate(payload)


def test_table_validation_allows_quality_defects_but_rejects_invalid_types() -> None:
    config = load_runtime_config()
    duplicate_vendors = pd.concat([_vendor(), _vendor()], ignore_index=True)
    invalid_transactions = _transaction()
    invalid_transactions["number_of_offers_received"] = invalid_transactions[
        "number_of_offers_received"
    ].astype(float)
    invalid_transactions.loc[0, "number_of_offers_received"] = 1.5

    validated = validate_table_frame(
        duplicate_vendors,
        schema=config.schema,
        table_name="vendor",
    )
    assert validated["vendor_id"].duplicated(keep=False).all()
    with pytest.raises(ValueError, match="must contain integers"):
        validate_table_frame(
            invalid_transactions,
            schema=config.schema,
            table_name="transaction",
        )


def test_paired_validation_rejects_orphan_vendor_references() -> None:
    config = load_runtime_config()
    transactions = _transaction()
    transactions.loc[0, "vendor_id"] = "UNKNOWN"

    with pytest.raises(ValueError, match="undeclared vendor keys"):
        validate_procurement_frames(
            _vendor(),
            transactions,
            schema=config.schema,
        )


def test_configuration_loader_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    missing_application = tmp_path / "missing.json"
    malformed_application = tmp_path / "malformed.json"
    malformed_application.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        load_runtime_config(missing_application)
    with pytest.raises(ValueError, match="not valid JSON"):
        load_runtime_config(malformed_application)


def test_table_validation_rejects_unknown_table_name() -> None:
    config = load_runtime_config()

    with pytest.raises(ValueError, match="Unknown table name"):
        validate_table_frame(
            _vendor(),
            schema=config.schema,
            table_name="invoice",
        )
