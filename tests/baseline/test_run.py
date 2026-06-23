import json
from pathlib import Path

import pandas as pd

from datalens.baseline import run
from datalens.configuration.loader import load_runtime_config


def _write_period_data(root: Path) -> None:
    period_dir = root / "fy2024"
    period_dir.mkdir(parents=True)
    vendors = pd.DataFrame(
        {
            "vendor_id": [f"UEI:V{index:011d}" for index in range(12)],
            "recipient_uei": [f"V{index:011d}" for index in range(12)],
            "recipient_name": [f"Vendor {index}" for index in range(12)],
            "recipient_country_code": ["USA"] * 12,
            "recipient_state_code": ["VA"] * 12,
        }
    )
    transactions = pd.DataFrame(
        {
            "contract_transaction_unique_key": [f"T{index}" for index in range(12)],
            "vendor_id": [f"UEI:V{index:011d}" for index in range(12)],
            "period_of_performance_start_date": pd.to_datetime(
                ["2024-01-01"] * 12,
                utc=True,
            ),
            "period_of_performance_current_end_date": pd.to_datetime(
                ["2024-12-31"] * 12,
                utc=True,
            ),
            "number_of_offers_received": [2] * 12,
            "action_date": pd.to_datetime(["2024-03-01"] * 12, utc=True),
        }
    )
    vendors.to_parquet(period_dir / "vendors.parquet", index=False)
    transactions.to_parquet(period_dir / "transactions.parquet", index=False)


def test_load_config_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = {
        "schema_version": 1,
        "seed": 42,
        "defects_per_type": 1,
    }
    path.write_text(json.dumps(config), encoding="utf-8")

    assert run.load_config(path) == run.BaselinePlan(
        schema_version=1,
        seed=42,
        defects_per_type=1,
    )


def test_run_period_persists_labels_findings_and_metrics(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    _write_period_data(processed)
    monkeypatch.setattr(run, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(run, "ARTIFACTS_DIR", artifacts)

    metrics = run.run_period(
        2024,
        run.BaselinePlan(schema_version=1, seed=42, defects_per_type=1),
        scoring_weights=load_runtime_config().schema.scoring_weights,
    )

    output_dir = artifacts / "baseline" / "fy2024"
    stored_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["controlled_defects"] == 9
    assert stored_metrics == metrics
    assert (output_dir / "controlled_defect_labels.parquet").exists()
    assert (output_dir / "findings.parquet").exists()
