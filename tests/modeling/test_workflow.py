import json
from pathlib import Path

import pandas as pd
import pytest

from datalens.baseline.run import BaselinePlan
from datalens.data.plan import (
    AgencyFilter,
    DatasetPlan,
    FiscalPeriod,
)
from datalens.modeling.workflow import (
    evaluate_temporal_holdout,
    run_development_experiment,
)


class RecordingTracker:
    def __init__(self) -> None:
        self.run_names: list[str] = []

    def log_run(self, *, run_name: str, **_: object) -> str:
        self.run_names.append(run_name)
        return f"run-{len(self.run_names)}"


def _plan() -> DatasetPlan:
    return DatasetPlan(
        source_name="test",
        api_base_url="https://example.test",
        dataset_name="test",
        agency=AgencyFilter(
            type="awarding",
            tier="subtier",
            name="test",
            toptier_name="test",
        ),
        award_type_codes=("A",),
        date_type="action_date",
        periods=(
            FiscalPeriod("2024", "development", "2023-10-01", "2024-09-30"),
            FiscalPeriod(
                "2025",
                "temporal_holdout",
                "2024-10-01",
                "2025-09-30",
            ),
        ),
    )


def _records(fiscal_year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    vendors = pd.DataFrame(
        {
            "vendor_id": [f"UEI:{fiscal_year}:{index}" for index in range(10)],
            "recipient_name": [f"Vendor {index}" for index in range(10)],
            "recipient_uei": [f"ABCDEF{index:06d}" for index in range(10)],
            "recipient_country_code": ["USA"] * 10,
            "recipient_state_code": ["VA", "MD"] * 5,
            "source_transaction_count": range(1, 11),
            "address_variant_count": [1 + index % 2 for index in range(10)],
            "contracting_officers_determination_of_business_size_code": [
                "S",
                "O",
            ]
            * 5,
        }
    )
    transactions = pd.DataFrame(
        {
            "contract_transaction_unique_key": [f"{fiscal_year}:T{index}" for index in range(20)],
            "vendor_id": [vendors.loc[index % 10, "vendor_id"] for index in range(20)],
            "period_of_performance_start_date": pd.to_datetime(
                ["2024-01-01"] * 20,
                utc=True,
            ),
            "period_of_performance_current_end_date": pd.to_datetime(
                ["2024-02-01"] * 20,
                utc=True,
            ),
            "action_date": pd.to_datetime(
                [f"{int(fiscal_year) - 1}-11-01"] * 20,
                utc=True,
            ),
            "federal_action_obligation": [float(index - 3) for index in range(20)],
            "total_dollars_obligated": [float(index * 10) for index in range(20)],
            "number_of_offers_received": [index % 4 for index in range(20)],
            "award_type_code": ["A", "B"] * 10,
            "type_of_contract_pricing_code": ["J", "K"] * 10,
            "action_type_code": ["A", "B"] * 10,
            "product_or_service_code": ["X", "Y"] * 10,
            "naics_code": ["1", "2"] * 10,
            "extent_competed_code": ["A", "B"] * 10,
            "solicitation_procedures_code": ["A", "B"] * 10,
            "type_of_set_aside_code": ["SBA", None] * 10,
        }
    )
    return vendors, transactions


def test_holdout_evaluation_requires_untampered_development_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracker = RecordingTracker()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for fiscal_year in ("2024", "2025"):
        (manifests / f"prepared_pbs_fy{fiscal_year}.json").write_text(
            json.dumps({"fiscal_year": int(fiscal_year)}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "datalens.modeling.workflow.MANIFEST_DIR",
        manifests,
    )
    monkeypatch.setattr(
        "datalens.modeling.workflow._load_clean_records",
        _records,
    )
    output_dir = tmp_path / "artifacts"
    baseline_plan = BaselinePlan(schema_version=1, seed=10, defects_per_type=1)

    development = run_development_experiment(
        tracker,
        output_dir=output_dir,
        dataset_plan=_plan(),
        baseline_plan=baseline_plan,
    )
    lock_path = Path(str(development["lock_path"]))
    holdout = evaluate_temporal_holdout(
        lock_path,
        tracker,
        output_dir=output_dir,
        dataset_plan=_plan(),
        baseline_plan=baseline_plan,
    )

    assert tracker.run_names == [
        "isolation-forest-fy2024",
        "temporal-evaluation-fy2025",
    ]
    assert holdout["metrics"]["evaluated_records"] == 32

    vendor_state = output_dir / "models" / "vendor" / "feature-pipeline.json"
    vendor_state.write_text(
        vendor_state.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has changed"):
        evaluate_temporal_holdout(
            lock_path,
            tracker,
            output_dir=output_dir,
            dataset_plan=_plan(),
            baseline_plan=baseline_plan,
        )
