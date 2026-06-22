from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from datalens.data.records import add_record_ids
from datalens.features.pipeline import FeatureTable
from datalens.modeling.models import ModelFamily, ModelSpec
from datalens.modeling.workflow import PeriodData, run_experiment


def _vendors(prefix: str, count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": [f"{prefix}:V{index}" for index in range(count)],
            "source_transaction_count": list(range(1, count + 1)),
            "address_variant_count": [1 + index % 3 for index in range(count)],
            "recipient_country_code": ["USA"] * count,
            "recipient_state_code": ["VA", "MD"] * (count // 2),
            "contracting_officers_determination_of_business_size_code": [
                "S",
                "O",
            ]
            * (count // 2),
        }
    )


def _transactions(prefix: str, count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_transaction_unique_key": [f"{prefix}:T{index}" for index in range(count)],
            "federal_action_obligation": [float(index - 5) for index in range(count)],
            "total_dollars_obligated": [float(index * 10) for index in range(count)],
            "number_of_offers_received": [index % 5 for index in range(count)],
            "award_type_code": ["A", "B"] * (count // 2),
            "type_of_contract_pricing_code": ["J", "K"] * (count // 2),
            "action_type_code": ["A", "B"] * (count // 2),
            "product_or_service_code": ["X", "Y"] * (count // 2),
            "naics_code": ["1", "2"] * (count // 2),
            "extent_competed_code": ["A", "B"] * (count // 2),
            "solicitation_procedures_code": ["A", "B"] * (count // 2),
            "type_of_set_aside_code": ["SBA", None] * (count // 2),
        }
    )


def _period(fiscal_year: str, role: str) -> PeriodData:
    clean_vendors = _vendors(f"FY{fiscal_year}")
    clean_transactions = _transactions(f"FY{fiscal_year}")
    defective_vendors, defective_transactions = add_record_ids(
        clean_vendors,
        clean_transactions,
    )
    defective_vendors.loc[0, "recipient_state_code"] = "ZZ"
    defective_transactions.loc[0, "number_of_offers_received"] = -10
    labels = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": [
                defective_vendors.loc[0, "_record_id"],
                defective_transactions.loc[0, "_record_id"],
            ],
            "severity": ["critical", "high"],
        }
    )
    deterministic_findings = pd.DataFrame(
        {
            "target_table": ["vendor", "transaction"],
            "record_id": labels["record_id"],
            "issue_type": ["invalid_domestic_state", "negative_offer_count"],
            "severity": labels["severity"],
            "risk_score": [100, 75],
        }
    )
    return PeriodData(
        fiscal_year=fiscal_year,
        role=role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defective_vendors=defective_vendors,
        defective_transactions=defective_transactions,
        labels=labels,
        deterministic_findings=deterministic_findings,
        dataset_identity={"fiscal_year": int(fiscal_year), "test_fixture": True},
    )


def _model_specs() -> dict[str, dict[FeatureTable, ModelSpec]]:
    return {
        ModelFamily.ISOLATION_FOREST.value: {
            table: ModelSpec(
                family=ModelFamily.ISOLATION_FOREST,
                review_fraction=0.2,
                seed=42,
                parameters={"n_estimators": 20, "n_jobs": 1},
            )
            for table in FeatureTable
        },
        ModelFamily.LOCAL_OUTLIER_FACTOR.value: {
            table: ModelSpec(
                family=ModelFamily.LOCAL_OUTLIER_FACTOR,
                review_fraction=0.2,
                seed=42,
                parameters={"n_neighbors": 5, "n_jobs": 1},
            )
            for table in FeatureTable
        },
    }


def test_end_to_end_workflow_tracks_models_and_keeps_holdout_evaluation_only(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"
    output_dir = tmp_path / "artifacts"

    summary = run_experiment(
        _period("2024", "development"),
        _period("2025", "temporal_holdout"),
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        experiment_name="test-model-comparison",
        model_specs=_model_specs(),
        top_k=2,
    )

    assert summary["selection_policy"]["holdout_used_for_selection"] is False
    assert summary["selection_policy"]["development_fiscal_year"] == "2024"
    assert summary["selection_policy"]["temporal_evaluation_fiscal_year"] == "2025"
    assert summary["promotion"]["gates"]["deterministic_critical_findings_preserved"]
    assert (output_dir / "comparison-summary.json").exists()
    assert (output_dir / "fy2025-bounded-anomaly-evidence.json").exists()
    assert (output_dir / "models" / "isolation_forest" / "vendor" / "bundle.json").exists()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-model-comparison")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 7
