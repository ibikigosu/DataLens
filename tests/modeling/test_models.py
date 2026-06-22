import json

import pandas as pd
import pytest

from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.schemas import VENDOR_FEATURE_SCHEMA
from datalens.modeling.evidence import MAX_EVIDENCE_CHARACTERS
from datalens.modeling.models import (
    IsolationForestSpec,
    load_trained_model,
    save_trained_model,
    train_isolation_forests,
)


def _dataset() -> DevelopmentFeatureDataset:
    return DevelopmentFeatureDataset(
        fiscal_year="2024",
        vendors=pd.DataFrame(
            {
                "_record_id": [f"vendor:{index}" for index in range(20)],
                "vendor_id": [f"V{index}" for index in range(20)],
                "source_transaction_count": range(1, 21),
                "address_variant_count": [1 + index % 3 for index in range(20)],
                "recipient_country_code": ["USA"] * 20,
                "recipient_state_code": ["VA", "MD"] * 10,
                "contracting_officers_determination_of_business_size_code": [
                    "S",
                    "O",
                ]
                * 10,
            }
        ),
        transactions=pd.DataFrame(
            {
                "_record_id": [f"transaction:{index}" for index in range(40)],
                "contract_transaction_unique_key": [f"T{index}" for index in range(40)],
                "federal_action_obligation": [float(index - 5) for index in range(40)],
                "total_dollars_obligated": [float(index * 10) for index in range(40)],
                "number_of_offers_received": [index % 5 for index in range(40)],
                "award_type_code": ["A", "B"] * 20,
                "type_of_contract_pricing_code": ["J", "K"] * 20,
                "action_type_code": ["A", "B"] * 20,
                "product_or_service_code": ["X", "Y"] * 20,
                "naics_code": ["1", "2"] * 20,
                "extent_competed_code": ["A", "B"] * 20,
                "solicitation_procedures_code": ["A", "B"] * 20,
                "type_of_set_aside_code": ["SBA", None] * 20,
            }
        ),
    )


def test_separate_models_are_reproducible_and_emit_bounded_evidence() -> None:
    spec = IsolationForestSpec(n_estimators=25, review_fraction=0.1, seed=7, n_jobs=1)

    first = train_isolation_forests(
        _dataset(),
        vendor_spec=spec,
        transaction_spec=spec,
    )
    second = train_isolation_forests(
        _dataset(),
        vendor_spec=spec,
        transaction_spec=spec,
    )

    first_scores = first.vendor.score(_dataset().vendors)
    second_scores = second.vendor.score(_dataset().vendors)
    assert first_scores["anomaly_score"].tolist() == pytest.approx(
        second_scores["anomaly_score"].tolist()
    )
    assert first.vendor is not first.transaction
    assert first_scores["evidence_json"].str.len().le(MAX_EVIDENCE_CHARACTERS).all()
    evidence = json.loads(first_scores.iloc[0]["evidence_json"])
    assert len(evidence["top_feature_deviations"]) <= 3
    assert "not business severity" in evidence["interpretation"]


def test_candidate_package_round_trips_without_pickle(tmp_path) -> None:
    models = train_isolation_forests(
        _dataset(),
        vendor_spec=IsolationForestSpec(n_estimators=10, n_jobs=1),
        transaction_spec=IsolationForestSpec(n_estimators=10, n_jobs=1),
    )
    package_dir = tmp_path / "vendor"

    save_trained_model(models.vendor, package_dir)
    restored = load_trained_model(package_dir, schema=VENDOR_FEATURE_SCHEMA)

    assert not list(package_dir.glob("*.joblib"))
    assert not list(package_dir.glob("*.pkl"))
    expected = models.vendor.score(_dataset().vendors)
    actual = restored.score(_dataset().vendors)
    assert actual["anomaly_score"].tolist() == pytest.approx(expected["anomaly_score"].tolist())
