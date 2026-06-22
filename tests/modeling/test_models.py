import json

import pandas as pd
import pytest

from datalens.features.pipeline import (
    CategoricalFeature,
    FeatureSchema,
    FeatureTable,
    NumericFeature,
)
from datalens.modeling.evidence import MAX_EVIDENCE_CHARACTERS
from datalens.modeling.models import (
    ModelFamily,
    ModelSpec,
    load_model_bundle,
    save_model_bundle,
    train_table_model,
)


def _schema() -> FeatureSchema:
    return FeatureSchema(
        table=FeatureTable.VENDOR,
        record_id_column="_record_id",
        numeric_features=(NumericFeature(source_column="amount", feature_name="amount"),),
        categorical_features=(CategoricalFeature(source_column="state", feature_name="state"),),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_record_id": [f"vendor:{index}" for index in range(20)],
            "amount": [float(index) for index in range(20)],
            "state": ["VA"] * 10 + ["MD"] * 10,
        }
    )


def _spec() -> ModelSpec:
    return ModelSpec(
        family=ModelFamily.ISOLATION_FOREST,
        review_fraction=0.1,
        seed=7,
        parameters={"n_estimators": 25, "n_jobs": 1},
    )


def test_training_and_scoring_are_reproducible_with_bounded_evidence() -> None:
    first = train_table_model(
        _frame(),
        schema=_schema(),
        spec=_spec(),
        fit_fiscal_year=2024,
        period_role="development",
    )
    second = train_table_model(
        _frame(),
        schema=_schema(),
        spec=_spec(),
        fit_fiscal_year=2024,
        period_role="development",
    )

    first_scores = first.score(_frame())
    second_scores = second.score(_frame())

    assert first_scores["anomaly_score"].tolist() == pytest.approx(
        second_scores["anomaly_score"].tolist()
    )
    assert first_scores["record_id"].tolist() == second_scores["record_id"].tolist()
    assert first_scores["rank_percentile"].between(0, 1).all()
    assert first_scores["evidence"].str.len().le(MAX_EVIDENCE_CHARACTERS).all()
    evidence = json.loads(first_scores.iloc[0]["evidence"])
    assert len(evidence["top_feature_deviations"]) <= 3
    assert "not business severity" in evidence["interpretation"]


def test_model_bundle_round_trips_for_later_scoring_integration(tmp_path) -> None:
    bundle = train_table_model(
        _frame(),
        schema=_schema(),
        spec=_spec(),
        fit_fiscal_year=2024,
        period_role="development",
    )
    path = tmp_path / "vendor.joblib"

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)

    expected = bundle.score(_frame())
    actual = loaded.score(_frame())
    assert actual["anomaly_score"].tolist() == pytest.approx(expected["anomaly_score"].tolist())
    assert loaded.fit_fiscal_year == "2024"


def test_training_rejects_temporal_holdout_data() -> None:
    with pytest.raises(ValueError, match="development period"):
        train_table_model(
            _frame(),
            schema=_schema(),
            spec=_spec(),
            fit_fiscal_year=2025,
            period_role="temporal_holdout",
        )
