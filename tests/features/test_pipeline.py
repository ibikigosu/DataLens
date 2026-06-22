import numpy as np
import pandas as pd
import pytest

from datalens.data.plan import load_dataset_plan
from datalens.features.builders import (
    build_transaction_features,
    build_vendor_features,
)
from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import (
    CategoricalFeature,
    FeaturePipeline,
    FeatureSchema,
    FeatureTable,
    NumericFeature,
    NumericTransform,
)
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA


def _schema() -> FeatureSchema:
    return FeatureSchema(
        table=FeatureTable.VENDOR,
        record_id_column="_record_id",
        numeric_features=(
            NumericFeature(
                source_column="source_transaction_count",
                feature_name="transaction_count",
                transform=NumericTransform.SIGNED_LOG1P,
            ),
        ),
        categorical_features=(
            CategoricalFeature(
                source_column="recipient_state_code",
                feature_name="state",
            ),
        ),
    )


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_record_id": ["V1", "V2", "V3", "V4"],
            "source_transaction_count": pd.Series([1, 3, 10, None], dtype="Float64"),
            "recipient_state_code": pd.Series(["VA", "VA", "MD", None], dtype="string"),
        }
    )


def test_fit_transform_produces_finite_model_matrix_with_stable_names() -> None:
    pipeline = FeaturePipeline(_schema())

    result = pipeline.fit_transform(_training_frame())

    assert result.table is FeatureTable.VENDOR
    assert result.record_ids.tolist() == ["V1", "V2", "V3", "V4"]
    assert result.feature_names == (
        "transaction_count__scaled",
        "transaction_count__missing",
        "state__frequency",
        "state__missing",
    )
    assert result.values.columns.tolist() == list(result.feature_names)
    assert result.values.shape == (4, 4)
    assert np.isfinite(result.values.to_numpy()).all()
    assert result.values["transaction_count__missing"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert result.values["state__frequency"].tolist() == [0.5, 0.5, 0.25, 0.25]
    assert result.values["state__missing"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_transform_reuses_training_statistics_and_maps_unseen_categories_to_zero() -> None:
    pipeline = FeaturePipeline(_schema()).fit(_training_frame())
    scoring_frame = pd.DataFrame(
        {
            "_record_id": ["S1", "S2"],
            "source_transaction_count": pd.Series([3, None], dtype="Float64"),
            "recipient_state_code": pd.Series(["VA", "ZZ"], dtype="string"),
        }
    )

    result = pipeline.transform(scoring_frame)

    assert result.values.loc[0, "transaction_count__scaled"] == pytest.approx(0.0)
    assert result.values.loc[1, "transaction_count__scaled"] == pytest.approx(0.0)
    assert result.values["state__frequency"].tolist() == [0.5, 0.0]


def test_fit_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty"):
        FeaturePipeline(_schema()).fit(_training_frame().iloc[0:0])


def test_pipeline_reports_missing_columns_and_invalid_numeric_values() -> None:
    pipeline = FeaturePipeline(_schema())

    with pytest.raises(ValueError, match="recipient_state_code"):
        pipeline.fit(
            _training_frame().drop(columns="recipient_state_code"),
        )

    invalid = _training_frame()
    invalid["source_transaction_count"] = invalid["source_transaction_count"].astype(object)
    invalid.loc[0, "source_transaction_count"] = "not-a-number"
    with pytest.raises(ValueError, match="source_transaction_count"):
        pipeline.fit(invalid)


@pytest.mark.parametrize(
    ("record_ids", "message"),
    [
        (["V1", None, "V3", "V4"], "missing"),
        (["V1", "V1", "V3", "V4"], "unique"),
    ],
)
def test_pipeline_requires_stable_record_identity(
    record_ids: list[str | None],
    message: str,
) -> None:
    frame = _training_frame()
    frame["_record_id"] = pd.Series(record_ids, dtype="string")

    with pytest.raises(ValueError, match=message):
        FeaturePipeline(_schema()).fit(frame)


def test_transform_requires_a_fitted_pipeline() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        FeaturePipeline(_schema()).transform(_training_frame())


def test_fitted_pipeline_state_round_trips_without_refitting() -> None:
    pipeline = FeaturePipeline(_schema()).fit(_training_frame())

    restored = FeaturePipeline.from_state(_schema(), pipeline.export_state())

    expected = pipeline.transform(_training_frame())
    actual = restored.transform(_training_frame())
    pd.testing.assert_frame_equal(actual.values, expected.values)
    pd.testing.assert_series_equal(actual.record_ids, expected.record_ids)


def test_default_schemas_keep_vendor_and_transaction_features_separate() -> None:
    assert VENDOR_FEATURE_SCHEMA.table is FeatureTable.VENDOR
    assert TRANSACTION_FEATURE_SCHEMA.table is FeatureTable.TRANSACTION
    assert VENDOR_FEATURE_SCHEMA.record_id_column == "_record_id"
    assert TRANSACTION_FEATURE_SCHEMA.record_id_column == "_record_id"
    assert len(VENDOR_FEATURE_SCHEMA.output_feature_names) == 10
    assert len(TRANSACTION_FEATURE_SCHEMA.output_feature_names) == 22
    assert not (
        set(VENDOR_FEATURE_SCHEMA.output_feature_names)
        & set(TRANSACTION_FEATURE_SCHEMA.output_feature_names)
    )


def test_development_dataset_rejects_temporal_holdout() -> None:
    plan = load_dataset_plan()

    with pytest.raises(ValueError, match="development period"):
        DevelopmentFeatureDataset.from_records(
            _vendor_source_frame(),
            _transaction_source_frame(),
            fiscal_year=2025,
            dataset_plan=plan,
        )


def test_development_dataset_builds_both_feature_tables() -> None:
    dataset = DevelopmentFeatureDataset.from_records(
        _vendor_source_frame(),
        _transaction_source_frame(),
        fiscal_year=2024,
    )

    vendor_matrix = FeaturePipeline(VENDOR_FEATURE_SCHEMA).fit_transform(dataset.vendors)
    transaction_matrix = FeaturePipeline(TRANSACTION_FEATURE_SCHEMA).fit_transform(
        dataset.transactions
    )

    assert dataset.fiscal_year == "2024"
    assert vendor_matrix.values.shape == (2, 10)
    assert transaction_matrix.values.shape == (2, 22)


def test_table_builders_allow_duplicate_business_keys_and_support_real_schemas() -> None:
    vendor_features = build_vendor_features(_vendor_source_frame())
    transaction_features = build_transaction_features(_transaction_source_frame())

    vendor_matrix = FeaturePipeline(VENDOR_FEATURE_SCHEMA).fit_transform(vendor_features)
    transaction_matrix = FeaturePipeline(TRANSACTION_FEATURE_SCHEMA).fit_transform(
        transaction_features
    )

    assert vendor_matrix.values.shape == (2, 10)
    assert transaction_matrix.values.shape == (2, 22)
    assert vendor_matrix.record_ids.is_unique
    assert transaction_matrix.record_ids.is_unique


def test_table_builders_preserve_existing_controlled_defect_record_ids() -> None:
    vendors = _vendor_source_frame()
    vendors["_record_id"] = ["controlled:vendor:1", "controlled:vendor:2"]
    transactions = _transaction_source_frame()
    transactions["_record_id"] = ["controlled:transaction:1", "controlled:transaction:2"]

    vendor_features = build_vendor_features(vendors)
    transaction_features = build_transaction_features(transactions)

    assert vendor_features["_record_id"].tolist() == [
        "controlled:vendor:1",
        "controlled:vendor:2",
    ]
    assert transaction_features["_record_id"].tolist() == [
        "controlled:transaction:1",
        "controlled:transaction:2",
    ]


def _vendor_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": ["UEI:A", "UEI:A"],
            "source_transaction_count": [2, 3],
            "address_variant_count": [1, 2],
            "recipient_country_code": ["USA", "USA"],
            "recipient_state_code": ["VA", "MD"],
            "contracting_officers_determination_of_business_size_code": ["S", "O"],
        }
    )


def _transaction_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_transaction_unique_key": ["T1", "T1"],
            "federal_action_obligation": [10.0, -5.0],
            "total_dollars_obligated": [100.0, 95.0],
            "number_of_offers_received": [2, None],
            "award_type_code": ["A", "B"],
            "type_of_contract_pricing_code": ["J", "K"],
            "action_type_code": ["A", None],
            "product_or_service_code": ["X", "Y"],
            "naics_code": ["1", "2"],
            "extent_competed_code": ["A", "B"],
            "solicitation_procedures_code": ["A", "B"],
            "type_of_set_aside_code": [None, "SBA"],
        }
    )
