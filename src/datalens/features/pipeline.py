"""Typed, leakage-resistant transformations for model-ready feature matrices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class FeatureTable(StrEnum):
    """Record table represented by a feature matrix."""

    VENDOR = "vendor"
    TRANSACTION = "transaction"


class NumericTransform(StrEnum):
    """Supported transformations applied before robust scaling."""

    IDENTITY = "identity"
    SIGNED_LOG1P = "signed_log1p"


@dataclass(frozen=True)
class NumericFeature:
    """One numeric source transformed into a scaled value and missingness signal."""

    source_column: str
    feature_name: str
    transform: NumericTransform = NumericTransform.IDENTITY


@dataclass(frozen=True)
class CategoricalFeature:
    """One categorical source represented by learned frequency and missingness."""

    source_column: str
    feature_name: str


@dataclass(frozen=True)
class FeatureSchema:
    """Versionable declaration of one table's model feature inputs."""

    table: FeatureTable
    record_id_column: str
    schema_version: int = 1
    numeric_features: tuple[NumericFeature, ...] = ()
    categorical_features: tuple[CategoricalFeature, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("Feature schema version must be positive")
        feature_names = [
            feature.feature_name for feature in (*self.numeric_features, *self.categorical_features)
        ]
        if not feature_names:
            raise ValueError("A feature schema must declare at least one feature")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return input columns required to fit or transform this schema."""
        return (self.record_id_column, *self.source_columns)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Return source columns consumed by statistical preprocessing."""
        source_columns = (
            *(feature.source_column for feature in self.numeric_features),
            *(feature.source_column for feature in self.categorical_features),
        )
        return tuple(dict.fromkeys(source_columns))

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        """Return the stable model matrix column order."""
        numeric_names = (
            output_name
            for feature in self.numeric_features
            for output_name in (
                f"{feature.feature_name}__scaled",
                f"{feature.feature_name}__missing",
            )
        )
        categorical_names = (
            output_name
            for feature in self.categorical_features
            for output_name in (
                f"{feature.feature_name}__frequency",
                f"{feature.feature_name}__missing",
            )
        )
        return (*numeric_names, *categorical_names)


@dataclass(frozen=True)
class FeatureMatrix:
    """Model-ready values paired with immutable source record identifiers."""

    table: FeatureTable
    record_ids: pd.Series
    values: pd.DataFrame
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.record_ids) != len(self.values):
            raise ValueError("Record identifiers and feature values must have equal length")
        if tuple(self.values.columns) != self.feature_names:
            raise ValueError("Feature matrix columns must match the declared feature names")
        if not np.isfinite(self.values.to_numpy(dtype=float)).all():
            raise ValueError("Feature matrix values must all be finite")


@dataclass(frozen=True)
class _NumericFit:
    median: float
    scale: float


@dataclass(frozen=True)
class _CategoricalFit:
    frequencies: dict[str, float]
    missing_frequency: float


class FeaturePipeline:
    """Fit preprocessing on development data and reuse it unchanged for scoring."""

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema
        self._numeric_fits: dict[str, _NumericFit] | None = None
        self._categorical_fits: dict[str, _CategoricalFit] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._numeric_fits is not None and self._categorical_fits is not None

    def fit(self, frame: pd.DataFrame) -> FeaturePipeline:
        """Learn transformation statistics from a validated feature frame."""
        if frame.empty:
            raise ValueError("Feature pipelines cannot be fitted on an empty frame")

        self._validate_frame(frame)
        numeric_fits: dict[str, _NumericFit] = {}
        for feature in self.schema.numeric_features:
            values = self._numeric_values(frame, feature)
            transformed = self._transform_numeric(values.dropna(), feature.transform)
            if transformed.empty:
                raise ValueError(
                    f"Numeric feature {feature.source_column!r} has no non-missing training values"
                )
            median = float(transformed.median())
            scale = float(transformed.quantile(0.75) - transformed.quantile(0.25))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            numeric_fits[feature.feature_name] = _NumericFit(median=median, scale=scale)

        categorical_fits: dict[str, _CategoricalFit] = {}
        for feature in self.schema.categorical_features:
            values, missing = self._categorical_values(frame, feature)
            frequencies = (values.value_counts(dropna=True) / len(values)).to_dict()
            categorical_fits[feature.feature_name] = _CategoricalFit(
                frequencies={
                    str(value): float(frequency) for value, frequency in frequencies.items()
                },
                missing_frequency=float(missing.mean()),
            )

        self._numeric_fits = numeric_fits
        self._categorical_fits = categorical_fits
        return self

    def transform(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Apply fitted transformations without learning from scoring data."""
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline must be fitted before transform")
        self._validate_frame(frame)
        assert self._numeric_fits is not None
        assert self._categorical_fits is not None

        columns: dict[str, pd.Series] = {}
        for feature in self.schema.numeric_features:
            values = self._numeric_values(frame, feature)
            missing = values.isna()
            fit = self._numeric_fits[feature.feature_name]
            transformed = self._transform_numeric(values, feature.transform)
            columns[f"{feature.feature_name}__scaled"] = (
                transformed.fillna(fit.median) - fit.median
            ) / fit.scale
            columns[f"{feature.feature_name}__missing"] = missing.astype(float)

        for feature in self.schema.categorical_features:
            values, missing = self._categorical_values(frame, feature)
            fit = self._categorical_fits[feature.feature_name]
            frequencies = values.map(fit.frequencies).fillna(0.0).astype(float)
            columns[f"{feature.feature_name}__frequency"] = frequencies.mask(
                missing,
                fit.missing_frequency,
            )
            columns[f"{feature.feature_name}__missing"] = missing.astype(float)

        feature_names = self.schema.output_feature_names
        values = (
            pd.DataFrame(columns, index=frame.index).loc[:, feature_names].reset_index(drop=True)
        )
        record_ids = self._normalized_record_ids(frame[self.schema.record_id_column]).reset_index(
            drop=True
        )
        return FeatureMatrix(
            table=self.schema.table,
            record_ids=record_ids,
            values=values,
            feature_names=feature_names,
        )

    def fit_transform(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Fit on a validated feature frame and transform that same frame."""
        return self.fit(frame).transform(frame)

    def metadata(self) -> dict[str, Any]:
        """Return bounded fitted-pipeline metadata for experiment tracking."""
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline must be fitted before exporting metadata")
        assert self._numeric_fits is not None
        assert self._categorical_fits is not None
        return {
            "table": self.schema.table.value,
            "schema_version": self.schema.schema_version,
            "record_id_column": self.schema.record_id_column,
            "feature_names": list(self.schema.output_feature_names),
            "numeric_fits": {
                feature_name: {
                    "median": fit.median,
                    "scale": fit.scale,
                }
                for feature_name, fit in self._numeric_fits.items()
            },
            "categorical_fits": {
                feature_name: {
                    "known_category_count": len(fit.frequencies),
                    "missing_frequency": fit.missing_frequency,
                }
                for feature_name, fit in self._categorical_fits.items()
            },
        }

    def export_state(self) -> dict[str, Any]:
        """Return complete JSON-compatible fitted state for safe model artifacts."""
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline must be fitted before exporting state")
        assert self._numeric_fits is not None
        assert self._categorical_fits is not None
        return {
            "schema": {
                "table": self.schema.table.value,
                "record_id_column": self.schema.record_id_column,
                "schema_version": self.schema.schema_version,
                "numeric_features": [
                    {
                        "source_column": feature.source_column,
                        "feature_name": feature.feature_name,
                        "transform": feature.transform.value,
                    }
                    for feature in self.schema.numeric_features
                ],
                "categorical_features": [
                    {
                        "source_column": feature.source_column,
                        "feature_name": feature.feature_name,
                    }
                    for feature in self.schema.categorical_features
                ],
            },
            "numeric_fits": {
                feature_name: {
                    "median": fit.median,
                    "scale": fit.scale,
                }
                for feature_name, fit in self._numeric_fits.items()
            },
            "categorical_fits": {
                feature_name: {
                    "frequencies": fit.frequencies,
                    "missing_frequency": fit.missing_frequency,
                }
                for feature_name, fit in self._categorical_fits.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FeaturePipeline:
        """Restore and validate complete fitted state from a data-only artifact."""
        schema_state = _mapping_value(state, "schema")
        numeric_features = tuple(
            NumericFeature(
                source_column=_string_value(feature, "source_column"),
                feature_name=_string_value(feature, "feature_name"),
                transform=NumericTransform(_string_value(feature, "transform")),
            )
            for feature in _mapping_list(schema_state, "numeric_features")
        )
        categorical_features = tuple(
            CategoricalFeature(
                source_column=_string_value(feature, "source_column"),
                feature_name=_string_value(feature, "feature_name"),
            )
            for feature in _mapping_list(schema_state, "categorical_features")
        )
        schema = FeatureSchema(
            table=FeatureTable(_string_value(schema_state, "table")),
            record_id_column=_string_value(schema_state, "record_id_column"),
            schema_version=_integer_value(schema_state, "schema_version"),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        pipeline = cls(schema)

        numeric_state = _mapping_value(state, "numeric_fits")
        expected_numeric = {feature.feature_name for feature in numeric_features}
        if set(numeric_state) != expected_numeric:
            raise ValueError("Numeric fitted state does not match the feature schema")
        pipeline._numeric_fits = {
            feature_name: _NumericFit(
                median=_finite_float(fit, "median"),
                scale=_positive_float(fit, "scale"),
            )
            for feature_name, fit in (
                (name, _mapping_value(numeric_state, name)) for name in expected_numeric
            )
        }

        categorical_state = _mapping_value(state, "categorical_fits")
        expected_categorical = {feature.feature_name for feature in categorical_features}
        if set(categorical_state) != expected_categorical:
            raise ValueError("Categorical fitted state does not match the feature schema")
        categorical_fits: dict[str, _CategoricalFit] = {}
        for feature_name in expected_categorical:
            fit = _mapping_value(categorical_state, feature_name)
            frequency_state = _mapping_value(fit, "frequencies")
            frequencies = {
                str(category): _bounded_fraction(frequency, f"frequency for {category!r}")
                for category, frequency in frequency_state.items()
            }
            if sum(frequencies.values()) > 1.0 + 1e-9:
                raise ValueError("Categorical frequencies cannot sum to more than one")
            categorical_fits[feature_name] = _CategoricalFit(
                frequencies=frequencies,
                missing_frequency=_bounded_fraction(
                    fit.get("missing_frequency"),
                    "missing_frequency",
                ),
            )
        pipeline._categorical_fits = categorical_fits
        return pipeline

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if frame.columns.has_duplicates:
            raise ValueError("Input frame column names must be unique")
        missing_columns = sorted(set(self.schema.required_columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Input frame is missing required columns: {missing_columns}")

        record_ids = self._normalized_record_ids(frame[self.schema.record_id_column])
        if record_ids.isna().any():
            raise ValueError(
                f"Record identity column {self.schema.record_id_column!r} has missing values"
            )
        if not record_ids.is_unique:
            raise ValueError(
                f"Record identity column {self.schema.record_id_column!r} must be unique"
            )

    @staticmethod
    def _normalized_record_ids(series: pd.Series) -> pd.Series:
        normalized = series.astype("string").str.strip()
        return normalized.mask(normalized.eq(""))

    @staticmethod
    def _transform_numeric(
        values: pd.Series,
        transform: NumericTransform,
    ) -> pd.Series:
        numeric = values.astype(float)
        if transform is NumericTransform.IDENTITY:
            return numeric
        if transform is NumericTransform.SIGNED_LOG1P:
            return np.sign(numeric) * np.log1p(numeric.abs())
        raise ValueError(f"Unsupported numeric transform: {transform}")

    @staticmethod
    def _numeric_values(frame: pd.DataFrame, feature: NumericFeature) -> pd.Series:
        source = frame[feature.source_column]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"Numeric feature {feature.source_column!r} contains non-numeric values"
            )
        finite_values = numeric.dropna().astype(float)
        if not np.isfinite(finite_values).all():
            raise ValueError(
                f"Numeric feature {feature.source_column!r} must contain finite values"
            )
        return numeric

    @staticmethod
    def _categorical_values(
        frame: pd.DataFrame,
        feature: CategoricalFeature,
    ) -> tuple[pd.Series, pd.Series]:
        values = frame[feature.source_column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        missing = values.isna()
        return values, missing


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_list(mapping: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _string_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer_value(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _finite_float(mapping: Mapping[str, Any], key: str) -> float:
    return _validated_float(mapping.get(key), key)


def _positive_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _validated_float(mapping.get(key), key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _bounded_fraction(value: Any, field_name: str) -> float:
    number = _validated_float(value, field_name)
    if not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between zero and one")
    return number


def _validated_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number
