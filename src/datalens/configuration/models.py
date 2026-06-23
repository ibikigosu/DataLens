"""Validated configuration models for schemas, models, and application settings."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from datalens.baseline.issues import ISSUES

TableName = Literal["vendor", "transaction"]


class DataType(StrEnum):
    """Supported procurement input types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    DATETIME = "datetime"


class StrictModel(BaseModel):
    """Base model that rejects misspelled or undeclared settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ColumnConfiguration(StrictModel):
    """One declared input column."""

    name: str = Field(min_length=1)
    data_type: DataType
    required: bool = True
    nullable: bool = True


class TableConfiguration(StrictModel):
    """Input contract for one procurement table."""

    business_key: str = Field(min_length=1)
    columns: tuple[ColumnConfiguration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_columns(self) -> TableConfiguration:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Table column names must be unique")
        if self.business_key not in names:
            raise ValueError("The table business key must be declared as a column")
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)


class RelationshipConfiguration(StrictModel):
    """One declared relationship between procurement tables."""

    name: str = Field(min_length=1)
    from_table: TableName
    from_column: str = Field(min_length=1)
    to_table: TableName
    to_column: str = Field(min_length=1)


class ProcurementSchema(StrictModel):
    """Approved schema, relationships, and quality scoring weights."""

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    vendor: TableConfiguration
    transaction: TableConfiguration
    relationships: tuple[RelationshipConfiguration, ...]
    scoring_weights: dict[str, int]

    @model_validator(mode="after")
    def validate_contract(self) -> ProcurementSchema:
        configured_issues = set(self.scoring_weights)
        canonical_issues = set(ISSUES)
        if configured_issues != canonical_issues:
            missing = sorted(canonical_issues - configured_issues)
            unexpected = sorted(configured_issues - canonical_issues)
            raise ValueError(
                f"Scoring weights must match canonical issues; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if any(weight < 0 or weight > 100 for weight in self.scoring_weights.values()):
            raise ValueError("Scoring weights must be between 0 and 100")
        for relationship in self.relationships:
            source = self.table(relationship.from_table)
            target = self.table(relationship.to_table)
            if relationship.from_column not in {
                column.name for column in source.columns
            } or relationship.to_column not in {column.name for column in target.columns}:
                raise ValueError(
                    f"Relationship {relationship.name!r} references undeclared columns"
                )
        return self

    def table(self, table: TableName) -> TableConfiguration:
        return self.vendor if table == "vendor" else self.transaction

    @property
    def transaction_vendor_relationship(self) -> RelationshipConfiguration:
        matches = [
            relationship
            for relationship in self.relationships
            if relationship.from_table == "transaction" and relationship.to_table == "vendor"
        ]
        if len(matches) != 1:
            raise ValueError(
                "The schema must declare exactly one transaction-to-vendor relationship"
            )
        return matches[0]


class TableModelConfiguration(StrictModel):
    """Parameters for one model family and table."""

    review_fraction: float = Field(gt=0, lt=1)
    parameters: dict[str, Any]


class FamilyModelConfiguration(StrictModel):
    """Vendor and transaction parameters for one model family."""

    vendor: TableModelConfiguration
    transaction: TableModelConfiguration

    def for_table(self, table: TableName) -> TableModelConfiguration:
        return self.vendor if table == "vendor" else self.transaction


class PromotionConfiguration(StrictModel):
    """Auditable model promotion gates."""

    minimum_top_k_ratio: float = Field(ge=0)
    minimum_macro_f1_ratio: float = Field(ge=0)
    maximum_false_alarm_increase_per_1000: float = Field(ge=0)
    required_guarded_high_critical_recall: float = Field(ge=0, le=1)


class FeedbackRerankerConfiguration(StrictModel):
    """Training sufficiency rules for the supervised feedback reranker."""

    minimum_examples: int = Field(ge=10)
    minimum_class_examples: int = Field(ge=3)
    validation_fraction: float = Field(gt=0.1, lt=0.5)


class ModelConfiguration(StrictModel):
    """Versioned feature, model, and promotion policy."""

    model_version: str = Field(min_length=1)
    active_model_version: str = Field(min_length=1)
    registry_name_prefix: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    feature_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    top_k: int = Field(gt=0)
    families: dict[str, FamilyModelConfiguration] = Field(min_length=1)
    promotion: PromotionConfiguration
    feedback_reranker: FeedbackRerankerConfiguration


class ApplicationSettings(BaseSettings):
    """Application settings with environment variables taking precedence."""

    model_config = SettingsConfigDict(
        env_prefix="DATALENS_",
        env_file=".env",
        extra="forbid",
        frozen=True,
    )

    environment: str = "local"
    schema_config_path: Path
    model_config_path: Path
    dataset_config_path: Path
    baseline_config_path: Path
    artifact_dir: Path
    manifest_dir: Path
    processed_data_dir: Path
    database_url: str
    mlflow_tracking_uri: str
    api_host: str
    api_port: int = Field(gt=0, le=65535)
    api_base_url: str
    maximum_upload_bytes: int = Field(gt=0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return env_settings, dotenv_settings, init_settings, file_secret_settings
