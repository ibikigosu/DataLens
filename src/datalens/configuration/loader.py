"""Load versioned configuration and resolve repository-relative paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from datalens.configuration.models import (
    ApplicationSettings,
    ModelConfiguration,
    ProcurementSchema,
)
from datalens.paths import PROJECT_ROOT

DEFAULT_APPLICATION_CONFIG = PROJECT_ROOT / "config" / "application" / "default.json"
ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


@dataclass(frozen=True)
class RuntimeConfig:
    """Complete validated runtime configuration."""

    settings: ApplicationSettings
    schema: ProcurementSchema
    model: ModelConfiguration

    @property
    def versions(self) -> dict[str, str]:
        return {
            "schema": self.schema.schema_version,
            "feature": self.model.feature_version,
            "model": self.model.model_version,
        }


def load_runtime_config(
    application_path: Path = DEFAULT_APPLICATION_CONFIG,
) -> RuntimeConfig:
    """Load application defaults, environment overrides, and versioned policies."""
    application_payload = _read_json(application_path)
    try:
        settings = ApplicationSettings(**application_payload)
    except ValidationError as error:
        raise ValueError(
            f"Invalid application configuration at {application_path}: {error}"
        ) from error

    path_fields = (
        "schema_config_path",
        "model_config_path",
        "dataset_config_path",
        "baseline_config_path",
        "artifact_dir",
        "manifest_dir",
        "processed_data_dir",
    )
    settings = settings.model_copy(
        update={field: _resolve(getattr(settings, field)) for field in path_fields}
    )
    return RuntimeConfig(
        settings=settings,
        schema=_load_model(settings.schema_config_path, ProcurementSchema),
        model=_load_model(settings.model_config_path, ModelConfiguration),
    )


def _load_model(path: Path, model: type[ConfigModel]) -> ConfigModel:
    try:
        return model.model_validate(_read_json(path))
    except ValidationError as error:
        raise ValueError(f"Invalid configuration at {path}: {error}") from error


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Configuration file is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return payload


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path
