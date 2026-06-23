"""MLflow candidate registration and application activation evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlflow import MlflowClient


@dataclass(frozen=True)
class CandidateRegistration:
    """One table-specific candidate model version in MLflow."""

    table: str
    model_family: str
    registered_model_name: str
    version: str


def register_candidate(
    client: MlflowClient,
    *,
    table: str,
    model_family: str,
    registered_model_name: str,
    version: str,
    promotion: dict[str, Any],
) -> CandidateRegistration:
    """Tag and alias one selected candidate without silently activating it."""
    status = "promoted" if promotion["promoted"] else "rejected"
    failed_gates = sorted(gate for gate, passed in promotion["gates"].items() if not passed)
    client.set_registered_model_alias(
        registered_model_name,
        "candidate",
        version,
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "datalens.status",
        status,
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "datalens.table",
        table,
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "datalens.model_family",
        model_family,
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "datalens.failed_gates",
        ",".join(failed_gates),
    )
    if promotion["promoted"]:
        client.set_registered_model_alias(
            registered_model_name,
            "active",
            version,
        )
    return CandidateRegistration(
        table=table,
        model_family=model_family,
        registered_model_name=registered_model_name,
        version=version,
    )


def build_registry_manifest(
    *,
    active_model_version: str,
    candidates: tuple[CandidateRegistration, ...],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    """Build an auditable distinction between active and candidate models."""
    failed_gates = sorted(gate for gate, passed in promotion["gates"].items() if not passed)
    if promotion["promoted"]:
        reason = "All configured promotion gates passed."
    else:
        reason = f"Candidate rejected because gates failed: {', '.join(failed_gates)}."
    return {
        "schema_version": 1,
        "active": {
            "model_version": active_model_version,
            "kind": "deterministic_rules",
            "status": "active",
            "reason": (
                "The deterministic baseline remains active until every candidate "
                "promotion gate passes."
            ),
        },
        "candidates": [
            {
                **asdict(candidate),
                "alias": "candidate",
                "status": "promoted" if promotion["promoted"] else "rejected",
                "reason": reason,
            }
            for candidate in candidates
        ],
        "promotion": promotion,
    }


def write_registry_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
