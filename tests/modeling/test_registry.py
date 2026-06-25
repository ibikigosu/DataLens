from pathlib import Path

from datalens.modeling.registry import (
    CandidateRegistration,
    build_registry_manifest,
    register_candidate,
    write_registry_manifest,
)


class RecordingMlflowClient:
    def __init__(self) -> None:
        self.aliases: list[tuple[str, str, str]] = []
        self.tags: list[tuple[str, str, str, str]] = []

    def set_registered_model_alias(
        self, registered_model_name: str, alias: str, version: str
    ) -> None:
        self.aliases.append((registered_model_name, alias, version))

    def set_model_version_tag(
        self, registered_model_name: str, version: str, key: str, value: str
    ) -> None:
        self.tags.append((registered_model_name, version, key, value))


def test_register_candidate_tags_rejected_candidate_without_active_alias() -> None:
    client = RecordingMlflowClient()
    promotion = {
        "promoted": False,
        "gates": {"top_50_precision": False, "temporal_holdout": True},
    }

    registration = register_candidate(
        client,
        table="vendor",
        model_family="isolation_forest",
        registered_model_name="datalens-vendor-anomaly",
        version="7",
        promotion=promotion,
    )

    assert registration == CandidateRegistration(
        table="vendor",
        model_family="isolation_forest",
        registered_model_name="datalens-vendor-anomaly",
        version="7",
    )
    assert client.aliases == [("datalens-vendor-anomaly", "candidate", "7")]
    assert (
        "datalens-vendor-anomaly",
        "7",
        "datalens.failed_gates",
        "top_50_precision",
    ) in client.tags


def test_register_candidate_marks_promoted_candidate_active() -> None:
    client = RecordingMlflowClient()

    register_candidate(
        client,
        table="transaction",
        model_family="random_forest",
        registered_model_name="datalens-transaction-anomaly",
        version="3",
        promotion={"promoted": True, "gates": {"top_50_precision": True}},
    )

    assert client.aliases == [
        ("datalens-transaction-anomaly", "candidate", "3"),
        ("datalens-transaction-anomaly", "active", "3"),
    ]


def test_registry_manifest_explains_rejected_candidate() -> None:
    manifest = build_registry_manifest(
        active_model_version="deterministic-baseline-v1",
        candidates=(
            CandidateRegistration(
                table="vendor",
                model_family="isolation_forest",
                registered_model_name="datalens-vendor-anomaly",
                version="7",
            ),
        ),
        promotion={
            "promoted": False,
            "gates": {"temporal_holdout": False, "top_50_precision": True},
        },
    )

    assert manifest["active"]["status"] == "active"
    assert manifest["candidates"][0]["status"] == "rejected"
    assert "temporal_holdout" in manifest["candidates"][0]["reason"]


def test_write_registry_manifest_creates_parent_directory(tmp_path: Path) -> None:
    manifest = {"schema_version": 1}
    path = tmp_path / "nested" / "model-registry.json"

    write_registry_manifest(path, manifest)

    assert path.read_text(encoding="utf-8") == '{\n  "schema_version": 1\n}\n'
