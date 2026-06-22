import pytest
from mlflow import MlflowClient

from datalens.modeling.tracking import ExperimentTracker, require_postgresql_tracking_uri


def test_production_tracking_requires_postgresql(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        require_postgresql_tracking_uri()
    with pytest.raises(ValueError, match="PostgreSQL"):
        require_postgresql_tracking_uri("sqlite:///mlflow.db")


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "postgresql://user:password@localhost/datalens",
        "postgresql+psycopg://user:password@localhost/datalens",
    ],
)
def test_postgresql_tracking_uris_are_accepted(tracking_uri: str) -> None:
    assert require_postgresql_tracking_uri(tracking_uri) == tracking_uri


def test_tracker_logs_to_an_explicit_test_database(tmp_path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}\n", encoding="utf-8")
    tracker = ExperimentTracker(
        tracking_uri,
        experiment_name="test-isolation-forest",
        artifact_root=tmp_path / "artifacts",
    )

    run_id = tracker.log_run(
        run_name="test-run",
        params={"seed": 42},
        metrics={"precision": 0.5},
        artifacts=(artifact,),
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["precision"] == 0.5
